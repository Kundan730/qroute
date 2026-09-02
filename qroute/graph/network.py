"""The road network model: nodes, edges, weights, and dynamic weight updates.

This is deliverable 1 of the problem statement in executable form. A
:class:`RoadNetwork` wraps an OSMnx ``MultiDiGraph`` (nodes = intersections and
depots, edges = road segments carrying travel time, distance and congestion)
and adds the two things the graph itself cannot provide:

* **A CSR adjacency** so that shortest paths and many-to-many matrices run in
  compiled code (``scipy.sparse.csgraph``) instead of in Python dictionaries.
  On the bundled Bengaluru extract this is the difference between a 200x200
  matrix taking seconds and taking tens of milliseconds.
* **An O(E) dynamic weight update.** Live traffic changes edge travel times
  every few seconds. Rebuilding the CSR from the NetworkX graph each time would
  dominate the runtime, so the sparsity structure is built exactly once and
  :meth:`RoadNetwork.update_weights` only overwrites the ``data`` arrays.

Design notes
------------
*Parallel edges.* A ``MultiDiGraph`` can hold several edges between the same
ordered pair of nodes (a service road alongside a main road, or two ways that
OSMnx simplification did not merge). The CSR holds one entry per ordered pair,
and that entry takes the **minimum** travel time of the parallel edges - a
vehicle chooses the fastest of them. Summing them, which is what a naive
``nx.to_scipy_sparse_array`` does, would invent a road that is slower than
either real road.

*Representative edge.* Distance and congestion for a pair are taken from
whichever parallel edge currently attains that minimum travel time, so the
three matrices always describe one physically consistent road, rather than a
mix of the shortest and the fastest.

*Units.* Lengths are metres, travel times are seconds, speeds km/h,
coordinates WGS84 decimal degrees. Congestion is a dimensionless level in
``[0, 1]``: the fraction of the current travel time that is delay, so 0 is
free flow and 0.5 means the trip takes twice as long as it would at free flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from qroute.graph import osm as osm_mod

#: Travel times are clamped to at least this many seconds. Zero-length edges do
#: exist in OSM extracts, and scipy's sparse containers treat a stored zero as
#: a structurally absent edge, which would silently disconnect the graph.
MIN_TRAVEL_TIME_S: float = 1e-6

#: Mean Earth radius in metres (spherical approximation, good to ~0.5% and far
#: more accurate than needed at city scale).
EARTH_RADIUS_M: float = 6_371_008.8


@dataclass(frozen=True)
class EdgeRef:
    """Identifies one edge of the underlying MultiDiGraph."""

    u: int          # OSM node id of the tail
    v: int          # OSM node id of the head
    key: int        # MultiDiGraph parallel-edge key


class RoadNetwork:
    """A routable road network backed by contiguous NumPy arrays.

    The instance owns two parallel views of the same road network:

    ``edge_*`` arrays
        One entry per edge of the original ``MultiDiGraph``, in a fixed order.
        This is the view live traffic updates are expressed in, because a
        traffic feed talks about physical road segments.

    ``csr_*`` matrices
        One entry per ordered node pair, which is what shortest-path algorithms
        need. Derived from the ``edge_*`` arrays by the minimum reduction
        described in the module docstring.
    """

    # ------------------------------------------------------------------ build
    def __init__(self, graph: nx.MultiDiGraph, name: str | None = None) -> None:
        if graph.number_of_nodes() == 0:
            raise ValueError("cannot build a RoadNetwork from an empty graph")
        self.graph = graph
        self.name = name or str(graph.graph.get("name", "road-network"))

        # --- nodes -------------------------------------------------------
        node_ids = np.fromiter(graph.nodes, dtype=np.int64, count=graph.number_of_nodes())
        self._node_ids = node_ids
        # Dense index lookup. A dict is used rather than a sorted-array
        # searchsorted because node ids are 64-bit OSM ids with huge gaps, and
        # the lookup happens once per API call, not in an inner loop.
        self._index_of: dict[int, int] = {int(nid): i for i, nid in enumerate(node_ids)}
        coords = np.empty((node_ids.size, 2), dtype=np.float64)
        for i, nid in enumerate(node_ids):
            data = graph.nodes[nid]
            coords[i, 0] = float(data["y"])   # latitude
            coords[i, 1] = float(data["x"])   # longitude
        self._coords = coords

        # --- edges -------------------------------------------------------
        m = graph.number_of_edges()
        eu = np.empty(m, dtype=np.int32)
        ev = np.empty(m, dtype=np.int32)
        ekey = np.empty(m, dtype=np.int32)
        length = np.empty(m, dtype=np.float64)
        free_flow = np.empty(m, dtype=np.float64)
        speed = np.empty(m, dtype=np.float64)
        capacity = np.empty(m, dtype=np.float64)
        importance = np.empty(m, dtype=np.int8)
        classes: list[str] = []
        geometries: list[object] = []
        for i, (u, v, k, data) in enumerate(graph.edges(keys=True, data=True)):
            eu[i] = self._index_of[int(u)]
            ev[i] = self._index_of[int(v)]
            ekey[i] = int(k)
            length[i] = float(data.get("length", 0.0))
            cls = data.get("highway_class") or osm_mod.highway_class(data.get("highway"))
            classes.append(cls)
            spd = float(data.get("speed_kph", osm_mod.DEFAULT_SPEED_KPH))
            speed[i] = spd
            tt = data.get("travel_time")
            free_flow[i] = float(tt) if tt is not None else length[i] / max(spd / 3.6, 1e-9)
            capacity[i] = float(
                data.get("capacity", osm_mod.DEFAULT_CAPACITY_VEH_PER_HOUR_PER_LANE)
            )
            importance[i] = osm_mod.CLASS_IMPORTANCE.get(cls, osm_mod.DEFAULT_IMPORTANCE)
            geometries.append(data.get("geometry"))

        self._edge_u = eu
        self._edge_v = ev
        self._edge_key = ekey
        self._edge_length = length
        self._edge_free_flow_time = np.maximum(free_flow, MIN_TRAVEL_TIME_S)
        self._edge_speed_kph = speed
        self._edge_capacity = capacity
        self._edge_class = np.array(classes, dtype=object)
        self._edge_importance = importance
        self._edge_geometry = geometries

        # Live state: current travel time and congestion level per edge.
        self._edge_travel_time = self._edge_free_flow_time.copy()
        self._edge_congestion = np.zeros(m, dtype=np.float64)

        self._build_csr_structure()
        self._refresh_csr_data()

    @classmethod
    def from_graphml(cls, path: str | Path, **kwargs) -> "RoadNetwork":
        """Load, clean and wrap a GraphML file (see :func:`qroute.graph.osm.load_graph`)."""
        graph = osm_mod.load_graph(path, **kwargs)
        return cls(graph, name=graph.graph.get("name", Path(str(path)).stem))

    def _build_csr_structure(self) -> None:
        """Compute the fixed CSR sparsity pattern; done exactly once."""
        n = self.n_nodes
        # np.unique returns the distinct keys in ascending order, which for
        # key = u * n + v is exactly row-major (CSR) order: rows ascending and,
        # within a row, columns ascending. So no separate sort is needed.
        pair_key = self._edge_u.astype(np.int64) * n + self._edge_v.astype(np.int64)
        uniq, inverse = np.unique(pair_key, return_inverse=True)
        self._pair_of_edge = inverse.astype(np.int64)
        pair_u = (uniq // n).astype(np.int32)
        pair_v = (uniq % n).astype(np.int32)
        indptr = np.zeros(n + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(np.bincount(pair_u, minlength=n))
        self._indptr = indptr
        self._indices = pair_v
        self._n_pairs = int(uniq.size)
        # Reverse map pair -> representative edge, filled by _refresh_csr_data.
        self._pair_representative = np.zeros(self._n_pairs, dtype=np.int64)

    def _refresh_csr_data(self) -> None:
        """Recompute the three CSR data arrays from the per-edge arrays.

        Complexity is O(E) with no allocation of graph structures, which is what
        makes :meth:`update_weights` cheap enough for a live traffic feed.
        """
        inv = self._pair_of_edge
        tt = np.full(self._n_pairs, np.inf, dtype=np.float64)
        # np.minimum.at is an unbuffered scatter-min: exactly the "collapse
        # parallel edges by minimum travel time" rule, in one pass over edges.
        np.minimum.at(tt, inv, self._edge_travel_time)

        # Identify, for every pair, an edge attaining the minimum. Ties are
        # broken by whichever edge is scanned last; any of them is equally
        # valid because they have identical travel time.
        rep = self._pair_representative
        rep.fill(-1)
        winners = self._edge_travel_time <= tt[inv]
        rep[inv[winners]] = np.nonzero(winners)[0]

        tt = np.maximum(tt, MIN_TRAVEL_TIME_S)
        shape = (self.n_nodes, self.n_nodes)
        self._csr_travel_time = csr_matrix((tt, self._indices, self._indptr), shape=shape)
        self._csr_length = csr_matrix(
            (self._edge_length[rep], self._indices, self._indptr), shape=shape
        )
        self._csr_congestion = csr_matrix(
            (self._edge_congestion[rep], self._indices, self._indptr), shape=shape
        )

    # ------------------------------------------------------------- properties
    @property
    def n_nodes(self) -> int:
        return int(self._node_ids.size)

    @property
    def n_edges(self) -> int:
        """Number of edges of the underlying MultiDiGraph (parallel edges counted)."""
        return int(self._edge_u.size)

    @property
    def n_arcs(self) -> int:
        """Number of distinct ordered node pairs, i.e. non-zeros in the CSR."""
        return self._n_pairs

    @property
    def node_ids(self) -> np.ndarray:
        """OSM node ids, in the internal index order."""
        return self._node_ids

    @property
    def coords(self) -> np.ndarray:
        """``(n_nodes, 2)`` array of ``[latitude, longitude]`` in degrees."""
        return self._coords

    @property
    def csr_travel_time(self) -> csr_matrix:
        """Current travel times in seconds, one entry per ordered node pair."""
        return self._csr_travel_time

    @property
    def csr_length(self) -> csr_matrix:
        """Lengths in metres of the currently fastest edge of each pair."""
        return self._csr_length

    @property
    def csr_congestion(self) -> csr_matrix:
        """Congestion level in ``[0, 1]`` of the currently fastest edge of each pair."""
        return self._csr_congestion

    @property
    def edge_free_flow_time(self) -> np.ndarray:
        return self._edge_free_flow_time

    @property
    def edge_travel_time(self) -> np.ndarray:
        return self._edge_travel_time

    @property
    def edge_length(self) -> np.ndarray:
        return self._edge_length

    @property
    def edge_capacity(self) -> np.ndarray:
        """Saturation flow in vehicles per hour, per edge."""
        return self._edge_capacity

    @property
    def edge_congestion(self) -> np.ndarray:
        return self._edge_congestion

    @property
    def edge_speed_kph(self) -> np.ndarray:
        return self._edge_speed_kph

    @property
    def edge_classes(self) -> np.ndarray:
        """Object array of OSM highway class strings, one per edge."""
        return self._edge_class

    def edge_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """``(tail, head)`` internal node indices, one entry per edge."""
        return self._edge_u, self._edge_v

    def index_of(self, node_id: int) -> int:
        """Internal index of an OSM node id."""
        try:
            return self._index_of[int(node_id)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"node {node_id} is not in network {self.name!r}") from exc

    def indices_of(self, node_ids: Iterable[int]) -> np.ndarray:
        return np.array([self.index_of(n) for n in node_ids], dtype=np.int64)

    def node_id_of(self, index: int) -> int:
        return int(self._node_ids[int(index)])

    # --------------------------------------------------------- weight updates
    def update_weights(
        self,
        travel_times: np.ndarray | None = None,
        *,
        factors: np.ndarray | float | None = None,
        congestion: np.ndarray | None = None,
    ) -> None:
        """Replace the live edge travel times. This is the dynamic weight update.

        Exactly one of ``travel_times`` or ``factors`` should be given.

        Parameters
        ----------
        travel_times:
            ``(n_edges,)`` new travel times in seconds, in edge order.
        factors:
            ``(n_edges,)`` (or a scalar) multipliers applied to the *free-flow*
            times. Multiplying free flow rather than the current time makes the
            update idempotent: applying the same feed twice gives the same
            network, which a feed that multiplies the current state would not.
        congestion:
            Optional explicit congestion levels. When omitted they are derived
            as ``1 - free_flow / current``, i.e. the share of the journey time
            that is delay, clipped to ``[0, 1]``.

        The cost is O(n_edges): the CSR sparsity pattern is untouched and only
        the three data arrays are recomputed. No NetworkX graph is rebuilt and
        no dictionary is traversed.
        """
        if travel_times is None and factors is None:
            raise ValueError("update_weights needs either travel_times or factors")
        if travel_times is not None and factors is not None:
            raise ValueError("give travel_times or factors, not both")

        if factors is not None:
            f = np.asarray(factors, dtype=np.float64)
            if f.ndim == 0:
                f = np.full(self.n_edges, float(f))
            if f.shape != (self.n_edges,):
                raise ValueError(f"factors must be scalar or ({self.n_edges},), got {f.shape}")
            if np.any(f <= 0):
                raise ValueError("congestion factors must be strictly positive")
            new_tt = self._edge_free_flow_time * f
        else:
            new_tt = np.asarray(travel_times, dtype=np.float64)
            if new_tt.shape != (self.n_edges,):
                raise ValueError(
                    f"travel_times must have shape ({self.n_edges},), got {new_tt.shape}"
                )
            if np.any(new_tt <= 0) or not np.all(np.isfinite(new_tt)):
                raise ValueError("travel times must be finite and strictly positive")

        self._edge_travel_time = np.maximum(new_tt, MIN_TRAVEL_TIME_S)
        if congestion is None:
            ratio = self._edge_free_flow_time / self._edge_travel_time
            self._edge_congestion = np.clip(1.0 - ratio, 0.0, 1.0)
        else:
            c = np.asarray(congestion, dtype=np.float64)
            if c.shape != (self.n_edges,):
                raise ValueError(f"congestion must have shape ({self.n_edges},)")
            self._edge_congestion = np.clip(c, 0.0, 1.0)
        self._refresh_csr_data()

    def reset_weights(self) -> None:
        """Restore free-flow travel times and zero congestion."""
        self._edge_travel_time = self._edge_free_flow_time.copy()
        self._edge_congestion = np.zeros(self.n_edges, dtype=np.float64)
        self._refresh_csr_data()

    def write_back_to_graph(self) -> None:
        """Copy the live travel times back onto the NetworkX graph.

        Only needed when handing the graph to an external tool (plotting,
        exporting). The routing path never reads the NetworkX attributes, which
        is why the update does not do this by default - it is O(E) *with* a
        dictionary write per edge and is roughly an order of magnitude slower
        than the array update.
        """
        for i, (u, v, k) in enumerate(self.graph.edges(keys=True)):
            data = self.graph[u][v][k]
            data["travel_time"] = float(self._edge_travel_time[i])
            data["congestion"] = float(self._edge_congestion[i])

    # ------------------------------------------------------------- geo lookups
    def nearest_node(
        self, lat: float | Sequence[float] | np.ndarray, lon: float | Sequence[float] | np.ndarray
    ) -> int | np.ndarray:
        """Index of the network node closest to each ``(lat, lon)`` query point.

        Vectorised over the query points and implemented with an equirectangular
        projection: at city scale (a few km) the error of treating a degree of
        longitude as ``cos(lat)`` degrees of latitude is far below the spacing
        between intersections, and it avoids a scikit-learn/BallTree dependency
        for what is a single argmin over 13k points.

        Returns a scalar index for scalar input, otherwise an array of indices.
        """
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        if lat_arr.shape != lon_arr.shape:
            raise ValueError("lat and lon must have the same shape")
        scale = np.cos(np.deg2rad(self._coords[:, 0].mean()))
        node_y = self._coords[:, 0]
        node_x = self._coords[:, 1] * scale
        dy = lat_arr[:, None] - node_y[None, :]
        dx = lon_arr[:, None] * scale - node_x[None, :]
        best = np.argmin(dy * dy + dx * dx, axis=1)
        if np.ndim(lat) == 0:
            return int(best[0])
        return best

    def haversine_to(self, index: int, others: np.ndarray | None = None) -> np.ndarray:
        """Great-circle distance in metres from node ``index`` to ``others``.

        Used as the admissible heuristic of the A* search in
        :mod:`qroute.graph.paths`.
        """
        target = self._coords[index]
        pts = self._coords if others is None else self._coords[others]
        lat1, lon1 = np.deg2rad(target[0]), np.deg2rad(target[1])
        lat2 = np.deg2rad(pts[:, 0])
        lon2 = np.deg2rad(pts[:, 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    @property
    def max_speed_mps(self) -> float:
        """Fastest free-flow speed anywhere in the network, in metres/second.

        Dividing a straight-line distance by this gives an admissible (never
        over-estimating) lower bound on travel time, which is what A* needs to
        stay exact.
        """
        return float(self._edge_speed_kph.max() / 3.6)

    # ---------------------------------------------------------------- exports
    def edge_geojson(
        self,
        *,
        min_importance: int = 0,
        max_edges: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        include_geometry: bool = True,
    ) -> dict:
        """GeoJSON ``FeatureCollection`` of the road edges, for the map frontend.

        Parameters
        ----------
        min_importance:
            Level-of-detail filter on the highway class ranking in
            :data:`qroute.graph.osm.CLASS_IMPORTANCE`. ``0`` draws everything;
            ``3`` draws only secondary roads and above ("arterials only"), which
            on the Bengaluru extract cuts 34k features down to about 2k - the
            difference between a map that pans smoothly and one that does not.
        max_edges:
            Hard cap; the most important edges are kept. Applied after
            ``min_importance`` and ``bbox``.
        bbox:
            ``(min_lat, min_lon, max_lat, max_lon)``; keeps edges whose tail or
            head lies inside.
        include_geometry:
            Emit the true OSM polyline when available. With ``False`` every edge
            becomes a straight two-point line, which is a further large size
            reduction for zoomed-out views.
        """
        keep = self._edge_importance >= min_importance
        if bbox is not None:
            min_lat, min_lon, max_lat, max_lon = bbox
            lat = self._coords[:, 0]
            lon = self._coords[:, 1]
            inside = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            keep &= inside[self._edge_u] | inside[self._edge_v]
        idx = np.nonzero(keep)[0]
        if max_edges is not None and idx.size > max_edges:
            # Keep the most important edges; stable so the selection is
            # deterministic for a given network and threshold.
            order = np.argsort(-self._edge_importance[idx], kind="stable")
            idx = idx[order[:max_edges]]
            idx.sort()

        features = []
        for i in idx:
            geom = self._edge_geometry[i] if include_geometry else None
            if geom is not None and hasattr(geom, "coords"):
                line = [[float(x), float(y)] for x, y in geom.coords]
            else:
                a = self._coords[self._edge_u[i]]
                b = self._coords[self._edge_v[i]]
                line = [[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": line},
                    "properties": {
                        "edge": int(i),
                        "u": int(self._node_ids[self._edge_u[i]]),
                        "v": int(self._node_ids[self._edge_v[i]]),
                        "highway": str(self._edge_class[i]),
                        "length_m": round(float(self._edge_length[i]), 2),
                        "free_flow_s": round(float(self._edge_free_flow_time[i]), 2),
                        "travel_time_s": round(float(self._edge_travel_time[i]), 2),
                        "congestion": round(float(self._edge_congestion[i]), 4),
                        "speed_kph": round(float(self._edge_speed_kph[i]), 1),
                    },
                }
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "network": self.name,
                "n_features": len(features),
                "min_importance": min_importance,
            },
        }

    def route_geojson(
        self, node_path: Sequence[int], *, by_id: bool = False, properties: dict | None = None
    ) -> dict:
        """GeoJSON ``Feature`` tracing a path given as a sequence of nodes.

        ``node_path`` holds internal indices unless ``by_id`` is set, in which
        case it holds OSM node ids. The polyline follows the true road geometry
        wherever OSM provides it, so the drawn route sits on the road rather
        than cutting corners between intersections.
        """
        path = [self.index_of(n) for n in node_path] if by_id else [int(n) for n in node_path]
        if len(path) < 2:
            pts = [[float(self._coords[p][1]), float(self._coords[p][0])] for p in path]
            return {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": pts},
                "properties": {"duration_s": 0.0, "distance_m": 0.0, "n_nodes": len(path)},
            }

        coords: list[list[float]] = []
        duration = 0.0
        distance = 0.0
        congestion_time = 0.0
        for a, b in zip(path[:-1], path[1:]):
            e = self._fastest_edge_between(a, b)
            if e < 0:
                raise ValueError(
                    f"nodes {self.node_id_of(a)} -> {self.node_id_of(b)} are not adjacent"
                )
            duration += float(self._edge_travel_time[e])
            distance += float(self._edge_length[e])
            congestion_time += float(self._edge_congestion[e]) * float(self._edge_travel_time[e])
            geom = self._edge_geometry[e]
            if geom is not None and hasattr(geom, "coords"):
                seg = [[float(x), float(y)] for x, y in geom.coords]
            else:
                pa, pb = self._coords[a], self._coords[b]
                seg = [[float(pa[1]), float(pa[0])], [float(pb[1]), float(pb[0])]]
            if coords and seg and coords[-1] == seg[0]:
                seg = seg[1:]
            coords.extend(seg)

        props = {
            "duration_s": round(duration, 2),
            "distance_m": round(distance, 2),
            "mean_congestion": round(congestion_time / duration, 4) if duration > 0 else 0.0,
            "n_nodes": len(path),
            "node_ids": [int(self._node_ids[p]) for p in path],
        }
        if properties:
            props.update(properties)
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        }

    def _fastest_edge_between(self, a: int, b: int) -> int:
        """Index of the fastest edge from internal node ``a`` to ``b``, or -1."""
        lo, hi = self._indptr[a], self._indptr[a + 1]
        row = self._indices[lo:hi]
        pos = np.searchsorted(row, b)
        if pos >= row.size or row[pos] != b:
            return -1
        return int(self._pair_representative[lo + pos])

    def to_json(self, path: str | Path, **geojson_kwargs) -> Path:
        """Write :meth:`edge_geojson` to disk (convenience for the frontend)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.edge_geojson(**geojson_kwargs)))
        return p

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_arcs": self.n_arcs,
            "total_length_km": round(float(self._edge_length.sum()) / 1000.0, 2),
            "mean_free_flow_kph": round(float(self._edge_speed_kph.mean()), 2),
            "mean_congestion": round(float(self._edge_congestion.mean()), 4),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RoadNetwork({self.name!r}, nodes={self.n_nodes}, "
            f"edges={self.n_edges}, arcs={self.n_arcs})"
        )
