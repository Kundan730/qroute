"""Turning a road network into a routing :class:`~qroute.problems.instance.Instance`.

The benchmark instances (CVRPLIB, Solomon) come with their own coordinates and
distance conventions. For the live demonstration we instead need instances that
sit on a real city: a depot at a real intersection, customers at real
addresses, and matrices that are genuine shortest paths through the road
network rather than straight-line distances. This module is that bridge.

What the produced instance contains
-----------------------------------
``duration``    shortest travel time in seconds between every pair of stops
``distance``    length in metres **of that same fastest path**, not of the
                separate shortest-by-distance path (see :mod:`qroute.graph.matrix`)
``congestion``  time-weighted mean congestion level along that path, in [0, 1]
``coords``      ``[latitude, longitude]`` of each stop, for the map
``node_ids``    OSM node id of each stop, so the frontend can ask the network
                for the full polyline of any leg
``meta``        the network name, the seed, and the sampling parameters, so any
                reported result can be reproduced exactly

Honesty note on demands and capacity
------------------------------------
Real delivery demands are not public data. They are sampled from a seeded
generator, and the instance metadata records the seed and the distribution.
Nothing here should be read as a measurement of actual Bengaluru logistics
demand; it is a reproducible synthetic workload on a real road topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from qroute.core.rng import make_rng
from qroute.graph.matrix import MatrixResult, build_matrices, route_node_path
from qroute.graph.network import RoadNetwork
from qroute.problems.instance import Instance, ObjectiveWeights

#: Default demand distribution: integer parcels per stop, uniform on this range.
DEFAULT_DEMAND_RANGE: tuple[int, int] = (1, 20)

#: Default fill ratio used when the caller does not fix a capacity. Capacity is
#: set so that the bin-packing lower bound is roughly ``n_customers / 8``
#: vehicles, which keeps instances non-trivial without being infeasible.
DEFAULT_STOPS_PER_VEHICLE: int = 8


@dataclass
class StopSelection:
    """The chosen depot and customer nodes, with their geography."""

    node_indices: np.ndarray     # internal RoadNetwork indices, depot first
    node_ids: list[int]          # corresponding OSM node ids
    coords: np.ndarray           # (k, 2) latitude, longitude


def _spread_sample(
    network: RoadNetwork, count: int, rng: np.random.Generator, candidates: np.ndarray
) -> np.ndarray:
    """Pick ``count`` well-separated nodes by farthest-point (k-centre) sampling.

    Uniform random sampling on an OSM graph over-represents dense residential
    grids, because that is where the intersections are; the resulting instances
    have most customers packed into a few blocks and are easier than they look.
    Farthest-point sampling starts from a random node and repeatedly takes the
    candidate furthest from everything chosen so far, which spreads stops over
    the whole extract. It is greedy and deterministic given the first pick, and
    the first pick comes from the seeded generator, so it is reproducible.
    """
    chosen = [int(rng.choice(candidates))]
    coords = network.coords
    scale = np.cos(np.deg2rad(coords[:, 0].mean()))
    pts = np.column_stack((coords[candidates, 0], coords[candidates, 1] * scale))
    first = np.array([coords[chosen[0], 0], coords[chosen[0], 1] * scale])
    best = ((pts - first) ** 2).sum(axis=1)
    for _ in range(count - 1):
        pick = int(np.argmax(best))
        chosen.append(int(candidates[pick]))
        d = ((pts - pts[pick]) ** 2).sum(axis=1)
        np.minimum(best, d, out=best)
    return np.array(chosen, dtype=np.int64)


def select_stops(
    network: RoadNetwork,
    n_customers: int,
    *,
    seed: int = 0,
    depot: Optional[int] = None,
    depot_latlon: Optional[tuple[float, float]] = None,
    sampling: str = "spread",
    min_street_count: int = 3,
) -> StopSelection:
    """Choose a depot and ``n_customers`` customer nodes on the network.

    Parameters
    ----------
    depot / depot_latlon:
        Fix the depot by internal index or by coordinates (snapped with
        :meth:`RoadNetwork.nearest_node`). When neither is given the depot is
        the node closest to the centroid of the extract, which is the natural
        choice for a city-centre distribution hub and keeps the instance from
        depending on the sampling seed for its most important node.
    sampling:
        ``'spread'`` (default, farthest-point) or ``'random'``.
    min_street_count:
        Only consider nodes with at least this many incident streets, so stops
        land on real junctions rather than on the artificial degree-2 nodes that
        OSMnx keeps where a way changes attributes. Relaxed automatically if too
        few such nodes exist.
    """
    rng = make_rng(seed)
    coords = network.coords

    street_count = np.array(
        [network.graph.nodes[int(nid)].get("street_count", 0) for nid in network.node_ids],
        dtype=np.int32,
    )
    candidates = np.nonzero(street_count >= min_street_count)[0]
    if candidates.size < n_customers + 1:
        candidates = np.arange(network.n_nodes, dtype=np.int64)

    if depot_latlon is not None:
        depot_idx = int(network.nearest_node(depot_latlon[0], depot_latlon[1]))
    elif depot is not None:
        depot_idx = int(depot)
    else:
        centre = coords.mean(axis=0)
        depot_idx = int(network.nearest_node(centre[0], centre[1]))

    pool = candidates[candidates != depot_idx]
    if pool.size < n_customers:
        raise ValueError(
            f"network {network.name!r} has only {pool.size} candidate nodes "
            f"but {n_customers} customers were requested"
        )

    if sampling == "random":
        picks = rng.choice(pool, size=n_customers, replace=False)
    elif sampling == "spread":
        picks = _spread_sample(network, n_customers, rng, pool)
    else:
        raise ValueError(f"unknown sampling {sampling!r}; use 'spread' or 'random'")

    node_indices = np.concatenate(([depot_idx], np.asarray(picks, dtype=np.int64)))
    return StopSelection(
        node_indices=node_indices,
        node_ids=[network.node_id_of(i) for i in node_indices],
        coords=np.ascontiguousarray(coords[node_indices]),
    )


def build_instance(
    network: RoadNetwork,
    n_customers: int = 50,
    *,
    seed: int = 0,
    name: Optional[str] = None,
    depot: Optional[int] = None,
    depot_latlon: Optional[tuple[float, float]] = None,
    sampling: str = "spread",
    demand_range: tuple[int, int] = DEFAULT_DEMAND_RANGE,
    capacity: Optional[float] = None,
    n_vehicles: Optional[int] = None,
    service_time_s: float = 0.0,
    max_route_duration_s: Optional[float] = None,
    weights: Optional[ObjectiveWeights] = None,
    stops: Optional[StopSelection] = None,
    return_matrices: bool = False,
) -> Instance | tuple[Instance, MatrixResult]:
    """Build a routing instance from a road network.

    The default objective weights minimise **travel time**, because that is what
    the problem statement asks for on a road network; the CVRPLIB default of
    pure distance is kept only for the benchmark instances where published
    best-known values are distances.

    Set ``return_matrices=True`` to also get the :class:`MatrixResult`, which
    carries the Dijkstra predecessors needed to draw real polylines for each
    leg (see :func:`leg_node_paths`).
    """
    if stops is None:
        stops = select_stops(
            network,
            n_customers,
            seed=seed,
            depot=depot,
            depot_latlon=depot_latlon,
            sampling=sampling,
        )
    k = stops.node_indices.size
    n_customers = k - 1

    matrices = build_matrices(network, stops.node_indices, keep_predecessors=True)
    if not matrices.is_finite:
        # Should be impossible on a strongly connected component; fail loudly
        # rather than let an inf reach the optimiser, where it would poison
        # every cost comparison silently.
        raise ValueError(
            "shortest-path matrix contains infinities: the network is not "
            "strongly connected (load it with strongly_connected=True)"
        )

    rng = make_rng(seed + 1)   # separate stream so demands do not shift when
                               # the stop sampling changes, and vice versa
    lo, hi = demand_range
    demand = np.zeros(k, dtype=np.float64)
    demand[1:] = rng.integers(lo, hi + 1, size=n_customers)

    if capacity is None:
        total = float(demand.sum())
        capacity = float(
            max(demand[1:].max(), np.ceil(total / max(1, n_customers / DEFAULT_STOPS_PER_VEHICLE)))
        )

    service = None
    if service_time_s > 0:
        service = np.full(k, float(service_time_s), dtype=np.float64)
        service[0] = 0.0

    if weights is None:
        weights = ObjectiveWeights(time=1.0, distance=0.0, congestion=0.0, vehicles=0.0)

    instance = Instance(
        name=name or f"{network.name}-n{n_customers}-s{seed}",
        distance=matrices.distance,
        duration=matrices.duration,
        demand=demand,
        capacity=float(capacity),
        n_vehicles=n_vehicles,
        service_time=service,
        max_route_duration=max_route_duration_s,
        congestion=matrices.congestion,
        coords=stops.coords,
        weights=weights,
        node_ids=list(stops.node_ids),
        meta={
            "source": "osm",
            "network": network.name,
            "seed": seed,
            "sampling": sampling,
            "demand_range": list(demand_range),
            "units": {"distance": "metres", "duration": "seconds"},
            "stop_node_indices": stops.node_indices.tolist(),
            "matrix_build_seconds": round(matrices.seconds, 4),
        },
    )
    if return_matrices:
        return instance, matrices
    return instance


def leg_node_paths(
    network: RoadNetwork, matrices: MatrixResult, routes: Sequence[Sequence[int]]
) -> list[list[int]]:
    """Full road-network node path of each route, depot to depot.

    ``routes`` uses the optimiser's stop indices (customers only; the depot is
    implicit). The returned lists are internal node indices ready for
    :meth:`RoadNetwork.route_geojson`.
    """
    paths: list[list[int]] = []
    for route in routes:
        if not len(route):
            continue
        sequence = [0, *[int(c) for c in route], 0]
        nodes: list[int] = []
        for a, b in zip(sequence[:-1], sequence[1:]):
            leg = route_node_path(network, matrices, a, b)
            if nodes and leg and nodes[-1] == leg[0]:
                leg = leg[1:]
            nodes.extend(leg)
        paths.append(nodes)
    return paths


def routes_geojson(
    network: RoadNetwork, matrices: MatrixResult, routes: Sequence[Sequence[int]]
) -> dict:
    """GeoJSON ``FeatureCollection`` with one polyline per vehicle route."""
    features = []
    for v, nodes in enumerate(leg_node_paths(network, matrices, routes)):
        features.append(network.route_geojson(nodes, properties={"vehicle": v}))
    return {"type": "FeatureCollection", "features": features}
