"""Loading, repairing and saving OpenStreetMap road graphs.

This module is the boundary between raw OSM data and the rest of the platform.
It exists because OSM extracts of Indian cities are not directly usable as a
routing network for three reasons, each of which is handled here:

1. **Speed tags are almost absent.** In the bundled Bengaluru extract only
   1398 of 34353 edges (4.1%) carry a ``maxspeed`` tag. OSMnx's default
   :func:`add_edge_speeds` fills the gap with the *mean of the tagged edges of
   the same highway class*, which produces figures such as 29 km/h for every
   residential street and 42 km/h for every secondary road - and, on extracts
   where a class has no tagged edge at all, a single global mean. That is a
   statistic of which roads happen to be tagged, not of how fast the road
   actually is. We therefore re-impute free-flow speeds from an explicit,
   auditable table keyed by highway class (:data:`FREE_FLOW_SPEED_KPH`).

2. **Lane and capacity information is needed by the congestion model** but OSM
   only tags ``lanes`` on 3690 of 34353 edges (10.7%). We default to one lane
   and document the resulting bias (capacity is under-estimated on arterials).

3. **The extract is not strongly connected.** A bounding-box download cuts
   roads at the boundary, leaving nodes that can be entered but not left (or
   vice versa). Any origin-destination matrix computed over such a graph
   contains infinities. :func:`largest_strongly_connected` removes them.

Everything here works offline from the GraphML files bundled under ``data/osm``;
:func:`download_graph` is provided for regenerating them but is never required.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import osmnx as ox

DATA_ROOT = Path(os.environ.get("QROUTE_DATA", "data"))
OSM_DIR = DATA_ROOT / "osm"

# --------------------------------------------------------------------------
# Assumption tables. These are modelling assumptions, not measurements. They
# are module-level constants precisely so that a reviewer can see and change
# every number the traffic model depends on in one place.
# --------------------------------------------------------------------------

#: Free-flow speed in km/h by OSM ``highway`` class. Chosen for dense Indian
#: urban arterials and side streets: these are *uncongested* speeds, so the
#: congestion model has room to slow them down. They are deliberately lower
#: than European defaults (a "primary" road in Koramangala is not a 60 km/h
#: road even at 4 a.m.). ``*_link`` values are the connecting ramps/slip roads.
FREE_FLOW_SPEED_KPH: dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 40.0,
    "primary": 45.0,
    "primary_link": 35.0,
    "secondary": 40.0,
    "secondary_link": 30.0,
    "tertiary": 35.0,
    "tertiary_link": 25.0,
    "unclassified": 30.0,
    "residential": 25.0,
    "living_street": 15.0,
    "service": 20.0,
    "busway": 30.0,
    "road": 25.0,
    "pedestrian": 10.0,
    "track": 15.0,
}

#: Fallback speed for any class not in the table above (km/h).
DEFAULT_SPEED_KPH: float = 25.0

#: Saturation flow in vehicles per hour per lane by highway class. Used by the
#: congestion model (volume/capacity ratio -> BPR-style delay). Urban Indian
#: values: an arterial lane in mixed traffic does not achieve the ~2200 veh/h
#: of a motorway lane, so the classes below trunk are penalised accordingly.
EDGE_CAPACITY_VEH_PER_HOUR_PER_LANE: dict[str, float] = {
    "motorway": 2000.0,
    "motorway_link": 1500.0,
    "trunk": 2000.0,
    "trunk_link": 1500.0,
    "primary": 1500.0,
    "primary_link": 1000.0,
    "secondary": 1000.0,
    "secondary_link": 800.0,
    "tertiary": 600.0,
    "tertiary_link": 600.0,
    "unclassified": 600.0,
    "residential": 600.0,
    "living_street": 300.0,
    "service": 300.0,
    "busway": 600.0,
    "road": 600.0,
}

#: Fallback capacity (veh/h/lane).
DEFAULT_CAPACITY_VEH_PER_HOUR_PER_LANE: float = 600.0

#: Ranking of highway classes for the map level-of-detail filter. Higher means
#: more important; the frontend can ask for "arterials only" by importance.
CLASS_IMPORTANCE: dict[str, int] = {
    "motorway": 6,
    "motorway_link": 5,
    "trunk": 5,
    "trunk_link": 4,
    "primary": 4,
    "primary_link": 3,
    "secondary": 3,
    "secondary_link": 2,
    "tertiary": 2,
    "tertiary_link": 1,
    "unclassified": 1,
    "residential": 1,
    "busway": 1,
    "road": 1,
    "living_street": 0,
    "service": 0,
    "pedestrian": 0,
    "track": 0,
}

DEFAULT_IMPORTANCE: int = 1


# --------------------------------------------------------------------------
# Tag parsing
# --------------------------------------------------------------------------

def highway_class(value: object) -> str:
    """Canonical highway class for an OSM ``highway`` tag value.

    OSMnx's graph simplification merges consecutive ways into one edge, and when
    those ways disagree the tag becomes a *list* (45 of 34353 edges in the
    Bengaluru extract). We resolve a list to its most important member rather
    than to its first, so a residential stub merged into a tertiary road is
    treated as tertiary: under-stating a road's class understates its speed,
    which biases routes away from real arterials.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return max(
            (str(v) for v in value),
            key=lambda v: CLASS_IMPORTANCE.get(v, DEFAULT_IMPORTANCE),
        )
    return "unclassified"


_MAXSPEED_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_maxspeed(value: object) -> Optional[float]:
    """Parse an OSM ``maxspeed`` tag into km/h, or ``None`` if unusable.

    Handles ``"40"``, ``"40 km/h"``, ``"30 mph"`` and lists (take the minimum,
    the conservative reading when a merged edge has two limits). Symbolic values
    such as ``"IN:urban"`` or ``"signals"`` yield ``None`` so the caller falls
    back to the class table.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parsed = [parse_maxspeed(v) for v in value]
        real = [p for p in parsed if p is not None]
        return min(real) if real else None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip().lower()
    match = _MAXSPEED_RE.search(text)
    if match is None:
        return None
    speed = float(match.group(1))
    if speed <= 0:
        return None
    if "mph" in text:
        speed *= 1.609344
    return speed


def parse_lanes(value: object) -> int:
    """Parse an OSM ``lanes`` tag, defaulting to one lane.

    Note the coverage caveat in the module docstring: only about 11% of edges
    carry the tag, so capacities are systematically under-estimated on
    multi-lane roads. That is the safe direction of error for a congestion
    model (it predicts congestion sooner rather than later), but it is a bias
    and it is stated here rather than hidden.
    """
    if value is None:
        return 1
    if isinstance(value, (list, tuple)):
        parsed = [parse_lanes(v) for v in value]
        return max(parsed) if parsed else 1
    try:
        lanes = int(float(str(value).split(";")[0].split(",")[0]))
    except (TypeError, ValueError):
        return 1
    return max(1, lanes)


# --------------------------------------------------------------------------
# Graph preparation
# --------------------------------------------------------------------------

def impute_speeds(
    G: nx.MultiDiGraph,
    *,
    respect_maxspeed: bool = True,
    speeds: Optional[dict[str, float]] = None,
    fallback_kph: float = DEFAULT_SPEED_KPH,
) -> nx.MultiDiGraph:
    """Set ``speed_kph`` and ``travel_time`` on every edge, in place.

    ``travel_time`` is in **seconds** and is the *free-flow* time; the dynamic
    congestion model multiplies it later and never overwrites it, so the
    baseline is always recoverable.

    Parameters
    ----------
    respect_maxspeed:
        When a usable ``maxspeed`` tag exists, prefer it over the class table.
        This keeps the small amount of genuine survey data in the extract.
        Set to ``False`` for a fully homogeneous, purely class-driven network.
    """
    table = FREE_FLOW_SPEED_KPH if speeds is None else speeds
    for _u, _v, data in G.edges(data=True):
        cls = highway_class(data.get("highway"))
        speed = None
        if respect_maxspeed:
            speed = parse_maxspeed(data.get("maxspeed"))
        if speed is None:
            speed = table.get(cls, fallback_kph)
        length = float(data.get("length", 0.0))
        data["highway_class"] = cls
        data["speed_kph"] = float(speed)
        # metres / (km/h -> m/s) = seconds.
        data["travel_time"] = length / (float(speed) / 3.6) if speed > 0 else 0.0
    return G


def annotate_capacity(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Set ``lanes_used`` and ``capacity`` (veh/h) on every edge, in place."""
    for _u, _v, data in G.edges(data=True):
        cls = data.get("highway_class") or highway_class(data.get("highway"))
        lanes = parse_lanes(data.get("lanes"))
        per_lane = EDGE_CAPACITY_VEH_PER_HOUR_PER_LANE.get(
            cls, DEFAULT_CAPACITY_VEH_PER_HOUR_PER_LANE
        )
        data["highway_class"] = cls
        data["lanes_used"] = lanes
        data["capacity"] = per_lane * lanes
    return G


def largest_strongly_connected(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the largest strongly connected component of ``G``.

    Strong connectivity, not weak: on a directed road network a weakly
    connected component can still contain a node that is reachable but from
    which the depot cannot be reached (a one-way street cut by the download
    bounding box). Such a node produces an infinite entry in the travel-time
    matrix, which silently destroys any optimiser that assumes finite costs.
    """
    return ox.truncate.largest_component(G, strongly=True)


def load_graph(
    path: str | Path,
    *,
    strongly_connected: bool = True,
    reimpute_speeds: bool = True,
    respect_maxspeed: bool = True,
) -> nx.MultiDiGraph:
    """Load a GraphML road network from disk and prepare it for routing.

    ``path`` may be a full path, or the bare name of a bundled graph such as
    ``"bengaluru_koramangala"``, which is resolved under ``data/osm``.
    """
    p = Path(path)
    if not p.exists():
        candidate = OSM_DIR / f"{p.name}.graphml" if p.suffix == "" else OSM_DIR / p.name
        if candidate.exists():
            p = candidate
        else:
            raise FileNotFoundError(f"no OSM graph at {path!r} or {candidate}")
    G = ox.load_graphml(p)
    if strongly_connected:
        G = largest_strongly_connected(G)
    if reimpute_speeds:
        impute_speeds(G, respect_maxspeed=respect_maxspeed)
    else:
        for _u, _v, data in G.edges(data=True):
            data.setdefault("highway_class", highway_class(data.get("highway")))
    annotate_capacity(G)
    G.graph.setdefault("name", p.stem)
    return G


def list_graphs(directory: str | Path = OSM_DIR) -> list[str]:
    """Names of the GraphML road networks available on disk."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(f.stem for f in d.glob("*.graphml"))


def download_graph(
    center: tuple[float, float],
    dist_m: float = 3000.0,
    *,
    network_type: str = "drive",
    simplify: bool = True,
) -> nx.MultiDiGraph:
    """Download a drivable road network around ``center`` (lat, lon).

    Requires internet access and is only used to regenerate the bundled
    extracts; nothing in the platform's runtime path calls it.
    """
    G = ox.graph.graph_from_point(
        center, dist=dist_m, network_type=network_type, simplify=simplify
    )
    G = largest_strongly_connected(G)
    impute_speeds(G)
    annotate_capacity(G)
    return G


def save_graph(G: nx.MultiDiGraph, path: str | Path) -> Path:
    """Write ``G`` to GraphML, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ox.io.save_graphml(G, p)
    return p


def graph_summary(G: nx.MultiDiGraph) -> dict[str, object]:
    """Auditable summary of a prepared graph, used in reports and tests."""
    classes: dict[str, int] = {}
    total_length = 0.0
    tagged_speed = 0
    tagged_lanes = 0
    for _u, _v, data in G.edges(data=True):
        cls = data.get("highway_class", "unknown")
        classes[cls] = classes.get(cls, 0) + 1
        total_length += float(data.get("length", 0.0))
        if parse_maxspeed(data.get("maxspeed")) is not None:
            tagged_speed += 1
        if data.get("lanes") is not None:
            tagged_lanes += 1
    return {
        "name": G.graph.get("name", "unnamed"),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "total_length_km": total_length / 1000.0,
        "maxspeed_tagged": tagged_speed,
        "lanes_tagged": tagged_lanes,
        "classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
    }


def timed_load(path: str | Path, **kwargs) -> tuple[nx.MultiDiGraph, float]:
    """Load a graph and report the wall-clock seconds it took."""
    start = time.perf_counter()
    G = load_graph(path, **kwargs)
    return G, time.perf_counter() - start
