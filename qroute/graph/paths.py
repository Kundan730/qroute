"""Exact shortest paths on a :class:`~qroute.graph.network.RoadNetwork`.

Three algorithms live here, each for a reason:

:func:`dijkstra`
    One-to-all (or one-to-one with early exit) label-setting search over the
    CSR adjacency. Used when the caller needs the *path*, not just the cost,
    and when a single source is involved. For many-to-many use
    :mod:`qroute.graph.matrix`, which hands the work to SciPy's compiled
    implementation.

:func:`astar`
    Same result as :func:`dijkstra` but goal-directed by a great-circle lower
    bound. On a road network A* typically expands a small fraction of the nodes
    that Dijkstra does for a point-to-point query. The heuristic is admissible
    (straight-line distance divided by the fastest free-flow speed anywhere in
    the network can never exceed the true remaining travel time), so the answer
    is exact, not approximate.

:func:`time_dependent_dijkstra`
    Shortest paths when edge speeds vary with the time of day. This is the
    algorithm the live-traffic demonstration runs on.

Why the time-dependent version is not just "multiply by the current factor"
--------------------------------------------------------------------------
The tempting shortcut is ``travel_time(e, t) = free_flow(e) * factor(period(t))``:
look up the congestion factor for the departure period and scale. It is wrong,
and not in a subtle way. Suppose an edge takes 10 minutes at free flow, the
peak period ends at 09:00, and the peak factor is 3. Departing at 08:59 gives
30 minutes and an arrival at 09:29; departing at 09:01 gives 10 minutes and an
arrival at 09:11. Leaving two minutes later gets you there eighteen minutes
earlier. That violates the **FIFO (non-overtaking) property**, and FIFO is
exactly the condition under which Dijkstra's label-setting argument survives
the move to time-dependent costs: without it, a longer-arrival label at an
intermediate node can still lead to an earlier arrival at the destination, so
settling a node permanently on first extraction is no longer valid and the
search silently returns non-optimal paths.

The fix is the step-speed traversal of Ichoua, Gendreau and Potvin (2003):
a vehicle travels at the speed of the period it is *currently in*, and when it
crosses a period boundary mid-edge it continues at the new speed for the rest
of the edge. Because speed is always positive, arrival time is a
non-decreasing function of departure time, so FIFO holds by construction and
label-setting Dijkstra is optimal again. :func:`traverse_edge` implements it
and ``tests/test_graph.py`` checks the FIFO property empirically on random
edges and random departure times.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numba import njit

from qroute.graph.network import RoadNetwork

#: Length of the cyclic time-of-day pattern, in seconds.
SECONDS_PER_DAY: float = 86_400.0


@dataclass
class PathResult:
    """A single shortest path and its cost breakdown."""

    nodes: list[int]           # internal node indices, source first
    cost: float                # cost in the units of the chosen weight
    duration_s: float
    distance_m: float
    mean_congestion: float = 0.0
    expanded: int = 0          # nodes settled, for comparing search effort

    @property
    def node_ids(self) -> list[int]:  # filled by the caller when useful
        return self.nodes

    def is_empty(self) -> bool:
        return len(self.nodes) == 0


# --------------------------------------------------------------------------
# Static shortest paths
# --------------------------------------------------------------------------

def _weight_arrays(network: RoadNetwork, weight: str) -> np.ndarray:
    if weight in ("travel_time", "time", "duration"):
        return network.csr_travel_time.data
    if weight in ("length", "distance"):
        return network.csr_length.data
    raise ValueError(f"unknown weight {weight!r}; use 'travel_time' or 'length'")


def _measure(network: RoadNetwork, path: Sequence[int]) -> tuple[float, float, float]:
    """Duration, distance and time-weighted mean congestion along ``path``."""
    duration = distance = congestion_time = 0.0
    for a, b in zip(path[:-1], path[1:]):
        e = network._fastest_edge_between(int(a), int(b))
        duration += float(network.edge_travel_time[e])
        distance += float(network.edge_length[e])
        congestion_time += float(network.edge_congestion[e]) * float(network.edge_travel_time[e])
    mean_cong = congestion_time / duration if duration > 0 else 0.0
    return duration, distance, mean_cong


def _reconstruct(pred: np.ndarray, source: int, target: int) -> list[int]:
    path = [int(target)]
    node = int(target)
    while node != source:
        node = int(pred[node])
        if node < 0:
            return []
        path.append(node)
    path.reverse()
    return path


def dijkstra(
    network: RoadNetwork,
    source: int,
    target: Optional[int] = None,
    *,
    weight: str = "travel_time",
) -> PathResult | tuple[np.ndarray, np.ndarray]:
    """Label-setting Dijkstra over the CSR adjacency.

    With ``target`` given, returns a :class:`PathResult` and stops as soon as
    the target is settled. Without it, returns ``(distances, predecessors)``
    over all nodes, with ``inf`` and ``-1`` for unreachable nodes (which cannot
    occur on a strongly connected network, and the tests assert that).

    A lazy binary heap is used: instead of decrease-key, a node may be pushed
    several times and stale pops are discarded on extraction. This is the
    standard trade for languages without an efficient decrease-key heap, and
    costs at most a constant factor in the heap size.
    """
    indptr = network._indptr
    indices = network._indices
    data = _weight_arrays(network, weight)
    n = network.n_nodes

    dist = np.full(n, np.inf, dtype=np.float64)
    pred = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(source))]
    expanded = 0

    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        expanded += 1
        if target is not None and u == target:
            break
        for p in range(indptr[u], indptr[u + 1]):
            v = int(indices[p])
            if done[v]:
                continue
            nd = d + data[p]
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    if target is None:
        return dist, pred
    if not np.isfinite(dist[target]):
        return PathResult([], float("inf"), float("inf"), float("inf"), 0.0, expanded)
    path = _reconstruct(pred, int(source), int(target))
    duration, distance, cong = _measure(network, path)
    return PathResult(path, float(dist[target]), duration, distance, cong, expanded)


def astar(
    network: RoadNetwork,
    source: int,
    target: int,
    *,
    weight: str = "travel_time",
) -> PathResult:
    """Goal-directed exact shortest path.

    The heuristic is the great-circle distance to the target, converted to a
    time bound by dividing by :attr:`RoadNetwork.max_speed_mps` when the weight
    is travel time. Both forms are admissible - a road can only be longer than
    the straight line, and no arc of the network is traversed faster than the
    fastest arc - and both are consistent (the triangle inequality holds for
    great-circle distance), so no node needs re-expansion and the first time
    the target is popped its label is final.

    The speed constant is taken from the *current* edge travel times, not from
    the free-flow speed table. Those coincide at free flow, but a call to
    :meth:`RoadNetwork.update_weights` with a factor below 1.0 speeds edges up
    past the table, and a heuristic pinned to the table would then over-estimate
    the remaining time and quietly return non-optimal paths. See
    :attr:`RoadNetwork.max_speed_mps` for the measurement behind that choice;
    ``tests/test_graph.py::test_astar_stays_exact_when_edges_are_sped_up`` is
    the regression test.
    """
    indptr = network._indptr
    indices = network._indices
    data = _weight_arrays(network, weight)
    n = network.n_nodes

    straight = network.haversine_to(int(target))
    if weight in ("travel_time", "time", "duration"):
        h = straight / network.max_speed_mps
    else:
        h = straight

    dist = np.full(n, np.inf, dtype=np.float64)
    pred = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(float(h[source]), int(source))]
    expanded = 0

    while heap:
        _f, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        expanded += 1
        if u == target:
            break
        du = dist[u]
        for p in range(indptr[u], indptr[u + 1]):
            v = int(indices[p])
            if done[v]:
                continue
            nd = du + data[p]
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd + h[v], v))

    if not np.isfinite(dist[target]):
        return PathResult([], float("inf"), float("inf"), float("inf"), 0.0, expanded)
    path = _reconstruct(pred, int(source), int(target))
    duration, distance, cong = _measure(network, path)
    return PathResult(path, float(dist[target]), duration, distance, cong, expanded)


# --------------------------------------------------------------------------
# Time-dependent travel times
# --------------------------------------------------------------------------

@dataclass
class SpeedProfile:
    """A cyclic, piecewise-constant time-of-day speed pattern.

    ``boundaries[p]`` is the start of period ``p`` in seconds since midnight,
    strictly increasing with ``boundaries[0] == 0``; the last period wraps
    around to ``cycle``. ``factors[p]`` multiplies the *free-flow speed* during
    period ``p``, so 1.0 is free flow and 0.4 means traffic moves at 40% of
    free-flow speed (and the edge takes 2.5x as long).

    ``factors`` may be one-dimensional (the same pattern everywhere) or of
    shape ``(n_periods, n_edges)`` when a traffic model gives every road
    segment its own profile.
    """

    boundaries: np.ndarray
    factors: np.ndarray
    cycle: float = SECONDS_PER_DAY
    name: str = "profile"

    def __post_init__(self) -> None:
        self.boundaries = np.ascontiguousarray(self.boundaries, dtype=np.float64)
        self.factors = np.ascontiguousarray(self.factors, dtype=np.float64)
        if self.boundaries.ndim != 1 or self.boundaries.size == 0:
            raise ValueError("boundaries must be a non-empty 1-D array")
        if self.boundaries[0] != 0.0:
            raise ValueError("boundaries must start at 0")
        if np.any(np.diff(self.boundaries) <= 0):
            raise ValueError("boundaries must be strictly increasing")
        if self.boundaries[-1] >= self.cycle:
            raise ValueError("boundaries must all be below the cycle length")
        if self.factors.shape[0] != self.boundaries.size:
            raise ValueError("factors must have one row per period")
        if np.any(self.factors <= 0):
            raise ValueError("speed factors must be strictly positive")

    @property
    def n_periods(self) -> int:
        return int(self.boundaries.size)

    @property
    def is_uniform(self) -> bool:
        """True when every edge shares the same pattern."""
        return self.factors.ndim == 1

    def edge_factors(self, edge: int) -> np.ndarray:
        """Per-period speed factors for one edge."""
        return self.factors if self.is_uniform else self.factors[:, edge]

    def factor_at(self, seconds: float, edge: int = 0) -> float:
        """Speed factor in force at an absolute time."""
        t = float(seconds) % self.cycle
        p = int(np.searchsorted(self.boundaries, t, side="right") - 1)
        return float(self.edge_factors(edge)[p])

    @classmethod
    def constant(cls, factor: float = 1.0, cycle: float = SECONDS_PER_DAY) -> "SpeedProfile":
        """A degenerate single-period profile; useful as a control in tests."""
        return cls(np.array([0.0]), np.array([float(factor)]), cycle, name="constant")


@njit(cache=True)
def _traverse(
    depart: float,
    free_flow_time: float,
    boundaries: np.ndarray,
    factors: np.ndarray,
    cycle: float,
) -> float:
    """Arrival time after traversing one edge, FIFO-correct (see module docstring).

    ``free_flow_time`` is the time the edge takes at factor 1.0. The vehicle
    covers the edge as a fraction in ``[0, 1]``; at factor ``f`` it progresses
    at ``f / free_flow_time`` per second, and the progress rate changes at every
    period boundary it crosses.
    """
    if free_flow_time <= 0.0:
        return depart
    n_periods = boundaries.shape[0]
    t = depart
    remaining = 1.0
    # A vehicle cannot cross more boundaries than this without the loop being a
    # bug; the bound also guarantees termination if a pathological factor is
    # ever supplied. 10 cycles at the finest period is far beyond any real trip.
    for _ in range(n_periods * 10 + 2):
        tod = t % cycle
        # Index of the period containing tod: last boundary <= tod.
        p = 0
        for i in range(n_periods):
            if boundaries[i] <= tod:
                p = i
            else:
                break
        f = factors[p]
        if p + 1 < n_periods:
            next_boundary = boundaries[p + 1]
        else:
            next_boundary = cycle
        dt = next_boundary - tod          # seconds left in this period
        progress = dt * f / free_flow_time
        if progress >= remaining:
            return t + remaining * free_flow_time / f
        remaining -= progress
        t += dt
    # Unreachable for positive factors; return a finite, clearly-wrong-if-hit value.
    return t


def traverse_edge(
    depart: float, free_flow_time: float, profile: SpeedProfile, edge: int = 0
) -> float:
    """Public wrapper around the step-speed traversal for one edge."""
    return float(
        _traverse(
            float(depart),
            float(free_flow_time),
            profile.boundaries,
            np.ascontiguousarray(profile.edge_factors(edge), dtype=np.float64),
            float(profile.cycle),
        )
    )


def _edge_adjacency(network: RoadNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Out-edge adjacency over *raw* MultiDiGraph edges, cached on the network.

    The static CSR keeps only the fastest parallel edge of each node pair, which
    is correct when weights are fixed. Under a time-dependent profile the
    fastest parallel edge can change with the hour, so the time-dependent search
    must consider all of them. The structure depends only on topology, so it is
    computed once and reused across departures and profiles.
    """
    cached = getattr(network, "_edge_adjacency_cache", None)
    if cached is not None:
        return cached
    tail, _head = network.edge_endpoints()
    order = np.argsort(tail, kind="stable").astype(np.int64)
    counts = np.bincount(tail, minlength=network.n_nodes)
    indptr = np.zeros(network.n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    cached = (indptr, order)
    network._edge_adjacency_cache = cached  # type: ignore[attr-defined]
    return cached


def time_dependent_dijkstra(
    network: RoadNetwork,
    source: int,
    depart_time: float,
    profile: SpeedProfile,
    *,
    target: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray] | PathResult:
    """Earliest-arrival paths from ``source`` leaving at ``depart_time``.

    Returns ``(arrival_times, predecessors)`` over all nodes, or a
    :class:`PathResult` when ``target`` is given. Arrival times are absolute
    seconds on the same clock as ``depart_time``.

    Correctness rests entirely on FIFO holding for :func:`traverse_edge`; see
    the module docstring. Given FIFO, the ordinary label-setting argument goes
    through unchanged: the smallest tentative arrival time in the queue can
    never be improved by first going somewhere else, because leaving later can
    never arrive earlier.
    """
    indptr, order = _edge_adjacency(network)
    head = network.edge_endpoints()[1]
    free_flow = network.edge_free_flow_time
    boundaries = profile.boundaries
    cycle = float(profile.cycle)
    uniform = profile.is_uniform
    uniform_factors = (
        np.ascontiguousarray(profile.factors, dtype=np.float64) if uniform else None
    )

    n = network.n_nodes
    arrive = np.full(n, np.inf, dtype=np.float64)
    pred = np.full(n, -1, dtype=np.int64)
    pred_edge = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    arrive[source] = float(depart_time)
    heap: list[tuple[float, int]] = [(float(depart_time), int(source))]
    expanded = 0

    while heap:
        t, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        expanded += 1
        if target is not None and u == target:
            break
        for p in range(indptr[u], indptr[u + 1]):
            e = int(order[p])
            v = int(head[e])
            if done[v]:
                continue
            factors = uniform_factors if uniform else np.ascontiguousarray(
                profile.factors[:, e], dtype=np.float64
            )
            arrival = _traverse(t, float(free_flow[e]), boundaries, factors, cycle)
            if arrival < arrive[v]:
                arrive[v] = arrival
                pred[v] = u
                pred_edge[v] = e
                heapq.heappush(heap, (arrival, v))

    if target is None:
        return arrive, pred
    if not np.isfinite(arrive[target]):
        return PathResult([], float("inf"), float("inf"), float("inf"), 0.0, expanded)

    path = _reconstruct(pred, int(source), int(target))
    distance = 0.0
    node = int(target)
    while node != int(source):
        distance += float(network.edge_length[int(pred_edge[node])])
        node = int(pred[node])
    duration = float(arrive[target] - depart_time)
    free_flow_total = 0.0
    node = int(target)
    while node != int(source):
        free_flow_total += float(free_flow[int(pred_edge[node])])
        node = int(pred[node])
    # Congestion here is the realised delay share, which is the same definition
    # RoadNetwork uses for its static congestion levels.
    cong = max(0.0, 1.0 - free_flow_total / duration) if duration > 0 else 0.0
    return PathResult(path, duration, duration, distance, cong, expanded)
