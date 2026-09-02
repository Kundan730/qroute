"""Many-to-many travel-time, distance and congestion matrices.

The optimiser never touches the road graph directly: it works on a dense
``(n+1, n+1)`` matrix over depot and customers. Producing that matrix is the
single most expensive preprocessing step in the platform, so it is worth doing
properly.

Why SciPy rather than NetworkX
------------------------------
``networkx.single_source_dijkstra`` walks Python dictionaries and allocates a
Python object per settled node. ``scipy.sparse.csgraph.dijkstra`` runs a
Fibonacci-heap implementation in compiled code over the CSR arrays and accepts
``indices=[...]`` to run many sources in one call. Measured on the bundled
Bengaluru network after taking the strongly connected component (13343 nodes,
34266 edges), a 50-source one-to-all solve takes 0.18-0.22 s with SciPy against
2.60-2.76 s with NetworkX on an already-built ``DiGraph`` - a factor of 12-14.
:func:`networkx_reference` reproduces the NetworkX version and the tests assert
that the two agree exactly, so the speed is not bought with a different answer.

Distance and congestion along the *time-optimal* path
-----------------------------------------------------
A subtlety that is easy to get wrong: the distance matrix must record the
length of the path the vehicle actually drives, which is the fastest path, not
the length of the separate shortest-by-distance path. Running Dijkstra twice
and pairing the results would report a distance the driver never covers, and
the reported total distance of a route would not correspond to its duration.

We therefore run Dijkstra once on travel time with ``return_predecessors=True``
and accumulate length (and congestion x time) along the resulting shortest-path
tree. Because every node's predecessor link is a tree edge, and the nodes
sorted by increasing distance from the source form a valid topological order of
that tree, one linear sweep per source suffices. That sweep is a genuine scalar
inner loop over ~13k nodes per source, so it is compiled with Numba; in pure
Python it dominated the runtime of the whole matrix build.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra

from qroute.graph.network import RoadNetwork


@dataclass
class MatrixResult:
    """Origin-destination matrices over a chosen set of network nodes."""

    nodes: np.ndarray            # internal node indices of the stops, in order
    duration: np.ndarray         # (k, k) seconds
    distance: np.ndarray         # (k, k) metres, along the fastest path
    congestion: np.ndarray       # (k, k) time-weighted mean congestion in [0, 1]
    predecessors: Optional[np.ndarray] = None   # (k, n_nodes), for path recovery
    seconds: float = 0.0         # wall clock spent building this

    @property
    def size(self) -> int:
        return int(self.nodes.size)

    @property
    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.duration)))


@njit(cache=True, parallel=True)
def _accumulate_along_tree(
    indptr: np.ndarray,
    indices: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    pred: np.ndarray,
    order: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
) -> None:
    """Sum two edge attributes along each source's shortest-path tree.

    ``pred[s, v]`` is the predecessor of ``v`` in the tree rooted at source
    ``s`` (-1 for the root and for unreachable nodes). ``order[s]`` lists the
    nodes in non-decreasing distance from the source, which guarantees a parent
    is processed before its children. The edge attribute for ``(u, v)`` is
    looked up by binary search in the CSR row of ``u``.

    Both attributes (path length and congestion x time) are accumulated in the
    same pass so the binary search is paid for once rather than twice, and the
    sources are split across cores with ``prange`` - the trees are independent,
    so there is no sharing and no reduction.
    """
    n_sources = pred.shape[0]
    n_nodes = pred.shape[1]
    for s in prange(n_sources):
        for i in range(n_nodes):
            v = order[s, i]
            u = pred[s, v]
            if u < 0:
                out_a[s, v] = 0.0
                out_b[s, v] = 0.0
                continue
            lo = indptr[u]
            hi = indptr[u + 1]
            # Binary search for column v in row u (CSR columns are sorted).
            found = -1
            while lo < hi:
                mid = (lo + hi) // 2
                if indices[mid] < v:
                    lo = mid + 1
                elif indices[mid] > v:
                    hi = mid
                else:
                    found = mid
                    break
            if found < 0:
                out_a[s, v] = np.inf
                out_b[s, v] = np.inf
            else:
                out_a[s, v] = out_a[s, u] + values_a[found]
                out_b[s, v] = out_b[s, u] + values_b[found]


def travel_time_matrix(
    network: RoadNetwork,
    sources: Sequence[int] | np.ndarray,
    targets: Optional[Sequence[int] | np.ndarray] = None,
    *,
    weight: str = "travel_time",
    return_predecessors: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Dense ``(len(sources), len(targets))`` matrix of shortest-path costs.

    ``targets`` defaults to ``sources``. Returns ``(matrix, predecessors)``
    where ``predecessors`` is ``None`` unless requested; it is a
    ``(len(sources), n_nodes)`` array in SciPy's convention (-9999 for "no
    predecessor"), which :func:`reconstruct_path` understands.
    """
    src = np.asarray(sources, dtype=np.int32)
    dst = src if targets is None else np.asarray(targets, dtype=np.int32)
    graph = network.csr_travel_time if weight in ("travel_time", "time", "duration") \
        else network.csr_length

    result = csgraph_dijkstra(
        graph, directed=True, indices=src, return_predecessors=return_predecessors
    )
    if return_predecessors:
        dist, pred = result
    else:
        dist, pred = result, None
    return np.ascontiguousarray(dist[:, dst]), pred


def build_matrices(
    network: RoadNetwork,
    nodes: Sequence[int] | np.ndarray,
    *,
    keep_predecessors: bool = True,
) -> MatrixResult:
    """Duration, distance and congestion matrices over ``nodes``.

    All three describe the same, time-optimal paths (see the module docstring).
    ``nodes`` are internal node indices; the depot is expected first when the
    caller is building a routing instance, but nothing here depends on that.
    """
    start = time.perf_counter()
    stops = np.asarray(nodes, dtype=np.int32)
    k = stops.size

    dist_all, pred_all = csgraph_dijkstra(
        network.csr_travel_time, directed=True, indices=stops, return_predecessors=True
    )
    duration = np.ascontiguousarray(dist_all[:, stops])

    # SciPy marks "no predecessor" with -9999; the kernel expects -1.
    pred = np.where(pred_all < 0, -1, pred_all).astype(np.int32)
    # Sorting by distance gives a topological order of each shortest-path tree.
    # Ties cannot occur along a tree edge because travel times are strictly
    # positive (clamped at MIN_TRAVEL_TIME_S), so an unstable sort is safe and
    # measurably faster. Unreachable nodes carry inf, sort last, and have
    # predecessor -1; they are masked back to inf below.
    order = np.argsort(dist_all, axis=1).astype(np.int32)

    n = network.n_nodes
    indptr = network._indptr.astype(np.int32)
    indices = network._indices.astype(np.int32)

    acc_len = np.zeros((k, n), dtype=np.float64)
    acc_cong = np.zeros((k, n), dtype=np.float64)
    congestion_time = network.csr_congestion.data * network.csr_travel_time.data
    _accumulate_along_tree(
        indptr, indices, network.csr_length.data, congestion_time,
        pred, order, acc_len, acc_cong,
    )

    unreachable = ~np.isfinite(dist_all)
    acc_len[unreachable] = np.inf
    acc_cong[unreachable] = np.inf

    distance = np.ascontiguousarray(acc_len[:, stops])
    with np.errstate(invalid="ignore", divide="ignore"):
        congestion = np.where(duration > 0, acc_cong[:, stops] / np.maximum(duration, 1e-12), 0.0)
    congestion = np.clip(np.nan_to_num(congestion, nan=0.0, posinf=0.0), 0.0, 1.0)

    # A node's path to itself is empty; force the diagonal to exact zeros so the
    # optimiser never sees a spurious self-loop cost from floating-point noise.
    np.fill_diagonal(duration, 0.0)
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(congestion, 0.0)

    return MatrixResult(
        nodes=stops.astype(np.int64),
        duration=duration,
        distance=distance,
        congestion=congestion,
        predecessors=pred_all if keep_predecessors else None,
        seconds=time.perf_counter() - start,
    )


def reconstruct_path(predecessors: np.ndarray, source_row: int, target: int) -> list[int]:
    """Recover the node path to ``target`` from a SciPy predecessor matrix.

    ``source_row`` indexes the row of ``predecessors`` (i.e. the position of the
    source in the ``indices`` list passed to Dijkstra), not the node itself.
    Returns internal node indices from source to target. A single-element list
    means the target is the source itself or was never reached; the caller
    distinguishes the two from the duration matrix.
    """
    row = predecessors[source_row]
    path = [int(target)]
    node = int(target)
    while True:
        prev = int(row[node])
        if prev < 0:
            break
        path.append(prev)
        node = prev
    path.reverse()
    return path


def route_node_path(
    network: RoadNetwork, result: MatrixResult, i: int, j: int
) -> list[int]:
    """Full road-network node path between stop ``i`` and stop ``j`` of a matrix.

    This is what turns an optimiser's answer (a sequence of stop indices) into a
    polyline the map can draw; feed the output to
    :meth:`RoadNetwork.route_geojson`.
    """
    if result.predecessors is None:
        raise ValueError("matrix was built without predecessors; pass keep_predecessors=True")
    if i == j:
        return [int(result.nodes[i])]
    return reconstruct_path(result.predecessors, i, int(result.nodes[j]))


def networkx_reference(
    network: RoadNetwork, sources: Sequence[int], weight: str = "travel_time"
) -> np.ndarray:
    """Same matrix computed with NetworkX, as a ground-truth check for tests.

    Deliberately slow and simple. It rebuilds a plain DiGraph collapsing
    parallel edges by minimum, exactly as :class:`RoadNetwork` does, so any
    disagreement points at a real bug rather than at a modelling difference.
    """
    import networkx as nx

    tail, head = network.edge_endpoints()
    values = network.edge_travel_time if weight == "travel_time" else network.edge_length
    simple = nx.DiGraph()
    simple.add_nodes_from(range(network.n_nodes))
    for e in range(network.n_edges):
        u, v, w = int(tail[e]), int(head[e]), float(values[e])
        existing = simple.get_edge_data(u, v)
        if existing is None or w < existing["weight"]:
            simple.add_edge(u, v, weight=w)

    out = np.full((len(sources), network.n_nodes), np.inf)
    for r, s in enumerate(sources):
        lengths = nx.single_source_dijkstra_path_length(simple, int(s), weight="weight")
        for node, d in lengths.items():
            out[r, node] = d
    return out
