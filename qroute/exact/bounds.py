"""Cheap, provably valid lower bounds on the CVRP optimum.

Purpose
-------
Exact solvers close instances of a few dozen customers. Every benchmark
instance larger than that has no known optimum, only a best-known solution,
and quoting a gap against a best-known value silently assumes that value is
optimal. Lower bounds fix this: together with any feasible solution they
*bracket* the optimum, so a claim like "our answer is at most 4.1% above
optimal" becomes a statement that can be checked rather than believed.

Validity is the whole point
---------------------------
A lower bound that can exceed the optimum is worse than no bound at all,
because it turns a correct solver into an apparently buggy one and can be used
to "prove" a wrong optimality claim. Every function here therefore states the
argument for its validity in its docstring, and
:func:`~tests.test_exact` checks each one against the proven optima and the
best-known solutions of the shipped benchmark set. Where a bound needs an
assumption that the data may not satisfy -- the radial bound needs the triangle
inequality, which integer-rounded Euclidean distances can violate -- the
assumption is *verified numerically* and the function returns ``-inf`` rather
than an unjustified number.

All bounds are stated for the weighted arc cost actually minimised, that is
``instance.cost_matrix``, and assume it is non-negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qroute.exact.scaling import integer_demands
from qroute.problems.instance import Instance

__all__ = [
    "BoundReport",
    "bin_packing_bound",
    "degree_bound",
    "mst_bound",
    "radial_bound",
    "one_tree_bound",
    "lp_bound",
    "bracket",
]

_NEG_INF = -float("inf")


@dataclass
class BoundReport:
    """Every bound computed for one instance, plus the best of them.

    ``best`` is the maximum over the individually valid bounds, which is itself
    valid: the maximum of lower bounds is a lower bound.
    """

    instance: str
    bounds: dict[str, float] = field(default_factory=dict)
    vehicles_lb: int = 0
    seconds: float = 0.0
    upper_bound: Optional[float] = None

    @property
    def best(self) -> float:
        finite = [v for v in self.bounds.values() if math.isfinite(v)]
        return max(finite) if finite else _NEG_INF

    @property
    def gap_percent(self) -> Optional[float]:
        """Width of the bracket relative to the upper bound, in percent."""
        if self.upper_bound is None or self.upper_bound <= 0 or not math.isfinite(self.best):
            return None
        return 100.0 * (self.upper_bound - self.best) / self.upper_bound

    def as_dict(self) -> dict:
        return {
            "instance": self.instance,
            "bounds": dict(self.bounds),
            "best_lower_bound": self.best,
            "vehicles_lower_bound": self.vehicles_lb,
            "upper_bound": self.upper_bound,
            "gap_percent": self.gap_percent,
            "seconds": self.seconds,
        }


# ------------------------------------------------------------------ vehicles
def bin_packing_bound(demand: np.ndarray, capacity: float) -> int:
    """Lower bound on the number of vehicles, via Martello and Toth's L2.

    The number of routes in any feasible CVRP solution is at least the number
    of bins needed to pack the demands into bins of size ``Q``, because each
    route is one such bin. ``L1 = ceil(sum d / Q)`` is the obvious bound; L2
    dominates it by reasoning about items too large to share a bin.

    For a threshold ``K`` in ``[0, Q/2]`` split the items into
    ``N1 = {d > Q - K}``, ``N2 = {Q/2 < d <= Q - K}`` and ``N3 = {K <= d <= Q/2}``.
    No two items of ``N1 union N2`` fit together, so they need
    ``|N1| + |N2|`` distinct bins, and only the ``N2`` bins have room left --
    at most ``|N2| * Q - sum(N2)`` in total. Whatever of ``N3`` does not fit in
    that residual space needs further bins. Maximising over ``K`` gives L2.
    """
    d = np.asarray(demand, dtype=np.float64)
    d = d[d > 0]
    if d.size == 0:
        return 0
    q = float(capacity)
    l1 = int(math.ceil(d.sum() / q - 1e-9))
    best = l1
    # Only item sizes matter as thresholds; K = 0 recovers a variant of L1.
    thresholds = sorted({0.0} | {float(v) for v in d if v <= q / 2.0})
    for k in thresholds:
        n1 = d[d > q - k]
        n2 = d[(d > q / 2.0) & (d <= q - k)]
        n3 = d[(d >= k) & (d <= q / 2.0)]
        free = n2.size * q - n2.sum()
        extra = max(0.0, n3.sum() - free)
        val = n1.size + n2.size + int(math.ceil(extra / q - 1e-9))
        best = max(best, val)
    return int(best)


# --------------------------------------------------------------- arc degrees
def degree_bound(instance: Instance, vehicles_lb: int | None = None) -> float:
    """Lower bound from the in/out degree structure of any feasible solution.

    In a solution with ``r`` routes every customer has out-degree exactly one
    and the depot has out-degree exactly ``r``, and those arcs partition the
    whole arc set. Hence

        cost = sum over customers of c[i, succ(i)] + sum over the r depot arcs

    The first term is at least the sum of each customer's cheapest outgoing
    arc. The ``r`` depot arcs have distinct heads and non-negative costs, and
    ``r >= r_min``, so the second term is at least the sum of the ``r_min``
    smallest entries of the depot row. The same argument on in-degrees gives a
    second bound; both are valid, so the larger is returned.
    """
    c = instance.cost_matrix
    n = c.shape[0]
    if n < 2:
        return 0.0
    k = vehicles_lb if vehicles_lb is not None else bin_packing_bound(instance.demand, instance.capacity)
    k = max(1, min(int(k), n - 1))

    off = c.copy()
    np.fill_diagonal(off, np.inf)

    cheapest_out = off[1:, :].min(axis=1).sum()
    depot_out = np.sort(c[0, 1:])[:k].sum()
    lb_out = float(cheapest_out + depot_out)

    cheapest_in = off[:, 1:].min(axis=0).sum()
    depot_in = np.sort(c[1:, 0])[:k].sum()
    lb_in = float(cheapest_in + depot_in)

    return max(lb_out, lb_in)


# ------------------------------------------------------------ spanning trees
def _symmetrised(c: np.ndarray) -> np.ndarray:
    """Undirected relaxation: keep the cheaper of the two directions."""
    return np.minimum(c, c.T)


def _mst(w: np.ndarray) -> tuple[float, object]:
    """Minimum spanning tree of the dense symmetric matrix ``w``.

    Returns its total weight under the *original* weights and the sparse tree
    itself, which the 1-tree bound needs in order to read off node degrees.

    SciPy's sparse MST treats a zero entry as "no edge", and the 1-tree bound
    feeds it weights shifted by node potentials that can be negative. Every
    weight is therefore translated so the smallest becomes 1, the tree is built,
    and the ``k - 1`` units of translation are subtracted again -- a spanning
    tree of a ``k``-node graph always has exactly ``k - 1`` edges, so the
    correction is exact and the choice of tree is unaffected.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    k = w.shape[0]
    if k < 2:
        return 0.0, None
    off = w.copy()
    np.fill_diagonal(off, np.inf)
    offset = 1.0 - float(off.min())
    shifted = w + offset
    np.fill_diagonal(shifted, 0.0)
    tree = minimum_spanning_tree(shifted)
    return float(tree.sum()) - (k - 1) * offset, tree


def mst_bound(instance: Instance, vehicles_lb: int | None = None) -> float:
    """Minimum-spanning-tree bound.

    Viewed as an undirected multigraph, a solution with ``r`` routes has
    ``n + r`` edges (a route over ``m`` customers contributes ``m + 1``) and is
    connected and spanning, because every route touches the depot. Any
    connected spanning multigraph contains a spanning tree, whose weight is at
    least the MST weight, and the ``r`` leftover edges are non-negative. Using
    ``min(c[i,j], c[j,i])`` as the undirected weight only weakens the bound, so
    it stays valid for asymmetric costs.

    The leftover edges are added back as ``r_min`` copies of the globally
    cheapest edge. Using the ``r_min`` cheapest *distinct* edges would be
    invalid, because a two-customer-free route ``depot -> i -> depot`` uses the
    same undirected edge twice.
    """
    c = _symmetrised(instance.cost_matrix)
    n = c.shape[0]
    if n < 2:
        return 0.0
    k = vehicles_lb if vehicles_lb is not None else bin_packing_bound(instance.demand, instance.capacity)

    mst_weight, _tree = _mst(c)

    off = c.copy()
    np.fill_diagonal(off, np.inf)
    cheapest_edge = float(off.min())
    return mst_weight + k * cheapest_edge


def one_tree_bound(cost: np.ndarray, iterations: int = 200, step: float = 2.0) -> float:
    """Held-Karp 1-tree bound for the **TSP** over ``cost``.

    A tour is a 1-tree: a spanning tree over nodes ``1..n-1`` plus two edges at
    node 0. Minimising over all 1-trees therefore under-estimates the optimal
    tour. Adding node potentials ``pi`` to every incident edge shifts every
    tour's cost by exactly ``2 * sum(pi)``, so

        LB(pi) = minimum 1-tree under c[i,j] + pi[i] + pi[j]  -  2 * sum(pi)

    is a valid bound for every ``pi``; subgradient ascent on ``pi`` tightens it.
    The best value seen is returned, so a badly chosen step size can only make
    the bound weaker, never invalid.

    This is a TSP bound. It is *not* valid for the CVRP, where the solution is
    not a single tour -- use :func:`mst_bound` there.
    """
    c = _symmetrised(np.asarray(cost, dtype=np.float64))
    n = c.shape[0]
    if n < 3:
        return float(c[0, 1] + c[1, 0]) if n == 2 else 0.0

    pi = np.zeros(n, dtype=np.float64)
    best = _NEG_INF
    # Scale the step by an estimate of the tour cost so the schedule is
    # instance independent.
    off = c.copy()
    np.fill_diagonal(off, np.inf)
    scale = float(off.min(axis=1).sum()) / max(n, 1)

    for it in range(iterations):
        w = c + pi[:, None] + pi[None, :]
        tree_weight, tree = _mst(np.ascontiguousarray(w[1:, 1:]))

        deg = np.zeros(n, dtype=np.float64)
        rows, cols = tree.nonzero()
        for r, cix in zip(rows, cols):
            deg[r + 1] += 1
            deg[cix + 1] += 1

        # Node 0 rejoins through its two cheapest edges.
        two = np.sort(w[0, 1:])[:2]
        value = tree_weight + float(two.sum()) - 2.0 * float(pi.sum())
        deg[0] = 2.0
        # Recover which two nodes were picked so their degrees are right.
        order = np.argsort(w[0, 1:])[:2]
        for o in order:
            deg[o + 1] += 1

        if value > best:
            best = value
        grad = deg - 2.0
        if not np.any(grad):
            break  # the 1-tree is a tour: the bound is exact
        norm = float(np.dot(grad, grad))
        if norm <= 0:
            break
        t = step * scale / (1.0 + it) / math.sqrt(norm)
        pi = pi + t * grad

    return best


# ------------------------------------------------------------- radial bound
def _violates_triangle(c: np.ndarray, tol: float = 1e-9) -> bool:
    """Exact check of ``c[i,j] <= c[i,k] + c[k,j]`` for every triple.

    Done as ``n`` vectorised passes rather than a triple loop; for ``n = 1000``
    this is about ``10^9`` element operations, a few seconds, and it is a proof
    rather than a sample.
    """
    n = c.shape[0]
    for k in range(n):
        if (c - (c[:, k][:, None] + c[k][None, :])).max() > tol:
            return True
    return False


def radial_bound(instance: Instance, allow_closure: bool = True, closure_limit: int = 400) -> float:
    """Haimovich and Rinnooy Kan's radial bound.

    For a metric CVRP,

        OPT >= (2 / Q) * sum over customers of  demand[i] * c[depot, i]

    The argument is that a vehicle carrying ``q`` units to customer ``i`` must
    cover the depot-to-``i`` distance in both directions, and each unit of
    demand can "share" a round trip with at most ``Q`` units.

    The bound needs the triangle inequality. Integer-rounded Euclidean
    distances can violate it by up to one unit per triple, so this function
    checks the matrix explicitly. If the check fails and the instance is small
    enough it falls back to the metric closure ``c*`` (all-pairs shortest
    paths), which satisfies the inequality by construction and whose optimum
    lower-bounds the optimum under ``c`` because ``c* <= c`` entrywise. If
    neither route is available it returns ``-inf`` instead of a number it
    cannot justify.
    """
    c = np.ascontiguousarray(instance.cost_matrix)
    n = c.shape[0]
    if n < 2:
        return 0.0
    if _violates_triangle(c):
        if not allow_closure or n > closure_limit:
            return _NEG_INF
        from scipy.sparse.csgraph import shortest_path

        c = shortest_path(c, method="FW", directed=True)

    _demand, capacity, _total = integer_demands(instance.demand, instance.capacity)
    radial = float(np.dot(instance.demand[1:], c[0, 1:]))
    return 2.0 * radial / float(capacity)


# ----------------------------------------------------------------- LP bound
def lp_bound(
    instance: Instance,
    formulation: str = "flow",
    max_nodes: int = 250,
    time_limit: float = 120.0,
) -> float:
    """Linear relaxation of the two-index model in :mod:`qroute.exact.milp`.

    The LP optimum of a relaxation of the MILP is a valid lower bound on the
    MILP optimum, which is the CVRP optimum. ``max_nodes`` guards against
    building an ``O(n^2)``-variable LP for a thousand-customer instance, where
    it would take longer than the whole benchmark; the bound is skipped
    (``-inf``) rather than attempted.
    """
    if instance.has_time_windows or instance.size > max_nodes:
        return _NEG_INF
    from qroute.exact.milp import lp_relaxation_value

    try:
        return lp_relaxation_value(instance, formulation=formulation, time_limit=time_limit)
    except Exception:  # pragma: no cover - backend dependent
        return _NEG_INF


# ------------------------------------------------------------------ bracket
def bracket(
    instance: Instance,
    upper_bound: float | None = None,
    include_lp: bool = True,
    lp_formulation: str = "flow",
    lp_max_nodes: int = 250,
    lp_time_limit: float = 120.0,
) -> BoundReport:
    """Compute every applicable bound and report the strongest.

    ``upper_bound`` is any feasible solution's cost (typically the best-known
    solution or the platform's own best answer); supplying it lets the report
    state the width of the bracket the optimum is known to lie in.
    """
    import time

    t0 = time.perf_counter()
    k = bin_packing_bound(instance.demand, instance.capacity)
    values: dict[str, float] = {
        "degree": degree_bound(instance, k),
        "mst": mst_bound(instance, k),
        "radial": radial_bound(instance),
    }
    if include_lp:
        values[f"lp_{lp_formulation}"] = lp_bound(
            instance, formulation=lp_formulation, max_nodes=lp_max_nodes, time_limit=lp_time_limit
        )
    if upper_bound is None:
        upper_bound = instance.meta.get("bks")
    return BoundReport(
        instance=instance.name,
        bounds=values,
        vehicles_lb=k,
        seconds=time.perf_counter() - t0,
        upper_bound=float(upper_bound) if upper_bound is not None else None,
    )
