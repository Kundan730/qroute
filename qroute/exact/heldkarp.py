"""Exact dynamic programmes over subsets: Held-Karp for the TSP and a
set-partitioning DP for very small capacitated VRPs.

These are the smallest, most transparent exact methods in the platform. They
have no solver dependency and no parameters to tune, so when CP-SAT and the
MILP agree with the DP on a tiny instance there is very little room left for a
shared modelling mistake. That cross-validation is their main job; they are not
competitive on anything larger than a toy.

Held-Karp (TSP)
---------------
With ``m = n - 1`` customers, ``dp[S][j]`` is the cheapest path that starts at
the depot, visits exactly the customer set ``S`` and ends at customer ``j``.
The recurrence ``dp[S | {k}][k] = min_j dp[S][j] + c[j][k]`` runs in
``O(2^m * m^2)`` time and ``O(2^m * m)`` memory. The memory is the binding
constraint in practice, so the entry point takes an explicit budget in
megabytes and refuses -- loudly -- to start a run that would exceed it, rather
than thrashing swap and appearing to hang.

Table sizes (float64 costs plus an int16 parent table, 10 bytes per state):

    n = 16   ->     5 MB
    n = 20   ->   100 MB
    n = 22   ->   440 MB
    n = 24   ->  1900 MB

The default budget of 512 MB therefore admits ``n <= 22``, which
:func:`max_tsp_nodes` reports. Running time grows as ``2^m * m^2`` and this is
a NumPy loop over masks, so anything past ``n = 20`` is slow even when it fits.
The documented, tested comfort zone is ``n <= 15``.

Exact CVRP by set partitioning
------------------------------
For the CVRP the DP has two stages. First the Held-Karp table is built once
over all *capacity-feasible* customer subsets, giving ``route[S]``, the optimal
single-vehicle round trip serving exactly ``S``. Since demand is monotone under
subset inclusion, every prefix of a feasible subset is feasible and the table
stays consistent. Second, a partition DP composes those routes::

    best[S] = min over T subset of S containing lowest(S), demand(T) <= Q
              of  route[T] + best[S \\ T]

Fixing the lowest-indexed customer of ``S`` to lie in ``T`` counts every
partition exactly once. The stage-two cost is ``O(3^m / 2)``, which is the real
limit: ``m = 13`` is 0.8 million submask steps, ``m = 15`` is 7 million,
``m = 16`` is 22 million and ``m = 18`` is 194 million. Measured on this
machine with numba, ``m = 15`` (P-n16-k8) takes 1.2 s end to end. Without
numba the same run is roughly two orders of magnitude slower, which is why the
default cap is ``max_customers = 16`` and larger values must be asked for
explicitly.

The result is a *proven optimum* for the unrestricted-fleet CVRP: the partition
DP considers every possible way of splitting the customers into capacity-
feasible routes, and each route is optimally sequenced.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from qroute.algorithms.base import OptimizationResult
from qroute.core.types import Solution
from qroute.exact.scaling import integer_demands
from qroute.problems.instance import Instance

__all__ = [
    "DPResult",
    "held_karp_tsp",
    "held_karp_cvrp",
    "solve_heldkarp",
    "max_tsp_nodes",
]

_INF = float("inf")

try:  # numba is a hard dependency of the project, but keep the DP usable without it
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is installed in this environment
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        return wrap if not args else args[0]


@dataclass
class DPResult:
    """Outcome of an exact dynamic programme.

    ``lower_bound`` equals ``cost`` because the DP enumerates the entire
    feasible set: there is nothing left to bound.
    """

    routes: list[list[int]]
    cost: float
    lower_bound: float
    status: str
    seconds: float
    proven_optimal: bool
    states: int = 0
    instance_name: str = ""

    def as_tuple(self) -> tuple[list[list[int]], float, float, str, float]:
        return self.routes, self.cost, self.lower_bound, self.status, self.seconds

    def to_optimization_result(self, instance: Instance) -> OptimizationResult:
        if self.routes:
            best = instance.make_solution(self.routes)
            best.validate(instance.n_customers)
        else:
            best = Solution()
        return OptimizationResult(
            algorithm="held-karp",
            instance=instance.name,
            best=best,
            history=[],
            iterations=0,
            evaluations=self.states,
            seconds=self.seconds,
            seed=None,
            params={
                "solver": "subset-dp",
                "status": self.status,
                "lower_bound": self.lower_bound,
                "proven_optimal": self.proven_optimal,
                "states": self.states,
                "n_routes": len(self.routes),
            },
        )


def max_tsp_nodes(memory_limit_mb: float = 512.0) -> int:
    """Largest ``n`` whose Held-Karp tables fit in ``memory_limit_mb``.

    Counts 8 bytes for the cost table and 2 for the parent table per state.
    """
    budget = memory_limit_mb * 1024 * 1024
    n = 2
    while True:
        m = n - 1
        need = (1 << m) * m * 10.0
        if need > budget:
            return n - 1
        n += 1
        if n > 40:  # pragma: no cover - unreachable for sane budgets
            return 40


# --------------------------------------------------------------------- TSP
def held_karp_tsp(cost: np.ndarray, memory_limit_mb: float = 512.0) -> DPResult:
    """Optimal TSP tour over ``cost``, starting and ending at node 0.

    Works for asymmetric matrices. Returns the tour as a single "route", i.e.
    the visiting order of nodes ``1..n-1`` with the depot implicit, matching
    the project's route convention.
    """
    c = np.ascontiguousarray(np.asarray(cost, dtype=np.float64))
    n = c.shape[0]
    if c.shape != (n, n):
        raise ValueError("cost matrix must be square")
    if n <= 1:
        return DPResult([], 0.0, 0.0, "OPTIMAL", 0.0, True, 0)
    if n == 2:
        return DPResult([[1]], float(c[0, 1] + c[1, 0]), float(c[0, 1] + c[1, 0]), "OPTIMAL", 0.0, True, 1)

    m = n - 1
    limit = max_tsp_nodes(memory_limit_mb)
    if n > limit:
        need_mb = (1 << m) * m * 10.0 / (1024 * 1024)
        raise MemoryError(
            f"Held-Karp on n={n} needs about {need_mb:.0f} MB for its state tables, "
            f"above the {memory_limit_mb:.0f} MB budget (largest allowed n is {limit}). "
            "Raise memory_limit_mb deliberately, or use qroute.exact.cpsat for a "
            "solver-based exact answer."
        )

    t0 = time.perf_counter()
    size = 1 << m
    # Sub-matrix over customers only; index k here is customer k+1.
    cc = np.ascontiguousarray(c[1:, 1:])
    from_depot = np.ascontiguousarray(c[0, 1:])
    to_depot = np.ascontiguousarray(c[1:, 0])

    dp = np.full((size, m), _INF, dtype=np.float64)
    parent = np.full((size, m), -1, dtype=np.int16)
    for j in range(m):
        dp[1 << j, j] = from_depot[j]

    for mask in range(1, size):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        # cand[k] = cheapest way to arrive at k from any end j already in mask.
        cand = row[:, None] + cc            # (m, m); infinite rows drop out on their own
        best_prev = np.argmin(cand, axis=0)
        best_val = cand[best_prev, np.arange(m)]
        for k in range(m):
            if mask >> k & 1:
                continue
            v = best_val[k]
            if v == _INF:
                continue
            nm = mask | (1 << k)
            if v < dp[nm, k]:
                dp[nm, k] = v
                parent[nm, k] = best_prev[k]

    full = size - 1
    closing = dp[full] + to_depot
    end = int(np.argmin(closing))
    total = float(closing[end])

    tour: list[int] = []
    mask, node = full, end
    while node >= 0:
        tour.append(node + 1)
        prev = int(parent[mask, node])
        mask ^= 1 << node
        node = prev
    tour.reverse()

    return DPResult(
        routes=[tour],
        cost=total,
        lower_bound=total,
        status="OPTIMAL",
        seconds=time.perf_counter() - t0,
        proven_optimal=True,
        states=size * m,
        instance_name="tsp",
    )


# -------------------------------------------------------------------- CVRP
@njit(cache=True)
def _route_costs(cc, from_depot, to_depot, demand, capacity, size, m):
    """Held-Karp restricted to capacity-feasible subsets.

    Returns ``route[S]`` (optimal round-trip cost for exactly ``S``) together
    with the ``dp``/``parent`` tables needed to rebuild each route's order.
    """
    dp = np.full((size, m), np.inf, dtype=np.float64)
    parent = np.full((size, m), -1, dtype=np.int16)
    load = np.zeros(size, dtype=np.int64)
    route = np.full(size, np.inf, dtype=np.float64)

    for mask in range(1, size):
        low = 0
        while not (mask >> low) & 1:
            low += 1
        load[mask] = load[mask ^ (1 << low)] + demand[low]

    for j in range(m):
        if demand[j] <= capacity:
            dp[1 << j, j] = from_depot[j]

    for mask in range(1, size):
        if load[mask] > capacity:
            continue
        best_end = np.inf
        for j in range(m):
            v = dp[mask, j]
            if v == np.inf:
                continue
            closing = v + to_depot[j]
            if closing < best_end:
                best_end = closing
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                if load[mask] + demand[k] > capacity:
                    continue
                nv = v + cc[j, k]
                nm = mask | (1 << k)
                if nv < dp[nm, k]:
                    dp[nm, k] = nv
                    parent[nm, k] = j
        route[mask] = best_end
    return dp, parent, route, load


@njit(cache=True)
def _partition(route, size, full):
    """Set-partitioning DP over capacity-feasible routes.

    ``best[S]`` is the optimal cost of serving exactly the customer set ``S``
    with any number of vehicles; ``choice[S]`` records the route removed first.
    """
    best = np.full(size, np.inf, dtype=np.float64)
    choice = np.zeros(size, dtype=np.int64)
    best[0] = 0.0
    for s in range(1, full + 1):
        low = 0
        while not (s >> low) & 1:
            low += 1
        lowbit = 1 << low
        rest = s ^ lowbit
        # Enumerate submasks of `rest`; every candidate route also contains
        # `lowbit`, so each partition is generated exactly once.
        sub = rest
        while True:
            t = sub | lowbit
            rc = route[t]
            if rc != np.inf:
                other = best[s ^ t]
                if other != np.inf:
                    v = rc + other
                    if v < best[s]:
                        best[s] = v
                        choice[s] = t
            if sub == 0:
                break
            sub = (sub - 1) & rest
    return best, choice


def held_karp_cvrp(
    instance: Instance,
    max_customers: int = 16,
    memory_limit_mb: float = 512.0,
) -> DPResult:
    """Exact CVRP for very small instances by enumerating optimal routes.

    The fleet is unrestricted unless ``instance.n_vehicles`` is set, in which
    case solutions using more routes than the fleet allows are rejected after
    the fact -- the DP itself does not track the route count, so a fleet cap
    that binds will make this raise rather than silently return an
    infeasible-for-the-fleet answer.

    Time windows are not supported: the partition DP assumes the cost of a
    route depends only on its customer *set*, which stops being true once
    waiting times and window feasibility enter.
    """
    if instance.has_time_windows:
        raise NotImplementedError(
            "the subset DP assumes route cost depends only on the customer set, "
            "which fails with time windows; use qroute.exact.cpsat instead"
        )
    m = instance.n_customers
    if m > max_customers:
        steps = 3.0 ** m / 2.0
        raise ValueError(
            f"exact subset DP refuses n={m} customers: stage two costs about "
            f"{steps:.3g} submask steps (limit is max_customers={max_customers}). "
            "This method is for cross-validating small instances only."
        )
    need_mb = (1 << m) * m * 10.0 / (1024 * 1024)
    if need_mb > memory_limit_mb:
        raise MemoryError(
            f"subset DP on n={m} customers needs about {need_mb:.0f} MB, above the "
            f"{memory_limit_mb:.0f} MB budget"
        )

    t0 = time.perf_counter()
    cost = instance.cost_matrix
    demand, capacity, _total = integer_demands(instance.demand, instance.capacity)
    cc = np.ascontiguousarray(cost[1:, 1:])
    from_depot = np.ascontiguousarray(cost[0, 1:])
    to_depot = np.ascontiguousarray(cost[1:, 0])
    dem = np.ascontiguousarray(demand[1:])

    size = 1 << m
    full = size - 1
    dp, parent, route, _load = _route_costs(cc, from_depot, to_depot, dem, capacity, size, m)
    best, choice = _partition(route, size, full)

    total = float(best[full])
    if not math.isfinite(total):
        return DPResult([], _INF, _INF, "INFEASIBLE", time.perf_counter() - t0, True, size * m,
                        instance.name)

    routes: list[list[int]] = []
    s = full
    while s:
        t = int(choice[s])
        routes.append(_rebuild_route(dp, parent, to_depot, t, m))
        s ^= t

    if instance.n_vehicles is not None and len(routes) > instance.n_vehicles:
        raise ValueError(
            f"the unrestricted-fleet optimum uses {len(routes)} routes but the instance "
            f"caps the fleet at {instance.n_vehicles}; this DP cannot enforce a fleet limit"
        )

    return DPResult(
        routes=routes,
        cost=total,
        lower_bound=total,
        status="OPTIMAL",
        seconds=time.perf_counter() - t0,
        proven_optimal=True,
        states=size * m,
        instance_name=instance.name,
    )


def _rebuild_route(dp, parent, to_depot, mask: int, m: int) -> list[int]:
    """Recover the visiting order of the optimal round trip serving ``mask``."""
    best_j, best_v = -1, _INF
    for j in range(m):
        if not (mask >> j) & 1:
            continue
        v = dp[mask, j] + to_depot[j]
        if v < best_v:
            best_v, best_j = v, j
    order: list[int] = []
    node, cur = best_j, mask
    while node >= 0:
        order.append(node + 1)
        prev = int(parent[cur, node])
        cur ^= 1 << node
        node = prev
    order.reverse()
    return order


def solve_heldkarp(instance: Instance, **kwargs) -> OptimizationResult:
    """Benchmark-runner entry point: subset DP as an :class:`OptimizationResult`."""
    return held_karp_cvrp(instance, **kwargs).to_optimization_result(instance)
