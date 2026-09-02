"""Compiled inner loops.

Everything in this module is JIT-compiled with numba and operates on plain
NumPy arrays. These are the routines executed millions of times inside the
metaheuristics; keeping them separate from the readable reference
implementations in :mod:`qroute.problems.instance` means the search is fast
without the model being obscure.

Route encoding used throughout
------------------------------
A solution is held as ``(flat, starts, n_routes)``:

* ``flat``     - customers of all routes concatenated, length ``n``
* ``starts``   - offsets, ``starts[r]:starts[r+1]`` is route ``r``
* ``n_routes`` - number of routes currently in use

The depot is node 0 and is implicit at both ends of every route.
"""

from __future__ import annotations

import numpy as np
from numba import njit

FASTMATH = True
CACHE = True


# --------------------------------------------------------------------------
# Split: optimal partition of a giant tour into feasible routes
# --------------------------------------------------------------------------
@njit(cache=CACHE, fastmath=FASTMATH)
def _split_free(tour, cost, demand, capacity, service, tw, veh_cost, pen_tw):
    """Optimal split with an unlimited fleet (single-dimension Bellman recursion)."""
    n = tour.shape[0]
    INF = 1e18
    has_tw = tw.shape[0] > 1
    v = np.full(n + 1, INF)
    p = np.zeros(n + 1, np.int32)
    v[0] = 0.0
    for i in range(n):
        if v[i] >= INF:
            continue
        load = 0.0
        open_cost = 0.0
        t = 0.0
        late = 0.0
        prev_node = 0
        j = i
        while j < n:
            c = tour[j]
            load += demand[c]
            if load > capacity:
                break
            # Extend the open path i..j-1 by one customer in O(1) instead of
            # re-summing the whole segment, which is what makes the split
            # O(n*L) rather than O(n*L^2).
            if j == i:
                open_cost = cost[0, c]
                t = cost[0, c]
                late = 0.0
            else:
                open_cost += cost[prev_node, c]
                t += cost[prev_node, c]
            if has_tw:
                if t < tw[c, 0]:
                    t = tw[c, 0]
                if t > tw[c, 1]:
                    late += t - tw[c, 1]
                t += service[c]
            prev_node = c
            rc = open_cost + cost[c, 0]
            if has_tw:
                t_end = t + cost[c, 0]
                late_total = late
                if t_end > tw[0, 1]:
                    late_total += t_end - tw[0, 1]
                rc += pen_tw * late_total
            tot = v[i] + rc + veh_cost
            if tot < v[j + 1]:
                v[j + 1] = tot
                p[j + 1] = i
            j += 1
    return p, v[n]


@njit(cache=CACHE, fastmath=FASTMATH)
def _split_fleet(tour, cost, demand, capacity, max_routes, service, tw, veh_cost, pen_tw):
    """Optimal split using at most ``max_routes`` vehicles.

    Returns ``(labels, cost)`` with cost ``1e18`` when no partition of this
    particular sequence fits in the fleet; the caller then falls back to the
    unrestricted split.
    """
    n = tour.shape[0]
    INF = 1e18
    has_tw = tw.shape[0] > 1
    K = max_routes
    v = np.full((K + 1, n + 1), INF)
    p = np.zeros((K + 1, n + 1), np.int32)
    v[0, 0] = 0.0
    for k in range(K):
        for i in range(n):
            if v[k, i] >= INF:
                continue
            load = 0.0
            open_cost = 0.0
            t = 0.0
            late = 0.0
            prev_node = 0
            j = i
            while j < n:
                c = tour[j]
                load += demand[c]
                if load > capacity:
                    break
                if j == i:
                    open_cost = cost[0, c]
                    t = cost[0, c]
                    late = 0.0
                else:
                    open_cost += cost[prev_node, c]
                    t += cost[prev_node, c]
                if has_tw:
                    if t < tw[c, 0]:
                        t = tw[c, 0]
                    if t > tw[c, 1]:
                        late += t - tw[c, 1]
                    t += service[c]
                prev_node = c
                rc = open_cost + cost[c, 0]
                if has_tw:
                    t_end = t + cost[c, 0]
                    late_total = late
                    if t_end > tw[0, 1]:
                        late_total += t_end - tw[0, 1]
                    rc += pen_tw * late_total
                tot = v[k, i] + rc + veh_cost
                if tot < v[k + 1, j + 1]:
                    v[k + 1, j + 1] = tot
                    p[k + 1, j + 1] = i
                j += 1
    best_k = 0
    best = INF
    for k in range(1, K + 1):
        if v[k, n] < best:
            best = v[k, n]
            best_k = k
    out = np.zeros(n + 1, np.int32)
    if best >= INF:
        return out, best
    k = best_k
    j = n
    while j > 0 and k > 0:
        i = p[k, j]
        out[j] = i
        j = i
        k -= 1
    return out, best


def split_tour(tour, cost, demand, capacity, max_routes, service, tw, veh_cost,
               pen_tw=1000.0):
    """Prins' optimal split of a giant tour into capacity-feasible routes.

    Given a fixed customer sequence ``tour``, this computes the *cheapest
    possible* way to cut it into routes, by solving a shortest-path problem on
    an auxiliary directed acyclic graph whose arc ``(i, j)`` represents serving
    ``tour[i..j-1]`` with one vehicle. Because the sub-problem is solved to
    optimality, the metaheuristic only has to search over permutations: the
    route boundaries are always placed optimally for the permutation at hand.

    Complexity is ``O(n * L)`` where ``L`` is the average number of customers
    that fit in one vehicle, not ``O(n^2)``, because the inner loop stops as
    soon as the load exceeds capacity.

    Parameters
    ----------
    tour:
        ``(n,)`` permutation of customers ``1..n``.
    cost:
        ``(n+1, n+1)`` weighted arc-cost matrix.
    demand, capacity:
        Load data; a candidate route is abandoned once its load exceeds capacity.
    max_routes:
        Fleet limit; ``<= 0`` means unlimited. When positive, the split is
        solved with a route-count dimension so the fleet constraint is respected
        exactly rather than penalised.
    service, tw:
        Optional service times and ``(n+1, 2)`` time windows. When ``tw`` has
        more than one row, a route that arrives late is charged a large penalty
        so the split prefers time-feasible cuts.
    veh_cost:
        Fixed cost charged per route used.
    pen_tw:
        Penalty per unit of lateness, so the split's notion of a good cut agrees
        with the penalty the rest of the search is currently applying.

    Returns
    -------
    (labels, total_cost)
        ``labels[j]`` is the index where the route ending at position ``j``
        starts, from which the routes are reconstructed by walking backwards.
    """
    if max_routes <= 0:
        return _split_free(tour, cost, demand, capacity, service, tw, veh_cost, pen_tw)
    labels, cost_val = _split_fleet(tour, cost, demand, capacity, max_routes,
                                    service, tw, veh_cost, pen_tw)
    if cost_val >= 1e18:
        # No partition of this particular sequence fits in the fleet. Rather than
        # return a meaningless infinity we fall back to the unrestricted split;
        # the extra routes then show up as a fleet violation, which the penalty
        # term prices and the search can repair by reordering.
        return _split_free(tour, cost, demand, capacity, service, tw, veh_cost, pen_tw)
    return labels, cost_val


@njit(cache=CACHE)
def labels_to_routes(tour, labels, n_max_routes):
    """Reconstruct ``(flat, starts, n_routes)`` from a split predecessor array."""
    n = tour.shape[0]
    # Walk the predecessor chain backwards collecting route end positions.
    ends = np.zeros(n + 1, np.int32)
    nb = 0
    j = n
    while j > 0:
        ends[nb] = j
        nb += 1
        i = labels[j]
        if i >= j:  # guard against a malformed label chain
            break
        j = i
    # Emit routes in forward order; route r spans labels[end]..end in the tour.
    starts = np.zeros(nb + 1, np.int32)
    flat = np.zeros(n, np.int32)
    pos = 0
    for r in range(nb):
        end = ends[nb - 1 - r]
        begin = labels[end]
        starts[r] = pos
        for q in range(begin, end):
            flat[pos] = tour[q]
            pos += 1
    starts[nb] = pos
    return flat, starts, nb


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@njit(cache=CACHE, fastmath=FASTMATH)
def route_cost(route, cost):
    """Closed-tour cost of one route, depot to depot."""
    if route.shape[0] == 0:
        return 0.0
    c = cost[0, route[0]]
    for i in range(route.shape[0] - 1):
        c += cost[route[i], route[i + 1]]
    c += cost[route[-1], 0]
    return c


@njit(cache=CACHE, fastmath=FASTMATH)
def evaluate(flat, starts, n_routes, cost, demand, capacity, service, tw,
             max_duration, pen_cap, pen_tw, pen_dur, veh_cost):
    """Penalised cost of a full solution.

    Returns ``(total, raw_cost, cap_violation, tw_violation, dur_violation)``
    so the caller can report the feasible-cost and the infeasibility separately.
    """
    has_tw = tw.shape[0] > 1
    raw = 0.0
    capv = 0.0
    twv = 0.0
    durv = 0.0
    for r in range(n_routes):
        a = starts[r]
        b = starts[r + 1]
        if b <= a:
            continue
        load = 0.0
        t = 0.0
        prev = 0
        rc = 0.0
        for k in range(a, b):
            c = flat[k]
            rc += cost[prev, c]
            t += cost[prev, c] if not has_tw else 0.0
            load += demand[c]
            prev = c
        rc += cost[prev, 0]
        raw += rc + veh_cost
        if load > capacity:
            capv += load - capacity
        if has_tw:
            t = 0.0
            prev = 0
            for k in range(a, b):
                c = flat[k]
                t += cost[prev, c]
                if t < tw[c, 0]:
                    t = tw[c, 0]
                if t > tw[c, 1]:
                    twv += t - tw[c, 1]
                t += service[c]
                prev = c
            t += cost[prev, 0]
            if t > tw[0, 1]:
                twv += t - tw[0, 1]
        if max_duration > 0.0 and rc > max_duration:
            durv += rc - max_duration
    total = raw + pen_cap * capv + pen_tw * twv + pen_dur * durv
    return total, raw, capv, twv, durv
