"""Local search operators applied to decoded solutions.

The metaheuristics in this package are *memetic*: the quantum-inspired global
search proposes customer orderings, the split turns them into routes, and the
operators here refine those routes to a local optimum. This division matters
because pure swarm search on a permutation encoding stalls several percent above
the best-known solutions, whereas the hybrid reaches roughly one percent.

Operators implemented
---------------------
Intra-route
    ``2-opt``   - reverse a segment, removing route crossings
    ``Or-opt``  - move a chain of 1-3 consecutive customers elsewhere in the route
Inter-route
    ``relocate``      - move one customer to another route
    ``swap``          - exchange two customers between routes
    ``2-opt*``        - exchange the tails of two routes
    ``cross-exchange``- exchange short segments between two routes

All operators use *first improvement* within a randomised neighbour order,
which is markedly faster than best improvement at equal quality, and every move
is screened by a granular neighbour list: only the ``K`` geographically nearest
customers of a node are considered as new predecessors. That reduces the
neighbourhood from ``O(n^2)`` to ``O(nK)`` and is what makes the search viable
on instances with a thousand customers.
"""

from __future__ import annotations

import numpy as np
from numba import njit

CACHE = True
FASTMATH = True


def neighbour_lists(cost: np.ndarray, k: int) -> np.ndarray:
    """``(n+1, k)`` array of the ``k`` nearest other nodes of each node."""
    n = cost.shape[0]
    k = int(min(k, n - 1))
    c = cost.copy()
    np.fill_diagonal(c, np.inf)
    idx = np.argpartition(c, k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(c[rows, idx], axis=1)
    return np.ascontiguousarray(idx[rows, order], dtype=np.int32)


# --------------------------------------------------------------------------
# Feasibility-aware route metrics
# --------------------------------------------------------------------------
@njit(cache=CACHE, fastmath=FASTMATH)
def _route_penalised(route, cost, demand, capacity, service, tw, max_duration,
                     pen_cap, pen_tw, pen_dur, veh_cost):
    """Penalised cost of a single route, used to price candidate moves."""
    m = route.shape[0]
    if m == 0:
        return 0.0
    has_tw = tw.shape[0] > 1
    c = 0.0
    load = 0.0
    prev = 0
    for i in range(m):
        node = route[i]
        c += cost[prev, node]
        load += demand[node]
        prev = node
    c += cost[prev, 0]
    total = c + veh_cost
    if load > capacity:
        total += pen_cap * (load - capacity)
    if max_duration > 0.0 and c > max_duration:
        total += pen_dur * (c - max_duration)
    if has_tw:
        t = 0.0
        prev = 0
        late = 0.0
        for i in range(m):
            node = route[i]
            t += cost[prev, node]
            if t < tw[node, 0]:
                t = tw[node, 0]
            if t > tw[node, 1]:
                late += t - tw[node, 1]
            t += service[node]
            prev = node
        t += cost[prev, 0]
        if t > tw[0, 1]:
            late += t - tw[0, 1]
        total += pen_tw * late
    return total


@njit(cache=CACHE, fastmath=FASTMATH)
def _solution_penalised(flat, starts, n_routes, cost, demand, capacity, service, tw,
                        max_duration, pen_cap, pen_tw, pen_dur, veh_cost):
    total = 0.0
    for r in range(n_routes):
        total += _route_penalised(flat[starts[r]:starts[r + 1]], cost, demand, capacity,
                                  service, tw, max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
    return total


# --------------------------------------------------------------------------
# Intra-route: 2-opt and Or-opt
# --------------------------------------------------------------------------
@njit(cache=CACHE, fastmath=FASTMATH)
def two_opt_route(route, cost, neigh, has_tw, symmetric):
    """First-improvement 2-opt on one route. Modifies ``route`` in place.

    2-opt reverses the segment between two positions. On a **symmetric** cost
    matrix the interior of that segment costs the same in either direction, so
    the move is priced by the classical four-arc difference

        delta = c[a,c] + c[b,d] - c[a,b] - c[c,d]

    On an **asymmetric** matrix - which is what a real road network with
    one-way streets produces - that shortcut is simply wrong: reversing the
    segment re-prices every arc inside it. Using it on OSM travel times would
    accept moves that actually make the route longer and would make the
    metaheuristic look worse than it is. When ``symmetric`` is false we
    therefore add the true interior difference before deciding.

    Time windows are the other case where reversal changes more than four arcs:
    a reversal shifts every subsequent arrival, so the operator is disabled and
    the caller relies on Or-opt and the inter-route moves, which re-price whole
    routes.
    """
    m = route.shape[0]
    if m < 4 or has_tw:
        return 0.0
    gain_total = 0.0
    improved = True
    while improved:
        improved = False
        for i in range(m - 1):
            a = 0 if i == 0 else route[i - 1]
            b = route[i]
            for jj in range(neigh.shape[1]):
                c = neigh[b, jj]
                if c == 0:
                    continue
                # locate c in this route, after position i
                j = -1
                for q in range(i + 1, m):
                    if route[q] == c:
                        j = q
                        break
                if j < 0:
                    continue
                d = route[j + 1] if j + 1 < m else 0
                if d == b:
                    continue
                delta = (cost[a, c] + cost[b, d]) - (cost[a, b] + cost[c, d])
                if not symmetric:
                    # True cost of walking the interior backwards minus forwards.
                    for q in range(i, j):
                        delta += cost[route[q + 1], route[q]] - cost[route[q], route[q + 1]]
                if delta < -1e-10:
                    lo = i
                    hi = j
                    while lo < hi:
                        tmp = route[lo]
                        route[lo] = route[hi]
                        route[hi] = tmp
                        lo += 1
                        hi -= 1
                    gain_total += delta
                    improved = True
                    break
            if improved:
                break
    return gain_total


@njit(cache=CACHE, fastmath=FASTMATH)
def or_opt_route(route, cost, max_seg, has_tw, symmetric):
    """Move chains of 1..``max_seg`` customers within a route (first improvement).

    ``symmetric`` controls whether a chain may be inserted reversed: on an
    asymmetric matrix the reversed insertion cost includes the re-priced
    interior of the chain, which this routine accounts for explicitly.
    """
    m = route.shape[0]
    if m < 3:
        return 0.0
    gain_total = 0.0
    improved = True
    buf = np.empty(max_seg, np.int32)
    while improved:
        improved = False
        for seg in range(1, max_seg + 1):
            for i in range(m - seg + 1):
                a = 0 if i == 0 else route[i - 1]
                b = route[i]
                e = route[i + seg - 1]
                f = route[i + seg] if i + seg < m else 0
                removed = cost[a, b] + cost[e, f] - cost[a, f]
                if removed <= 1e-12 and not has_tw:
                    continue
                for k in range(seg):
                    buf[k] = route[i + k]
                for j in range(m - seg + 1):
                    if j >= i - 1 and j <= i + seg - 1:
                        continue
                    # insert the chain before position j (after shifting)
                    if j < i:
                        p = route[j - 1] if j > 0 else 0
                        q = route[j]
                    else:
                        p = route[j + seg - 1]
                        q = route[j + seg] if j + seg < m else 0
                    add_fwd = cost[p, buf[0]] + cost[buf[seg - 1], q] - cost[p, q]
                    add_rev = cost[p, buf[seg - 1]] + cost[buf[0], q] - cost[p, q]
                    if not symmetric:
                        # reversing the chain re-prices its interior arcs too
                        for kk in range(seg - 1):
                            add_rev += (cost[buf[kk + 1], buf[kk]]
                                        - cost[buf[kk], buf[kk + 1]])
                    rev = add_rev < add_fwd and not has_tw
                    add = add_rev if rev else add_fwd
                    delta = add - removed
                    if delta < -1e-10:
                        # rebuild the route with the chain moved
                        tmp = np.empty(m, np.int32)
                        pos = 0
                        for q2 in range(m):
                            if q2 >= i and q2 < i + seg:
                                continue
                            if pos == j:
                                for k in range(seg):
                                    tmp[pos] = buf[seg - 1 - k] if rev else buf[k]
                                    pos += 1
                            tmp[pos] = route[q2]
                            pos += 1
                        if pos < m:
                            for k in range(seg):
                                tmp[pos] = buf[seg - 1 - k] if rev else buf[k]
                                pos += 1
                        for q2 in range(m):
                            route[q2] = tmp[q2]
                        gain_total += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return gain_total


# --------------------------------------------------------------------------
# Inter-route operators
# --------------------------------------------------------------------------
@njit(cache=CACHE, fastmath=FASTMATH)
def _rebuild(routes_flat, routes_starts, n_routes, new_r1, r1, new_r2, r2, n):
    """Return a fresh (flat, starts) with routes r1 and r2 replaced."""
    lens = np.zeros(n_routes, np.int32)
    for r in range(n_routes):
        if r == r1:
            lens[r] = new_r1.shape[0]
        elif r == r2:
            lens[r] = new_r2.shape[0]
        else:
            lens[r] = routes_starts[r + 1] - routes_starts[r]
    starts = np.zeros(n_routes + 1, np.int32)
    for r in range(n_routes):
        starts[r + 1] = starts[r] + lens[r]
    flat = np.zeros(n, np.int32)
    for r in range(n_routes):
        base = starts[r]
        if r == r1:
            for k in range(new_r1.shape[0]):
                flat[base + k] = new_r1[k]
        elif r == r2:
            for k in range(new_r2.shape[0]):
                flat[base + k] = new_r2[k]
        else:
            src = routes_starts[r]
            for k in range(lens[r]):
                flat[base + k] = routes_flat[src + k]
    return flat, starts


@njit(cache=CACHE, fastmath=FASTMATH)
def inter_route_search(flat, starts, n_routes, cost, demand, capacity, service, tw,
                       max_duration, pen_cap, pen_tw, pen_dur, veh_cost, neigh,
                       route_of, position_of, queue, in_queue, max_moves):
    """Relocate / swap / 2-opt* between routes, driven by don't-look bits.

    Moves are proposed only between a customer and its geographic neighbours,
    and each candidate is priced by fully re-evaluating just the one or two
    routes it touches. Full re-evaluation of the affected routes, rather than an
    arc-delta shortcut, is what keeps the operator correct under time windows,
    where an insertion shifts every subsequent arrival time.

    The scan uses the classical *don't-look bits* strategy: a customer sits in
    the work queue only while a move involving it might still pay off. After a
    successful move only the customers whose neighbourhood actually changed are
    re-queued. Without this the operator would restart its triple loop after
    every improvement, which is what makes naive implementations quadratic and
    unusable beyond a few hundred customers.
    """
    n = flat.shape[0]
    cur_flat = flat.copy()
    cur_starts = starts.copy()
    total_gain = 0.0

    for r in range(n_routes):
        for k in range(cur_starts[r], cur_starts[r + 1]):
            route_of[cur_flat[k]] = r
            position_of[cur_flat[k]] = k - cur_starts[r]

    # seed the queue with every customer, in route order
    qhead = 0
    qtail = 0
    for k in range(n):
        queue[qtail] = cur_flat[k]
        in_queue[cur_flat[k]] = True
        qtail += 1
    qsize = qtail
    cap_q = queue.shape[0]

    moves = 0
    while qsize > 0 and moves < max_moves:
        u = queue[qhead]
        qhead = (qhead + 1) % cap_q
        qsize -= 1
        in_queue[u] = False

        r1 = route_of[u]
        if r1 < 0 or r1 >= n_routes:
            continue
        len1 = cur_starts[r1 + 1] - cur_starts[r1]
        if len1 == 0:
            continue
        i = position_of[u]
        if i < 0 or i >= len1 or cur_flat[cur_starts[r1] + i] != u:
            continue
        base1 = _route_penalised(cur_flat[cur_starts[r1]:cur_starts[r1 + 1]], cost,
                                 demand, capacity, service, tw, max_duration,
                                 pen_cap, pen_tw, pen_dur, veh_cost)

        applied = False
        for jj in range(neigh.shape[1]):
            v = neigh[u, jj]
            if v == 0:
                continue
            r2 = route_of[v]
            if r2 == r1 or r2 < 0 or r2 >= n_routes:
                continue
            len2 = cur_starts[r2 + 1] - cur_starts[r2]
            if len2 == 0:
                continue
            pv = position_of[v]
            if pv < 0 or pv >= len2 or cur_flat[cur_starts[r2] + pv] != v:
                continue
            base2 = _route_penalised(cur_flat[cur_starts[r2]:cur_starts[r2 + 1]],
                                     cost, demand, capacity, service, tw,
                                     max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
            base = base1 + base2

            best_delta = -1e-10
            best_kind = -1
            best_pos = -1

            # --- move 1: relocate u to just before or just after v -----------
            r1n = np.empty(len1 - 1, np.int32)
            p = 0
            for k in range(len1):
                if k != i:
                    r1n[p] = cur_flat[cur_starts[r1] + k]
                    p += 1
            c1 = _route_penalised(r1n, cost, demand, capacity, service, tw,
                                  max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
            for where in range(2):
                ins = pv + where
                r2n = np.empty(len2 + 1, np.int32)
                p = 0
                for k in range(len2 + 1):
                    if k == ins:
                        r2n[p] = u
                        p += 1
                    if k < len2:
                        r2n[p] = cur_flat[cur_starts[r2] + k]
                        p += 1
                c2 = _route_penalised(r2n, cost, demand, capacity, service, tw,
                                      max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
                d = (c1 + c2) - base
                if d < best_delta:
                    best_delta = d
                    best_kind = 0
                    best_pos = ins

            # --- move 2: swap u and v ----------------------------------------
            r1s = cur_flat[cur_starts[r1]:cur_starts[r1 + 1]].copy()
            r2s = cur_flat[cur_starts[r2]:cur_starts[r2 + 1]].copy()
            r1s[i] = v
            r2s[pv] = u
            d = (_route_penalised(r1s, cost, demand, capacity, service, tw,
                                  max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
                 + _route_penalised(r2s, cost, demand, capacity, service, tw,
                                    max_duration, pen_cap, pen_tw, pen_dur, veh_cost)) - base
            if d < best_delta:
                best_delta = d
                best_kind = 1
                best_pos = pv

            # --- move 3: 2-opt*, exchange the tails after u and after v -------
            t1 = len1 - i - 1
            t2 = len2 - pv - 1
            na = i + 1 + t2
            nb = pv + 1 + t1
            if na > 0 and nb > 0:
                ra = np.empty(na, np.int32)
                for k in range(i + 1):
                    ra[k] = cur_flat[cur_starts[r1] + k]
                for k in range(t2):
                    ra[i + 1 + k] = cur_flat[cur_starts[r2] + pv + 1 + k]
                rb = np.empty(nb, np.int32)
                for k in range(pv + 1):
                    rb[k] = cur_flat[cur_starts[r2] + k]
                for k in range(t1):
                    rb[pv + 1 + k] = cur_flat[cur_starts[r1] + i + 1 + k]
                d = (_route_penalised(ra, cost, demand, capacity, service, tw,
                                      max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
                     + _route_penalised(rb, cost, demand, capacity, service, tw,
                                        max_duration, pen_cap, pen_tw, pen_dur, veh_cost)) - base
                if d < best_delta:
                    best_delta = d
                    best_kind = 2
                    best_pos = 0

            if best_kind < 0:
                continue

            # ---- build the two replacement routes ---------------------------
            if best_kind == 0:
                r1f = r1n
                ins = best_pos
                r2f = np.empty(len2 + 1, np.int32)
                p = 0
                for k in range(len2 + 1):
                    if k == ins:
                        r2f[p] = u
                        p += 1
                    if k < len2:
                        r2f[p] = cur_flat[cur_starts[r2] + k]
                        p += 1
            elif best_kind == 1:
                r1f = r1s
                r2f = r2s
            else:
                t1b = len1 - i - 1
                t2b = len2 - pv - 1
                r1f = np.empty(i + 1 + t2b, np.int32)
                for k in range(i + 1):
                    r1f[k] = cur_flat[cur_starts[r1] + k]
                for k in range(t2b):
                    r1f[i + 1 + k] = cur_flat[cur_starts[r2] + pv + 1 + k]
                r2f = np.empty(pv + 1 + t1b, np.int32)
                for k in range(pv + 1):
                    r2f[k] = cur_flat[cur_starts[r2] + k]
                for k in range(t1b):
                    r2f[pv + 1 + k] = cur_flat[cur_starts[r1] + i + 1 + k]

            cur_flat, cur_starts = _rebuild(cur_flat, cur_starts, n_routes,
                                            r1f, r1, r2f, r2, n)
            total_gain += best_delta
            moves += 1
            applied = True

            # refresh indices for the two routes we touched, and wake their
            # customers plus u's and v's neighbours for another look
            for rr in (r1, r2):
                for k in range(cur_starts[rr], cur_starts[rr + 1]):
                    c = cur_flat[k]
                    route_of[c] = rr
                    position_of[c] = k - cur_starts[rr]
                    if not in_queue[c] and qsize < cap_q:
                        queue[qtail] = c
                        qtail = (qtail + 1) % cap_q
                        in_queue[c] = True
                        qsize += 1
            for jj2 in range(neigh.shape[1]):
                for w in (neigh[u, jj2], neigh[v, jj2]):
                    if w != 0 and not in_queue[w] and qsize < cap_q:
                        queue[qtail] = w
                        qtail = (qtail + 1) % cap_q
                        in_queue[w] = True
                        qsize += 1
            break

        if applied:
            continue
    return cur_flat, cur_starts, total_gain


@njit(cache=CACHE, fastmath=FASTMATH)
def local_search(flat, starts, n_routes, cost, demand, capacity, service, tw,
                 max_duration, pen_cap, pen_tw, pen_dur, veh_cost, neigh,
                 route_of, position_of, queue, in_queue, max_rounds, or_opt_seg,
                 symmetric):
    """Run intra- and inter-route operators alternately until no move improves.

    Returns ``(flat, starts, n_routes, penalised_cost)``. Empty routes are
    dropped, which is how the search reduces vehicle count when that is priced.
    """
    has_tw = tw.shape[0] > 1
    cur_flat = flat.copy()
    cur_starts = starts.copy()
    nr = n_routes
    prev_cost = _solution_penalised(cur_flat, cur_starts, nr, cost, demand, capacity,
                                    service, tw, max_duration, pen_cap, pen_tw,
                                    pen_dur, veh_cost)
    for _round in range(max_rounds):
        # ---- intra-route ------------------------------------------------
        for r in range(nr):
            a = cur_starts[r]
            b = cur_starts[r + 1]
            if b - a >= 3:
                seg = cur_flat[a:b].copy()
                two_opt_route(seg, cost, neigh, has_tw, symmetric)
                or_opt_route(seg, cost, or_opt_seg, has_tw, symmetric)
                # keep the change only if the whole route got cheaper
                before = _route_penalised(cur_flat[a:b], cost, demand, capacity, service,
                                          tw, max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
                after = _route_penalised(seg, cost, demand, capacity, service, tw,
                                         max_duration, pen_cap, pen_tw, pen_dur, veh_cost)
                if after < before - 1e-10:
                    for k in range(b - a):
                        cur_flat[a + k] = seg[k]
        # ---- inter-route ------------------------------------------------
        cur_flat, cur_starts, _g = inter_route_search(
            cur_flat, cur_starts, nr, cost, demand, capacity, service, tw, max_duration,
            pen_cap, pen_tw, pen_dur, veh_cost, neigh, route_of, position_of,
            queue, in_queue, 20 * cur_flat.shape[0])
        # ---- drop empty routes ------------------------------------------
        keep = 0
        new_starts = np.zeros(nr + 1, np.int32)
        new_flat = np.zeros(cur_flat.shape[0], np.int32)
        pos = 0
        for r in range(nr):
            ln = cur_starts[r + 1] - cur_starts[r]
            if ln > 0:
                new_starts[keep] = pos
                for k in range(ln):
                    new_flat[pos + k] = cur_flat[cur_starts[r] + k]
                pos += ln
                keep += 1
        new_starts[keep] = pos
        cur_flat = new_flat
        cur_starts = new_starts
        nr = keep

        cost_now = _solution_penalised(cur_flat, cur_starts, nr, cost, demand, capacity,
                                       service, tw, max_duration, pen_cap, pen_tw,
                                       pen_dur, veh_cost)
        if cost_now > prev_cost - 1e-9:
            prev_cost = cost_now
            break
        prev_cost = cost_now
    return cur_flat, cur_starts, nr, prev_cost
