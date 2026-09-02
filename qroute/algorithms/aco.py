"""Ant Colony System for vehicle routing.

ACO is the fourth control, and it is included because it is the only baseline
here whose memory is *structural*. PSO, QPSO and the GA all remember good
solutions - points in a search space. An ant colony instead remembers good
*arcs*: the pheromone matrix ``tau`` accumulates evidence about which pairs of
customers are worth visiting consecutively, independently of any particular
solution that used them. Two solutions that share no customers in the same
positions can still reinforce the same arcs.

That difference matters for the project's central claim. If a swarm beats a GA,
one can object that both are point-based and the comparison only shows that one
point-based rule is better tuned than another. ACO is a genuinely different kind
of memory, so beating it is evidence about the search paradigm rather than about
parameter tuning. It also gives the honest counter-case: on instances with
strong local structure - clustered customers, a road network with obvious
corridors - arc memory is the right inductive bias and ACO can be expected to do
well.

Algorithm
---------
This is Dorigo and Gambardella's Ant Colony System (1997) with the
Stuetzle-Hoos MAX-MIN trail bounds (2000):

* **Pseudo-random-proportional rule.** At each step an ant at node ``i`` picks
  the next customer ``j`` from its candidate list either greedily, with
  probability ``q0``, by maximising ``tau[i,j]^alpha * eta[i,j]^beta``, or
  otherwise by sampling proportionally to the same quantity. ``q0`` is the
  explicit exploitation/exploration dial that plain Ant System lacks. ACS fixes
  ``alpha = 1``, which is the default here: with only one pheromone matrix the
  exponent is redundant with the trail scale, and it is exposed only so the
  effect of that convention can be checked rather than assumed.
* **Candidate lists.** Only the ``k`` nearest unvisited nodes are considered,
  which is what makes construction ``O(n k)`` instead of ``O(n^2)``. The
  candidate lists are the same ones the local search uses.
* **Local pheromone update.** Immediately after traversing ``(i, j)`` the ant
  evaporates that arc towards ``tau0``. This is a diversification device, not a
  learning one: it makes an arc less attractive to the *next* ant in the same
  iteration, so a colony does not collapse onto one tour within an iteration.
* **Global update on the best-so-far solution only**, with evaporation and
  deposit applied to its arcs.
* **MAX-MIN bounds.** ``tau`` is clipped to ``[tau_min, tau_max]`` with
  ``tau_max = 1 / (rho * L_best)`` and ``tau_min = tau_max / (2 n)``. Without
  bounds the best-so-far update drives non-best arcs towards zero and the
  colony stops exploring entirely after a few hundred iterations.

Coupling to the shared decoder
------------------------------
An ant builds a *giant tour* over the customers, not a set of routes. That tour
is handed to the same :class:`~qroute.algorithms.decoder.Decoder` every other
algorithm uses, which splits it optimally into vehicle routes and refines them
with local search. Route boundaries are therefore never the ants' problem, which
is the standard and much stronger formulation - an ant that has to decide when
to return to the depot is making a decision that dynamic programming can make
optimally in linear time.

Pheromone is deposited on the arcs of the solution *after* local search, so the
colony learns from the refined structure rather than from its own raw
construction. This is the usual ACO-with-local-search arrangement and it matters
a great deal in practice; depositing on the unimproved tour teaches the colony to
reproduce mistakes the local search then has to undo.

Determinism note: the construction loop is compiled with numba, and numba's
random state is not the NumPy generator this project seeds. All random numbers
an ant needs are therefore drawn from ``self.rng`` up front and passed into the
kernel, so a seeded run is exactly reproducible.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder
from qroute.core.types import Solution


@njit(cache=True)
def _construct_tour(tau, weight, neigh, rand_q, rand_r, q0, xi, tau0, n,
                    symmetric, alpha):
    """Build one giant tour with the ACS pseudo-random-proportional rule.

    ``weight[i, j]`` is ``eta[i, j] ** beta``, precomputed once per run because
    the heuristic term never changes and the exponentiation would otherwise
    dominate the inner loop. The pheromone term cannot be precomputed - ``tau``
    changes during the construction - so ``alpha != 1`` costs one ``pow`` per
    candidate, and the common case ``alpha == 1`` is branched around. ``tau`` is
    modified in place by the local pheromone update.
    """
    plain = alpha == 1.0
    size = tau.shape[0]
    k = neigh.shape[1]
    visited = np.zeros(size, np.bool_)
    visited[0] = True
    tour = np.empty(n, np.int32)
    cand = np.empty(k, np.int32)
    scores = np.empty(k)
    cur = 0
    for step in range(n):
        m = 0
        for a in range(k):
            j = neigh[cur, a]
            if j == 0 or visited[j]:
                continue
            cand[m] = j
            t = tau[cur, j] if plain else tau[cur, j] ** alpha
            scores[m] = t * weight[cur, j]
            m += 1

        if m == 0:
            # Every candidate is taken: fall back to a full scan. This happens
            # rarely (only near the end of a construction), so the O(n) cost
            # does not change the overall O(n k) complexity in practice.
            best = -1.0
            nxt = -1
            for j in range(1, size):
                if not visited[j]:
                    t = tau[cur, j] if plain else tau[cur, j] ** alpha
                    s = t * weight[cur, j]
                    if s > best:
                        best = s
                        nxt = j
            if nxt < 0:
                break
        elif rand_q[step] < q0:
            bi = 0
            for a in range(1, m):
                if scores[a] > scores[bi]:
                    bi = a
            nxt = cand[bi]
        else:
            total = 0.0
            for a in range(m):
                total += scores[a]
            nxt = cand[m - 1]
            if total > 0.0:
                threshold = rand_r[step] * total
                acc = 0.0
                for a in range(m):
                    acc += scores[a]
                    if acc >= threshold:
                        nxt = cand[a]
                        break

        visited[nxt] = True
        tour[step] = nxt
        # Local update: pull this arc back towards tau0 so the next ant in the
        # same iteration is less likely to copy it.
        t = (1.0 - xi) * tau[cur, nxt] + xi * tau0
        tau[cur, nxt] = t
        if symmetric:
            tau[nxt, cur] = t
        cur = nxt
    return tour


class AntColony(Optimizer):
    """Ant Colony System over giant tours, split and refined by the shared decoder."""

    name = "ACO"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 n_ants: int = 10,
                 alpha: float = 1.0,
                 beta: float = 3.0,
                 q0: float = 0.9,
                 rho: float = 0.1,
                 xi: float = 0.1,
                 candidate_list: int = 15,
                 tau_min_factor: float = 0.5,
                 local_search: bool = True,
                 neighbours: int = 15,
                 local_search_rounds: int = 30,
                 penalty_capacity: float | None = None,
                 penalty_time_window: float | None = None,
                 penalty_duration: float | None = None,
                 vehicle_cost: float = 0.0,
                 decoder: Decoder | None = None,
                 **kw):
        super().__init__(instance, stop, seed, callback,
                         n_ants=n_ants, alpha=alpha, beta=beta, q0=q0, rho=rho, xi=xi,
                         candidate_list=candidate_list, tau_min_factor=tau_min_factor,
                         local_search=local_search, neighbours=neighbours, **kw)
        if not 0.0 <= q0 < 1.0:
            raise ValueError("q0 must lie in [0, 1)")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 < rho <= 1.0 or not 0.0 < xi <= 1.0:
            raise ValueError("rho and xi must lie in (0, 1]")

        self.n_ants = max(1, int(n_ants))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.q0 = float(q0)
        self.rho = float(rho)
        self.xi = float(xi)
        self.tau_min_factor = float(tau_min_factor)
        self.decoder = decoder or Decoder(
            instance, neighbours=neighbours, local_search_rounds=local_search_rounds,
            penalty_capacity=penalty_capacity, penalty_time_window=penalty_time_window,
            penalty_duration=penalty_duration, vehicle_cost=vehicle_cost,
            use_local_search=local_search)
        self.n = instance.n_customers
        self.candidate_list = int(candidate_list)

    # ------------------------------------------------------------------- run
    def _run(self) -> int:
        rng = self.rng
        dec = self.decoder
        n = self.n
        size = self.instance.size
        cost = dec.cost

        # Candidate lists. A separate, possibly wider list than the local
        # search uses, because construction benefits from more choice than
        # improvement does.
        from qroute.algorithms.localsearch import neighbour_lists
        k = int(min(max(self.candidate_list, 3), max(n, 1)))
        neigh = neighbour_lists(cost, k) if k != dec.neigh.shape[1] else dec.neigh

        # Heuristic desirability eta = 1 / cost, pre-raised to the power beta.
        # Zero-length arcs (duplicate coordinates do occur in the X set) would
        # give an infinite eta, so the matrix is floored at a small fraction of
        # the smallest positive arc.
        positive = cost[cost > 0.0]
        floor = float(positive.min()) * 1e-3 if positive.size else 1.0
        weight = np.power(1.0 / np.maximum(cost, floor), self.beta)
        np.fill_diagonal(weight, 0.0)
        weight = np.ascontiguousarray(weight, dtype=np.float64)

        # tau0 = 1 / (n * L_nn), the ACS convention: the trail starts just above
        # what a single nearest-neighbour tour would deposit, so early ants are
        # steered mostly by eta and only gradually by accumulated experience.
        l_nn = self._nearest_neighbour_length(cost)
        tau0 = 1.0 / max(n * l_nn, 1e-9)
        tau = np.full((size, size), tau0, dtype=np.float64)

        best_cost = float("inf")
        best_routes: list[list[int]] = []
        keys = np.empty(n, dtype=np.float64)
        ranks = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1)

        it = 0
        while not self.should_stop(it):
            it += 1
            iter_costs = np.empty(self.n_ants)
            for a in range(self.n_ants):
                rand_q = rng.random(n)
                rand_r = rng.random(n)
                tour = _construct_tour(tau, weight, neigh, rand_q, rand_r,
                                       self.q0, self.xi, tau0, n, dec.symmetric,
                                       self.alpha)
                keys[tour - 1] = ranks
                routes, c, _ = dec.decode(keys)
                self.evaluations += 1
                iter_costs[a] = c
                if c < best_cost - 1e-10:
                    best_cost = c
                    best_routes = routes
                    self.offer(Solution([list(r) for r in routes], c))

            if best_routes:
                self._global_update(tau, best_routes, best_cost, dec.symmetric)

            self.record(it, float(best_cost), float(iter_costs.mean()),
                        float(self._diversity(tau, tau0)), True)
        return it

    # -------------------------------------------------------------- pheromone
    def _global_update(self, tau: np.ndarray, routes, best_cost: float,
                       symmetric: bool) -> None:
        """Evaporate everywhere, deposit on the best-so-far arcs, clip to bounds."""
        deposit = 1.0 / max(best_cost, 1e-9)
        tau *= (1.0 - self.rho)
        add = self.rho * deposit
        for r in routes:
            prev = 0
            for c in r:
                tau[prev, c] += add
                if symmetric:
                    tau[c, prev] = tau[prev, c]
                prev = c
            tau[prev, 0] += add
            if symmetric:
                tau[0, prev] = tau[prev, 0]

        tau_max = 1.0 / (self.rho * max(best_cost, 1e-9))
        tau_min = tau_max * self.tau_min_factor / max(self.n, 1)
        np.clip(tau, tau_min, tau_max, out=tau)

    def _nearest_neighbour_length(self, cost: np.ndarray) -> float:
        """Length of a greedy nearest-neighbour tour, used only to scale tau0."""
        size = cost.shape[0]
        visited = np.zeros(size, dtype=bool)
        visited[0] = True
        cur = 0
        total = 0.0
        for _ in range(size - 1):
            c = cost[cur].copy()
            c[visited] = np.inf
            nxt = int(np.argmin(c))
            total += float(c[nxt])
            visited[nxt] = True
            cur = nxt
        total += float(cost[cur, 0])
        return max(total, 1e-9)

    @staticmethod
    def _diversity(tau: np.ndarray, tau0: float) -> float:
        """Normalised spread of the pheromone matrix.

        The colony's analogue of swarm diversity: while ``tau`` is still close
        to uniform the ants are exploring, and a large spread means the colony
        has committed to a small set of arcs. Reported so the convergence plots
        of the arc-memory and position-memory methods can be read side by side,
        with the caveat that the two quantities are not on the same scale.
        """
        return float(tau.std() / max(tau0, 1e-12))
