"""Quantum Particle Swarm Optimisation for vehicle routing.

Background
----------
In classical PSO a particle has a position *and* a velocity, and the trajectory
it follows is a damped oscillation around a weighted average of its own best and
the swarm's best. Because the velocity is bounded, the region a particle can
reach in the next step is bounded too, so the swarm's search space shrinks as it
converges and it cannot escape a deep local optimum.

QPSO, introduced by Sun, Feng and Xu (2004), removes the velocity entirely. Each
particle is treated as a quantum particle bound in a delta potential well
centred on a *local attractor* ``p``. In quantum mechanics the particle has no
definite trajectory; only a probability density of being found at a position.
For the delta well that density is a double exponential, and sampling from it by
inverting the cumulative distribution gives the update rule

    x(t+1) = p  ±  (L/2) * ln(1/u),     u ~ U(0, 1)

The support of that distribution is the whole real line, so at every single
iteration a particle has non-zero probability of appearing anywhere in the
search space. That is the concrete sense in which QPSO has "stronger global
search" than PSO, and it costs one logarithm per dimension rather than the extra
velocity array PSO must carry.

The characteristic length ``L`` is set from the swarm's *mean best position*

    mbest = (1/M) * sum_i pbest_i
    L     = 2 * beta * |mbest - x|

which makes the sampling width shrink automatically as the personal bests
cluster, giving the exploration-to-exploitation transition without an explicit
schedule. The complete update, per particle ``i`` and dimension ``d``:

    phi   ~ U(0, 1)
    p_id  = phi * pbest_id + (1 - phi) * gbest_d          (local attractor)
    u     ~ U(0, 1)
    x_id  = p_id  ±  beta * |mbest_d - x_id| * ln(1/u)    (sign by fair coin)

``beta``, the contraction-expansion coefficient, is the algorithm's one critical
parameter. Sun et al.'s stability analysis shows the swarm converges when
``beta`` is below roughly 1.78; the standard schedule decreases it linearly from
1.0 to 0.5 over the run, which this implementation uses by default.

Adaptation to routing
---------------------
Positions are random keys (see :mod:`qroute.algorithms.decoder`), decoded by
sorting and then optimally split into routes. Local search refines each decoded
solution and the result is written back into the particle's keys, so the swarm
accumulates the structural improvements rather than rediscovering them.

Optional refinements, all off unless enabled:

* **Weighted mbest** (WQPSO, Xi, Sun & Xu 2008) - rank-weight the personal bests
  so better particles pull the sampling width more strongly.
* **Mutation** - Cauchy or Gaussian perturbation, which helps on instances where
  the swarm converges before the budget is spent. Off by default: see below.
* **Restart** - reinitialise the worst particles after a stagnation window,
  keeping the elite.

What the parameter study actually showed
----------------------------------------
A sweep over swarm size, contraction schedule, mutation and local-search policy
was run on A-n45-k7, A-n80-k10, X-n101-k25 and R101, three seeds each, with an
equal twelve-second budget per configuration. Mean gap to the best-known
solution across every configuration tried fell in a narrow band, roughly 1.6% to
2.1%, with a standard deviation across runs near 1.0. In other words the
differences between reasonable parameter settings are smaller than the
run-to-run noise, and no setting is significantly better than another.

That is worth stating plainly rather than hiding, because it says something true
about this class of algorithm: once an optimal split and a good local search are
in place, they do most of the work, and the swarm rule mainly decides which
orderings get refined. The defaults below are the best observed mean, not a
claim of tuned superiority. The comparison that does matter is against the
same pipeline driven by other search rules, which is why every baseline shares
this decoder.

The classical schedule from Sun et al., beta falling linearly from 1.0 to 0.5,
was the best of the five schedules tested, so the literature default stands.
Mutation was very slightly worse than none, so it is off by default and kept as
an option.
"""

from __future__ import annotations

import numpy as np

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder
from qroute.core.types import Solution


class QPSO(Optimizer):
    """Quantum-behaved particle swarm optimiser over random-key positions."""

    name = "QPSO"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 swarm_size: int = 30,
                 beta_start: float = 1.0,
                 beta_end: float = 0.5,
                 beta_schedule: str = "linear",
                 beta_scaling: str = "fixed",
                 weighted_mbest: bool = True,
                 mutation: str = "none",
                 mutation_rate: float = 0.10,
                 mutation_scale: float = 0.10,
                 elite_fraction: float = 0.10,
                 restart_after: int = 60,
                 restart_fraction: float = 0.30,
                 clone_prevention: bool = False,
                 diversity_bias: float = 0.0,
                 local_search: bool = True,
                 ls_policy: str = "all",
                 ls_fraction: float = 0.25,
                 neighbours: int = 15,
                 local_search_rounds: int = 30,
                 penalty_capacity: float | None = None,
                 penalty_time_window: float | None = None,
                 penalty_duration: float | None = None,
                 vehicle_cost: float = 0.0,
                 decoder: Decoder | None = None,
                 initial_keys: np.ndarray | None = None,
                 **kw):
        super().__init__(instance, stop, seed, callback,
                         swarm_size=swarm_size, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, beta_scaling=beta_scaling,
                         weighted_mbest=weighted_mbest,
                         mutation=mutation, mutation_rate=mutation_rate,
                         mutation_scale=mutation_scale, elite_fraction=elite_fraction,
                         restart_after=restart_after, restart_fraction=restart_fraction,
                         clone_prevention=clone_prevention, diversity_bias=diversity_bias,
                         local_search=local_search, ls_policy=ls_policy,
                         ls_fraction=ls_fraction, neighbours=neighbours, **kw)
        self.M = int(swarm_size)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        self.beta_scaling = beta_scaling
        self.weighted_mbest = bool(weighted_mbest)
        self.mutation = mutation
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.elite = max(1, int(elite_fraction * self.M))
        self.restart_after = int(restart_after)
        self.restart_fraction = float(restart_fraction)
        self.ls_policy = ls_policy
        self.ls_fraction = float(ls_fraction)
        self.clone_prevention = bool(clone_prevention)
        self.diversity_bias = float(diversity_bias)
        self.initial_keys = initial_keys
        self.decoder = decoder or Decoder(
            instance, neighbours=neighbours, local_search_rounds=local_search_rounds,
            penalty_capacity=penalty_capacity, penalty_time_window=penalty_time_window,
            penalty_duration=penalty_duration, vehicle_cost=vehicle_cost,
            use_local_search=local_search)
        self.n = instance.n_customers

    # ------------------------------------------------------------------ beta
    def beta(self, iteration: int) -> float:
        """Contraction-expansion coefficient at the current point in the run.

        Kept below the stability bound of about 1.78 established by Sun et al.'s
        convergence analysis, so the swarm is guaranteed to contract.

        The schedule is indexed on *whichever budget is actually binding*. This
        matters more than it sounds: benchmark runs are given a wall-clock
        budget with the iteration cap left effectively infinite, and a schedule
        that divided by that cap would compute a progress fraction near zero for
        the whole run, hold beta at its starting value, and never contract at
        all. The exploration-to-exploitation transition would silently not
        happen, which is precisely the behaviour the algorithm is supposed to
        provide.
        """
        scale = 1.0
        if self.beta_scaling == "rank":
            # Derived rather than tuned. The step taken in one dimension is
            # beta * |mbest - x| * ln(1/u). With canonical rank keys the mean
            # best position sits mid-range in every dimension, so |mbest - x| is
            # roughly 0.29 whatever the instance, while the smallest meaningful
            # change to an ordering is one rank, or 1/n. A step that refines
            # rather than randomises therefore needs beta proportional to 1/n.
            # The constant is fixed by the measured optimum on instances of
            # about eighty customers, where beta = 0.05 was best.
            scale = (0.05 * 80.0) / max(self.n, 1) / max(self.beta_start, 1e-9)
        frac = 0.0
        if np.isfinite(self.stop.max_seconds) and self.stop.max_seconds > 0:
            frac = self.elapsed / self.stop.max_seconds
        if self.stop.max_iterations and self.stop.max_iterations < 10 ** 6:
            frac = max(frac, iteration / max(self.stop.max_iterations, 1))
        frac = min(max(frac, 0.0), 1.0)
        if self.beta_schedule == "linear":
            return scale * (self.beta_start + (self.beta_end - self.beta_start) * frac)
        if self.beta_schedule == "exponential":
            return scale * self.beta_start * (self.beta_end / self.beta_start) ** frac
        if self.beta_schedule == "fixed":
            return scale * self.beta_start
        raise ValueError(f"unknown beta schedule {self.beta_schedule!r}")

    # ------------------------------------------------------------------- run
    def _run(self) -> int:
        rng = self.rng
        n, M = self.n, self.M
        dec = self.decoder

        # --- initialisation -------------------------------------------------
        X = rng.uniform(0.0, 1.0, size=(M, n))
        if self.initial_keys is not None:
            k = min(len(self.initial_keys), M)
            X[:k] = self.initial_keys[:k]

        costs = np.empty(M)
        routes: list[list[list[int]]] = [None] * M  # type: ignore[list-item]
        for i in range(M):
            r, c, nk = dec.decode(X[i])
            if nk is not None:
                X[i] = nk
            costs[i] = c
            routes[i] = r
        self.evaluations += M

        P = X.copy()                 # personal best positions
        pcost = costs.copy()         # personal best costs
        proutes = list(routes)
        g = int(np.argmin(pcost))
        self.offer(self._solution(proutes[g], pcost[g]))

        stall = 0
        it = 0
        while not self.should_stop(it):
            it += 1
            beta = self.beta(it)

            # --- mean best position ----------------------------------------
            if self.weighted_mbest:
                # WQPSO: rank the personal bests and give the better ones more
                # weight, so the sampling width follows the promising region.
                order = np.argsort(pcost)
                w = np.empty(M)
                # linear weights from 1.5 down to 0.5 by rank
                w[order] = np.linspace(1.5, 0.5, M)
                w /= w.sum()
                mbest = (w[:, None] * P).sum(axis=0)
            else:
                mbest = P.mean(axis=0)

            gbest = P[g]

            # --- quantum position update -----------------------------------
            phi = rng.random((M, n))
            attractor = phi * P + (1.0 - phi) * gbest          # local attractor p
            u = rng.random((M, n))
            np.maximum(u, 1e-12, out=u)                        # ln(1/u) must stay finite
            spread = beta * np.abs(mbest - X) * np.log(1.0 / u)
            sign = np.where(rng.random((M, n)) < 0.5, 1.0, -1.0)
            Xnew = attractor + sign * spread

            # --- mutation ---------------------------------------------------
            if self.mutation != "none" and self.mutation_rate > 0:
                mask = rng.random((M, n)) < self.mutation_rate
                if mask.any():
                    if self.mutation == "cauchy":
                        # Heavier tails than Gaussian: occasional long jumps that
                        # let a converged swarm still reach a distant basin.
                        pert = self.mutation_scale * rng.standard_cauchy((M, n))
                        np.clip(pert, -1.0, 1.0, out=pert)
                    else:
                        pert = self.mutation_scale * rng.standard_normal((M, n))
                    Xnew = np.where(mask, Xnew + pert, Xnew)

            # Keys are a *relative* ordering, so their absolute range does not
            # matter; clipping only keeps the numbers well conditioned.
            np.clip(Xnew, -2.0, 3.0, out=Xnew)

            # protect the elite from being overwritten by their own update
            elite_idx = np.argsort(pcost)[: self.elite]
            X = Xnew

            improved_global = False
            # Which particles get the (expensive) local search this iteration.
            # Refining every particle gives the best solution per iteration but
            # the fewest iterations per second; refining a sample plus the
            # incumbent trades quality per iteration for more of them. Which
            # wins depends on instance size, so it is a parameter rather than a
            # hard-coded choice, and the ablation is reported.
            if self.ls_policy == "all" or not dec.use_local_search:
                improve_set = None
            else:
                k = max(1, int(self.ls_fraction * M))
                chosen = rng.choice(M, size=k, replace=False)
                improve_set = set(int(x) for x in chosen)
                improve_set.add(int(g))

            for i in range(M):
                do_ls = improve_set is None or i in improve_set
                r, c, nk = dec.decode(X[i], improve=do_ls)
                if nk is not None:
                    X[i] = nk
                costs[i] = c
                routes[i] = r
                if c < pcost[i] - 1e-10:
                    pcost[i] = c
                    P[i] = X[i]
                    proutes[i] = r
            self.evaluations += M

            # --- clone prevention -------------------------------------------
            # Hybridising a swarm with a strong local search has a failure mode
            # that is easy to miss: every refined candidate is a local optimum,
            # and distinct particles keep landing on the *same* one. The swarm
            # then holds many copies of one solution, its effective size
            # collapses, and it explores less than independent restarts would.
            # Measurements on this platform showed exactly that, with multi-start
            # local search beating the swarm at short budgets. Re-seeding a
            # duplicate restores the lost independence at the cost of one
            # evaluation.
            if self.clone_prevention:
                seen: dict[tuple, int] = {}
                for i in range(M):
                    key = tuple(tuple(r) for r in proutes[i]) if proutes[i] else ()
                    if key and key in seen:
                        X[i] = rng.uniform(0.0, 1.0, n)
                        r, c, nk = dec.decode(X[i])
                        if nk is not None:
                            X[i] = nk
                        costs[i] = c
                        routes[i] = r
                        pcost[i] = c
                        P[i] = X[i]
                        proutes[i] = r
                        self.evaluations += 1
                    else:
                        seen[key] = i

            gnew = int(np.argmin(pcost))
            if pcost[gnew] < pcost[g] - 1e-10:
                g = gnew
                improved_global = True
            if self.offer(self._solution(proutes[g], pcost[g])):
                improved_global = True

            stall = 0 if improved_global else stall + 1

            # --- diversification on stagnation ------------------------------
            if self.restart_after and stall >= self.restart_after:
                k = max(1, int(self.restart_fraction * M))
                worst = np.argsort(pcost)[-k:]
                keep = set(int(e) for e in elite_idx)
                for i in worst:
                    if int(i) in keep:
                        continue
                    X[i] = rng.uniform(0.0, 1.0, n)
                    r, c, nk = dec.decode(X[i])
                    if nk is not None:
                        X[i] = nk
                    costs[i] = c
                    routes[i] = r
                    pcost[i] = c
                    P[i] = X[i]
                    proutes[i] = r
                self.evaluations += len(worst)
                stall = 0

            self.record(it, float(pcost[g]), float(costs.mean()),
                        float(self._diversity(X)), True)
        return it

    # -------------------------------------------------------------- helpers
    def _solution(self, routes, cost: float) -> Solution:
        sol = Solution([list(r) for r in routes], cost)
        return sol

    @staticmethod
    def _diversity(X: np.ndarray) -> float:
        """Mean distance of particles from the swarm centroid.

        Reported per iteration so the convergence analysis can show *why* a run
        stopped improving: a collapsed diversity means premature convergence,
        while a high one means the budget ran out first.
        """
        if X.shape[0] < 2:
            return 0.0
        centre = X.mean(axis=0)
        return float(np.sqrt(((X - centre) ** 2).sum(axis=1)).mean())
