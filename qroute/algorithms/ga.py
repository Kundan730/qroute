"""Steady-state genetic algorithm over the random-key encoding.

The GA is the second control experiment. Like :mod:`qroute.algorithms.pso` it
shares the representation, the decoder (optimal split plus local search), the
Lamarckian write-back and the stopping rules with QPSO, so the comparison
isolates the search rule: population-based recombination against swarm dynamics.

Why steady state rather than generational
-----------------------------------------
A generational GA replaces the whole population at once, which under a fixed
wall-clock budget means the best individual found early cannot influence
anything until the next generation is complete. A steady-state GA inserts each
child immediately, so a good child starts contributing to selection at once.
With an expensive decoder - split plus local search dominates the cost of an
evaluation here - the number of evaluations is the scarce resource, and
steady-state extracts more search per evaluation. It also makes elitism trivial
and makes the "one iteration" unit configurable, which is what lets the
convergence histories of the GA and the swarms be plotted on the same axes:
``offspring_per_iteration`` defaults to half the population so that one recorded
GA iteration costs about as many evaluations as one swarm iteration.

Everything is really a permutation
----------------------------------
Three crossovers are provided. Order crossover (OX) works directly on the
decoded permutation; blend and uniform crossover work on the key values. They
are not as different as they look, because the decoder only ever sees the
*ordering* the keys induce, and because the Lamarckian write-back rewrites every
surviving chromosome into canonical form ``(rank + 0.5) / n`` anyway. This
implementation therefore canonicalises each child immediately after
recombination. That is not a loss: it discards only information the decoder
cannot read, and it keeps the mutation operators (swap and inversion) meaningful
on every child regardless of how it was produced.

Duplicate filtering
-------------------
A steady-state population with elitism converges to clones very quickly, and a
population of clones evaluates the same solution over and over. Children whose
penalised cost matches an incumbent's to within a tolerance are rejected unless
they improve the global best. This is a cheap surrogate for a genuine diversity
measure (a broken-pairs distance over the routes would be better, and slower);
it is honest to call it a heuristic, and it is the reason the population does
not collapse under a 15-second budget.
"""

from __future__ import annotations

import numpy as np

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder
from qroute.core.types import Solution

_CROSSOVERS = ("ox", "blend", "uniform", "mixed")


class GeneticAlgorithm(Optimizer):
    """Steady-state GA with order/blend crossover on random keys."""

    name = "GA"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 population: int = 50,
                 offspring_per_iteration: int = 0,
                 crossover: str = "mixed",
                 ox_probability: float = 0.7,
                 blend_alpha: float = 0.25,
                 mutation_rate: float = 0.30,
                 inversion_probability: float = 0.5,
                 mutation_strength: float = 0.05,
                 tournament_size: int = 2,
                 elite_fraction: float = 0.10,
                 duplicate_tolerance: float = 1e-6,
                 restart_after: int = 60,
                 restart_fraction: float = 0.30,
                 local_search: bool = True,
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
                         population=population,
                         offspring_per_iteration=offspring_per_iteration,
                         crossover=crossover, ox_probability=ox_probability,
                         blend_alpha=blend_alpha, mutation_rate=mutation_rate,
                         inversion_probability=inversion_probability,
                         mutation_strength=mutation_strength,
                         tournament_size=tournament_size,
                         elite_fraction=elite_fraction,
                         restart_after=restart_after, restart_fraction=restart_fraction,
                         local_search=local_search, neighbours=neighbours, **kw)
        if crossover not in _CROSSOVERS:
            raise ValueError(f"unknown crossover {crossover!r}, expected one of {_CROSSOVERS}")

        self.P = max(4, int(population))
        self.children = int(offspring_per_iteration) or max(1, self.P // 2)
        self.crossover = crossover
        self.ox_probability = float(ox_probability)
        self.blend_alpha = float(blend_alpha)
        self.mutation_rate = float(mutation_rate)
        self.inversion_probability = float(inversion_probability)
        self.mutation_strength = float(mutation_strength)
        self.tournament = max(2, int(tournament_size))
        self.elite = max(1, int(elite_fraction * self.P))
        self.duplicate_tolerance = float(duplicate_tolerance)
        self.restart_after = int(restart_after)
        self.restart_fraction = float(restart_fraction)
        self.initial_keys = initial_keys
        self.decoder = decoder or Decoder(
            instance, neighbours=neighbours, local_search_rounds=local_search_rounds,
            penalty_capacity=penalty_capacity, penalty_time_window=penalty_time_window,
            penalty_duration=penalty_duration, vehicle_cost=vehicle_cost,
            use_local_search=local_search)
        self.n = instance.n_customers

    # ------------------------------------------------------------------- run
    def _run(self) -> int:
        rng = self.rng
        n, P = self.n, self.P
        dec = self.decoder

        X = rng.uniform(0.0, 1.0, size=(P, n))
        if self.initial_keys is not None:
            k = min(len(self.initial_keys), P)
            X[:k] = self.initial_keys[:k]

        costs = np.empty(P)
        routes: list[list[list[int]]] = [None] * P  # type: ignore[list-item]
        for i in range(P):
            r, c, nk = dec.decode(X[i])
            if nk is not None:
                X[i] = nk
            costs[i] = c
            routes[i] = r
        self.evaluations += P

        best = int(np.argmin(costs))
        self.offer(self._solution(routes[best], costs[best]))

        stall = 0
        it = 0
        while not self.should_stop(it):
            it += 1
            improved = False
            for _ in range(self.children):
                # Two independent tournaments; drawing the same parent twice is
                # allowed and simply yields a mutated copy, which is a useful
                # small-step move rather than a bug.
                a = self._tournament(rng, costs)
                b = self._tournament(rng, costs)
                child = self._recombine(rng, X[a], X[b])
                child = self._mutate(rng, child)

                r, c, nk = dec.decode(child)
                if nk is not None:
                    child = nk
                self.evaluations += 1

                # --- steady-state replacement -----------------------------
                order = np.argsort(costs)
                worst = int(order[-1])
                is_new_best = c < costs[order[0]] - 1e-10
                if not is_new_best and self._is_duplicate(c, costs):
                    continue
                if c >= costs[worst] and not is_new_best:
                    continue
                if self.elite >= self.P:
                    continue  # every slot is protected; nothing can be replaced
                X[worst] = child
                costs[worst] = c
                routes[worst] = r

                if self.offer(self._solution(r, c)):
                    improved = True

            stall = 0 if improved else stall + 1

            if self.restart_after and stall >= self.restart_after:
                order = np.argsort(costs)
                k = max(1, int(self.restart_fraction * P))
                for i in order[-k:]:
                    i = int(i)
                    X[i] = rng.uniform(0.0, 1.0, n)
                    r, c, nk = dec.decode(X[i])
                    if nk is not None:
                        X[i] = nk
                    costs[i] = c
                    routes[i] = r
                    self.evaluations += 1
                stall = 0

            # The incumbent, not the population best: a restart can evict a good
            # individual, and the history's best-cost column must be monotone if
            # it is to be plotted against the other algorithms' curves.
            self.record(it, float(self._best.cost), float(costs.mean()),
                        float(self._diversity(costs)), True)
        return it

    # ------------------------------------------------------------- operators
    def _tournament(self, rng: np.random.Generator, costs: np.ndarray) -> int:
        """Binary (or k-ary) tournament selection: pick k at random, keep the best."""
        picks = rng.integers(0, costs.shape[0], size=self.tournament)
        return int(picks[int(np.argmin(costs[picks]))])

    def _recombine(self, rng: np.random.Generator, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
        mode = self.crossover
        if mode == "mixed":
            mode = "ox" if rng.random() < self.ox_probability else "blend"
        if mode == "ox":
            perm = self._order_crossover(rng, np.argsort(pa, kind="stable"),
                                         np.argsort(pb, kind="stable"))
            return self._canonical(perm)
        if mode == "blend":
            # BLX-alpha: sample each gene from an interval slightly wider than
            # the two parent values, so the child can lie just outside the
            # parents' span and the population does not contract in key space.
            lo = np.minimum(pa, pb)
            hi = np.maximum(pa, pb)
            span = hi - lo
            u = rng.random(pa.shape[0])
            child = (lo - self.blend_alpha * span) + u * (span * (1 + 2 * self.blend_alpha))
        else:  # uniform
            mask = rng.random(pa.shape[0]) < 0.5
            child = np.where(mask, pa, pb)
        return self._canonical(np.argsort(child, kind="stable"))

    @staticmethod
    def _order_crossover(rng: np.random.Generator, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
        """Order crossover (OX) on two permutations of ``0..n-1``.

        A contiguous slice of the first parent is copied to the child in place;
        the remaining positions are filled with the elements of the second
        parent in the order they appear there, skipping those already present.
        OX is the standard choice for permutation problems because it preserves
        the *relative order* of the second parent rather than absolute
        positions, and relative order is what the split step actually consumes.
        """
        n = pa.shape[0]
        if n < 2:
            return pa.copy()
        i, j = np.sort(rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=pa.dtype)
        child[i:j + 1] = pa[i:j + 1]
        taken = np.zeros(n, dtype=bool)
        taken[pa[i:j + 1]] = True
        fill = pb[~taken[pb]]
        holes = np.concatenate([np.arange(j + 1, n), np.arange(0, i)])
        child[holes] = fill
        return child

    def _mutate(self, rng: np.random.Generator, keys: np.ndarray) -> np.ndarray:
        """Swap and inversion mutation, plus a small key jitter.

        Swap exchanges two customers, which is a minimal perturbation of the
        giant tour. Inversion reverses a segment, which on a symmetric instance
        costs nothing inside the segment and only changes its two endpoints -
        exactly the 2-opt move, so it explores a structurally different
        direction from swap. The key jitter is a small continuous perturbation
        that can reorder near-ties; it is what keeps the encoding continuous
        rather than purely combinatorial.
        """
        if rng.random() >= self.mutation_rate:
            return keys
        n = keys.shape[0]
        perm = np.argsort(keys, kind="stable")
        if n >= 2:
            if rng.random() < self.inversion_probability:
                i, j = np.sort(rng.choice(n, size=2, replace=False))
                perm[i:j + 1] = perm[i:j + 1][::-1]
            else:
                i, j = rng.choice(n, size=2, replace=False)
                perm[i], perm[j] = perm[j], perm[i]
        out = self._canonical(perm)
        if self.mutation_strength > 0:
            out = out + self.mutation_strength * rng.standard_normal(n) / max(n, 1)
        return out

    @staticmethod
    def _canonical(perm: np.ndarray) -> np.ndarray:
        """Keys that reproduce ``perm`` exactly, in the decoder's canonical form."""
        n = perm.shape[0]
        keys = np.empty(n, dtype=np.float64)
        keys[perm] = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1)
        return keys

    # --------------------------------------------------------------- helpers
    def _is_duplicate(self, cost: float, costs: np.ndarray) -> bool:
        return bool(np.any(np.abs(costs - cost) <= self.duplicate_tolerance))

    @staticmethod
    def _solution(routes, cost: float) -> Solution:
        return Solution([list(r) for r in routes], cost)

    @staticmethod
    def _diversity(costs: np.ndarray) -> float:
        """Coefficient of variation of the population's costs.

        A population-level diversity measure in *objective* space rather than
        key space, because after canonicalisation two chromosomes encoding the
        same tour are identical vectors and a key-space spread would understate
        how varied the population really is.
        """
        mean = float(costs.mean())
        if not np.isfinite(mean) or mean == 0.0:
            return 0.0
        return float(costs.std() / abs(mean))
