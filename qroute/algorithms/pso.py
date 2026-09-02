"""Classical Particle Swarm Optimisation over the random-key encoding.

This is the control experiment for the whole project. The claim under test is
that the quantum-behaved swarm searches better than the classical one, and that
claim is only meaningful if the two differ in exactly one thing: the position
update rule. Everything else here is deliberately identical to
:mod:`qroute.algorithms.qpso` - the same random-key representation, the same
:class:`~qroute.algorithms.decoder.Decoder` (optimal split plus local search),
the same Lamarckian write-back, the same stopping criteria, the same evaluation
accounting, and the same optional restart-on-stagnation. If PSO loses, it loses
because of the update rule.

The update rule
---------------
Each particle carries a position ``x`` and a velocity ``v``::

    v <- w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    x <- x + v

with ``r1, r2 ~ U(0, 1)`` drawn independently per dimension. The default
coefficients ``w = 0.7298`` and ``c1 = c2 = 1.49618`` are Clerc and Kennedy's
constriction values (2002): they are the algebraically derived setting at which
the particle's trajectory is a damped oscillation that converges without needing
an explicit velocity clamp. They remain the standard baseline, which is why they
are the default rather than the older ``w = 0.9 -> 0.4`` inertia schedule - that
schedule is available as ``inertia="linear"`` for comparison.

Why this is the interesting comparison
--------------------------------------
The velocity term is what bounds a classical particle's reachable set. In one
step a particle can only move as far as its velocity allows, so once the swarm
contracts, the region it can sample contracts with it and the search cannot
recover without an explicit restart. QPSO discards velocity and samples from a
double-exponential density whose support is the whole space, so its reachable
set never closes. That is the entire mechanistic difference, and it is the thing
this file exists to measure.

Not a strawman
--------------
Three deliberate choices keep this PSO competitive rather than convenient:

* A **ring topology** is available and is the default. In a global-best swarm
  every particle is pulled towards the same point and the swarm collapses fast;
  with a ring, information about the best position diffuses around the
  neighbourhood graph over several iterations, which is well documented to
  perform better on multimodal problems (Kennedy and Mendes, 2002). Choosing
  the weaker ``gbest`` topology would have made PSO look worse for a reason that
  has nothing to do with the quantum update.
* **Velocity clamping** to a fraction of the key range, so a particle that
  receives a large attraction term does not leave the useful part of the space
  in one step.
* The **same restart mechanism** QPSO has, so neither algorithm gets a
  diversification device the other lacks.
"""

from __future__ import annotations

import numpy as np

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder
from qroute.core.types import Solution


class PSO(Optimizer):
    """Canonical particle swarm optimiser over random-key positions."""

    name = "PSO"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 swarm_size: int = 30,
                 inertia: str = "constriction",
                 w: float = 0.7298,
                 w_start: float = 0.9,
                 w_end: float = 0.4,
                 c1: float = 1.49618,
                 c2: float = 1.49618,
                 velocity_clamp: float = 0.2,
                 topology: str = "ring",
                 neighbourhood: int = 2,
                 mutation: str = "none",
                 mutation_rate: float = 0.05,
                 mutation_scale: float = 0.10,
                 elite_fraction: float = 0.10,
                 restart_after: int = 60,
                 restart_fraction: float = 0.30,
                 local_search: bool = True,
                 neighbours: int = 15,
                 local_search_rounds: int = 30,
                 penalty_capacity: float = 1000.0,
                 penalty_time_window: float = 1000.0,
                 penalty_duration: float = 1000.0,
                 vehicle_cost: float = 0.0,
                 decoder: Decoder | None = None,
                 initial_keys: np.ndarray | None = None,
                 **kw):
        super().__init__(instance, stop, seed, callback,
                         swarm_size=swarm_size, inertia=inertia, w=w, c1=c1, c2=c2,
                         velocity_clamp=velocity_clamp, topology=topology,
                         neighbourhood=neighbourhood, mutation=mutation,
                         mutation_rate=mutation_rate, mutation_scale=mutation_scale,
                         elite_fraction=elite_fraction, restart_after=restart_after,
                         restart_fraction=restart_fraction, local_search=local_search,
                         neighbours=neighbours, **kw)
        if inertia not in ("constriction", "linear", "fixed"):
            raise ValueError(f"unknown inertia scheme {inertia!r}")
        if topology not in ("ring", "gbest"):
            raise ValueError(f"unknown topology {topology!r}")

        self.M = int(swarm_size)
        self.inertia = inertia
        self.w = float(w)
        self.w_start = float(w_start)
        self.w_end = float(w_end)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.vmax = float(velocity_clamp)
        self.topology = topology
        self.neighbourhood = max(1, int(neighbourhood))
        self.mutation = mutation
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.elite = max(1, int(elite_fraction * self.M))
        self.restart_after = int(restart_after)
        self.restart_fraction = float(restart_fraction)
        self.initial_keys = initial_keys
        self.decoder = decoder or Decoder(
            instance, neighbours=neighbours, local_search_rounds=local_search_rounds,
            penalty_capacity=penalty_capacity, penalty_time_window=penalty_time_window,
            penalty_duration=penalty_duration, vehicle_cost=vehicle_cost,
            use_local_search=local_search)
        self.n = instance.n_customers

    # --------------------------------------------------------------- schedule
    def progress(self, iteration: int) -> float:
        """Fraction of the budget consumed, in ``[0, 1]``.

        Taking the maximum over the iteration, wall-clock and evaluation limits
        means a time-budgeted run still gets a meaningful inertia schedule. A
        schedule driven only by ``max_iterations`` silently degenerates to a
        constant when the run is stopped by the clock instead, which is exactly
        the situation the benchmark uses.
        """
        s = self.stop
        frac = 0.0
        if s.max_iterations > 0:
            frac = max(frac, iteration / s.max_iterations)
        if np.isfinite(s.max_seconds) and s.max_seconds > 0:
            frac = max(frac, self.elapsed / s.max_seconds)
        if s.max_evaluations:
            frac = max(frac, self.evaluations / s.max_evaluations)
        return float(min(frac, 1.0))

    def inertia_weight(self, iteration: int) -> float:
        if self.inertia == "linear":
            return self.w_start + (self.w_end - self.w_start) * self.progress(iteration)
        return self.w  # "constriction" and "fixed" both hold w constant

    def coefficients(self) -> tuple[float, float]:
        if self.inertia == "linear":
            # The classical decreasing-inertia variant is normally paired with
            # c1 = c2 = 2.0; keeping the constriction coefficients with a
            # decreasing w would put the trajectory outside its stability region.
            return 2.0, 2.0
        return self.c1, self.c2

    # ------------------------------------------------------------------- run
    def _run(self) -> int:
        rng = self.rng
        n, M = self.n, self.M
        dec = self.decoder

        X = rng.uniform(0.0, 1.0, size=(M, n))
        if self.initial_keys is not None:
            k = min(len(self.initial_keys), M)
            X[:k] = self.initial_keys[:k]
        # Half-range initial velocities: large enough to explore, small enough
        # that the first few steps do not scramble the seeded positions.
        V = rng.uniform(-self.vmax, self.vmax, size=(M, n)) * 0.5

        costs = np.empty(M)
        routes: list[list[list[int]]] = [None] * M  # type: ignore[list-item]
        for i in range(M):
            r, c, nk = dec.decode(X[i])
            if nk is not None:
                X[i] = nk
            costs[i] = c
            routes[i] = r
        self.evaluations += M

        P = X.copy()              # personal best positions
        pcost = costs.copy()      # personal best costs
        proutes = list(routes)
        g = int(np.argmin(pcost))
        self.offer(self._solution(proutes[g], pcost[g]))

        nbr_index = self._ring_index(M) if self.topology == "ring" else None

        stall = 0
        it = 0
        while not self.should_stop(it):
            it += 1
            w = self.inertia_weight(it)
            c1, c2 = self.coefficients()

            # --- social attractor -------------------------------------------
            if nbr_index is None:
                attractor = np.broadcast_to(P[g], (M, n))
            else:
                # Best personal best inside each particle's ring neighbourhood.
                best_in_nbr = nbr_index[np.arange(M), np.argmin(pcost[nbr_index], axis=1)]
                attractor = P[best_in_nbr]

            # --- velocity and position --------------------------------------
            r1 = rng.random((M, n))
            r2 = rng.random((M, n))
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (attractor - X)
            np.clip(V, -self.vmax, self.vmax, out=V)
            Xnew = X + V

            if self.mutation != "none" and self.mutation_rate > 0:
                mask = rng.random((M, n)) < self.mutation_rate
                if mask.any():
                    if self.mutation == "cauchy":
                        pert = self.mutation_scale * rng.standard_cauchy((M, n))
                        np.clip(pert, -1.0, 1.0, out=pert)
                    else:
                        pert = self.mutation_scale * rng.standard_normal((M, n))
                    Xnew = np.where(mask, Xnew + pert, Xnew)

            # Absorbing walls: a particle pushed outside the key range is placed
            # on the boundary and its velocity in that dimension is zeroed, so it
            # does not keep accelerating into a wall it cannot cross.
            out_lo = Xnew < 0.0
            out_hi = Xnew > 1.0
            if out_lo.any() or out_hi.any():
                V[out_lo | out_hi] = 0.0
                np.clip(Xnew, 0.0, 1.0, out=Xnew)
            X = Xnew

            elite_idx = np.argsort(pcost)[: self.elite]

            for i in range(M):
                r, c, nk = dec.decode(X[i])
                if nk is not None:
                    # Lamarckian write-back. The velocity is intentionally left
                    # alone: it still points in the direction the particle was
                    # travelling, and re-deriving a velocity for the rewritten
                    # position would be arbitrary.
                    X[i] = nk
                costs[i] = c
                routes[i] = r
                if c < pcost[i] - 1e-10:
                    pcost[i] = c
                    P[i] = X[i]
                    proutes[i] = r
            self.evaluations += M

            improved_global = False
            gnew = int(np.argmin(pcost))
            if pcost[gnew] < pcost[g] - 1e-10:
                g = gnew
                improved_global = True
            if self.offer(self._solution(proutes[g], pcost[g])):
                improved_global = True
            stall = 0 if improved_global else stall + 1

            if self.restart_after and stall >= self.restart_after:
                k = max(1, int(self.restart_fraction * M))
                worst = np.argsort(pcost)[-k:]
                keep = {int(e) for e in elite_idx}
                for i in worst:
                    if int(i) in keep:
                        continue
                    X[i] = rng.uniform(0.0, 1.0, n)
                    V[i] = rng.uniform(-self.vmax, self.vmax, n) * 0.5
                    r, c, nk = dec.decode(X[i])
                    if nk is not None:
                        X[i] = nk
                    costs[i] = c
                    routes[i] = r
                    pcost[i] = c
                    P[i] = X[i]
                    proutes[i] = r
                    self.evaluations += 1
                stall = 0

            self.record(it, float(pcost[g]), float(costs.mean()),
                        float(self._diversity(X)), True)
        return it

    # --------------------------------------------------------------- helpers
    def _ring_index(self, M: int) -> np.ndarray:
        """``(M, 2k+1)`` table of each particle's ring neighbours, itself included."""
        k = min(self.neighbourhood, max((M - 1) // 2, 0))
        offsets = np.arange(-k, k + 1)
        return (np.arange(M)[:, None] + offsets[None, :]) % M

    @staticmethod
    def _solution(routes, cost: float) -> Solution:
        return Solution([list(r) for r in routes], cost)

    @staticmethod
    def _diversity(X: np.ndarray) -> float:
        """Mean distance of particles from the swarm centroid.

        Identical to the QPSO measure so the two convergence plots are directly
        comparable; a collapsing curve is the signature of the velocity-bounded
        contraction described in the module docstring.
        """
        if X.shape[0] < 2:
            return 0.0
        centre = X.mean(axis=0)
        return float(np.sqrt(((X - centre) ** 2).sum(axis=1)).mean())
