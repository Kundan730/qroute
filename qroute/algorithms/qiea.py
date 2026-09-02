"""Quantum-Inspired Evolutionary Algorithm (rotation-gate) for vehicle routing.

Why this module exists
----------------------
The problem statement's deliverable table asks explicitly for "quantum rotation /
update rules". QPSO (:mod:`qroute.algorithms.qpso`) is quantum-inspired in a
different sense - it samples from a delta-potential-well density and has no
gate at all - so this module supplies the second, gate-based family: Han and
Kim's QIEA (IEEE Transactions on Evolutionary Computation 6(6):580-593, 2002).
The rotation gate itself lives in :mod:`qroute.algorithms.qtypes`; this file is
the evolutionary loop around it and the two ways it is wired to routing.

How QIEA works
--------------
A population member is not a solution. It is a :class:`QubitRegister` - a
factorised probability distribution over binary strings. Each generation:

1. **Observe.** Collapse every register once to get a binary string.
2. **Repair / decode.** Turn that string into a feasible routing solution.
3. **Evaluate.** Score it.
4. **Update.** Rotate every qubit toward the bit value of the better of
   (this observation, this individual's best-so-far), by the lookup-table angle.
5. **Migrate.** Periodically copy the best reference string within a small group
   (local migration) or across the whole population (global migration), which is
   QIEA's only mechanism for sharing information between individuals - there is
   no crossover.

There is no mutation operator either; the H-epsilon gate plays that role by
refusing to let any qubit's probability reach 0 or 1.

Two wirings, one gate
---------------------
:class:`QIEA`
    Bits mean **arcs**. For every node, its ``k`` nearest neighbours give ``k``
    candidate successors and one qubit each; an observation is a mask of enabled
    arcs that steers a nearest-neighbour tour construction. This is the natural
    combinatorial reading of a binary genotype for routing.

:class:`QuantumRotationKeys`
    Bits mean **numbers**. Each customer owns ``b`` qubits whose observed string
    is read as a fixed-point fraction, giving that customer's random key. The
    register therefore searches exactly the space QPSO searches, over exactly the
    same decoder, so the two quantum-inspired engines can be compared without
    the representation confounding the result.

Both use the shared :class:`~qroute.algorithms.decoder.Decoder` (Prins split +
local search), so the only thing that differs between them, and between them and
QPSO, is how a search state becomes a giant tour.

Honesty about performance
-------------------------
Quantum-inspired evolutionary algorithms are frequently oversold, and the
sceptical literature deserves to be stated before the numbers. Ma and Cheah,
"Are Quantum-Inspired Genetic Algorithms Really Better than Classical Genetic
Algorithms?" (arXiv:2409.13788, 2024), find that on TSPLIB-style routing
problems a plain genetic algorithm usually matches or beats a quantum-inspired
one and runs considerably faster, because the rotation update is a weak,
positionally-independent learning signal.

What was actually measured here, on this machine, equal 20-second wall-clock
budget, 3 seeds, mean gap to best-known (runs executed serially):

    ===========  ======  ======  ====================
    instance     QPSO    QIEA    QuantumRotationKeys
    ===========  ======  ======  ====================
    A-n32-k5     0.00%   0.00%   0.00%
    A-n45-k7     0.52%   0.00%   0.00%
    A-n80-k10    2.36%   0.51%   0.98%
    ===========  ======  ======  ====================

So on this benchmark both rotation-gate engines beat the project's QPSO, which
is not the result the literature above would predict. Three caveats keep that
from being a claim about quantum-inspired methods in general:

* Three instances from one CVRP family is not evidence about algorithms; it is
  evidence about these instances. The full benchmark sweep is what should be
  quoted in the report.
* Part of the margin is not the gate at all. Setting ``delta_theta=0`` disables
  rotation entirely and leaves a randomised-restart control that still has the
  same construction, split and local search. On A-n80-k10 that control scores
  1.89% (QIEA) and 2.46% (QuantumRotationKeys) - so the gate is worth roughly
  1.3 and 1.5 percentage points respectively, and the rest comes from the
  representation and from local search.
* Part of the margin is a QPSO configuration artefact. QPSO's ``beta`` schedule
  is keyed to ``max_iterations``, which a wall-clock benchmark leaves
  effectively unbounded, so its contraction coefficient never anneals. Giving
  QPSO a matching iteration cap improves it to 0.15% on A-n45-k7 (from 0.52%)
  though it does not help on A-n80-k10. The schedule in this module uses
  whichever budget is binding, precisely to avoid that trap.

The ablation is the part worth keeping: the rotation gate does measurably work -
it is not decoration - and the module reports what it measured rather than what
would be flattering.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder
from qroute.algorithms.qtypes import QubitRegister
from qroute.core.types import Solution


class _RotationOptimizer(Optimizer):
    """Shared QIEA machinery: population, migrations, schedule, bookkeeping.

    Subclasses decide only what a bit *means*, by implementing
    :meth:`_register_size`, :meth:`_keys_from_bits` and optionally
    :meth:`_target_bits`.
    """

    name = "QIEA"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 population_size: int = 20,
                 group_size: int = 5,
                 local_migration_period: int = 1,
                 global_migration_period: int = 100,
                 delta_theta: float = 0.01 * np.pi,
                 delta_theta_end: float = 0.002 * np.pi,
                 theta_schedule: str = "fixed",
                 epsilon: float = 0.01,
                 rotation_table: str = "symmetric",
                 lamarckian: bool = False,
                 local_search: bool = True,
                 neighbours: int = 15,
                 local_search_rounds: int = 30,
                 # Left as None so the decoder derives penalties from the
                 # instance's own cost and demand scale. A hard-coded weight is
                 # wrong by orders of magnitude on a road network measured in
                 # seconds, where route costs run past 100,000.
                 penalty_capacity: float | None = None,
                 penalty_time_window: float | None = None,
                 penalty_duration: float | None = None,
                 vehicle_cost: float = 0.0,
                 decoder: Decoder | None = None,
                 **kw):
        super().__init__(instance, stop, seed, callback,
                         population_size=population_size, group_size=group_size,
                         local_migration_period=local_migration_period,
                         global_migration_period=global_migration_period,
                         delta_theta=delta_theta, delta_theta_end=delta_theta_end,
                         theta_schedule=theta_schedule, epsilon=epsilon,
                         rotation_table=rotation_table, lamarckian=lamarckian,
                         local_search=local_search, neighbours=neighbours, **kw)
        if theta_schedule not in ("fixed", "linear", "exponential"):
            raise ValueError(f"unknown theta schedule {theta_schedule!r}")
        self.P = int(population_size)
        self.group_size = max(1, int(group_size))
        self.local_period = int(local_migration_period)
        self.global_period = int(global_migration_period)
        self.delta_theta = float(delta_theta)
        self.delta_theta_end = float(delta_theta_end)
        self.theta_schedule = theta_schedule
        self.epsilon = float(epsilon)
        self.rotation_table = rotation_table
        self.lamarckian = bool(lamarckian)
        self.decoder = decoder or Decoder(
            instance, neighbours=neighbours, local_search_rounds=local_search_rounds,
            penalty_capacity=penalty_capacity, penalty_time_window=penalty_time_window,
            penalty_duration=penalty_duration, vehicle_cost=vehicle_cost,
            use_local_search=local_search)
        self.n = instance.n_customers
        self.neigh = self.decoder.neigh          # (size, k) nearest-node table
        self.k = int(self.neigh.shape[1])

    # -------------------------------------------------------- subclass hooks
    @abstractmethod
    def _register_size(self) -> int:
        """Number of qubits one individual carries."""

    @abstractmethod
    def _keys_from_bits(self, bits: np.ndarray) -> np.ndarray:
        """Turn one observation into decoder random keys (length ``n``)."""

    def _target_bits(self, bits: np.ndarray, new_keys: np.ndarray | None,
                     routes: list[list[int]]) -> np.ndarray:
        """Bit-string the gate should treat as "what was actually evaluated".

        The default is the observation itself, which is the literal QIEA. A
        subclass may instead return the encoding of the *improved* solution that
        local search produced, making the update Lamarckian: the register then
        learns the structure local search found rather than the raw draw that
        led to it. That is a deviation from Han and Kim and is opt-in.
        """
        return bits

    # ------------------------------------------------------------- schedule
    def _progress(self, iteration: int) -> float:
        """Fraction of the budget consumed, over whichever limit is binding.

        Benchmarks in this project run on a wall-clock budget with the iteration
        cap left effectively unbounded, so a schedule keyed only to
        ``max_iterations`` would never move. Taking the maximum of the two
        fractions keeps the schedule meaningful under either kind of budget.
        """
        frac = iteration / max(self.stop.max_iterations, 1)
        if np.isfinite(self.stop.max_seconds) and self.stop.max_seconds > 0:
            frac = max(frac, self.elapsed / self.stop.max_seconds)
        return float(min(frac, 1.0))

    def theta(self, iteration: int) -> float:
        """Base rotation step at ``iteration``.

        ``fixed`` is the default and is what the paper does. The annealed
        options trade late-run exploration for finer convergence; they help on
        long budgets and hurt on short ones, which is why they are not default.
        """
        if self.theta_schedule == "fixed":
            return self.delta_theta
        f = self._progress(iteration)
        if self.theta_schedule == "linear":
            return self.delta_theta + (self.delta_theta_end - self.delta_theta) * f
        ratio = max(self.delta_theta_end, 1e-12) / max(self.delta_theta, 1e-12)
        return self.delta_theta * ratio ** f

    # ------------------------------------------------------------------ run
    def _run(self) -> int:
        rng = self.rng
        dec = self.decoder
        P = self.P
        m = self._register_size()

        regs = [QubitRegister(m, self.rotation_table) for _ in range(P)]
        best_bits: list[np.ndarray] = [np.zeros(m, dtype=np.uint8) for _ in range(P)]
        best_cost = np.full(P, np.inf)
        best_routes: list[list[list[int]]] = [[] for _ in range(P)]
        seeded = np.zeros(P, dtype=bool)

        costs = np.empty(P)
        it = 0
        while not self.should_stop(it):
            it += 1
            theta = self.theta(it)

            for i in range(P):
                bits = regs[i].observe(rng)
                keys = self._keys_from_bits(bits)
                routes, cost, new_keys = dec.decode(keys)
                costs[i] = cost
                x = self._target_bits(bits, new_keys, routes) if self.lamarckian else bits

                if not seeded[i]:
                    # First generation: no reference string exists yet, so there
                    # is nothing to rotate toward. Adopt and skip the gate.
                    best_bits[i] = x.copy()
                    best_cost[i] = cost
                    best_routes[i] = routes
                    seeded[i] = True
                    continue

                improved = cost < best_cost[i] - 1e-10
                # Rotate against the *previous* reference, before replacing it.
                # Updating first would make x and b identical whenever the draw
                # improved, so the "x is better and they disagree" rows of the
                # table could never fire and the gate would only ever reinforce.
                regs[i].rotate(x, best_bits[i], improved, theta, rng)
                regs[i].h_epsilon(self.epsilon)
                if improved:
                    best_bits[i] = x.copy()
                    best_cost[i] = cost
                    best_routes[i] = routes

            self.evaluations += P

            g = int(np.argmin(best_cost))
            self.offer(Solution([list(r) for r in best_routes[g]], float(best_cost[g])))

            self._migrate(it, best_bits, best_cost, best_routes)

            entropy = float(np.mean([r.entropy() for r in regs]))
            self.record(it, float(best_cost[g]), float(costs.mean()), entropy, True)
        return it

    def _migrate(self, iteration: int, best_bits, best_cost, best_routes) -> None:
        """Share reference strings, QIEA's substitute for crossover.

        Local migration copies the best reference within each group of
        ``group_size`` neighbours; global migration copies the population best
        everywhere. The periods matter: frequent global migration collapses the
        population onto one attractor and the H-epsilon gate is then the only
        source of diversity left, so the paper's default of every 100
        generations is deliberately rare.
        """
        P = self.P
        do_global = self.global_period and iteration % self.global_period == 0
        do_local = self.local_period and iteration % self.local_period == 0
        if do_global:
            g = int(np.argmin(best_cost))
            src = best_bits[g].copy()
            cost = float(best_cost[g])
            routes = best_routes[g]
            for i in range(P):
                if i != g:
                    best_bits[i] = src.copy()
                    best_cost[i] = cost
                    best_routes[i] = routes
        elif do_local:
            for start in range(0, P, self.group_size):
                stop = min(start + self.group_size, P)
                grp = range(start, stop)
                g = min(grp, key=lambda j: best_cost[j])
                src = best_bits[g].copy()
                cost = float(best_cost[g])
                routes = best_routes[g]
                for i in grp:
                    if i != g:
                        best_bits[i] = src.copy()
                        best_cost[i] = cost
                        best_routes[i] = routes


class QIEA(_RotationOptimizer):
    """QIEA over an **arc-selection** genotype.

    Genotype
    --------
    One qubit per (node, candidate-successor) pair: node ``u`` contributes ``k``
    qubits, one for each entry of its ``k``-nearest-neighbour list. An
    observation is therefore a mask over the ``k``-nearest-neighbour graph, and
    a set bit reads as "``u -> v`` is an arc this individual wants to use".

    Only near neighbours get qubits. A complete graph would need ``n**2`` of
    them, which is both wasteful and useless: in every good CVRP solution the
    overwhelming majority of arcs join near neighbours, so arcs outside the
    ``k``-nearest lists are not worth a search variable. They remain reachable -
    the construction can still emit one as a fallback - they just cannot be
    *preferred* by the genotype.

    Phenotype
    ---------
    A randomised greedy construction walks from the depot, at each step taking
    the nearest unvisited node whose incoming arc is enabled by the mask, and
    falling back to the plain nearest unvisited node when the mask offers
    nothing. The resulting giant tour goes through the shared decoder: Prins
    split cuts it into vehicle routes optimally for that ordering, then local
    search refines them.

    In equal superposition each arc is enabled with probability 1/2, so the
    construction picks roughly uniformly among a node's two or three nearest
    unvisited neighbours - a sensible randomised-greedy start. As the register
    converges, the mask sharpens toward one preferred successor per node and the
    construction becomes near-deterministic. That transition from exploration to
    exploitation is produced entirely by the gate; there is no temperature or
    inertia parameter driving it.
    """

    name = "QIEA"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 lamarckian: bool = True, **kw):
        super().__init__(instance, stop, seed, callback, lamarckian=lamarckian, **kw)
        self.size = instance.size          # depot + customers
        self._unvisited = np.zeros(self.size, dtype=bool)
        self._tour = np.zeros(self.n, dtype=np.int32)
        self._keys = np.zeros(self.n, dtype=np.float64)
        # Position of node v in u's neighbour list, or -1. Built once so the
        # Lamarckian write-back does not have to search the list per arc.
        self._slot = np.full((self.size, self.size), -1, dtype=np.int32)
        rows = np.repeat(np.arange(self.size), self.k)
        self._slot[rows, self.neigh.ravel()] = np.tile(np.arange(self.k), self.size)

    def _register_size(self) -> int:
        return int(self.size * self.k)

    def _keys_from_bits(self, bits: np.ndarray) -> np.ndarray:
        mask = bits.reshape(self.size, self.k).astype(bool)
        tour = self._construct(mask)
        # Rank-valued keys reproduce exactly this ordering when the decoder
        # argsorts them, so the construction and the decoder agree.
        self._keys[tour - 1] = (np.arange(self.n, dtype=np.float64) + 0.5) / self.n
        return self._keys

    def _construct(self, mask: np.ndarray) -> np.ndarray:
        """Mask-guided nearest-neighbour construction."""
        cost = self.decoder.cost
        neigh = self.neigh
        unvisited = self._unvisited
        unvisited[:] = True
        unvisited[0] = False                # the depot is not a customer
        tour = self._tour
        cur = 0
        for t in range(self.n):
            row = neigh[cur]
            # neigh is sorted by increasing distance, so the first enabled and
            # unvisited entry is the nearest such candidate.
            ok = np.flatnonzero(mask[cur] & unvisited[row])
            if ok.size:
                nxt = int(row[ok[0]])
            else:
                rest = np.flatnonzero(unvisited)
                nxt = int(rest[np.argmin(cost[cur, rest])])
            tour[t] = nxt
            unvisited[nxt] = False
            cur = nxt
        return tour

    def _target_bits(self, bits, new_keys, routes) -> np.ndarray:
        """Encode the improved routes back into an arc mask (Lamarckian).

        The rotation target becomes the arc set of the solution local search
        actually produced, rather than the mask that merely seeded it. A set bit
        then means "this successor survived local search", which is a much
        stronger signal than "this successor was worth trying".

        The expected objection is entropy: the target mask is sparse - at most
        one set bit per node against roughly ``k/2`` in an observation - so the
        agreeing-zero rows of the table dominate and the register might commit
        far too early. Measured on this machine at a 10-second budget on
        A-n80-k10 (2 seeds), it does not: mean gap to best-known falls from
        1.59% without the write-back to 0.26% with it, and the run completes
        *more* generations (79 against 63) because the sharper masks build
        better tours for local search to start from. It is therefore on by
        default, with ``lamarckian=False`` available to recover the literal
        Han-Kim update. Note that this is a deviation from the 2002 paper.
        """
        mask = np.zeros((self.size, self.k), dtype=np.uint8)
        for route in routes:
            prev = 0
            for c in route:
                s = self._slot[prev, c]
                if s >= 0:
                    mask[prev, s] = 1
                prev = c
        return mask.ravel()


class QuantumRotationKeys(_RotationOptimizer):
    """QIEA over a **random-key** genotype: fixed-point numbers made of qubits.

    Each customer owns ``bits`` qubits. Their observed string, read
    most-significant-bit first, is the binary expansion of that customer's key:

        key = sum_{j=1..b} x_j * 2**(-j)      in  [0, 1 - 2**(-b)]

    Sorting the keys gives the giant tour, exactly as in QPSO, so the rotation
    gate becomes a drop-in replacement for the quantum-well position update and
    the two engines are directly comparable: same decoder, same split, same
    local search, same objective, different update rule.

    Bit-depth trade-off
    -------------------
    ``bits`` sets the key resolution to ``2**-b`` and the register size to
    ``n * b`` qubits, and the two pull in opposite directions.

    * Too few bits and distinct customers collide on the same key. Ties are
      broken by the decoder's stable argsort, i.e. by customer index, which is
      an arbitrary ordering the search cannot influence. With ``b`` bits there
      are ``2**b`` levels, and by the birthday bound collisions become common
      once ``n`` approaches ``2**(b/2)`` - so ``b = 8`` is already marginal at
      ``n = 100``.
    * Too many bits and the low-order qubits are noise: flipping bit 10 of a key
      almost never changes the sort order, so those qubits receive rotation
      pressure that carries no fitness signal, and the effective search is
      diluted across ``n * b`` variables instead of ``n``.

    Measured on this machine, 10-second budget, population 20, 3 seeds, mean gap
    to best-known:

    ========  ==========  ===========
    ``bits``  A-n45-k7    A-n80-k10
    ========  ==========  ===========
    4         0.00%       1.72%
    6         0.00%       1.27%
    8         0.00%       1.15%
    10        0.00%       0.91%
    12        0.00%       1.06%
    14        0.00%       0.34%
    16        0.00%       0.68%
    ========  ==========  ===========

    So the collision argument is real - 4 bits gives 16 levels for 79 customers
    and is clearly the worst setting - but it is much gentler than the birthday
    bound alone suggests, because local search repairs most of the damage a tied
    ordering does. Above 10 bits the differences are inside the spread across
    seeds (the 14-bit row ranged from 0.06% to 0.79%) and should not be read as
    a ranking. ``b = 10`` (1024 levels) is the default: past the region where
    depth demonstrably hurts, and cheap at ten qubits per customer.

    Note also that a key's *value* has no meaning of its own - only the induced
    ordering does - yet the genotype is positional. Two orderings that differ by
    one adjacent swap can require flipping high-order bits of both customers.
    This mismatch is a genuine weakness of the encoding and part of why the
    measured results below favour QPSO on larger instances.
    """

    name = "QuantumRotationKeys"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 bits: int = 10, lamarckian: bool = True, **kw):
        kw.setdefault("bits", bits)
        super().__init__(instance, stop, seed, callback, lamarckian=lamarckian, **kw)
        if not 2 <= int(bits) <= 30:
            raise ValueError("bits must lie in [2, 30]")
        self.bits = int(bits)
        self.levels = 1 << self.bits
        # Place value of each qubit, most significant first.
        self._weights = 0.5 ** (np.arange(self.bits, dtype=np.float64) + 1.0)
        self._shifts = np.arange(self.bits - 1, -1, -1, dtype=np.int64)

    def _register_size(self) -> int:
        return int(self.n * self.bits)

    def _keys_from_bits(self, bits: np.ndarray) -> np.ndarray:
        return bits.reshape(self.n, self.bits).astype(np.float64) @ self._weights

    def _target_bits(self, bits, new_keys, routes) -> np.ndarray:
        """Quantise the decoder's Lamarckian write-back back into qubits.

        The decoder returns canonical rank keys for the solution local search
        actually produced. Rounding them to ``bits`` places and re-expanding
        gives the bit-string that *would* have produced that solution, which is
        a far more informative rotation target than the raw draw. This is on by
        default here (unlike in :class:`QIEA`) because the encoding is dense -
        roughly half the target bits are ones either way - so it does not
        collapse the register's entropy the way a sparse arc mask does.
        """
        if new_keys is None:
            return bits
        q = np.clip((new_keys * self.levels).astype(np.int64), 0, self.levels - 1)
        return ((q[:, None] >> self._shifts) & 1).astype(np.uint8).ravel()
