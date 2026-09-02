"""Simulated annealing on an explicit route representation.

Simulated annealing is the natural third control, and it is the one that differs
most from the swarm methods: it maintains a single solution rather than a
population, and its only source of diversification is the probability of
accepting a worse solution. Including it answers an obvious objection to the
whole project - that a population is not actually needed and a well-tuned
single-trajectory search would do as well.

Representation
--------------
Unlike PSO, QPSO and the GA, this solver does not use random keys. It works
directly on the routes, because the classical SA neighbourhoods (relocate, swap,
2-opt, Or-opt) are defined on routes and translating them through a key encoding
would change what they mean. It still shares the objective with every other
algorithm: candidate solutions are priced with the same compiled evaluator and
the same penalty weights the :class:`~qroute.algorithms.decoder.Decoder` holds,
and periodic intensification calls ``decoder.improve_routes``, which is the same
local search the swarms use. So the comparison is still like for like on cost;
only the search rule and the neighbourhood differ.

Internally the solution is a flat ``int32`` array of customers plus a per-route
length array. Every move is a small number of NumPy slice concatenations, and
the candidate is priced by a full pass of the compiled evaluator. A full
re-evaluation is ``O(n)`` where an incremental delta would be ``O(1)``, and that
is a real cost. It was chosen anyway because an incremental delta has to
replicate the capacity, time-window and duration accounting exactly, and any
discrepancy between the incremental and the reference objective silently
invalidates the comparison this module exists to make. At the instance sizes
here the compiled evaluator makes a full pass cheap enough that correctness is
worth more than the constant factor; the honest consequence is that SA's
throughput advantage over the decoder-based methods shrinks as ``n`` grows.

Initial temperature
-------------------
A hand-picked initial temperature is meaningless across instances whose costs
differ by four orders of magnitude, so it is calibrated instead. Starting from
the initial solution the search performs a short random walk in which *every*
move is accepted, and records the size of each uphill (worsening) step. If those
steps have mean magnitude ``dbar``, then choosing

    T0 = dbar / -ln(chi)

makes the acceptance probability ``exp(-d / T0)`` of an *average* uphill move
equal to the target ``chi`` (default 0.4). This is the coarse form of the
estimator; Ben-Ameur (2004) gives an iterative version that matches the target
acceptance rate exactly rather than at the mean, which is not used here because
the extra fidelity does not survive the geometric cooling that follows. A random
walk that accepts everything is used rather than sampling around a fixed
solution because it visits the cost landscape the search will actually see.

Cooling and reheating
---------------------
Geometric cooling ``T <- alpha * T`` applied once per temperature level, where a
level is a fixed number of proposed moves proportional to ``n``. When the
incumbent has not improved for a number of levels, the temperature is reset
upward and the search restarts from the best solution found. Reheating is what
keeps a single-trajectory search useful under a long budget: without it, once
``T`` has decayed the search is a deterministic descent that has already
finished.

How the defaults were chosen, and what they imply
-------------------------------------------------
Measured on A-n80-k10, X-n101-k25 and R101, three seeds each, five-second
budget (mean gap to best known)::

    2 n moves per level, intensify every level    +3.17%   <- the default
    2 n moves per level, intensify every 2        +3.47%
    4 n, intensify every 2, alpha = 0.98          +3.44%
    4 n, intensify every 2                        +3.66%
    4 n, intensify every 5                        +4.32%
    8 n, intensify every 2                        +4.97%
    8 n, intensify every 5                        +5.55%
    8 n, intensify every 20 (initial guess)       +6.67%

The trend is monotone and it is worth being honest about what it means: the
better SA gets, the less of the work the annealing itself is doing. At the
tuned setting a level is short and local search runs at the end of every level,
which makes the method close to an annealing-driven iterated local search
rather than textbook SA. That is what the measurements support, and it is the
configuration used in the comparison, because a baseline should be reported at
its best rather than at its most canonical. The textbook setting is one
argument away (``moves_per_customer=8, intensify_every=0``) for anyone who
wants to see the difference the intensification makes.
"""

from __future__ import annotations

import numpy as np

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder, random_keys
from qroute.algorithms.kernels import evaluate as kernel_evaluate
from qroute.core.types import Solution


class SimulatedAnnealing(Optimizer):
    """Single-trajectory annealing over route-level neighbourhoods."""

    name = "SA"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 alpha: float = 0.995,
                 moves_per_customer: int = 2,
                 min_moves_per_level: int = 100,
                 target_acceptance: float = 0.40,
                 calibration_moves: int = 300,
                 min_temperature_fraction: float = 1e-4,
                 reheat_after: int = 25,
                 reheat_fraction: float = 0.5,
                 intensify_every: int = 1,
                 or_opt_max_segment: int = 3,
                 move_weights: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
                 guided_probability: float = 0.5,
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
                         alpha=alpha, moves_per_customer=moves_per_customer,
                         min_moves_per_level=min_moves_per_level,
                         min_temperature_fraction=min_temperature_fraction,
                         target_acceptance=target_acceptance,
                         calibration_moves=calibration_moves,
                         reheat_after=reheat_after, reheat_fraction=reheat_fraction,
                         intensify_every=intensify_every,
                         or_opt_max_segment=or_opt_max_segment,
                         move_weights=tuple(move_weights),
                         guided_probability=guided_probability,
                         local_search=local_search, neighbours=neighbours, **kw)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not 0.0 < target_acceptance < 1.0:
            raise ValueError("target_acceptance must lie strictly between 0 and 1")

        self.alpha = float(alpha)
        self.moves_per_customer = int(moves_per_customer)
        self.min_moves_per_level = int(min_moves_per_level)
        self.target_acceptance = float(target_acceptance)
        self.calibration_moves = int(calibration_moves)
        self.min_temperature_fraction = float(min_temperature_fraction)
        self.reheat_after = int(reheat_after)
        self.reheat_fraction = float(reheat_fraction)
        self.intensify_every = int(intensify_every)
        self.or_opt_max = max(2, int(or_opt_max_segment))
        w = np.asarray(move_weights, dtype=np.float64)
        if w.shape != (4,) or w.min() < 0 or w.sum() <= 0:
            raise ValueError("move_weights must be four non-negative numbers")
        self.move_cdf = np.cumsum(w / w.sum())
        self.guided_probability = float(guided_probability)

        self.decoder = decoder or Decoder(
            instance, neighbours=neighbours, local_search_rounds=local_search_rounds,
            penalty_capacity=penalty_capacity, penalty_time_window=penalty_time_window,
            penalty_duration=penalty_duration, vehicle_cost=vehicle_cost,
            use_local_search=local_search)
        self.n = instance.n_customers
        # Fleet capacity of the internal representation. With a fixed fleet the
        # array has exactly K slots, so no move can ever exceed the fleet; with
        # an unlimited fleet a couple of spare empty routes are enough, because
        # opening a route always costs two extra depot arcs and is only ever
        # worth it to repair an overload.
        self._max_routes = self.decoder.max_routes

    # ------------------------------------------------------------------- run
    def _run(self) -> int:
        rng = self.rng
        dec = self.decoder

        # --- starting solution ---------------------------------------------
        # Built with exactly the machinery the other algorithms use for their
        # initial population, so SA does not start from a systematically better
        # or worse point than the swarms.
        keys = random_keys(rng, self.n)
        routes0, _cost0, _ = dec.decode(keys)
        self.evaluations += 1

        flat, lengths = self._pack(routes0)
        cost = self._cost(flat, lengths)
        best_flat, best_lengths, best_cost = flat.copy(), lengths.copy(), cost
        self.offer(self._solution(flat, lengths, cost))

        # --- initial temperature -------------------------------------------
        T0 = self._calibrate(rng, flat.copy(), lengths.copy(), cost)
        T = T0
        T_min = max(T0 * self.min_temperature_fraction, 1e-12)

        moves_per_level = max(self.min_moves_per_level,
                              self.moves_per_customer * self.n)
        stall_levels = 0
        it = 0
        # The stopping rules are consulted once per temperature level rather
        # than once per move. A level is a few hundred to a few thousand moves
        # of a few microseconds each, so the wall-clock overshoot is small,
        # whereas calling ``perf_counter`` on every move would be a measurable
        # share of the run.
        #
        # The temperature calibration above already spent evaluations, so under
        # a very small evaluation budget ``should_stop`` can already be true
        # here. One level always runs, so a run never returns an empty history.
        while it == 0 or not self.should_stop(it):
            it += 1
            accepted = 0
            level_improved = False
            for _ in range(moves_per_level):
                cand = self._propose(rng, flat, lengths)
                if cand is None:
                    continue
                new_flat, new_lengths = cand
                new_cost = self._cost(new_flat, new_lengths)
                self.evaluations += 1
                delta = new_cost - cost
                if delta <= 0.0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                    flat, lengths, cost = new_flat, new_lengths, new_cost
                    accepted += 1
                    if cost < best_cost - 1e-10:
                        best_flat, best_lengths, best_cost = flat.copy(), lengths.copy(), cost
                        self.offer(self._solution(flat, lengths, cost))
                        level_improved = True

            # --- periodic intensification -----------------------------------
            if self.intensify_every and it % self.intensify_every == 0:
                routes = self._unpack(flat, lengths)
                improved, icost = dec.improve_routes(routes)
                self.evaluations += 1
                if icost < cost - 1e-10:
                    flat, lengths = self._pack(improved)
                    cost = icost
                    if cost < best_cost - 1e-10:
                        best_flat, best_lengths, best_cost = flat.copy(), lengths.copy(), cost
                        self.offer(self._solution(flat, lengths, cost))
                        level_improved = True

            stall_levels = 0 if level_improved else stall_levels + 1

            # --- cooling and reheating --------------------------------------
            T = max(T * self.alpha, T_min)
            if self.reheat_after and stall_levels >= self.reheat_after:
                T = max(T, T0 * self.reheat_fraction)
                flat, lengths, cost = best_flat.copy(), best_lengths.copy(), best_cost
                stall_levels = 0

            # The history's ``diversity`` slot carries SA's acceptance rate.
            # It plays the same diagnostic role as swarm diversity - it says
            # whether the search is still moving or has frozen - but the two
            # are different quantities and must not be plotted on one axis.
            self.record(it, float(best_cost), float(cost),
                        float(accepted / max(moves_per_level, 1)), True)
        return it

    # ----------------------------------------------------------- temperature
    def _calibrate(self, rng: np.random.Generator, flat: np.ndarray,
                   lengths: np.ndarray, cost: float) -> float:
        """Estimate ``T0`` from the mean uphill step of an accept-everything walk."""
        uphill: list[float] = []
        for _ in range(max(self.calibration_moves, 10)):
            cand = self._propose(rng, flat, lengths)
            if cand is None:
                continue
            new_flat, new_lengths = cand
            new_cost = self._cost(new_flat, new_lengths)
            self.evaluations += 1
            if new_cost > cost:
                uphill.append(new_cost - cost)
            flat, lengths, cost = new_flat, new_lengths, new_cost
        if not uphill:
            # Degenerate landscape (for instance a single-customer instance):
            # any positive temperature behaves identically.
            return 1.0
        dbar = float(np.mean(uphill))
        return max(dbar / -np.log(self.target_acceptance), 1e-9)

    # ----------------------------------------------------------------- moves
    def _propose(self, rng: np.random.Generator, flat: np.ndarray,
                 lengths: np.ndarray):
        """Draw one neighbour. Returns ``None`` when the draw was degenerate."""
        u = rng.random()
        kind = int(np.searchsorted(self.move_cdf, u, side="right"))
        kind = min(kind, 3)
        if kind == 0:
            return self._segment_move(rng, flat, lengths, seg_len=1, allow_reverse=False)
        if kind == 1:
            return self._swap(rng, flat, lengths)
        if kind == 2:
            return self._two_opt(rng, flat, lengths)
        seg = int(rng.integers(2, self.or_opt_max + 1))
        return self._segment_move(rng, flat, lengths, seg_len=seg, allow_reverse=True)

    def _segment_move(self, rng, flat, lengths, seg_len: int, allow_reverse: bool):
        """Relocate (``seg_len == 1``) or Or-opt: move a run of consecutive customers.

        The destination is chosen next to one of the segment head's nearest
        neighbours with probability ``guided_probability``. An unguided random
        insertion point is almost always terrible on a routing instance, so a
        purely random SA neighbourhood wastes most of its proposals; biasing
        towards the candidate lists is the standard fix and is the same
        neighbour structure the local search uses.
        """
        starts = self._starts(lengths)
        nonempty = np.flatnonzero(lengths >= seg_len)
        if nonempty.size == 0:
            return None
        r1 = int(rng.choice(nonempty))
        i = int(rng.integers(0, lengths[r1] - seg_len + 1))
        s1 = int(starts[r1])
        seg = flat[s1 + i: s1 + i + seg_len].copy()
        if allow_reverse and seg_len > 1 and rng.random() < 0.5:
            seg = seg[::-1].copy()

        rest = np.concatenate((flat[: s1 + i], flat[s1 + i + seg_len:]))
        new_lengths = lengths.copy()
        new_lengths[r1] -= seg_len
        rstarts = self._starts(new_lengths)

        pos = self._insertion_point(rng, rest, new_lengths, rstarts, int(seg[0]))
        if pos is None:
            return None
        r2, j = pos
        at = int(rstarts[r2]) + j
        new_flat = np.concatenate((rest[:at], seg, rest[at:]))
        new_lengths[r2] += seg_len
        return new_flat, new_lengths

    def _insertion_point(self, rng, rest, lengths, starts, head: int):
        """Pick ``(route, offset)`` for an insertion, guided by candidate lists."""
        n_routes = lengths.shape[0]
        if rng.random() < self.guided_probability:
            cands = self.decoder.neigh[head]
            target = int(cands[int(rng.integers(0, cands.shape[0]))])
            if target != 0:
                where = np.flatnonzero(rest == target)
                if where.size:
                    p = int(where[0])
                    r2 = int(np.searchsorted(starts, p, side="right") - 1)
                    # Insert immediately before or after the neighbour.
                    j = p - int(starts[r2]) + int(rng.integers(0, 2))
                    return r2, min(j, int(lengths[r2]))
        r2 = int(rng.integers(0, n_routes))
        return r2, int(rng.integers(0, lengths[r2] + 1))

    def _swap(self, rng, flat, lengths):
        """Exchange two customers, possibly in different routes."""
        n = flat.shape[0]
        if n < 2:
            return None
        p, q = rng.integers(0, n, size=2)
        if p == q:
            return None
        new_flat = flat.copy()
        new_flat[p], new_flat[q] = new_flat[q], new_flat[p]
        return new_flat, lengths.copy()

    def _two_opt(self, rng, flat, lengths):
        """Reverse a segment inside one route (the intra-route 2-opt move)."""
        starts = self._starts(lengths)
        nonempty = np.flatnonzero(lengths >= 2)
        if nonempty.size == 0:
            return None
        r = int(rng.choice(nonempty))
        s, L = int(starts[r]), int(lengths[r])
        a, b = np.sort(rng.choice(L, size=2, replace=False))
        new_flat = flat.copy()
        new_flat[s + a: s + b + 1] = new_flat[s + a: s + b + 1][::-1]
        return new_flat, lengths.copy()

    # --------------------------------------------------------------- packing
    def _pack(self, routes) -> tuple[np.ndarray, np.ndarray]:
        """Routes -> ``(flat, lengths)`` padded out to the working fleet size."""
        used = [r for r in routes if len(r)]
        slots = self._max_routes if self._max_routes > 0 else len(used) + 2
        slots = max(slots, len(used))
        lengths = np.zeros(slots, dtype=np.int64)
        for k, r in enumerate(used):
            lengths[k] = len(r)
        flat = np.concatenate([np.asarray(r, dtype=np.int32) for r in used]) \
            if used else np.zeros(0, dtype=np.int32)
        return np.ascontiguousarray(flat, dtype=np.int32), lengths

    @staticmethod
    def _starts(lengths: np.ndarray) -> np.ndarray:
        out = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
        np.cumsum(lengths, out=out[1:])
        return out

    def _unpack(self, flat: np.ndarray, lengths: np.ndarray) -> list[list[int]]:
        starts = self._starts(lengths)
        return [[int(c) for c in flat[starts[r]:starts[r + 1]]]
                for r in range(lengths.shape[0]) if lengths[r] > 0]

    # ------------------------------------------------------------ evaluation
    def _cost(self, flat: np.ndarray, lengths: np.ndarray) -> float:
        d = self.decoder
        starts = self._starts(lengths).astype(np.int32)
        total, *_ = kernel_evaluate(
            flat, starts, lengths.shape[0], d.cost, d.demand, d.capacity,
            d.service, d.tw, d.max_duration, d.pen_cap, d.pen_tw, d.pen_dur,
            d.vehicle_cost)
        return float(total)

    def _solution(self, flat: np.ndarray, lengths: np.ndarray, cost: float) -> Solution:
        return Solution(self._unpack(flat, lengths), cost)
