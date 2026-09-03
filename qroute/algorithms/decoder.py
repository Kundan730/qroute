"""Continuous-to-discrete decoding.

Quantum-inspired particle swarms move points through a continuous space, but a
vehicle routing solution is a set of ordered customer sequences. The bridge is a
*random-key* representation, also called smallest-position-value encoding:

1. A particle is a vector ``x`` in ``R^n``, one real key per customer.
2. Sorting the keys ascending yields a permutation - the *giant tour*.
3. Prins' split cuts that tour into vehicle routes, optimally for that ordering.
4. Local search refines the routes.
5. The improved routes are written back into the keys (Lamarckian learning), so
   the swarm's memory carries the refined solution rather than the raw one.

Step 5 matters more than it looks. Without it the swarm keeps re-deriving the
same improvements from unchanged positions and the search stagnates; with it the
particles inherit the structure that local search discovered. The write-back
assigns each customer a key equal to its rank in the improved giant tour, scaled
into the same range as the search space.
"""

from __future__ import annotations

import numpy as np

from qroute.algorithms.kernels import labels_to_routes, split_tour
from qroute.algorithms.localsearch import local_search
from qroute.core.types import Solution
from qroute.problems.instance import Instance


class Decoder:
    """Turns real-valued position vectors into evaluated routing solutions.

    A single decoder instance is shared by all particles of a run: it owns the
    pre-computed matrices, the neighbour lists and the scratch buffers, so
    decoding does not allocate on the hot path.
    """

    def __init__(self, instance: Instance, neighbours: int = 15,
                 local_search_rounds: int = 30, or_opt_segment: int = 3,
                 penalty_capacity: float | None = None,
                 penalty_time_window: float | None = None,
                 penalty_duration: float | None = None, vehicle_cost: float = 0.0,
                 use_local_search: bool = True, respect_fleet: bool = True,
                 writeback: str = "canonical"):
        from qroute.algorithms.localsearch import neighbour_lists

        self.instance = instance
        self.cost = instance.cost_matrix
        self.demand = instance.demand
        self.capacity = float(instance.capacity)
        self.n = instance.n_customers
        self.use_local_search = use_local_search
        self.ls_rounds = int(local_search_rounds)
        self.or_opt_segment = int(or_opt_segment)
        self.vehicle_cost = float(vehicle_cost)
        if writeback not in ("canonical", "preserve", "none"):
            raise ValueError(f"unknown writeback scheme {writeback!r}")
        self.writeback = writeback

        # Time-window data: a 1x2 zero array is the sentinel for "no windows",
        # which the kernels test with tw.shape[0] > 1.
        if instance.has_time_windows:
            self.tw = np.ascontiguousarray(instance.time_windows, dtype=np.float64)
            self.service = np.ascontiguousarray(instance.service_time, dtype=np.float64)
        else:
            self.tw = np.zeros((1, 2), dtype=np.float64)
            self.service = np.zeros(1, dtype=np.float64)

        self.max_duration = float(instance.max_route_duration or 0.0)
        # Penalties must be expressed in the instance's own units. A fixed value
        # is meaningless across problem families: benchmark instances have arc
        # costs around 10 to 100 distance units, while a road network measured in
        # seconds has arcs in the hundreds and route costs above 100,000. A flat
        # penalty of 1000 is prohibitive in the first case and negligible in the
        # second, which is how an overloaded route can slip through on a road
        # network while looking impossible on A-n32-k5. Scaling by the exchange
        # rate between cost and demand fixes that; see
        # qroute.algorithms.penalty.AdaptivePenalty for the same rule.
        self.pen_cap = float(penalty_capacity) if penalty_capacity is not None \
            else self.default_capacity_penalty(instance)
        scale = float(np.max(self.cost)) if np.isfinite(self.cost).all() else 1.0
        self.pen_tw = float(penalty_time_window) if penalty_time_window is not None \
            else max(1.0, scale)
        self.pen_dur = float(penalty_duration) if penalty_duration is not None \
            else max(1.0, scale)

        self.max_routes = int(instance.n_vehicles) if (respect_fleet and instance.n_vehicles) else 0
        # One-way streets make road-network matrices asymmetric, which changes
        # how segment-reversing moves must be priced.
        self.symmetric = bool(np.allclose(self.cost, self.cost.T, rtol=1e-9, atol=1e-9))
        k = int(min(max(neighbours, 5), max(self.n, 1)))
        self.neigh = neighbour_lists(self.cost, k)
        self._route_of = np.full(instance.size, -1, dtype=np.int32)
        self._position_of = np.full(instance.size, -1, dtype=np.int32)
        # Ring buffer for the don't-look-bit work queue, sized generously so a
        # burst of re-queued neighbours never overflows it.
        self._queue = np.zeros(max(instance.size * 4, 64), dtype=np.int32)
        self._in_queue = np.zeros(instance.size, dtype=np.bool_)

    # ------------------------------------------------------------------ API
    @staticmethod
    def default_capacity_penalty(instance: Instance) -> float:
        """Cost charged per unit of overload, in the instance's own units.

        One unit of overload is priced at roughly the longest arc divided by the
        largest single demand, so overloading a vehicle by its biggest customer
        costs about as much as the worst detour in the instance. That makes the
        weight comparable across problem families without hand tuning.

        The multiplier of three was chosen by measurement, not taste. Sweeping
        it over 1, 3, 10, 30, 100 and 300 on A-n45-k7, A-n80-k10, B-n78-k10,
        X-n101-k25, C101 and R101 with three seeds and an eight-second budget
        gave mean gaps of 2.27, 1.66, 1.67, 1.78, 1.79 and 1.74 percent. The
        unscaled rate is clearly too permissive; everything from three upwards is
        indistinguishable, so the smallest value in that plateau is used, which
        keeps the search free to cross infeasible ground without letting it
        wander. Every setting ended feasible because the reporting repair pass
        runs regardless.
        """
        max_cost = float(np.max(instance.cost_matrix))
        max_demand = float(np.max(instance.demand))
        if max_demand <= 0.0 or not np.isfinite(max_cost):
            return 1.0
        return float(np.clip(3.0 * max_cost / max_demand, 0.1, 1e6))

    def keys_to_tour(self, keys: np.ndarray) -> np.ndarray:
        """Sort the random keys into a customer permutation (1-based)."""
        order = np.argsort(keys, kind="stable")
        return (order + 1).astype(np.int32)

    def decode(self, keys: np.ndarray, improve: bool | None = None):
        """Decode a position vector.

        Returns ``(routes, penalised_cost, new_keys)``. ``new_keys`` is the
        Lamarckian write-back: ``None`` when local search is disabled or made no
        change, otherwise the keys that reproduce the improved solution.
        """
        improve = self.use_local_search if improve is None else improve
        tour = self.keys_to_tour(keys)
        labels, split_cost = split_tour(tour, self.cost, self.demand, self.capacity,
                                        self.max_routes, self.service, self.tw,
                                        self.vehicle_cost, self.pen_tw)
        flat, starts, n_routes = labels_to_routes(tour, labels)
        if not improve or n_routes == 0:
            cost = self._penalised(flat, starts, n_routes)
            return self._routes(flat, starts, n_routes), cost, None

        flat2, starts2, nr2, cost2 = local_search(
            flat, starts, n_routes, self.cost, self.demand, self.capacity,
            self.service, self.tw, self.max_duration, self.pen_cap, self.pen_tw,
            self.pen_dur, self.vehicle_cost, self.neigh, self._route_of,
            self._position_of, self._queue, self._in_queue,
            self.ls_rounds, self.or_opt_segment, self.symmetric)
        new_keys = None if self.writeback == "none" else self.tour_to_keys(flat2, keys)
        return self._routes(flat2, starts2, nr2), cost2, new_keys

    def tour_to_keys(self, flat: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Write a customer sequence back into random keys (Lamarckian update).

        Two schemes are available and the choice matters more than it looks:

        ``canonical``
            Key of the customer visited *i*-th becomes ``(i + 0.5) / n``. Every
            particle then expresses the same ordering with the same numbers, so
            the mean best position that QPSO samples around is the genuine
            average visit position of each customer. This is the representation
            that biased random-key methods rely on.
        ``preserve``
            The particle keeps its own sorted key values and only their
            assignment changes. Diversity in key space stays high, but the mean
            best position becomes close to meaningless because two particles can
            encode the same tour with very different vectors.

        ``canonical`` is the default because the swarm has no way to exploit a
        mean position that does not correspond to an ordering.
        """
        n = flat.shape[0]
        new = np.empty_like(template)
        if self.writeback == "canonical":
            new[flat - 1] = (np.arange(n, dtype=np.float64) + 0.5) / n
        else:
            new[flat - 1] = np.sort(template)[:n]
        return new

    def evaluate_routes(self, routes) -> float:
        """Penalised cost of an explicit route list (used by non-key algorithms)."""
        from qroute.core.types import routes_to_array

        flat, starts = routes_to_array(routes)
        return self._penalised(flat, starts, len(routes))

    def improve_routes(self, routes):
        """Run local search on an explicit route list."""
        from qroute.core.types import routes_to_array

        flat, starts = routes_to_array(routes)
        flat2, starts2, nr2, cost2 = local_search(
            flat.astype(np.int32), starts.astype(np.int32), len(routes), self.cost,
            self.demand, self.capacity, self.service, self.tw, self.max_duration,
            self.pen_cap, self.pen_tw, self.pen_dur, self.vehicle_cost, self.neigh,
            self._route_of, self._position_of, self._queue, self._in_queue,
            self.ls_rounds, self.or_opt_segment, self.symmetric)
        return self._routes(flat2, starts2, nr2), cost2

    def to_solution(self, routes) -> Solution:
        return self.instance.make_solution(routes)

    def repair(self, routes, max_rounds: int = 4):
        """Restore feasibility of a solution before it is reported.

        The search deliberately allows mildly infeasible solutions, because the
        feasible region is disconnected under the usual move operators and a
        search confined to it gets stuck. That is a good bargain during the
        search and a bad one at the end: a route two units over capacity is not
        a solution a depot can dispatch.

        This performs the standard escalation. The capacity, time-window and
        duration penalties are multiplied by ten and local search is run again;
        if a violation survives, they are multiplied again, up to
        ``max_rounds`` times. Because the penalties grow geometrically, a move
        that removes a violation becomes worth far more than any routing saving,
        so the search is driven back into the feasible region rather than
        merely nudged.

        If escalation still leaves an overloaded route - which happens when the
        fleet is so tight that no arrangement of the current routes fits - the
        overflowing customers are split off into an additional route. That is
        always feasible when the fleet is unlimited, and when it is not, the
        result is reported with an explicit fleet violation rather than a
        capacity one, because exceeding a stated fleet size is a decision an
        operator can act on while an overloaded vehicle is not.

        Returns ``(routes, cost)`` with the cost recomputed under the original
        penalties, so it stays comparable with everything else.
        """
        current = [list(r) for r in routes if len(r) > 0]
        best = current
        saved = (self.pen_cap, self.pen_tw, self.pen_dur)
        try:
            for k in range(max_rounds):
                stats = self.instance.evaluate(best)
                if stats.total_violation <= 1e-9:
                    break
                self.pen_cap = saved[0] * (10.0 ** (k + 1))
                self.pen_tw = saved[1] * (10.0 ** (k + 1))
                self.pen_dur = saved[2] * (10.0 ** (k + 1))
                improved, _ = self.improve_routes(best)
                if improved and self.instance.evaluate(improved).total_violation <= stats.total_violation + 1e-9:
                    best = improved
        finally:
            self.pen_cap, self.pen_tw, self.pen_dur = saved

        stats = self.instance.evaluate(best)
        if stats.capacity_violation > 1e-9:
            best = self._split_overloaded(best)
        return best, self._penalised_routes(best)

    def _split_overloaded(self, routes):
        """Move the tail of every overloaded route into fresh routes.

        Customers are removed from the end of an overloaded route until it fits,
        which preserves the order of the customers that remain, and the removed
        ones are packed greedily into new routes. This never fails and never
        loses a customer; it trades cost for feasibility, which is the right
        direction at reporting time.
        """
        demand = self.instance.demand
        keep: list[list[int]] = []
        overflow: list[int] = []
        for route in routes:
            r = list(route)
            while r and float(demand[r].sum()) > self.capacity + 1e-9:
                overflow.append(r.pop())
            if r:
                keep.append(r)
        # Greedy first-fit for the displaced customers, largest demand first.
        overflow.sort(key=lambda c: -float(demand[c]))
        for c in overflow:
            placed = False
            for r in keep:
                if float(demand[r].sum()) + float(demand[c]) <= self.capacity + 1e-9:
                    r.append(c)
                    placed = True
                    break
            if not placed:
                keep.append([c])

        # ``keep`` is feasible by construction. Polishing it with local search is
        # worth a few percent, but only under a penalty large enough that no
        # routing saving can pay for re-creating an overload; under the ordinary
        # weight the search happily merges two routes again and undoes the
        # repair. The polished result is therefore accepted only if it is still
        # feasible, so this step can improve the answer and never spoil it.
        saved = self.pen_cap
        try:
            self.pen_cap = max(saved, 1.0) * 1e6
            polished, _ = self.improve_routes(keep)
        finally:
            self.pen_cap = saved
        if polished and self.instance.evaluate(polished).capacity_violation <= 1e-9:
            return polished
        return keep

    def _penalised_routes(self, routes) -> float:
        from qroute.core.types import routes_to_array

        flat, starts = routes_to_array(routes)
        return self._penalised(flat, starts, len(routes))

    # -------------------------------------------------------------- internals
    def _penalised(self, flat, starts, n_routes) -> float:
        from qroute.algorithms.kernels import evaluate

        total, *_ = evaluate(flat.astype(np.int32), starts.astype(np.int32), n_routes,
                             self.cost, self.demand, self.capacity, self.service,
                             self.tw, self.max_duration, self.pen_cap, self.pen_tw,
                             self.pen_dur, self.vehicle_cost)
        return float(total)

    @staticmethod
    def _routes(flat, starts, n_routes) -> list[list[int]]:
        return [[int(c) for c in flat[starts[r]:starts[r + 1]]]
                for r in range(n_routes) if starts[r + 1] > starts[r]]


def random_keys(rng: np.random.Generator, n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return rng.uniform(lo, hi, size=n)
