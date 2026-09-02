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
                 penalty_capacity: float = 1000.0, penalty_time_window: float = 1000.0,
                 penalty_duration: float = 1000.0, vehicle_cost: float = 0.0,
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
        self.pen_cap = float(penalty_capacity)
        self.pen_tw = float(penalty_time_window)
        self.pen_dur = float(penalty_duration)

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
        flat, starts, n_routes = labels_to_routes(tour, labels, max(self.max_routes, self.n))
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
