"""Problem instance model and the objective function.

This module is the executable form of the mathematical formulation required by
deliverable 2 of the problem statement. A single :class:`Instance` covers the
capacitated VRP, the VRP with time windows, and the time-dependent variant used
for the live-traffic demonstration, because they differ only in which optional
arrays are populated.

Formulation
-----------
Let ``G = (V, A)`` with ``V = {0, 1, ..., n}`` where 0 is the depot and
``1..n`` are customers. Decision variables are the binary arc-usage variables
``x[i][j][k] = 1`` iff vehicle ``k`` traverses arc ``(i, j)``. The platform
searches over route permutations, which is an equivalent and far more compact
encoding of the same feasible set.

Objective (weighted, as required by "minimise travel time, distance and
congestion")::

    min  w_time * sum(travel_time(i, j))
       + w_dist * sum(distance(i, j))
       + w_congestion * sum(congestion_penalty(i, j))
       + w_vehicles * (number of routes used)

Constraints:

1. every customer is visited exactly once             (enforced by the encoding)
2. route load must not exceed vehicle capacity Q      (``capacity_violation``)
3. service at customer i starts within [e_i, l_i]     (``time_window_violation``)
4. route duration must not exceed the shift limit     (``duration_violation``)
5. the number of routes must not exceed the fleet K   (``fleet_violation``)
6. arc flow must not exceed edge capacity             (``edge_load_violation``)

Constraints 1 is structural. Constraints 2-6 are handled by an adaptive penalty
(see :mod:`qroute.algorithms.penalty`) rather than by rejection, because on
tightly constrained instances a search that only ever visits feasible solutions
explores far more slowly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qroute.core.types import Solution, SolutionStats


@dataclass
class ObjectiveWeights:
    """Weights of the multi-objective cost function.

    Defaults reproduce the classical single-objective CVRP/VRPTW literature
    (pure distance), so benchmark numbers are directly comparable with
    published best-known solutions. The traffic demonstration overrides them.
    """

    time: float = 0.0
    distance: float = 1.0
    congestion: float = 0.0
    vehicles: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "time": self.time,
            "distance": self.distance,
            "congestion": self.congestion,
            "vehicles": self.vehicles,
        }


@dataclass
class Instance:
    """A routing problem instance.

    Attributes
    ----------
    name:
        Identifier used in reports.
    distance:
        ``(n+1, n+1)`` matrix of distances between depot and customers.
    duration:
        ``(n+1, n+1)`` matrix of travel times. Defaults to ``distance`` when the
        instance has no separate time information (the classical CVRP case).
    demand:
        ``(n+1,)`` demands; ``demand[0]`` is 0 for the depot.
    capacity:
        Vehicle capacity ``Q``.
    n_vehicles:
        Fleet size ``K``. ``None`` means unlimited, which is the convention used
        by CVRPLIB instances whose optimal route count is not fixed.
    time_windows:
        Optional ``(n+1, 2)`` array of ``[earliest, latest]`` service start times.
    service_time:
        Optional ``(n+1,)`` service durations.
    max_route_duration:
        Optional per-route duration limit (a driver shift).
    congestion:
        Optional ``(n+1, n+1)`` matrix in ``[0, 1]`` giving the congestion level
        of the shortest path between each pair, used by the congestion term of
        the objective.
    coords:
        Optional ``(n+1, 2)`` coordinates, used for plotting and for geographic
        instances the projected x/y or lon/lat of each stop.
    """

    name: str
    distance: np.ndarray
    demand: np.ndarray
    capacity: float
    duration: Optional[np.ndarray] = None
    n_vehicles: Optional[int] = None
    time_windows: Optional[np.ndarray] = None
    service_time: Optional[np.ndarray] = None
    max_route_duration: Optional[float] = None
    congestion: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    node_ids: Optional[list] = None  # original graph node ids, for map rendering
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        self.distance = np.ascontiguousarray(self.distance, dtype=np.float64)
        if self.duration is None:
            self.duration = self.distance
        else:
            self.duration = np.ascontiguousarray(self.duration, dtype=np.float64)
        self.demand = np.ascontiguousarray(self.demand, dtype=np.float64)
        if self.time_windows is not None:
            self.time_windows = np.ascontiguousarray(self.time_windows, dtype=np.float64)
        if self.service_time is None and self.time_windows is not None:
            self.service_time = np.zeros(self.size, dtype=np.float64)
        if self.service_time is not None:
            self.service_time = np.ascontiguousarray(self.service_time, dtype=np.float64)
        if self.congestion is not None:
            self.congestion = np.ascontiguousarray(self.congestion, dtype=np.float64)
        self._validate()
        self._cost_matrix = self._build_cost_matrix()

    def _validate(self) -> None:
        n = self.size
        if self.distance.shape != (n, n):
            raise ValueError(f"distance must be square ({n}, {n}), got {self.distance.shape}")
        if self.duration.shape != (n, n):
            raise ValueError(f"duration must be ({n}, {n}), got {self.duration.shape}")
        if self.demand[0] != 0:
            raise ValueError("depot demand must be zero")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        over = np.where(self.demand > self.capacity)[0]
        if over.size:
            raise ValueError(f"customers {over.tolist()} have demand above vehicle capacity")
        if self.time_windows is not None and self.time_windows.shape != (n, 2):
            raise ValueError(f"time_windows must be ({n}, 2), got {self.time_windows.shape}")

    def _build_cost_matrix(self) -> np.ndarray:
        """Pre-combine the weighted arc cost so inner loops touch one matrix."""
        w = self.weights
        c = w.distance * self.distance
        if w.time:
            c = c + w.time * self.duration
        if w.congestion and self.congestion is not None:
            # Congestion is charged proportionally to the time spent on the arc,
            # so a congested but very short link is not over-penalised.
            c = c + w.congestion * self.congestion * self.duration
        return np.ascontiguousarray(c, dtype=np.float64)

    def with_weights(self, weights: ObjectiveWeights) -> "Instance":
        """Return a copy of this instance scored under different weights."""
        import copy as _copy

        other = _copy.copy(self)
        other.weights = weights
        other._cost_matrix = other._build_cost_matrix()
        return other

    def with_matrices(
        self, distance: np.ndarray | None = None, duration: np.ndarray | None = None,
        congestion: np.ndarray | None = None,
    ) -> "Instance":
        """Return a copy with updated travel matrices (the dynamic weight update)."""
        import copy as _copy

        other = _copy.copy(self)
        if distance is not None:
            other.distance = np.ascontiguousarray(distance, dtype=np.float64)
        if duration is not None:
            other.duration = np.ascontiguousarray(duration, dtype=np.float64)
        if congestion is not None:
            other.congestion = np.ascontiguousarray(congestion, dtype=np.float64)
        other._cost_matrix = other._build_cost_matrix()
        return other

    # ------------------------------------------------------------- properties
    @property
    def size(self) -> int:
        """Number of nodes including the depot."""
        return int(self.demand.shape[0])

    @property
    def n_customers(self) -> int:
        return self.size - 1

    @property
    def cost_matrix(self) -> np.ndarray:
        """Weighted arc-cost matrix actually minimised by the search."""
        return self._cost_matrix

    @property
    def has_time_windows(self) -> bool:
        return self.time_windows is not None

    @property
    def min_vehicles(self) -> int:
        """Bin-packing lower bound on the number of vehicles."""
        return int(np.ceil(self.demand.sum() / self.capacity - 1e-9))

    # ------------------------------------------------------------- evaluation
    def evaluate(self, routes) -> SolutionStats:
        """Compute the full cost breakdown of ``routes``.

        Pure NumPy/Python: the hot path used inside the metaheuristics is the
        compiled version in :mod:`qroute.algorithms.kernels`; this one is the
        readable reference implementation that the tests check against.
        """
        d = self.distance
        t = self.duration
        dist = dur = cap_v = tw_v = dur_v = 0.0
        cong = 0.0
        used = 0
        for route in routes:
            if not route:
                continue
            used += 1
            load = 0.0
            prev = 0
            elapsed = 0.0
            rd = 0.0
            for c in route:
                dist += d[prev, c]
                rd += t[prev, c]
                if self.congestion is not None:
                    cong += self.congestion[prev, c] * t[prev, c]
                elapsed += t[prev, c]
                if self.time_windows is not None:
                    early, late = self.time_windows[c]
                    if elapsed < early:            # wait until the window opens
                        elapsed = early
                    if elapsed > late:             # late arrival is a violation
                        tw_v += elapsed - late
                if self.service_time is not None:
                    elapsed += self.service_time[c]
                    rd += self.service_time[c]
                load += self.demand[c]
                prev = c
            dist += d[prev, 0]
            rd += t[prev, 0]
            elapsed += t[prev, 0]
            if self.congestion is not None:
                cong += self.congestion[prev, 0] * t[prev, 0]
            dur += rd
            if load > self.capacity:
                cap_v += load - self.capacity
            if self.time_windows is not None:
                depot_close = self.time_windows[0, 1]
                if elapsed > depot_close:
                    tw_v += elapsed - depot_close
            if self.max_route_duration is not None and rd > self.max_route_duration:
                dur_v += rd - self.max_route_duration

        fleet_v = 0
        if self.n_vehicles is not None and used > self.n_vehicles:
            fleet_v = used - self.n_vehicles
        return SolutionStats(
            distance=dist,
            duration=dur,
            congestion_delay=cong,
            capacity_violation=cap_v,
            time_window_violation=tw_v,
            duration_violation=dur_v,
            fleet_violation=fleet_v,
            edge_load_violation=0.0,
        )

    def objective(self, stats: SolutionStats, n_routes: int = 0) -> float:
        """Weighted objective value (excluding constraint penalties).

        This must agree with :attr:`cost_matrix`, which is what the search
        actually minimises. It previously did not: the congestion term was
        folded into the matrix but written here as ``+ w.congestion * 0.0``, so
        with a non-zero congestion weight a solution's reported cost was lower
        than the cost the optimiser had minimised. The discrepancy was hidden
        only because every construction site pinned the weight to zero.
        """
        w = self.weights
        val = w.distance * stats.distance + w.time * stats.duration
        if w.congestion:
            val += w.congestion * stats.congestion_delay
        val += w.vehicles * n_routes
        return val

    def make_solution(self, routes) -> Solution:
        """Build a :class:`Solution` with cost and statistics filled in."""
        clean = [list(r) for r in routes if len(r) > 0]
        stats = self.evaluate(clean)
        sol = Solution(clean, 0.0, stats)
        sol.cost = self.objective(stats, sol.n_routes)
        return sol

    def __repr__(self) -> str:  # pragma: no cover
        kind = "VRPTW" if self.has_time_windows else "CVRP"
        return f"Instance({self.name!r}, {kind}, n={self.n_customers}, Q={self.capacity:g}, K={self.n_vehicles})"
