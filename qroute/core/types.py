"""Core solution data structures shared by every optimiser in the platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# A route is the ordered list of customer indices served between two depot visits.
# The depot (index 0) is implicit: the vehicle starts and ends there.
Route = list[int]


@dataclass(frozen=True)
class SolutionStats:
    """Cost breakdown of a solution.

    All quantities are in the instance's native units. ``distance`` is in
    distance units, ``duration`` in time units (seconds for road networks),
    and the violation terms are raw amounts of infeasibility, *not* penalties.
    """

    distance: float = 0.0
    duration: float = 0.0
    #: Vehicle-seconds spent in congestion: the congestion level of each arc
    #: weighted by the time spent on it. Zero unless the instance carries a
    #: congestion matrix, which only road-network instances do.
    congestion_delay: float = 0.0
    capacity_violation: float = 0.0
    time_window_violation: float = 0.0
    duration_violation: float = 0.0
    fleet_violation: int = 0
    #: Reserved for the edge-capacity extension of the flow constraints, which
    #: is formulated in the documentation but NOT implemented: nothing computes
    #: this and it is always zero. It is kept as a named field so the gap is
    #: visible rather than silently absent.
    edge_load_violation: float = 0.0

    @property
    def total_violation(self) -> float:
        return (
            self.capacity_violation
            + self.time_window_violation
            + self.duration_violation
            + float(self.fleet_violation)
            + self.edge_load_violation
        )

    @property
    def is_feasible(self) -> bool:
        return self.total_violation <= 1e-9

    def as_dict(self) -> dict[str, float]:
        return {
            "distance": self.distance,
            "duration": self.duration,
            "congestion_delay": self.congestion_delay,
            "capacity_violation": self.capacity_violation,
            "time_window_violation": self.time_window_violation,
            "duration_violation": self.duration_violation,
            "fleet_violation": float(self.fleet_violation),
            "edge_load_violation": self.edge_load_violation,
            "total_violation": self.total_violation,
            "feasible": float(self.is_feasible),
        }


@dataclass
class Solution:
    """A set of vehicle routes together with its evaluated cost.

    ``cost`` is the penalised objective actually minimised by the search.
    ``stats`` carries the interpretable breakdown used for reporting, so a
    solution can be reported honestly as feasible or infeasible regardless of
    how the penalty weights were tuned.
    """

    routes: list[Route] = field(default_factory=list)
    cost: float = float("inf")
    stats: SolutionStats = field(default_factory=SolutionStats)

    def copy(self) -> "Solution":
        return Solution([list(r) for r in self.routes], self.cost, self.stats)

    @property
    def n_routes(self) -> int:
        return sum(1 for r in self.routes if r)

    @property
    def is_feasible(self) -> bool:
        return self.stats.is_feasible

    def customers(self) -> list[int]:
        return [c for r in self.routes for c in r]

    def giant_tour(self) -> list[int]:
        """Flatten to a single permutation of customers (route delimiters dropped)."""
        return self.customers()

    def validate(self, n_customers: int) -> None:
        """Raise ``ValueError`` unless every customer 1..n is visited exactly once.

        This is deliberately strict: it is called in tests and after every
        optimiser run so a subtly broken operator cannot silently produce a
        cheap-looking but invalid answer.
        """
        seen = self.customers()
        expected = set(range(1, n_customers + 1))
        got = set(seen)
        if len(seen) != len(got):
            dupes = sorted({c for c in seen if seen.count(c) > 1})
            raise ValueError(f"solution visits customers more than once: {dupes}")
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(f"invalid customer set (missing={missing}, extra={extra})")

    def to_json(self) -> dict:
        return {
            "routes": [list(r) for r in self.routes],
            "cost": self.cost,
            "n_routes": self.n_routes,
            "feasible": self.is_feasible,
            "stats": self.stats.as_dict(),
        }

    @staticmethod
    def from_routes(routes: Iterable[Sequence[int]]) -> "Solution":
        return Solution([list(r) for r in routes if len(r) > 0])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        flag = "feasible" if self.is_feasible else "INFEASIBLE"
        return f"Solution(cost={self.cost:.2f}, routes={self.n_routes}, {flag})"


def routes_to_array(routes: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten routes into a contiguous array plus start offsets.

    The flat representation is what the numba kernels consume: ``flat`` holds
    the customers of every route back to back and ``starts`` has ``len(routes)+1``
    entries delimiting them.
    """
    starts = np.zeros(len(routes) + 1, dtype=np.int32)
    for i, r in enumerate(routes):
        starts[i + 1] = starts[i] + len(r)
    flat = np.zeros(int(starts[-1]), dtype=np.int32)
    for i, r in enumerate(routes):
        flat[starts[i] : starts[i + 1]] = np.asarray(r, dtype=np.int32)
    return flat, starts


def array_to_routes(flat: np.ndarray, starts: np.ndarray) -> list[Route]:
    """Inverse of :func:`routes_to_array`, dropping empty routes."""
    out: list[Route] = []
    for i in range(len(starts) - 1):
        seg = flat[starts[i] : starts[i + 1]]
        if seg.size:
            out.append([int(x) for x in seg])
    return out
