"""Optimiser interface, run configuration and convergence recording."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from qroute.core.rng import make_rng
from qroute.core.types import Solution
from qroute.problems.instance import Instance


@dataclass
class StopCriteria:
    """When to stop searching.

    A run stops at whichever limit is hit first. Benchmarks that compare
    algorithms fairly use ``max_seconds`` (equal wall-clock budget) rather than
    ``max_iterations``, because one iteration means something different for a
    swarm of 30 than for a population of 100.
    """

    max_iterations: int = 500
    max_seconds: float = float("inf")
    max_evaluations: int = 0            # 0 = unlimited
    target_cost: float = -float("inf")  # stop early once reached
    stall_iterations: int = 0           # 0 = never stop on stagnation


@dataclass
class IterationRecord:
    """One row of the convergence history."""

    iteration: int
    elapsed: float
    evaluations: int
    best_cost: float
    mean_cost: float
    diversity: float
    feasible: bool


@dataclass
class OptimizationResult:
    """Everything a single run produced.

    ``history`` is what the convergence analysis required by the problem
    statement is computed from, and what the live UI streams.
    """

    algorithm: str
    instance: str
    best: Solution
    history: list[IterationRecord] = field(default_factory=list)
    iterations: int = 0
    evaluations: int = 0
    seconds: float = 0.0
    seed: Optional[int] = None
    params: dict = field(default_factory=dict)

    @property
    def best_cost(self) -> float:
        return self.best.cost

    def gap_to(self, reference: float) -> float:
        """Percentage gap above a reference cost (typically the best known)."""
        if reference <= 0:
            return float("nan")
        return 100.0 * (self.best.cost - reference) / reference

    def convergence_curve(self) -> tuple[np.ndarray, np.ndarray]:
        it = np.array([h.iteration for h in self.history], dtype=float)
        bc = np.array([h.best_cost for h in self.history], dtype=float)
        return it, bc

    def iterations_to_within(self, reference: float, pct: float) -> Optional[int]:
        """First iteration whose best cost is within ``pct``% of ``reference``.

        This is the honest way to state "converges faster": it measures when a
        run reached a given quality, not merely how its curve looks.
        """
        threshold = reference * (1.0 + pct / 100.0)
        for h in self.history:
            if h.best_cost <= threshold:
                return h.iteration
        return None

    def time_to_within(self, reference: float, pct: float) -> Optional[float]:
        threshold = reference * (1.0 + pct / 100.0)
        for h in self.history:
            if h.best_cost <= threshold:
                return h.elapsed
        return None

    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "instance": self.instance,
            "best_cost": self.best.cost,
            "routes": self.best.routes,
            "n_routes": self.best.n_routes,
            "feasible": self.best.is_feasible,
            "stats": self.best.stats.as_dict(),
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "seconds": self.seconds,
            "seed": self.seed,
            "params": self.params,
            "history": [
                {
                    "iteration": h.iteration,
                    "elapsed": round(h.elapsed, 4),
                    "evaluations": h.evaluations,
                    "best_cost": h.best_cost,
                    "mean_cost": h.mean_cost,
                    "diversity": h.diversity,
                }
                for h in self.history
            ],
        }


ProgressCallback = Callable[[IterationRecord], None]


class Optimizer(ABC):
    """Base class for every solver in the platform.

    Subclasses implement :meth:`_run`. The base class owns the clock, the
    evaluation counter, the convergence log and the stopping rules, so that
    every algorithm is measured the same way and the benchmark comparison is
    apples to apples.
    """

    name: str = "optimizer"

    def __init__(self, instance: Instance, stop: StopCriteria | None = None,
                 seed: int | None = None, callback: ProgressCallback | None = None,
                 **params):
        self.instance = instance
        self.stop = stop or StopCriteria()
        self.seed = seed
        self.rng = make_rng(seed)
        self.callback = callback
        self.params = params
        self.history: list[IterationRecord] = []
        self.evaluations = 0
        self._t0 = 0.0
        self._best: Solution = Solution()
        self._stall = 0

    # ------------------------------------------------------------- lifecycle
    def solve(self) -> OptimizationResult:
        self._t0 = time.perf_counter()
        self.history = []
        self.evaluations = 0
        self._stall = 0
        self._best = Solution()
        iterations = self._run()
        elapsed = time.perf_counter() - self._t0
        best = self._best
        if best.routes:
            # Re-score with the reference evaluator so the reported cost never
            # depends on the compiled kernel or on penalty weights.
            best = self.instance.make_solution(best.routes)
            best.validate(self.instance.n_customers)
        return OptimizationResult(
            algorithm=self.name,
            instance=self.instance.name,
            best=best,
            history=self.history,
            iterations=iterations,
            evaluations=self.evaluations,
            seconds=elapsed,
            seed=self.seed,
            params=self.describe(),
        )

    @abstractmethod
    def _run(self) -> int:
        """Execute the search, returning the number of iterations performed."""

    def describe(self) -> dict:
        return dict(self.params)

    # -------------------------------------------------------------- plumbing
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def should_stop(self, iteration: int) -> bool:
        s = self.stop
        if iteration >= s.max_iterations:
            return True
        if self.elapsed >= s.max_seconds:
            return True
        if s.max_evaluations and self.evaluations >= s.max_evaluations:
            return True
        if self._best.cost <= s.target_cost:
            return True
        if s.stall_iterations and self._stall >= s.stall_iterations:
            return True
        return False

    def record(self, iteration: int, best_cost: float, mean_cost: float,
               diversity: float, feasible: bool = True) -> None:
        rec = IterationRecord(
            iteration=iteration,
            elapsed=self.elapsed,
            evaluations=self.evaluations,
            best_cost=best_cost,
            mean_cost=mean_cost,
            diversity=diversity,
            feasible=feasible,
        )
        self.history.append(rec)
        if self.callback is not None:
            self.callback(rec)

    def offer(self, solution: Solution) -> bool:
        """Update the incumbent if ``solution`` is better. Returns True if it was."""
        if solution.cost < self._best.cost - 1e-10:
            self._best = solution.copy()
            self._stall = 0
            return True
        self._stall += 1
        return False
