"""PyVRP hybrid genetic search wrapped as the state-of-the-art reference.

PyVRP is the open-source descendant of Vidal's HGS-CVRP, the algorithm family
that holds most of the best-known solutions on the CVRPLIB X set. It is here as
the *ceiling*, not as a competitor we expect to beat. Being honest about that
is the point: a benchmark whose strongest baseline is a weak one proves
nothing. The quantum-inspired optimiser is interesting if it is competitive
with OR-Tools and lands close to PyVRP, not if it wins a race against
straw men.

Integer costs
-------------
PyVRP is an integer solver, so the same exact scaling used everywhere else in
the project applies (factor 1 for CVRPLIB, 10 for Solomon). As with every other
wrapper the routes are re-scored with :meth:`Instance.make_solution`, so the
number reported here is produced by the project's own evaluator and is directly
comparable with the rest of the benchmark.

Availability
------------
PyVRP ships binary wheels and can fail to install on a new Python release
before its wheels catch up. The import is therefore deferred and guarded: if
PyVRP is unavailable, :func:`available` returns ``False`` and the solver raises
a clear :class:`PyVRPUnavailable` explaining how to install it. It never
silently falls back to another algorithm, because a benchmark row labelled
"PyVRP" that was produced by something else is worse than a missing row.

Measured on this machine: PyVRP 0.14.0 installs and runs cleanly on
CPython 3.13.7 (macOS, arm64), so the fallback path is defensive rather than
routinely exercised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qroute.algorithms.base import IterationRecord, OptimizationResult
from qroute.core.types import Solution
from qroute.exact.scaling import integer_demands, integer_scaling
from qroute.problems.instance import Instance

__all__ = ["PyVRPUnavailable", "available", "version", "PyVRPResult", "solve_pyvrp",
           "solve_pyvrp_result"]


class PyVRPUnavailable(RuntimeError):
    """Raised when PyVRP is not importable in this environment."""


def _import_pyvrp():
    try:
        import pyvrp  # noqa: F401
        from pyvrp import Model
        from pyvrp.stop import MaxIterations, MaxRuntime

        return pyvrp, Model, MaxRuntime, MaxIterations
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise PyVRPUnavailable(
            "PyVRP is not available in this environment "
            f"({type(exc).__name__}: {exc}). Install it with `pip install pyvrp`. "
            "No substitute solver is used in its place."
        ) from exc


def available() -> bool:
    """Whether PyVRP can be imported here."""
    try:
        _import_pyvrp()
        return True
    except PyVRPUnavailable:
        return False


def version() -> Optional[str]:
    """Installed PyVRP version, or ``None`` when unavailable."""
    try:
        from importlib.metadata import version as _v

        return _v("pyvrp")
    except Exception:
        return None


@dataclass
class PyVRPResult:
    """Routes and the anytime curve of one PyVRP run."""

    routes: list[list[int]]
    cost: float
    seconds: float
    feasible: bool
    iterations: int = 0
    n_vehicles: int = 0
    curve: list[tuple[float, float]] = field(default_factory=list)
    scale_factor: int = 1
    instance_name: str = ""

    def to_optimization_result(self, instance: Instance, params: dict | None = None) -> OptimizationResult:
        if self.routes:
            best = instance.make_solution(self.routes)
            best.validate(instance.n_customers)
        else:
            best = Solution()
        history = [
            IterationRecord(
                iteration=k,
                elapsed=t,
                evaluations=0,
                best_cost=cost,
                mean_cost=cost,
                diversity=0.0,
                feasible=True,
            )
            for k, (t, cost) in enumerate(self.curve)
        ]
        merged = {
            "solver": "pyvrp-hgs",
            "pyvrp_version": version(),
            "scale_factor": self.scale_factor,
            "pyvrp_feasible": self.feasible,
            "n_routes": self.n_vehicles,
        }
        merged.update(params or {})
        return OptimizationResult(
            algorithm="pyvrp-hgs",
            instance=instance.name,
            best=best,
            history=history,
            iterations=self.iterations,
            evaluations=0,
            seconds=self.seconds,
            seed=params.get("seed") if params else None,
            params=merged,
        )


def solve_pyvrp_result(
    instance: Instance,
    seconds: float = 10.0,
    seed: int = 0,
    max_iterations: int | None = None,
    n_vehicles: int | None = None,
    vehicle_slack: int = 2,
    display: bool = False,
) -> PyVRPResult:
    """Run PyVRP's hybrid genetic search on ``instance``.

    Parameters
    ----------
    seconds:
        Wall-clock budget, passed as ``MaxRuntime``. Ignored when
        ``max_iterations`` is given, which is the reproducible alternative:
        PyVRP is deterministic in the seed for a fixed iteration count, but a
        time limit makes the iteration count machine dependent.
    seed:
        PyVRP's own RNG seed; runs with the same seed and iteration limit
        reproduce exactly.
    """
    _pyvrp, Model, MaxRuntime, MaxIterations = _import_pyvrp()

    n = instance.size
    scaling = integer_scaling(instance.cost_matrix, instance.duration, instance.time_windows,
                              instance.service_time)
    cost_i = scaling.to_int(instance.cost_matrix)
    dur_i = scaling.to_int(instance.duration)
    demand, capacity, _total = integer_demands(instance.demand, instance.capacity)

    if n_vehicles is not None:
        fleet = int(n_vehicles)
    elif instance.n_vehicles is not None:
        fleet = int(instance.n_vehicles)
    else:
        fleet = instance.min_vehicles + max(0, int(vehicle_slack))
    fleet = max(1, min(fleet, instance.n_customers))

    has_tw = instance.has_time_windows
    if has_tw:
        tw_i = scaling.to_int(instance.time_windows)
        service_i = scaling.to_int(
            instance.service_time if instance.service_time is not None else np.zeros(n)
        )
    coords = instance.coords if instance.coords is not None else np.zeros((n, 2))

    model = Model()
    # Coordinates are only used by PyVRP for bookkeeping and plotting; all
    # travel costs come from the explicit edges added below, so a coordinate-
    # free instance (a road-network matrix, say) still works.
    locations = [model.add_location(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
    depot = model.add_depot(
        location=locations[0],
        tw_early=int(tw_i[0, 0]) if has_tw else 0,
        tw_late=int(tw_i[0, 1]) if has_tw else 2**40,
    )
    for i in range(1, n):
        model.add_client(
            location=locations[i],
            delivery=[int(demand[i])],
            service_duration=int(service_i[i]) if has_tw else 0,
            tw_early=int(tw_i[i, 0]) if has_tw else 0,
            tw_late=int(tw_i[i, 1]) if has_tw else 2**40,
        )
    model.add_vehicle_type(
        num_available=fleet,
        capacity=[capacity],
        start_depot=depot,
        end_depot=depot,
        tw_early=int(tw_i[0, 0]) if has_tw else 0,
        tw_late=int(tw_i[0, 1]) if has_tw else 2**40,
        max_distance=2**40,
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            model.add_edge(locations[i], locations[j], distance=int(cost_i[i, j]),
                           duration=int(dur_i[i, j]) if has_tw else 0)

    stop = MaxIterations(int(max_iterations)) if max_iterations is not None else MaxRuntime(float(seconds))

    t0 = time.perf_counter()
    result = model.solve(stop=stop, seed=int(seed), display=bool(display), collect_stats=True)
    elapsed = time.perf_counter() - t0

    # A route iterates over scheduled activities including its depot visits.
    # `activity.idx` is PyVRP's *client* index, and clients were added in
    # instance order 1..n-1, so the instance node is `idx + 1`.
    routes = [
        [int(act.idx) + 1 for act in route if act.is_client()]
        for route in result.best.routes()
    ]
    routes = [r for r in routes if r]

    # PyVRP's statistics hold a per-iteration best-cost series and the duration
    # of each iteration; converting to the project's (elapsed, cost) curve just
    # needs a cumulative sum. The layout differs between PyVRP versions, so a
    # failure here costs the curve but not the result.
    curve: list[tuple[float, float]] = []
    try:
        stats = result.stats
        runtimes = np.cumsum(np.asarray(stats.runtimes, dtype=float))
        for t, datum in zip(runtimes, stats.data):
            best = datum.best_cost
            if best is not None and np.isfinite(best):
                curve.append((float(t), scaling.to_float(best)))
    except Exception:  # pragma: no cover - statistics layout is version dependent
        curve = []

    return PyVRPResult(
        routes=routes,
        cost=scaling.to_float(result.cost()),
        seconds=elapsed,
        feasible=bool(result.is_feasible()),
        iterations=int(result.num_iterations),
        n_vehicles=len(routes),
        curve=curve,
        scale_factor=scaling.factor,
        instance_name=instance.name,
    )


def solve_pyvrp(instance: Instance, **kwargs) -> OptimizationResult:
    """Benchmark-runner entry point: PyVRP as an :class:`OptimizationResult`."""
    raw = solve_pyvrp_result(instance, **kwargs)
    return raw.to_optimization_result(instance, params=dict(kwargs))
