"""OR-Tools routing solver wrapped as a project baseline.

Why this baseline
-----------------
OR-Tools' routing library with guided local search is the solver most teams
reach for first, and it is what a sceptical judge will ask about. Comparing the
quantum-inspired optimiser against a strong, widely used classical heuristic on
an equal wall-clock budget is the only comparison that means anything. This
module makes that comparison mechanical: same instance object, same evaluation
function, same :class:`~qroute.algorithms.base.OptimizationResult`.

Integer costs
-------------
The routing library works in integers. Costs are multiplied by the factor from
:mod:`qroute.exact.scaling` -- 1 for CVRPLIB, whose distances are already
integral, and 10 for Solomon, whose distances are truncated to one decimal --
so the transformation is exact and no arc cost is distorted. The scaled value
OR-Tools reports is nonetheless *not* what this module returns: the extracted
routes are re-scored with :meth:`Instance.make_solution`, exactly like every
other solver in the platform, so a scaling mistake would show up as a
disagreement rather than as a quietly wrong number.

Determinism
-----------
The routing library has no random seed to set; its search is deterministic
given identical parameters. Under a *wall-clock* limit, however, the amount of
search completed depends on machine load, so repeated runs can differ. That is
a property of time-limited search, not a bug, and it is why the benchmark
reports several runs rather than one. Set ``solution_limit`` instead of
``seconds`` when a bit-for-bit reproducible run is needed.

Interface choice
----------------
As with the exact solvers this is a function rather than an ``Optimizer``
subclass: OR-Tools owns its search loop and exposes only an at-solution
callback, so there is no per-iteration hook for the base class to drive. The
callback is used to build a genuine convergence curve, so the anytime plots
include this baseline on the same axes as the metaheuristics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from qroute.algorithms.base import IterationRecord, OptimizationResult
from qroute.core.types import Solution
from qroute.exact.scaling import integer_demands, integer_scaling
from qroute.problems.instance import Instance

__all__ = ["ORToolsResult", "solve_ortools", "solve_ortools_result"]

_FIRST_SOLUTION = {
    "path_cheapest_arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "savings": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    "parallel_cheapest_insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    "christofides": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    "automatic": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
}

_METAHEURISTIC = {
    "guided_local_search": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "tabu_search": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    "simulated_annealing": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    "greedy_descent": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    "automatic": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
}


@dataclass
class ORToolsResult:
    """Routes plus the anytime curve recorded during the search."""

    routes: list[list[int]]
    cost: float
    status: str
    seconds: float
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
        merged = {"solver": "ortools-routing", "status": self.status,
                  "scale_factor": self.scale_factor, "n_routes": self.n_vehicles}
        merged.update(params or {})
        return OptimizationResult(
            algorithm="ortools-gls",
            instance=instance.name,
            best=best,
            history=history,
            iterations=len(history),
            evaluations=0,
            seconds=self.seconds,
            seed=None,
            params=merged,
        )


def solve_ortools_result(
    instance: Instance,
    seconds: float = 10.0,
    first_solution: str = "path_cheapest_arc",
    metaheuristic: str = "guided_local_search",
    n_vehicles: int | None = None,
    vehicle_slack: int = 2,
    solution_limit: int | None = None,
    log: bool = False,
) -> ORToolsResult:
    """Run the OR-Tools routing solver on ``instance``.

    Parameters
    ----------
    seconds:
        Wall-clock budget handed to the search.
    n_vehicles:
        Fleet size for the model. ``None`` derives one from the instance: its
        own fleet limit if set, else the bin-packing minimum plus
        ``vehicle_slack``. Unused vehicles stay at the depot and cost nothing,
        so a slightly generous fleet does not distort the objective; too small
        a fleet makes the model infeasible.
    solution_limit:
        Stop after this many improving solutions instead of on the clock. Use
        it when a reproducible run matters more than a fixed budget.
    """
    n = instance.size
    scaling = integer_scaling(instance.cost_matrix, instance.duration, instance.time_windows,
                              instance.service_time)
    cost_i = scaling.to_int(instance.cost_matrix)
    demand, capacity, _total = integer_demands(instance.demand, instance.capacity)

    if n_vehicles is not None:
        fleet = int(n_vehicles)
    elif instance.n_vehicles is not None:
        fleet = int(instance.n_vehicles)
    else:
        fleet = instance.min_vehicles + max(0, int(vehicle_slack))
    fleet = max(1, min(fleet, instance.n_customers))

    manager = pywrapcp.RoutingIndexManager(n, fleet, 0)
    routing = pywrapcp.RoutingModel(manager)

    def arc_cost(from_index: int, to_index: int) -> int:
        return int(cost_i[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # ------------------------------------------------------------- capacity
    def demand_cb(from_index: int) -> int:
        return int(demand[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                       # no slack: load cannot be dropped
        [capacity] * fleet,
        True,                    # start cumul at zero
        "Capacity",
    )

    # --------------------------------------------------------- time windows
    if instance.has_time_windows:
        travel_i = scaling.to_int(instance.duration)
        service_i = scaling.to_int(
            instance.service_time if instance.service_time is not None else np.zeros(n)
        )
        tw_i = scaling.to_int(instance.time_windows)
        horizon = int(tw_i[0, 1])

        def time_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            # Service happens at the node we are leaving, the usual convention.
            return int(travel_i[i, j] + service_i[i])

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_idx, horizon, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for node in range(1, n):
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(int(tw_i[node, 0]), int(tw_i[node, 1]))
        for v in range(fleet):
            start, end = routing.Start(v), routing.End(v)
            time_dim.CumulVar(start).SetRange(int(tw_i[0, 0]), horizon)
            time_dim.CumulVar(end).SetRange(int(tw_i[0, 0]), horizon)
            # Finalising start and end cumuls lets the solver shrink waiting time.
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(start))
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(end))
    elif instance.max_route_duration is not None:
        dur_i = scaling.to_int(instance.duration)
        limit = int(round(instance.max_route_duration * scaling.factor))

        def dur_cb(from_index: int, to_index: int) -> int:
            return int(dur_i[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

        dur_idx = routing.RegisterTransitCallback(dur_cb)
        routing.AddDimension(dur_idx, 0, limit, True, "Duration")

    # ----------------------------------------------------------- parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = _FIRST_SOLUTION[first_solution]
    params.local_search_metaheuristic = _METAHEURISTIC[metaheuristic]
    if solution_limit is not None:
        params.solution_limit = int(solution_limit)
    else:
        params.time_limit.FromMilliseconds(int(seconds * 1000))
    params.log_search = bool(log)

    curve: list[tuple[float, float]] = []
    t0 = time.perf_counter()

    def on_solution() -> None:
        # CostVar().Max() is the objective of the solution just accepted.
        curve.append((time.perf_counter() - t0, scaling.to_float(routing.CostVar().Max())))

    routing.AddAtSolutionCallback(on_solution)

    assignment = routing.SolveWithParameters(params)
    seconds_taken = time.perf_counter() - t0

    routes: list[list[int]] = []
    cost_value = float("inf")
    if assignment is not None:
        for v in range(fleet):
            index = routing.Start(v)
            route: list[int] = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    route.append(int(node))
                index = assignment.Value(routing.NextVar(index))
            if route:
                routes.append(route)
        cost_value = scaling.to_float(assignment.ObjectiveValue())

    status = routing_enums_pb2.RoutingSearchStatus.Value.Name(routing.status())

    return ORToolsResult(
        routes=routes,
        cost=cost_value,
        status=status,
        seconds=seconds_taken,
        n_vehicles=len(routes),
        curve=curve,
        scale_factor=scaling.factor,
        instance_name=instance.name,
    )


def solve_ortools(instance: Instance, **kwargs) -> OptimizationResult:
    """Benchmark-runner entry point: OR-Tools as an :class:`OptimizationResult`."""
    raw = solve_ortools_result(instance, **kwargs)
    return raw.to_optimization_result(instance, params={k: v for k, v in kwargs.items()})
