"""Exact CVRP and TSP models for OR-Tools CP-SAT.

Why this module exists
----------------------
Every claim of the form "the quantum-inspired optimiser is within x% of
optimal" needs an optimum to be within x% of. For the small benchmark
instances CP-SAT can *prove* optimality, and this module is the component that
produces those proofs. For instances it cannot close it still returns the best
dual bound it reached, which brackets the optimum from below and is therefore
still usable in a benchmark table.

Formulation
-----------
The model is an arc-based multi-circuit formulation. For every ordered pair
``(i, j)``, ``i != j``, there is a Boolean ``x[i, j]`` that is true when some
vehicle traverses that arc. ``AddMultipleCircuit`` constrains the selected arcs
to decompose into circuits that all pass through node 0, which is exactly the
structure of a set of vehicle routes out of a single depot. No self-loop
literals are supplied, so every node must lie on some circuit, which encodes
"visit every customer exactly once".

Capacity is imposed through a load variable per node::

    load[0] = 0
    x[i, j] = 1  and  j != 0   =>   load[j] = load[i] + demand[j]
    demand[j] <= load[j] <= Q

This is the natural CP-SAT analogue of the MTZ/Miller-Tucker-Zemlin lifting.
Note that ``AddMultipleCircuit`` already removes subtours, so these constraints
carry capacity information only; they are not doing subtour elimination duty,
which is why the usual complaint about MTZ (a weak linear relaxation) matters
much less here than it does in :mod:`qroute.exact.milp`. CP-SAT reasons about
the circuit constraint propagator directly rather than about its LP projection.

Time windows are supported by an optional arrival-time variable per node with
the standard "arrival[j] >= arrival[i] + service[i] + travel[i][j]" implication
on each selected arc. This is included for completeness; in practice CP-SAT
closes VRPTW instances of only a handful of customers, and the module reports
an unproven bound rather than pretending otherwise.

Interface choice
----------------
This is a plain function rather than an :class:`~qroute.algorithms.base.Optimizer`
subclass, because CP-SAT owns its own search loop and there is no meaningful
per-iteration hook to drive the base class's clock with. The richer
:class:`CPSATResult` carries the dual bound and the optimality proof, and
:meth:`CPSATResult.to_optimization_result` converts it to the project-wide
:class:`~qroute.algorithms.base.OptimizationResult` so the benchmark runner can
treat it exactly like any metaheuristic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from ortools.sat.python import cp_model

from qroute.algorithms.base import IterationRecord, OptimizationResult
from qroute.core.types import Solution
from qroute.exact.scaling import Scaling, integer_demands, integer_scaling
from qroute.problems.instance import Instance

__all__ = ["CPSATResult", "solve_cvrp_cpsat", "solve_tsp_cpsat", "solve_cpsat"]


@dataclass
class CPSATResult:
    """Outcome of one CP-SAT run.

    ``lower_bound`` is CP-SAT's best dual bound converted back to the
    instance's native units. It is a valid lower bound on the optimum whenever
    the integer scaling was exact, which is the case for both benchmark
    families we ship.
    """

    routes: list[list[int]]
    cost: float
    lower_bound: float
    status: str
    seconds: float
    proven_optimal: bool
    scaling: Scaling = field(default_factory=lambda: Scaling(1, True, 0.0))
    n_vehicles: int = 0
    curve: list[tuple[float, float, float]] = field(default_factory=list)
    instance_name: str = ""

    @property
    def gap(self) -> float:
        """Relative optimality gap in percent, or ``inf`` with no incumbent."""
        if not math.isfinite(self.cost) or self.cost <= 0:
            return float("inf")
        return 100.0 * (self.cost - self.lower_bound) / self.cost

    def as_tuple(self) -> tuple[list[list[int]], float, float, str, float]:
        """``(routes, cost, lower_bound, status, seconds)``."""
        return self.routes, self.cost, self.lower_bound, self.status, self.seconds

    def to_optimization_result(self, instance: Instance) -> OptimizationResult:
        """Re-score with the project evaluator and wrap for the benchmark runner."""
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
                best_cost=obj,
                mean_cost=obj,
                diversity=0.0,
                feasible=True,
            )
            for k, (t, obj, _bound) in enumerate(self.curve)
        ]
        return OptimizationResult(
            algorithm="cpsat",
            instance=instance.name,
            best=best,
            history=history,
            iterations=len(history),
            evaluations=0,
            seconds=self.seconds,
            seed=None,
            params={
                "solver": "ortools-cpsat",
                "status": self.status,
                "lower_bound": self.lower_bound,
                "proven_optimal": self.proven_optimal,
                "gap_percent": self.gap,
                "scale_factor": self.scaling.factor,
                "scaling_exact": self.scaling.exact,
                "n_routes": self.n_vehicles,
            },
        )


class _SolutionRecorder(cp_model.CpSolverSolutionCallback):
    """Records ``(elapsed, incumbent, dual bound)`` at every improving solution.

    Without this the exact solver would be the only method in the benchmark
    with no convergence curve, which would make the anytime-behaviour plots
    inconsistent.
    """

    def __init__(self, scaling: Scaling, t0: float) -> None:
        super().__init__()
        self._scaling = scaling
        self._t0 = t0
        self.curve: list[tuple[float, float, float]] = []

    def on_solution_callback(self) -> None:  # pragma: no cover - solver driven
        self.curve.append(
            (
                time.perf_counter() - self._t0,
                self._scaling.to_float(self.objective_value),
                self._scaling.to_float(self.best_objective_bound),
            )
        )


_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


def solve_cvrp_cpsat(
    instance: Instance,
    time_limit: float = 60.0,
    workers: int = 8,
    max_vehicles: int | None = None,
    min_vehicles: int | None = None,
    seed: int = 0,
    log: bool = False,
    hint: Sequence[Sequence[int]] | None = None,
    relative_gap: float = 0.0,
) -> CPSATResult:
    """Solve a CVRP (optionally with time windows) exactly with CP-SAT.

    Parameters
    ----------
    time_limit:
        Wall-clock budget in seconds. On expiry the best incumbent and the best
        dual bound found so far are returned with ``proven_optimal=False``.
    workers:
        Number of parallel search workers. Eight is the project default and
        matches the ten-core development machine without starving it.
    max_vehicles:
        Upper bound on the number of routes. ``None`` means the instance's own
        fleet limit if it has one, otherwise the number of customers, i.e. an
        unrestricted fleet. Do not lower this to the reference ``k`` value when
        you intend to claim a proof: capping the fleet at the best-known route
        count assumes part of the answer.
    min_vehicles:
        Lower bound on the number of routes. Defaults to the bin-packing bound,
        which is always valid.
    hint:
        Optional warm-start routes. CP-SAT uses them as a solution hint only,
        so a bad hint cannot make the result incorrect.
    relative_gap:
        Stop once the relative gap drops below this value. Leave at zero for a
        genuine optimality proof.
    """
    n = instance.size
    cost = instance.cost_matrix
    scaling = integer_scaling(cost, instance.duration if instance.has_time_windows else None)
    cost_i = scaling.to_int(cost)
    demand, capacity, total_demand = integer_demands(instance.demand, instance.capacity)

    k_min = int(math.ceil(total_demand / capacity - 1e-9))
    if min_vehicles is not None:
        k_min = max(k_min, int(min_vehicles))
    if max_vehicles is not None:
        k_max = int(max_vehicles)
    elif instance.n_vehicles is not None:
        k_max = int(instance.n_vehicles)
    else:
        k_max = instance.n_customers
    k_max = max(k_max, k_min)

    model = cp_model.CpModel()

    # ---------------------------------------------------------------- arcs
    lit: dict[tuple[int, int], cp_model.IntVar] = {}
    arcs: list[tuple[int, int, cp_model.IntVar]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = model.new_bool_var(f"x_{i}_{j}")
            lit[(i, j)] = v
            arcs.append((i, j, v))
    # No self-loop literals are added, so every node is forced onto a circuit.
    model.add_multiple_circuit(arcs)

    # ------------------------------------------------------------ capacity
    load = [model.new_int_var(0, capacity, f"load_{i}") for i in range(n)]
    model.add(load[0] == 0)
    for j in range(1, n):
        model.add(load[j] >= int(demand[j]))
    for (i, j), v in lit.items():
        if j == 0:
            continue
        # Depot start is covered by the same equation because load[0] == 0.
        model.add(load[j] == load[i] + int(demand[j])).only_enforce_if(v)

    # ------------------------------------------------------------- fleet
    depot_out = [lit[(0, j)] for j in range(1, n)]
    depot_in = [lit[(j, 0)] for j in range(1, n)]
    n_routes = model.new_int_var(k_min, k_max, "n_routes")
    model.add(sum(depot_out) == n_routes)
    model.add(sum(depot_in) == n_routes)
    # Redundant but helpful: a solution with r routes over n customers uses
    # exactly n + r arcs. Stating it explicitly gives the propagators a global
    # counting argument they would otherwise have to rediscover.
    model.add(sum(lit.values()) == instance.n_customers + n_routes)

    # ------------------------------------------------------- time windows
    if instance.has_time_windows:
        tw = instance.time_windows
        service = instance.service_time if instance.service_time is not None else np.zeros(n)
        travel = scaling.to_int(instance.duration)
        serv_i = scaling.to_int(service)
        early = scaling.to_int(tw[:, 0])
        late = scaling.to_int(tw[:, 1])
        horizon = int(late[0])
        arrival = [
            model.new_int_var(int(early[i]), int(late[i]) if i else horizon, f"t_{i}")
            for i in range(n)
        ]
        model.add(arrival[0] == int(early[0]))
        ret = [model.new_int_var(int(early[0]), horizon, f"ret_{i}") for i in range(n)]
        for (i, j), v in lit.items():
            if j == 0:
                # Returning to the depot must happen before the depot closes.
                model.add(ret[i] == arrival[i] + int(serv_i[i]) + int(travel[i, 0])).only_enforce_if(v)
                model.add(ret[i] <= horizon).only_enforce_if(v)
            else:
                model.add(
                    arrival[j] >= arrival[i] + int(serv_i[i]) + int(travel[i, j])
                ).only_enforce_if(v)

    # --------------------------------------------------------- objective
    model.minimize(sum(int(cost_i[i, j]) * v for (i, j), v in lit.items()))

    if hint:
        for route in hint:
            prev = 0
            for c in route:
                if (prev, c) in lit:
                    model.add_hint(lit[(prev, c)], 1)
                prev = c
            if (prev, 0) in lit:
                model.add_hint(lit[(prev, 0)], 1)

    # ------------------------------------------------------------- solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(log)
    if relative_gap > 0:
        solver.parameters.relative_gap_limit = float(relative_gap)

    t0 = time.perf_counter()
    recorder = _SolutionRecorder(scaling, t0)
    status = solver.solve(model, recorder)
    seconds = time.perf_counter() - t0

    status_name = _STATUS_NAMES.get(status, str(status))
    routes: list[list[int]] = []
    cost_value = float("inf")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        routes = _extract_routes(solver, lit, n)
        cost_value = scaling.to_float(solver.objective_value)
    bound = scaling.to_float(solver.best_objective_bound) if status != cp_model.INFEASIBLE else float("inf")
    proven = status == cp_model.OPTIMAL and scaling.exact and relative_gap == 0.0

    return CPSATResult(
        routes=routes,
        cost=cost_value,
        lower_bound=bound,
        status=status_name,
        seconds=seconds,
        proven_optimal=proven,
        scaling=scaling,
        n_vehicles=len(routes),
        curve=recorder.curve,
        instance_name=instance.name,
    )


def _extract_routes(solver: cp_model.CpSolver, lit: dict, n: int) -> list[list[int]]:
    """Walk the selected arcs from the depot into one list per circuit."""
    succ: dict[int, int] = {}
    starts: list[int] = []
    for (i, j), v in lit.items():
        if solver.boolean_value(v):
            if i == 0:
                starts.append(j)
            else:
                succ[i] = j
    routes: list[list[int]] = []
    for s in sorted(starts):
        route = []
        node = s
        guard = 0
        while node != 0:
            route.append(int(node))
            node = succ.get(node, 0)
            guard += 1
            if guard > n:  # pragma: no cover - would mean a broken model
                raise RuntimeError("cycle detected while extracting CP-SAT routes")
        routes.append(route)
    return routes


def solve_tsp_cpsat(
    cost: np.ndarray,
    time_limit: float = 60.0,
    workers: int = 8,
    seed: int = 0,
    log: bool = False,
) -> CPSATResult:
    """Exact TSP over an arbitrary (possibly asymmetric) cost matrix.

    Uses ``AddCircuit``, the single-circuit specialisation of the constraint
    above. Node 0 is the start and end of the tour. The returned "routes" list
    holds one route, the tour without the depot, so the result can be handed to
    the same evaluation code as the CVRP models.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n = cost.shape[0]
    if cost.shape != (n, n):
        raise ValueError("cost matrix must be square")
    scaling = integer_scaling(cost)
    cost_i = scaling.to_int(cost)

    model = cp_model.CpModel()
    lit = {}
    arcs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = model.new_bool_var(f"x_{i}_{j}")
            lit[(i, j)] = v
            arcs.append((i, j, v))
    model.add_circuit(arcs)
    model.minimize(sum(int(cost_i[i, j]) * v for (i, j), v in lit.items()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(log)

    t0 = time.perf_counter()
    recorder = _SolutionRecorder(scaling, t0)
    status = solver.solve(model, recorder)
    seconds = time.perf_counter() - t0

    tour: list[int] = []
    value = float("inf")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        succ = {i: j for (i, j), v in lit.items() if solver.boolean_value(v)}
        node = succ[0]
        while node != 0:
            tour.append(int(node))
            node = succ[node]
        value = scaling.to_float(solver.objective_value)

    return CPSATResult(
        routes=[tour] if tour else [],
        cost=value,
        lower_bound=scaling.to_float(solver.best_objective_bound),
        status=_STATUS_NAMES.get(status, str(status)),
        seconds=seconds,
        proven_optimal=status == cp_model.OPTIMAL and scaling.exact,
        scaling=scaling,
        n_vehicles=1 if tour else 0,
        curve=recorder.curve,
        instance_name="tsp",
    )


def solve_cpsat(instance: Instance, **kwargs) -> OptimizationResult:
    """Benchmark-runner entry point: CP-SAT as an :class:`OptimizationResult`."""
    return solve_cvrp_cpsat(instance, **kwargs).to_optimization_result(instance)
