"""Two-index mixed-integer programmes for the CVRP, solved through pywraplp.

This module exists for two reasons. First, it is the textbook statement of the
problem: the SIH deliverable asks for a mathematical formulation, and a working
MILP is that formulation in executable form rather than a picture in a slide.
Second, on tiny instances it independently reproduces the CP-SAT optimum, which
is a genuine cross-check -- two different solvers on two different models
agreeing on the same number is much stronger evidence than one solver's word.

It is *not* the workhorse. A compact two-index MILP is far weaker than a
branch-and-cut code with rounded-capacity cuts, and it is much weaker than
CP-SAT's circuit propagator on these instances. Expect it to close instances of
roughly fifteen to twenty customers and to stall well before A-n32-k5.

The two formulations
--------------------
Both share the degree constraints on binary arc variables ``x[i, j]``::

    sum_j x[i, j] = 1                     for every customer i
    sum_i x[i, j] = 1                     for every customer j
    sum_j x[0, j] = sum_j x[j, 0] = K     K routes leave and return

and differ only in how they forbid subtours and enforce capacity.

**MTZ** (Miller-Tucker-Zemlin, in the Kara/Christofides capacity form) adds a
continuous "load so far" variable ``u[i]`` per customer with
``d_i <= u_i <= Q`` and the big-M implication

    x[i, j] = 1  =>  u_j >= u_i + d_j        i.e.   u_i - u_j + Q*x[i,j] <= Q - d_j

**SCF** (single-commodity flow, Gavish-Graves) adds a continuous flow ``f[i, j]``
carrying the goods still on board while traversing the arc::

    sum_i f[i, j] - sum_k f[j, k] = d_j                      for every customer j
    d_j * x[i, j] <= f[i, j] <= (Q - d_i) * x[i, j]          arc capacity linking
    f[j, 0] = 0                                             vehicles return empty

Why MTZ is weaker
-----------------
The MTZ constraint is only active when ``x[i, j]`` is near one. As soon as the
relaxation sets ``x[i, j] = 0.5`` the big-M term contributes ``Q/2`` of slack
and the constraint stops saying anything about ``u``. Its linear relaxation is
therefore close to the pure assignment relaxation and gives a bound that is
often 20-40% below the optimum, which is useless for pruning: branch and bound
then has to enumerate its way to the answer.

The flow formulation has no big-M in that sense. Every unit of demand must be
routed from the depot along selected arcs, so a fractional ``x`` immediately
forces fractional flow to be paid for on arcs leaving the depot, and the
relaxation "sees" the capacity structure. It is a classical result that the
projection of the SCF polytope onto the ``x`` variables is contained in the MTZ
one (Gouveia, 1995), that is, the SCF linear bound dominates the MTZ bound for
every instance -- never worse, usually much better. The price is ``O(n^2)``
extra continuous variables instead of ``O(n)``, which on the sizes where either
formulation is usable is irrelevant.

Neither compact formulation approaches the strength of the exponential
two-index model with rounded-capacity inequalities separated on the fly; that
is what a real branch-and-cut CVRP code does, and it is out of scope here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

from ortools.linear_solver import pywraplp

from qroute.algorithms.base import OptimizationResult
from qroute.core.types import Solution
from qroute.exact.scaling import integer_demands
from qroute.problems.instance import Instance

__all__ = [
    "MILPResult",
    "available_solvers",
    "solve_cvrp_milp",
    "solve_milp",
    "lp_relaxation_value",
]

#: Backends worth trying, in the order we prefer them. SCIP is the strongest
#: MIP solver bundled with OR-Tools; HiGHS is a fast open-source alternative;
#: CBC is the historical default and the slowest of the three.
_MIP_BACKENDS = ("SCIP", "HIGHS", "CBC")
#: Pure-LP backends used for relaxation bounds.
_LP_BACKENDS = ("GLOP", "HIGHS", "CLP")

_STATUS_NAMES = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


def available_solvers(candidates: Iterable[str] = ("SCIP", "CBC", "HIGHS", "GLOP", "CLP", "SAT")) -> dict[str, bool]:
    """Which pywraplp backends this OR-Tools build actually links in.

    OR-Tools ships a different set of backends depending on how the wheel was
    built, so this is checked at runtime rather than assumed.
    """
    out: dict[str, bool] = {}
    for name in candidates:
        try:
            out[name] = pywraplp.Solver.CreateSolver(name) is not None
        except Exception:
            out[name] = False
    return out


@dataclass
class MILPResult:
    """Outcome of one MILP run."""

    routes: list[list[int]]
    cost: float
    lower_bound: float
    status: str
    seconds: float
    proven_optimal: bool
    formulation: str = "flow"
    backend: str = ""
    n_variables: int = 0
    n_constraints: int = 0
    n_vehicles: int = 0
    instance_name: str = ""

    def as_tuple(self) -> tuple[list[list[int]], float, float, str, float]:
        return self.routes, self.cost, self.lower_bound, self.status, self.seconds

    def to_optimization_result(self, instance: Instance) -> OptimizationResult:
        """Re-score with the project evaluator so costs compare like for like."""
        if self.routes:
            best = instance.make_solution(self.routes)
            best.validate(instance.n_customers)
        else:
            best = Solution()
        return OptimizationResult(
            algorithm=f"milp-{self.formulation}",
            instance=instance.name,
            best=best,
            history=[],
            iterations=0,
            evaluations=0,
            seconds=self.seconds,
            seed=None,
            params={
                "solver": f"pywraplp-{self.backend}",
                "formulation": self.formulation,
                "status": self.status,
                "lower_bound": self.lower_bound,
                "proven_optimal": self.proven_optimal,
                "n_variables": self.n_variables,
                "n_constraints": self.n_constraints,
                "n_routes": self.n_vehicles,
            },
        )


def _pick_backend(preferred: str | None, pool: Iterable[str]) -> tuple[pywraplp.Solver, str]:
    names = [preferred] if preferred else list(pool)
    for name in names:
        solver = pywraplp.Solver.CreateSolver(name)
        if solver is not None:
            return solver, name
    raise RuntimeError(
        f"none of the requested pywraplp backends are available: {list(names)}. "
        f"Available in this build: {[k for k, v in available_solvers().items() if v]}"
    )


def _build(
    solver: pywraplp.Solver,
    instance: Instance,
    formulation: str,
    integral: bool,
    min_vehicles: int | None,
    max_vehicles: int | None,
):
    """Construct the shared degree model plus the chosen subtour/capacity part.

    ``integral=False`` builds the linear relaxation, which is what
    :func:`lp_relaxation_value` and :mod:`qroute.exact.bounds` use.
    """
    n = instance.size
    cost = instance.cost_matrix
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

    inf = solver.infinity()
    make_bin = solver.BoolVar if integral else (lambda name: solver.NumVar(0.0, 1.0, name))

    x: dict[tuple[int, int], object] = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = make_bin(f"x_{i}_{j}")

    # Degree constraints: every customer entered once and left once.
    for i in range(1, n):
        solver.Add(solver.Sum(x[(i, j)] for j in range(n) if j != i) == 1)
        solver.Add(solver.Sum(x[(j, i)] for j in range(n) if j != i) == 1)

    routes_var = solver.IntVar(k_min, k_max, "K") if integral else solver.NumVar(k_min, k_max, "K")
    solver.Add(solver.Sum(x[(0, j)] for j in range(1, n)) == routes_var)
    solver.Add(solver.Sum(x[(j, 0)] for j in range(1, n)) == routes_var)

    if formulation == "mtz":
        u = {i: solver.NumVar(float(demand[i]), float(capacity), f"u_{i}") for i in range(1, n)}
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                # u_j >= u_i + d_j whenever arc (i,j) is used. With x = 0 the
                # constraint reduces to u_i - u_j <= Q - d_j, which holds for
                # every point of the box d <= u <= Q and is therefore vacuous:
                # that vacuousness is precisely the weakness discussed above.
                solver.Add(u[i] - u[j] + capacity * x[(i, j)] <= capacity - float(demand[j]))
        for j in range(1, n):
            # Leaving the depot resets the accumulated load to this customer's demand.
            solver.Add(u[j] <= float(demand[j]) + capacity * (1 - x[(0, j)]))
        extra = u
    elif formulation == "flow":
        f: dict[tuple[int, int], object] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ub = float(capacity) if i == 0 else float(capacity - demand[i])
                ub = max(ub, 0.0)
                f[(i, j)] = solver.NumVar(0.0, ub, f"f_{i}_{j}")
        for j in range(1, n):
            inflow = solver.Sum(f[(i, j)] for i in range(n) if i != j)
            outflow = solver.Sum(f[(j, k)] for k in range(n) if k != j)
            solver.Add(inflow - outflow == float(demand[j]))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ub = float(capacity) if i == 0 else float(capacity - demand[i])
                solver.Add(f[(i, j)] <= max(ub, 0.0) * x[(i, j)])
                if j != 0:
                    # The vehicle must still be carrying j's demand on the way in.
                    solver.Add(f[(i, j)] >= float(demand[j]) * x[(i, j)])
        for i in range(1, n):
            solver.Add(f[(i, 0)] == 0.0)  # vehicles come back empty
        solver.Add(solver.Sum(f[(0, j)] for j in range(1, n)) == float(total_demand))
        extra = f
    else:
        raise ValueError(f"unknown formulation {formulation!r}; use 'mtz' or 'flow'")

    solver.Minimize(solver.Sum(float(cost[i, j]) * v for (i, j), v in x.items()))
    return x, extra, routes_var


def solve_cvrp_milp(
    instance: Instance,
    formulation: str = "flow",
    time_limit: float = 60.0,
    backend: str | None = None,
    threads: int = 8,
    log: bool = False,
    min_vehicles: int | None = None,
    max_vehicles: int | None = None,
) -> MILPResult:
    """Solve a CVRP with a compact two-index MILP.

    Parameters
    ----------
    formulation:
        ``'flow'`` for the single-commodity-flow model (default, stronger) or
        ``'mtz'`` for Miller-Tucker-Zemlin.
    backend:
        pywraplp backend name. ``None`` picks the first available of SCIP,
        HiGHS, CBC.

    Time windows are not modelled here. Adding them to a two-index formulation
    needs another big-M block and makes an already weak model weaker, so the
    VRPTW path in this project goes through CP-SAT and the heuristics instead.
    """
    if instance.has_time_windows or instance.max_route_duration is not None:
        raise NotImplementedError(
            "the two-index MILP models capacity only; time windows and route-duration "
            "limits are not represented. Use qroute.exact.cpsat for VRPTW and "
            "qroute.baselines.ortools_gls for duration limits."
        )
    solver, backend_name = _pick_backend(backend, _MIP_BACKENDS)
    solver.SetTimeLimit(int(time_limit * 1000))
    try:
        solver.SetNumThreads(int(threads))
    except Exception:  # not every backend exposes threading
        pass
    if log:
        solver.EnableOutput()

    x, _extra, _k = _build(solver, instance, formulation, True, min_vehicles, max_vehicles)
    n_vars, n_cons = solver.NumVariables(), solver.NumConstraints()

    t0 = time.perf_counter()
    status = solver.Solve()
    seconds = time.perf_counter() - t0

    routes: list[list[int]] = []
    cost_value = float("inf")
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        routes = _extract_routes(x, instance.size)
        cost_value = float(solver.Objective().Value())
    try:
        bound = float(solver.Objective().BestBound())
    except Exception:  # pragma: no cover - backend dependent
        bound = -float("inf")
    if status == pywraplp.Solver.OPTIMAL:
        bound = min(bound, cost_value) if math.isfinite(cost_value) else bound

    return MILPResult(
        routes=routes,
        cost=cost_value,
        lower_bound=bound,
        status=_STATUS_NAMES.get(status, str(status)),
        seconds=seconds,
        proven_optimal=status == pywraplp.Solver.OPTIMAL,
        formulation=formulation,
        backend=backend_name,
        n_variables=n_vars,
        n_constraints=n_cons,
        n_vehicles=len(routes),
        instance_name=instance.name,
    )


def _extract_routes(x: dict, n: int) -> list[list[int]]:
    succ: dict[int, int] = {}
    starts: list[int] = []
    for (i, j), var in x.items():
        if var.solution_value() > 0.5:
            if i == 0:
                starts.append(j)
            else:
                succ[i] = j
    routes: list[list[int]] = []
    for s in sorted(starts):
        route: list[int] = []
        node = s
        guard = 0
        while node != 0:
            route.append(int(node))
            node = succ.get(node, 0)
            guard += 1
            if guard > n:  # pragma: no cover
                raise RuntimeError("cycle detected while extracting MILP routes")
        routes.append(route)
    return routes


def lp_relaxation_value(
    instance: Instance,
    formulation: str = "flow",
    backend: str | None = None,
    time_limit: float = 60.0,
    min_vehicles: int | None = None,
    max_vehicles: int | None = None,
) -> float:
    """Optimal value of the linear relaxation, a valid lower bound.

    Returns ``-inf`` if the LP could not be solved within the time limit, so a
    caller taking the maximum over several bounds is never misled.
    """
    if instance.has_time_windows or instance.max_route_duration is not None:
        raise NotImplementedError("LP relaxation is defined for the capacity model only")
    solver, _name = _pick_backend(backend, _LP_BACKENDS)
    solver.SetTimeLimit(int(time_limit * 1000))
    _build(solver, instance, formulation, False, min_vehicles, max_vehicles)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return -float("inf")
    return float(solver.Objective().Value())


def solve_milp(instance: Instance, **kwargs) -> OptimizationResult:
    """Benchmark-runner entry point: MILP as an :class:`OptimizationResult`."""
    return solve_cvrp_milp(instance, **kwargs).to_optimization_result(instance)
