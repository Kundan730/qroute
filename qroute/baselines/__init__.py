"""Strong classical solvers, wrapped so the comparison is fair.

A quantum-inspired metaheuristic is only interesting if it is measured against
what a competent engineer would otherwise use. These wrappers provide that
reference point, and they are deliberately configured to do well rather than to
lose gracefully.

:mod:`~qroute.baselines.ortools_gls`
    OR-Tools routing with guided local search. The default industrial choice
    for CVRP and VRPTW, and the baseline most people will ask about.
:mod:`~qroute.baselines.pyvrp_hgs`
    PyVRP's hybrid genetic search, the open-source state of the art for the
    CVRP. This is the ceiling of the benchmark, not a target to beat.

Both are plain functions returning
:class:`~qroute.algorithms.base.OptimizationResult` rather than
:class:`~qroute.algorithms.base.Optimizer` subclasses. The base class owns an
iteration loop, a clock and an evaluation counter, and neither of these solvers
exposes an inner loop for it to drive -- forcing them into the ABC would mean
faking iteration counts. A function returning the shared result type gives the
benchmark runner everything it needs (best solution, wall-clock time, and a
convergence curve built from each library's own solution callback) without
inventing numbers.

Both wrappers re-score their routes with
:meth:`~qroute.problems.instance.Instance.make_solution`, so every cost in the
benchmark comes from one evaluator regardless of which library produced the
routes.
"""

from __future__ import annotations

from qroute.baselines.ortools_gls import ORToolsResult, solve_ortools, solve_ortools_result
from qroute.baselines.pyvrp_hgs import (
    PyVRPResult,
    PyVRPUnavailable,
    solve_pyvrp,
    solve_pyvrp_result,
)
from qroute.baselines.pyvrp_hgs import available as pyvrp_available
from qroute.baselines.pyvrp_hgs import version as pyvrp_version

__all__ = [
    "ORToolsResult",
    "solve_ortools",
    "solve_ortools_result",
    "PyVRPResult",
    "PyVRPUnavailable",
    "solve_pyvrp",
    "solve_pyvrp_result",
    "pyvrp_available",
    "pyvrp_version",
]
