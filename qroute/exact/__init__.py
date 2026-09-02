"""Exact methods and valid lower bounds: the ground truth of the benchmark.

Nothing in this package is meant to be fast. Its job is to say, for as many
instances as possible, what the answer actually is -- and where that is out of
reach, to bracket it between a feasible solution and a bound that is provably
below the optimum. Without this package a phrase like "within 2% of optimal"
would be an assertion about a best-known value someone else published; with it,
it is a measurement.

Contents
--------
:mod:`~qroute.exact.cpsat`
    CP-SAT multi-circuit model for the CVRP (and a single-circuit TSP mode).
    The strongest exact method here: it proves A-n32-k5 optimal at 784.
:mod:`~qroute.exact.milp`
    Compact two-index MILPs -- MTZ and single-commodity flow -- through
    pywraplp. Used on tiny instances and as an independent cross-check of the
    CP-SAT model, and as the source of the LP relaxation bound.
:mod:`~qroute.exact.heldkarp`
    Solver-free subset dynamic programmes: Held-Karp for the TSP and an exact
    set-partitioning CVRP for very small instances.
:mod:`~qroute.exact.bounds`
    Cheap lower bounds (bin packing, degree, spanning tree, radial, LP) that
    bracket instances no exact method can close.
:mod:`~qroute.exact.scaling`
    Shared float-to-integer conversion, so every integer solver in the project
    optimises exactly the same objective.

Three independent methods -- CP-SAT, the flow MILP and the subset DP -- agree
on 450 for P-n16-k8, which is the cross-validation that makes the rest of the
numbers trustworthy.
"""

from __future__ import annotations

from qroute.exact.bounds import (
    BoundReport,
    bin_packing_bound,
    bracket,
    degree_bound,
    lp_bound,
    mst_bound,
    one_tree_bound,
    radial_bound,
)
from qroute.exact.cpsat import CPSATResult, solve_cpsat, solve_cvrp_cpsat, solve_tsp_cpsat
from qroute.exact.heldkarp import DPResult, held_karp_cvrp, held_karp_tsp, solve_heldkarp
from qroute.exact.milp import MILPResult, available_solvers, solve_cvrp_milp, solve_milp
from qroute.exact.scaling import Scaling, integer_scaling

__all__ = [
    "CPSATResult",
    "solve_cvrp_cpsat",
    "solve_tsp_cpsat",
    "solve_cpsat",
    "MILPResult",
    "solve_cvrp_milp",
    "solve_milp",
    "available_solvers",
    "DPResult",
    "held_karp_tsp",
    "held_karp_cvrp",
    "solve_heldkarp",
    "BoundReport",
    "bracket",
    "bin_packing_bound",
    "degree_bound",
    "mst_bound",
    "radial_bound",
    "one_tree_bound",
    "lp_bound",
    "Scaling",
    "integer_scaling",
]
