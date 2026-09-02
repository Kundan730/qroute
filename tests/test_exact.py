"""Tests for the exact methods, the lower bounds and the classical baselines.

The tests are organised around the three properties that make this part of the
platform worth trusting:

1. **Agreement.** Independent methods must produce the same optimum on the same
   instance. CP-SAT, the flow MILP, the MTZ MILP and the subset DP all have to
   agree on P-n16-k8, and Held-Karp has to agree with brute force on random
   tiny TSPs.
2. **Validity.** Every lower bound must be at or below a known optimum, and at
   or below the best-known solution of instances whose optimum is unknown. A
   bound that can exceed the optimum is the single most damaging bug this
   package could ship, so it is tested on every instance we ship.
3. **Comparability.** Every wrapper must return routes that the project's own
   evaluator accepts and re-scores to the value the wrapper reported. This is
   what catches an integer-scaling mistake.

Runtimes are kept modest so the file stays usable as a pre-commit check. The
one genuinely slow test -- the fifty-second proof that A-n32-k5 is optimal at
784 -- runs only when ``QROUTE_SLOW_TESTS=1`` is set, so it does not have to be
registered as a custom pytest marker in configuration this module does not own.
"""

from __future__ import annotations

import os
from itertools import permutations

import numpy as np
import pytest

from qroute.baselines.ortools_gls import solve_ortools, solve_ortools_result
from qroute.baselines.pyvrp_hgs import available as pyvrp_available
from qroute.baselines.pyvrp_hgs import solve_pyvrp, solve_pyvrp_result
from qroute.exact.bounds import (
    bin_packing_bound,
    bracket,
    degree_bound,
    lp_bound,
    mst_bound,
    one_tree_bound,
    radial_bound,
)
from qroute.exact.cpsat import solve_cvrp_cpsat, solve_tsp_cpsat
from qroute.exact.heldkarp import held_karp_cvrp, held_karp_tsp, max_tsp_nodes
from qroute.exact.milp import available_solvers, lp_relaxation_value, solve_cvrp_milp
from qroute.exact.scaling import integer_demands, integer_scaling
from qroute.problems.loaders import load

# Instances small enough for an exact method inside a test run.
TINY = "P-n16-k8"
TINY_OPT = 450.0

# Every instance in the reported table, with its best-known cost. For A/B/P the
# best-known value is the proven optimum; for X-n101-k25 and C101 it is an
# upper bound, which is all a lower-bound validity test needs.
BENCHMARK = [
    ("P-n16-k8", 450.0),
    ("A-n32-k5", 784.0),
    ("A-n33-k5", 661.0),
    ("A-n37-k5", 669.0),
    ("A-n45-k7", 1146.0),
    ("A-n80-k10", 1763.0),
    ("X-n101-k25", 27591.0),
    ("C101", 827.3),
]


def _brute_force_tsp(cost: np.ndarray) -> float:
    n = cost.shape[0]
    return min(
        sum(cost[a, b] for a, b in zip((0,) + p, p + (0,)))
        for p in permutations(range(1, n))
    )


# --------------------------------------------------------------------- scaling
def test_scaling_is_exact_for_both_benchmark_families():
    """CVRPLIB needs no scaling; Solomon needs exactly a factor of ten."""
    cvrp = load("A-n32-k5")
    s = integer_scaling(cvrp.cost_matrix)
    assert s.factor == 1 and s.exact

    solomon = load("C101")
    s = integer_scaling(solomon.cost_matrix, solomon.duration, solomon.time_windows,
                        solomon.service_time)
    assert s.factor == 10 and s.exact
    # Round-tripping must not move a single arc cost.
    back = s.to_int(solomon.cost_matrix) / s.factor
    assert np.allclose(back, solomon.cost_matrix, atol=1e-12)


def test_integer_demands_rejects_fractional_input():
    inst = load(TINY)
    demand = inst.demand.copy()
    demand[1] += 0.5
    with pytest.raises(ValueError):
        integer_demands(demand, inst.capacity)


# ----------------------------------------------------------------- Held-Karp
@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_held_karp_matches_brute_force(n):
    """The subset DP must equal exhaustive enumeration on random asymmetric costs."""
    rng = np.random.default_rng(1234 + n)
    cost = rng.random((n, n)) * 10.0
    np.fill_diagonal(cost, 0.0)
    result = held_karp_tsp(cost)
    assert result.proven_optimal
    assert result.cost == pytest.approx(_brute_force_tsp(cost), abs=1e-9)
    assert sorted(result.routes[0]) == list(range(1, n))


def test_held_karp_refuses_to_exceed_its_memory_budget():
    limit = max_tsp_nodes(1.0)
    cost = np.ones((limit + 2, limit + 2))
    with pytest.raises(MemoryError):
        held_karp_tsp(cost, memory_limit_mb=1.0)


def test_held_karp_cvrp_solves_the_tiny_instance_exactly():
    inst = load(TINY)
    result = held_karp_cvrp(inst)
    assert result.proven_optimal
    assert result.cost == pytest.approx(TINY_OPT)
    sol = inst.make_solution(result.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible
    assert sol.cost == pytest.approx(TINY_OPT)


def test_held_karp_cvrp_refuses_large_instances_and_time_windows():
    with pytest.raises(ValueError):
        held_karp_cvrp(load("A-n32-k5"))
    with pytest.raises(NotImplementedError):
        held_karp_cvrp(load("C101"))


# --------------------------------------------------------------------- CP-SAT
def test_cpsat_proves_the_tiny_instance():
    inst = load(TINY)
    result = solve_cvrp_cpsat(inst, time_limit=60, workers=8)
    assert result.status == "OPTIMAL"
    assert result.proven_optimal
    assert result.cost == pytest.approx(TINY_OPT)
    assert result.lower_bound == pytest.approx(TINY_OPT)
    sol = inst.make_solution(result.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible
    assert sol.cost == pytest.approx(result.cost)


def test_cpsat_tsp_mode_matches_held_karp():
    """Integer costs scale exactly, so CP-SAT may claim a proof here."""
    rng = np.random.default_rng(99)
    cost = np.rint(rng.random((9, 9)) * 100.0)
    np.fill_diagonal(cost, 0.0)
    exact = held_karp_tsp(cost)
    sat = solve_tsp_cpsat(cost, time_limit=30)
    assert sat.proven_optimal
    assert sat.cost == pytest.approx(exact.cost, rel=1e-9)
    assert sorted(sat.routes[0]) == list(range(1, 9))


def test_cpsat_withholds_the_optimality_claim_when_scaling_is_inexact():
    """Costs needing more than four decimals are rounded, so no proof is claimed.

    The answer is still returned and is still right to within the rounding, but
    ``proven_optimal`` must be False: the model solved is not exactly the model
    asked about, and saying otherwise would be the kind of quiet overclaim this
    package exists to prevent.
    """
    rng = np.random.default_rng(99)
    cost = rng.random((9, 9)) * 100.0  # irrational-looking floats
    np.fill_diagonal(cost, 0.0)
    sat = solve_tsp_cpsat(cost, time_limit=30)
    assert sat.status == "OPTIMAL"
    assert not sat.scaling.exact
    assert not sat.proven_optimal
    # Still close to the true optimum, just not certified.
    assert sat.cost == pytest.approx(held_karp_tsp(cost).cost, rel=1e-3)


def test_cpsat_is_reproducible_only_under_deterministic_time():
    """Pins what the ``seed`` argument does and does not buy.

    Under a *wall-clock* limit CP-SAT is a race against the machine, so an
    unproven incumbent and dual bound vary between identically seeded runs at
    any worker count. ``deterministic_time`` with a single worker is the one
    stopping rule that repeats exactly. A benchmark quoting a CP-SAT dual bound
    as though it were a constant is quoting one sample, and this test is what
    keeps that distinction from being forgotten.
    """
    inst = load("A-n45-k7")  # far too hard to close in the budgets below
    a = solve_cvrp_cpsat(inst, workers=1, seed=0, deterministic_time=1.0)
    b = solve_cvrp_cpsat(inst, workers=1, seed=0, deterministic_time=1.0)
    assert not a.proven_optimal
    assert a.cost == pytest.approx(b.cost)
    assert a.lower_bound == pytest.approx(b.lower_bound)
    assert a.routes == b.routes


def test_cpsat_result_converts_to_optimization_result():
    inst = load(TINY)
    result = solve_cvrp_cpsat(inst, time_limit=60).to_optimization_result(inst)
    assert result.best_cost == pytest.approx(TINY_OPT)
    assert result.params["proven_optimal"] is True
    assert result.params["lower_bound"] == pytest.approx(TINY_OPT)
    assert result.gap_to(TINY_OPT) == pytest.approx(0.0)


@pytest.mark.skipif(
    os.environ.get("QROUTE_SLOW_TESTS") != "1",
    reason="takes about a minute; set QROUTE_SLOW_TESTS=1 to run",
)
def test_cpsat_proves_a_n32_k5_optimal_at_784():
    """The headline proof the rest of the benchmark rests on."""
    inst = load("A-n32-k5")
    result = solve_cvrp_cpsat(inst, time_limit=180, workers=8)
    assert result.proven_optimal
    assert result.cost == pytest.approx(784.0)
    assert result.lower_bound == pytest.approx(784.0)
    sol = inst.make_solution(result.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible and sol.cost == pytest.approx(784.0)


# ----------------------------------------------------------------------- MILP
def test_reports_which_pywraplp_backends_exist():
    backends = available_solvers()
    # SCIP is bundled with every OR-Tools wheel we support; if it disappears we
    # want a loud failure rather than a silent fallback to CBC.
    assert backends["SCIP"] is True
    assert set(backends) >= {"SCIP", "CBC", "HIGHS", "GLOP"}


@pytest.mark.parametrize("formulation", ["flow", "mtz"])
def test_milp_reproduces_the_tiny_optimum(formulation):
    inst = load(TINY)
    result = solve_cvrp_milp(inst, formulation=formulation, time_limit=180)
    assert result.proven_optimal, f"{formulation} did not close {TINY}"
    assert result.cost == pytest.approx(TINY_OPT)
    sol = inst.make_solution(result.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible
    assert sol.cost == pytest.approx(result.cost)


def test_flow_relaxation_dominates_mtz_relaxation():
    """The claim made in the module docstring, checked rather than asserted."""
    for name in ("P-n16-k8", "A-n32-k5", "A-n33-k5"):
        inst = load(name)
        flow = lp_relaxation_value(inst, formulation="flow")
        mtz = lp_relaxation_value(inst, formulation="mtz")
        assert flow >= mtz - 1e-6, f"{name}: flow {flow} < mtz {mtz}"
        assert flow <= inst.meta["bks"] + 1e-6


def test_milp_rejects_time_windows_loudly():
    with pytest.raises(NotImplementedError):
        solve_cvrp_milp(load("C101"))


# --------------------------------------------------------------------- bounds
def test_bin_packing_bound_is_valid_and_at_least_l1():
    for name, _bks in BENCHMARK:
        inst = load(name)
        lb = bin_packing_bound(inst.demand, inst.capacity)
        assert lb >= inst.min_vehicles
        reference_k = inst.meta.get("bks_routes") or inst.meta.get("reference_k")
        if reference_k:
            assert lb <= reference_k, f"{name}: vehicle bound {lb} above known {reference_k}"


def test_bin_packing_bound_beats_l1_when_it_should():
    # Three items of size 6 in bins of size 10: L1 says 2, but no two fit
    # together, so the true answer -- and L2 -- is 3.
    demand = np.array([0.0, 6.0, 6.0, 6.0])
    assert bin_packing_bound(demand, 10.0) == 3


@pytest.mark.parametrize("name,bks", BENCHMARK)
def test_every_lower_bound_is_below_the_best_known_solution(name, bks):
    """The property that makes bounds worth having at all."""
    inst = load(name)
    report = bracket(inst, upper_bound=bks, include_lp=inst.size <= 120 and not inst.has_time_windows)
    for label, value in report.bounds.items():
        if np.isfinite(value):
            assert value <= bks + 1e-6, f"{name}: bound {label}={value} exceeds {bks}"
    assert report.best <= bks + 1e-6
    assert np.isfinite(report.best)


def test_bounds_are_valid_against_a_proven_optimum():
    inst = load(TINY)
    for value in (degree_bound(inst), mst_bound(inst), radial_bound(inst), lp_bound(inst)):
        if np.isfinite(value):
            assert value <= TINY_OPT + 1e-6


def test_one_tree_bound_never_exceeds_the_exact_tsp_optimum():
    rng = np.random.default_rng(31337)
    for _ in range(5):
        n = int(rng.integers(6, 11))
        pts = rng.random((n, 2)) * 100.0
        cost = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        opt = held_karp_tsp(cost).cost
        lb = one_tree_bound(cost, iterations=200)
        assert lb <= opt + 1e-6


def test_radial_bound_declines_rather_than_guess_when_it_cannot_verify_metricity():
    """A matrix that badly violates the triangle inequality, too big to close."""
    inst = load("A-n32-k5")
    broken = inst.cost_matrix.copy()
    broken[1, 2] = 10_000.0  # a shortcut through node 3 is now far cheaper
    hacked = inst.with_matrices(distance=broken)
    value = radial_bound(hacked, allow_closure=False)
    assert value == -np.inf


# ------------------------------------------------------------------ baselines
def test_ortools_solves_a_cvrp_and_its_cost_survives_rescoring():
    inst = load("A-n32-k5")
    raw = solve_ortools_result(inst, seconds=5)
    assert raw.routes
    sol = inst.make_solution(raw.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible
    # The scaled objective OR-Tools reports must match the project evaluator
    # exactly; a mismatch means the integer scaling is wrong.
    assert sol.cost == pytest.approx(raw.cost, abs=1e-6)
    assert raw.curve, "no at-solution callback fired, so there is no curve"


def test_ortools_respects_time_windows():
    inst = load("C101")
    raw = solve_ortools_result(inst, seconds=10)
    sol = inst.make_solution(raw.routes)
    sol.validate(inst.n_customers)
    assert sol.stats.time_window_violation == pytest.approx(0.0)
    assert sol.stats.capacity_violation == pytest.approx(0.0)
    assert sol.cost == pytest.approx(raw.cost, abs=1e-6)


def test_ortools_recovers_when_the_first_solution_strategy_fails():
    """R101 defeats path_cheapest_arc; the baseline must still produce a row.

    Measured: with a five-second budget seven Solomon instances (R101-R103,
    R105, R106, RC101, RC105) end in ROUTING_FAIL_TIMEOUT with *zero* routes
    under the default construction heuristic. Silently returning an empty
    solution there would delete the strongest classical baseline from precisely
    the hardest time-window instances, so a fallback strategy is tried.
    """
    inst = load("R101")

    without = solve_ortools_result(inst, seconds=5, fallback_first_solution=())
    assert not without.routes, "R101 no longer defeats path_cheapest_arc; retune this test"

    with_fallback = solve_ortools_result(inst, seconds=5)
    assert with_fallback.routes
    assert with_fallback.attempts == 2
    assert with_fallback.first_solution_used == "parallel_cheapest_insertion"
    sol = inst.make_solution(with_fallback.routes)
    sol.validate(inst.n_customers)
    assert sol.stats.time_window_violation == pytest.approx(0.0)
    assert sol.stats.capacity_violation == pytest.approx(0.0)
    assert sol.cost == pytest.approx(with_fallback.cost, abs=1e-6)


@pytest.mark.parametrize("name", ["P-n16-k8", "A-n45-k7", "C201"])
def test_baseline_reported_cost_equals_the_project_evaluator(name):
    """The docstring claim that re-scoring catches a scaling mistake, checked.

    Both wrappers report their own solver's objective. If the integer scaling
    were wrong that number would differ from what the project evaluator makes
    of the same routes, so the two are compared here rather than merely
    asserted in prose.
    """
    inst = load(name)
    raw = solve_ortools_result(inst, seconds=5)
    sol = inst.make_solution(raw.routes)
    sol.validate(inst.n_customers)
    assert sol.cost == pytest.approx(raw.cost, abs=1e-6)

    if pyvrp_available():
        praw = solve_pyvrp_result(inst, max_iterations=300, seed=0)
        psol = inst.make_solution(praw.routes)
        psol.validate(inst.n_customers)
        assert psol.cost == pytest.approx(praw.cost, abs=1e-6)


def test_ortools_returns_an_optimization_result():
    inst = load("A-n32-k5")
    result = solve_ortools(inst, seconds=5)
    assert result.algorithm == "ortools-gls"
    assert result.best.is_feasible
    assert result.history and result.history[-1].best_cost >= result.best_cost - 1e-6


@pytest.mark.skipif(not pyvrp_available(), reason="PyVRP is not installed")
def test_pyvrp_solves_a_cvrp_and_its_cost_survives_rescoring():
    inst = load("A-n32-k5")
    raw = solve_pyvrp_result(inst, max_iterations=500, seed=1)
    assert raw.feasible
    sol = inst.make_solution(raw.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible
    assert sol.cost == pytest.approx(raw.cost, abs=1e-6)


@pytest.mark.skipif(not pyvrp_available(), reason="PyVRP is not installed")
def test_pyvrp_is_reproducible_for_a_fixed_seed_and_iteration_count():
    inst = load("A-n33-k5")
    a = solve_pyvrp_result(inst, max_iterations=400, seed=7)
    b = solve_pyvrp_result(inst, max_iterations=400, seed=7)
    assert a.cost == pytest.approx(b.cost)
    assert a.routes == b.routes


@pytest.mark.skipif(not pyvrp_available(), reason="PyVRP is not installed")
def test_pyvrp_respects_time_windows():
    inst = load("C101")
    raw = solve_pyvrp_result(inst, max_iterations=800, seed=3)
    sol = inst.make_solution(raw.routes)
    sol.validate(inst.n_customers)
    assert sol.stats.time_window_violation == pytest.approx(0.0)
    assert sol.stats.capacity_violation == pytest.approx(0.0)


@pytest.mark.skipif(not pyvrp_available(), reason="PyVRP is not installed")
def test_pyvrp_returns_an_optimization_result():
    inst = load("A-n32-k5")
    result = solve_pyvrp(inst, max_iterations=300, seed=1)
    assert result.algorithm == "pyvrp-hgs"
    assert result.best.is_feasible
    assert result.params["pyvrp_version"]


# --------------------------------------------------------------- cross-checks
def test_four_independent_methods_agree_on_the_tiny_optimum():
    """CP-SAT, both MILP formulations and the subset DP must all say 450.

    This is the strongest single check in the package: four different models
    solved by three different pieces of software. A shared mistake in the
    problem definition would have to survive all of them.
    """
    inst = load(TINY)
    costs = {
        "cpsat": solve_cvrp_cpsat(inst, time_limit=60).cost,
        "milp-flow": solve_cvrp_milp(inst, formulation="flow", time_limit=180).cost,
        "milp-mtz": solve_cvrp_milp(inst, formulation="mtz", time_limit=180).cost,
        "subset-dp": held_karp_cvrp(inst).cost,
    }
    for label, value in costs.items():
        assert value == pytest.approx(TINY_OPT), f"{label} disagrees: {value}"


def test_free_fleet_and_fixed_fleet_are_different_problems():
    """A CVRPLIB best-known value can assume a fixed fleet.

    On P-n22-k8 the published solution costs 603 using 8 routes, but the
    unrestricted-fleet optimum is 590 using 9. Both are correct answers to
    different questions, and a benchmark that mixes them up will report a
    negative gap and look like it has a bug. This test pins the distinction so
    nobody "fixes" the free-fleet default by accident.
    """
    inst = load("P-n22-k8")
    assert inst.meta["bks"] == pytest.approx(603.0)

    free = solve_cvrp_cpsat(inst, time_limit=120, workers=8)
    assert free.proven_optimal
    assert free.cost == pytest.approx(590.0)
    assert len(free.routes) == 9
    sol = inst.make_solution(free.routes)
    sol.validate(inst.n_customers)
    assert sol.is_feasible and sol.cost == pytest.approx(590.0)

    # Constrained to the reference fleet, the reference value is reachable.
    fixed = solve_cvrp_cpsat(inst, time_limit=120, workers=8,
                             min_vehicles=8, max_vehicles=8)
    assert fixed.cost == pytest.approx(603.0)
    assert len(fixed.routes) == 8


def test_solution_callback_failure_cannot_truncate_the_search():
    """A broken recorder must cost the curve, never the solve.

    An exception raised inside a CP-SAT callback aborts the search, which would
    silently return a weaker bound than the time limit was meant to buy. The
    recorder therefore swallows its own errors. Here the conversion it performs
    is sabotaged, so every callback fails internally.
    """
    from qroute.exact import cpsat as cpsat_mod

    class _BrokenScaling:
        factor = 1
        exact = True

        def to_float(self, value):
            raise RuntimeError("conversion is broken")

    class _BrokenRecorder(cpsat_mod._SolutionRecorder):
        def __init__(self, scaling, t0):
            super().__init__(scaling, t0)
            self._scaling = _BrokenScaling()

    inst = load(TINY)
    original = cpsat_mod._SolutionRecorder
    try:
        cpsat_mod._SolutionRecorder = _BrokenRecorder
        result = solve_cvrp_cpsat(inst, time_limit=60, workers=8)
    finally:
        cpsat_mod._SolutionRecorder = original

    # The proof still goes through even though every callback raised inside.
    assert result.proven_optimal
    assert result.cost == pytest.approx(TINY_OPT)
    assert result.curve == []
