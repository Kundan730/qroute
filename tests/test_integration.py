"""End-to-end checks across the whole platform.

These tests exist to catch the failures that unit tests structurally cannot: a
solver that reports a cost computed under different assumptions from the one the
report prints, a decoder that loses a customer only on a particular instance
family, a repair pass that silently makes a solution worse, or a road-network
instance whose matrices do not correspond to the graph they came from.

Everything here is verified against an independent recomputation. No test asserts
that a number equals a value that was itself produced by the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from qroute.algorithms.base import StopCriteria
from qroute.algorithms.decoder import Decoder
from qroute.algorithms.registry import build, names
from qroute.problems.loaders import load, list_instances, read_reference_solution

SMALL_CVRP = "A-n32-k5"
SMALL_VRPTW = "C101"


# --------------------------------------------------------------------------
# Reference solutions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("A-n32-k5", 784.0), ("A-n80-k10", 1763.0), ("P-n16-k8", 450.0),
    ("B-n31-k5", 672.0), ("X-n101-k25", 27591.0),
])
def test_published_optima_reproduce_exactly(name, expected):
    """The platform's objective must reproduce each published best-known cost.

    If this drifts, every gap in every table is wrong by the same amount, so it
    is checked for both instance families and for several sets within them.
    """
    inst = load(name)
    assert inst.meta["bks"] == pytest.approx(expected)
    routes = read_reference_solution(inst)
    assert routes, f"no reference solution shipped for {name}"
    solution = inst.make_solution(routes)
    solution.validate(inst.n_customers)
    assert solution.cost == pytest.approx(expected, abs=1e-6)
    assert solution.is_feasible


@pytest.mark.parametrize("name,expected", [("C101", 827.3), ("R101", 1637.7), ("R201", 1143.2)])
def test_vrptw_optima_reproduce_exactly(name, expected):
    inst = load(name)
    routes = read_reference_solution(inst)
    solution = inst.make_solution(routes)
    solution.validate(inst.n_customers)
    assert solution.cost == pytest.approx(expected, abs=1e-6)


def test_the_instance_library_is_present():
    available = list_instances()
    assert len(available["cvrp"]) >= 50
    assert len(available["vrptw"]) >= 30


# --------------------------------------------------------------------------
# Every algorithm, end to end
# --------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", names())
def test_algorithm_returns_a_valid_and_correctly_priced_solution(algorithm):
    """A solver's reported cost must equal an independent recomputation.

    This is the check that makes every benchmark number trustworthy: it closes
    the gap between what a solver believes it found and what its routes actually
    cost under the published objective.
    """
    inst = load(SMALL_CVRP)
    result = build(algorithm, inst, stop=StopCriteria(max_iterations=10 ** 6, max_seconds=4),
                   seed=0).solve()
    result.best.validate(inst.n_customers)
    recomputed = inst.make_solution(result.best.routes)
    assert recomputed.cost == pytest.approx(result.best.cost, rel=1e-9)
    assert result.best.cost >= inst.meta["bks"] - 1e-6, "claimed to beat a proven optimum"
    assert result.history, "no convergence history was recorded"
    assert result.evaluations > 0


@pytest.mark.parametrize("algorithm", names())
def test_algorithm_is_reproducible(algorithm):
    inst = load(SMALL_CVRP)
    stop = StopCriteria(max_iterations=25)
    a = build(algorithm, inst, stop=stop, seed=7).solve()
    b = build(algorithm, inst, stop=stop, seed=7).solve()
    assert a.best.cost == pytest.approx(b.best.cost, rel=1e-12)
    assert a.best.routes == b.best.routes


@pytest.mark.parametrize("algorithm", names())
def test_algorithm_respects_a_time_budget(algorithm):
    inst = load("A-n80-k10")
    result = build(algorithm, inst, stop=StopCriteria(max_iterations=10 ** 9, max_seconds=2.0),
                   seed=0).solve()
    assert result.seconds < 12.0, "overran its budget by more than compilation can explain"


def test_time_windows_are_honoured_or_reported():
    """On a time-window instance the solution is feasible, or says it is not."""
    inst = load(SMALL_VRPTW)
    result = build("qpso", inst, stop=StopCriteria(max_iterations=10 ** 6, max_seconds=6),
                   seed=1).solve()
    result.best.validate(inst.n_customers)
    stats = inst.evaluate(result.best.routes)
    assert stats.time_window_violation == pytest.approx(result.best.stats.time_window_violation)
    if result.best.is_feasible:
        assert stats.time_window_violation <= 1e-9
        assert stats.capacity_violation <= 1e-9


# --------------------------------------------------------------------------
# Decoder and repair
# --------------------------------------------------------------------------
def test_decoder_never_loses_or_duplicates_a_customer():
    rng = np.random.default_rng(0)
    for name in (SMALL_CVRP, SMALL_VRPTW, "A-n80-k10"):
        inst = load(name)
        decoder = Decoder(inst)
        for _ in range(5):
            routes, _cost, _keys = decoder.decode(rng.random(inst.n_customers))
            inst.make_solution(routes).validate(inst.n_customers)


def test_repair_restores_feasibility_without_losing_customers():
    """Hand the repair pass a deliberately overloaded solution and check it."""
    inst = load(SMALL_CVRP)
    decoder = Decoder(inst)
    everyone_in_one_vehicle = [list(range(1, inst.size))]
    before = inst.evaluate(everyone_in_one_vehicle)
    assert before.capacity_violation > 0, "the test instance is not actually overloaded"
    repaired, _cost = decoder.repair(everyone_in_one_vehicle)
    solution = inst.make_solution(repaired)
    solution.validate(inst.n_customers)
    assert solution.stats.capacity_violation <= 1e-9


def test_penalties_scale_with_the_instance():
    """A fixed penalty is meaningless across cost scales; check it adapts."""
    small = load(SMALL_CVRP)
    scaled = small.with_matrices(distance=small.distance * 1000.0)
    assert Decoder(scaled).pen_cap > Decoder(small).pen_cap * 100


# --------------------------------------------------------------------------
# Exact methods as ground truth
# --------------------------------------------------------------------------
def test_no_solver_beats_a_proven_optimum():
    """CP-SAT closes a small instance; nothing may report a cheaper cost."""
    cpsat = pytest.importorskip("qroute.exact.cpsat")
    inst = load("P-n16-k8")
    from qroute.benchmark.runner import _call_with_supported
    exact = _call_with_supported(cpsat.solve_cpsat, inst, seconds=30)
    exact.best.validate(inst.n_customers)
    recomputed = inst.make_solution(exact.best.routes)
    assert recomputed.cost == pytest.approx(exact.best.cost, rel=1e-6)
    assert exact.best.cost >= inst.meta["bks"] - 1e-6


def test_lower_bounds_never_exceed_the_known_optimum():
    """A bound above the optimum is worse than no bound at all."""
    bounds = pytest.importorskip("qroute.exact.bounds")
    for name in ("P-n16-k8", "A-n32-k5", "A-n45-k7"):
        inst = load(name)
        report = bounds.bracket(inst)
        values = report if isinstance(report, dict) else getattr(report, "__dict__", {})
        for key, value in values.items():
            if isinstance(value, (int, float)) and "bound" in key.lower() and value > 0:
                assert value <= inst.meta["bks"] + 1e-6, \
                    f"{key} = {value} exceeds the optimum {inst.meta['bks']} on {name}"
