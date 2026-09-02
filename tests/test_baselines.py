"""Contract tests for the classical baselines and the adaptive penalty manager.

The point of the baselines is to make the QPSO comparison fair, and a
comparison is only fair if every solver obeys the same contract. These tests
check that contract rather than solution quality: quality is measured by the
benchmark runner against best-known solutions, and asserting a particular cost
here would make the suite fail whenever a parameter default is retuned.

The contract each optimiser must satisfy:

1. it returns a :class:`~qroute.core.types.Solution` that visits every customer
   exactly once (``validate`` raises otherwise);
2. it stops within its wall-clock budget;
3. the same seed and the same iteration budget give the same best cost;
4. it records a non-empty convergence history.

Reproducibility is checked under an *iteration* budget, not a time budget. Two
runs limited by the clock legitimately perform different numbers of iterations
on a loaded machine, so requiring identical costs there would be testing the
machine rather than the code.

The whole file is written to run in a few seconds: budgets are small and the
instance is the smallest CVRPLIB one.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qroute.algorithms.base import StopCriteria
from qroute.algorithms.decoder import Decoder
from qroute.algorithms.penalty import CONSTRAINTS, AdaptivePenalty
from qroute.algorithms.registry import ALGORITHMS, build, catalogue, get, names
from qroute.core.types import SolutionStats
from qroute.problems.loaders import load

ALGO_NAMES = list(ALGORITHMS)


@pytest.fixture(scope="module")
def instance():
    return load("A-n32-k5")


@pytest.fixture(scope="module", autouse=True)
def _warm_up(instance):
    """Trigger numba compilation before anything is timed.

    ACO's construction kernel and the shared routing kernels compile on first
    use. That cost is paid once per machine (numba caches to disk) but it would
    otherwise land inside whichever test ran first and make the budget
    assertion flaky on a cold checkout.
    """
    for name in ALGO_NAMES:
        build(name, instance, stop=StopCriteria(max_iterations=1), seed=0).solve()


# --------------------------------------------------------------------- registry
def test_registry_lists_every_algorithm():
    assert set(names()) == {"qpso", "pso", "ga", "sa", "aco"}
    assert {c["name"] for c in catalogue()} == set(names())
    assert all(c["description"] for c in catalogue())


def test_registry_rejects_unknown_name():
    with pytest.raises(KeyError):
        get("simulated-quantum-annealing")


def test_registry_resolves_classes_lazily():
    for name in ALGO_NAMES:
        cls = get(name)
        assert isinstance(cls, type)
        assert get(name) is cls  # memoised, not re-imported


# ------------------------------------------------------------------- contract
@pytest.mark.parametrize("name", ALGO_NAMES)
def test_returns_a_valid_solution(instance, name):
    result = build(name, instance, stop=StopCriteria(max_iterations=5), seed=7).solve()
    result.best.validate(instance.n_customers)
    assert result.best.routes
    assert np.isfinite(result.best_cost)
    # Reported cost must come from the reference evaluator, not the penalised
    # objective the search minimises, otherwise cross-algorithm costs are not
    # comparable.
    assert result.best_cost == pytest.approx(
        instance.make_solution(result.best.routes).cost)


@pytest.mark.parametrize("name", ALGO_NAMES)
def test_records_non_empty_history(instance, name):
    result = build(name, instance, stop=StopCriteria(max_iterations=5), seed=7).solve()
    assert len(result.history) >= 1
    assert result.iterations == len(result.history)
    costs = [h.best_cost for h in result.history]
    # The incumbent can only improve.
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))
    assert all(h.evaluations > 0 for h in result.history)
    assert result.evaluations > 0


@pytest.mark.parametrize("name", ALGO_NAMES)
def test_is_reproducible_for_a_fixed_seed(instance, name):
    stop = StopCriteria(max_iterations=6)
    a = build(name, instance, stop=stop, seed=12345).solve()
    b = build(name, instance, stop=stop, seed=12345).solve()
    assert a.best_cost == b.best_cost
    assert a.best.routes == b.best.routes
    assert a.evaluations == b.evaluations


@pytest.mark.parametrize("name", ALGO_NAMES)
def test_different_seeds_explore_differently(instance, name):
    """A seed must actually change the trajectory, not just the reported value."""
    stop = StopCriteria(max_iterations=4)
    a = build(name, instance, stop=stop, seed=1).solve()
    b = build(name, instance, stop=stop, seed=2).solve()
    curves_differ = [h.best_cost for h in a.history] != [h.best_cost for h in b.history]
    assert curves_differ or a.best.routes != b.best.routes


@pytest.mark.parametrize("name", ALGO_NAMES)
def test_respects_a_wall_clock_budget(instance, name):
    budget = 0.5
    t0 = time.perf_counter()
    result = build(name, instance, stop=StopCriteria(max_iterations=10**9,
                                                     max_seconds=budget),
                   seed=3).solve()
    elapsed = time.perf_counter() - t0
    # The budget is checked between iterations, so an overshoot of up to one
    # iteration is expected and allowed; anything larger means an inner loop is
    # not consulting the clock.
    #
    # The allowance is stated in units of this run's own mean iteration cost
    # rather than as a fixed number of seconds. A constant slack silently
    # encodes an assumption about machine speed and load: this suite is run on
    # shared CI and alongside other benchmark sweeps, where a single iteration
    # can genuinely take longer than any constant a developer would pick, and
    # the test then fails for a solver that is obeying the contract perfectly.
    # Scaling by the measured cost keeps the assertion sharp -- a solver that
    # ignores the clock runs to max_iterations (10**9) and overshoots by
    # thousands of iterations, not by two.
    per_iteration = elapsed / max(result.iterations, 1)
    assert elapsed < budget + 2.0 * per_iteration + 0.05
    # The relative bound above goes slack when a run manages only one
    # iteration, since then "one iteration" is the whole elapsed time. Keep an
    # absolute ceiling as well, sized so that even a heavily loaded machine has
    # room: one iteration of the largest of these (10 ACO ants, each a decode
    # of about a millisecond on this instance) is milliseconds, so seconds of
    # overshoot means an inner loop is genuinely not looking at the clock.
    assert elapsed < budget + 3.0
    assert result.seconds <= elapsed + 1e-6
    assert result.iterations >= 1


@pytest.mark.parametrize("name", ALGO_NAMES)
def test_respects_an_evaluation_budget(instance, name):
    result = build(name, instance,
                   stop=StopCriteria(max_iterations=10**9, max_evaluations=200),
                   seed=3).solve()
    # Checked between iterations, so one iteration's worth of overshoot is
    # allowed; the point is that the run terminates on the counter at all.
    assert result.evaluations >= 200
    assert result.iterations >= 1


def test_all_algorithms_share_one_objective(instance):
    """Every solver must price a given route set identically.

    This is the assumption the whole comparison rests on. If two algorithms
    disagreed about what a solution costs, their gap columns would not be
    comparable no matter how carefully the budgets were matched.
    """
    # Pin ONE fixed route set that every solver must price identically. Asking
    # each solver to price its own output instead would be a tautology:
    # make_solution is deterministic, so pricing the same routes twice can
    # never disagree, and the loop would assert nothing.
    #
    # The probe has to be feasible. On a feasible route set every penalty term
    # is zero, so the decoders agree if and only if they share a cost matrix
    # and a vehicle cost -- which is the property under test. An infeasible
    # probe would instead measure whether they share *penalty weights*, and
    # they deliberately do not: qpso.py still hard-codes 1000.0 while the
    # baselines inherit the decoder's instance-scaled defaults.
    probe = build("ga", instance, stop=StopCriteria(max_iterations=3),
                  seed=4).solve().best.routes
    assert instance.evaluate(probe).total_violation <= 1e-9, "probe must be feasible"
    reference = instance.make_solution(probe).cost

    for name in ALGO_NAMES:
        solver = build(name, instance, stop=StopCriteria(max_iterations=3), seed=4)
        # Every solver reaches the objective through its own Decoder, so ask
        # that decoder directly. A solver that quietly used a different cost
        # matrix or vehicle cost shows up right here.
        assert solver.decoder.evaluate_routes(probe) == pytest.approx(reference)

        result = solver.solve()
        # ... and the cost it reports must come from the reference evaluator,
        # not from the penalised objective it minimises internally.
        assert result.best_cost == pytest.approx(
            instance.make_solution(result.best.routes).cost)


# ----------------------------------------------------------- adaptive penalty
def test_penalty_initial_values_scale_with_the_instance(instance):
    from qroute.algorithms.decoder import Decoder

    ap = AdaptivePenalty(instance)
    # The starting weight is the decoder's own instance-scaled default, so the
    # adaptive scheme begins exactly where a fixed-penalty run would.
    assert ap.capacity == pytest.approx(Decoder.default_capacity_penalty(instance))
    # ... and that default is derived from the instance's own units rather than
    # being a constant, so it is of the order of the cost-to-demand exchange rate.
    rate = instance.cost_matrix.max() / instance.demand.max()
    assert rate <= ap.capacity <= 100.0 * rate
    # The same must hold for the two time-unit constraints, and for the same
    # reason: a controller that starts two orders of magnitude below the
    # decoder's own weight spends its whole budget climbing back to the
    # starting line. Compare against a real Decoder rather than a literal so
    # this fails if either side is retuned independently.
    dec = Decoder(instance)
    assert ap.time_window == pytest.approx(dec.pen_tw)
    assert ap.duration == pytest.approx(dec.pen_dur)
    assert ap.time_window > 1.0        # instance-scaled, not the neutral default
    assert set(ap.as_dict()) == set(CONSTRAINTS)


def test_penalty_without_an_instance_stays_neutral():
    """With no instance there is nothing to scale to, so all three start at 1."""
    ap = AdaptivePenalty(None)
    assert (ap.capacity, ap.time_window, ap.duration) == (1.0, 1.0, 1.0)


def test_penalty_rises_when_nothing_is_feasible():
    ap = AdaptivePenalty(None, interval=10, target=0.2)
    before = ap.capacity
    for _ in range(10):
        ap.register(SolutionStats(capacity_violation=5.0))
    assert ap.capacity == pytest.approx(before * 1.2)
    assert len(ap.history) == 1
    assert ap.history[0].feasible_capacity == 0.0


def test_penalty_falls_when_everything_is_feasible():
    ap = AdaptivePenalty(None, interval=10, target=0.2)
    before = ap.capacity
    for _ in range(10):
        ap.register(SolutionStats())
    assert ap.capacity == pytest.approx(before * 0.85)


def test_penalty_holds_inside_the_dead_band():
    ap = AdaptivePenalty(None, interval=10, target=0.2, tolerance=0.05)
    before = ap.as_dict()
    for i in range(10):
        ap.register(SolutionStats(capacity_violation=0.0 if i < 2 else 3.0))
    # Capacity was feasible in exactly 20% of the window, which is on target, so
    # its weight is untouched. The other two constraints were never violated at
    # all, so theirs are relaxed - each constraint is controlled independently.
    assert ap.capacity == pytest.approx(before["capacity"])
    assert ap.time_window == pytest.approx(before["time_window"] * 0.85)
    assert len(ap.history) == 1            # an unchanged window is still logged


def test_penalty_respects_its_bounds():
    ap = AdaptivePenalty(None, interval=1, floor=0.5, ceiling=2.0, capacity=1.0)
    for _ in range(50):
        ap.register(SolutionStats(capacity_violation=1.0))
    assert ap.capacity == 2.0
    for _ in range(50):
        ap.register(SolutionStats())
    assert ap.capacity == 0.5


def test_penalty_accepts_several_evidence_shapes():
    ap = AdaptivePenalty(None, interval=4)
    ap.register(SolutionStats(time_window_violation=1.0))
    ap.register({"capacity": 0.0, "time_window": 2.0})
    ap.register({"capacity_violation": 3.0})
    ap.register([True, False, True])
    assert ap.total_registered == 4
    # Only the third registration left the time window unmentioned, and an
    # absent constraint counts as satisfied.
    assert ap.history[0].feasible_time_window == pytest.approx(0.25)
    assert ap.history[0].feasible_capacity == pytest.approx(0.75)
    with pytest.raises(TypeError):
        ap.register(object())


def test_penalty_applies_to_a_decoder(instance):
    from qroute.algorithms.decoder import Decoder

    dec = Decoder(instance)
    ap = AdaptivePenalty(instance, capacity=7.0, time_window=8.0, duration=9.0)
    ap.apply_to(dec)
    assert (dec.pen_cap, dec.pen_tw, dec.pen_dur) == (7.0, 8.0, 9.0)


def test_penalty_reset_restores_the_starting_point():
    ap = AdaptivePenalty(None, interval=2, capacity=1.0)
    for _ in range(6):
        ap.register(SolutionStats(capacity_violation=1.0))
    assert ap.capacity > 1.0 and ap.history
    ap.reset()
    assert ap.capacity == 1.0
    assert ap.history == [] and ap.total_registered == 0


def test_penalty_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        AdaptivePenalty(None, target=0.0)
    with pytest.raises(ValueError):
        AdaptivePenalty(None, interval=0)
    with pytest.raises(ValueError):
        AdaptivePenalty(None, floor=10.0, ceiling=1.0)
    with pytest.raises(ValueError):
        AdaptivePenalty(None, increase=0.9)


# ------------------------------------------------------- algorithm-specific
def test_pso_ring_topology_covers_the_swarm():
    from qroute.algorithms.pso import PSO

    inst = load("A-n32-k5")
    pso = PSO(inst, neighbourhood=2, swarm_size=8)
    idx = pso._ring_index(8)
    assert idx.shape == (8, 5)
    assert set(idx[0].tolist()) == {6, 7, 0, 1, 2}
    assert (idx >= 0).all() and (idx < 8).all()


def test_ga_order_crossover_produces_a_permutation():
    from qroute.algorithms.ga import GeneticAlgorithm

    rng = np.random.default_rng(0)
    pa = rng.permutation(20)
    pb = rng.permutation(20)
    for _ in range(50):
        child = GeneticAlgorithm._order_crossover(rng, pa, pb)
        assert sorted(child.tolist()) == list(range(20))


def test_ga_canonical_keys_reproduce_their_permutation():
    from qroute.algorithms.ga import GeneticAlgorithm

    rng = np.random.default_rng(1)
    perm = rng.permutation(30)
    keys = GeneticAlgorithm._canonical(perm)
    assert np.array_equal(np.argsort(keys, kind="stable"), perm)


def test_sa_moves_preserve_the_customer_set(instance):
    from qroute.algorithms.sa import SimulatedAnnealing

    sa = SimulatedAnnealing(instance, seed=0)
    routes = [[1, 2, 3, 4], [5, 6, 7], list(range(8, instance.n_customers + 1))]
    flat, lengths = sa._pack(routes)
    expected = sorted(range(1, instance.n_customers + 1))
    rng = np.random.default_rng(0)
    for _ in range(400):
        cand = sa._propose(rng, flat, lengths)
        if cand is None:
            continue
        new_flat, new_lengths = cand
        assert sorted(new_flat.tolist()) == expected
        assert int(new_lengths.sum()) == new_flat.shape[0]
        flat, lengths = new_flat, new_lengths


def test_sa_calibrated_temperature_hits_its_acceptance_target(instance):
    """The calibrated T0 should accept roughly ``target`` of average uphill moves."""
    from qroute.algorithms.sa import SimulatedAnnealing

    sa = SimulatedAnnealing(instance, seed=0, target_acceptance=0.4,
                            calibration_moves=400)
    routes, _, _ = sa.decoder.decode(np.random.default_rng(0).random(instance.n_customers))
    flat, lengths = sa._pack(routes)
    cost = sa._cost(flat, lengths)
    t0 = sa._calibrate(np.random.default_rng(0), flat.copy(), lengths.copy(), cost)
    assert t0 > 0.0

    # Measure the realised acceptance rate of uphill moves at that temperature.
    rng = np.random.default_rng(5)
    uphill, accepted = 0, 0
    cur_flat, cur_lengths, cur_cost = flat.copy(), lengths.copy(), cost
    for _ in range(800):
        cand = sa._propose(rng, cur_flat, cur_lengths)
        if cand is None:
            continue
        c = sa._cost(*cand)
        if c > cur_cost:
            uphill += 1
            if rng.random() < np.exp(-(c - cur_cost) / t0):
                accepted += 1
                cur_flat, cur_lengths, cur_cost = cand[0], cand[1], c
        else:
            cur_flat, cur_lengths, cur_cost = cand[0], cand[1], c
    assert uphill > 50
    rate = accepted / uphill
    # A generous band: the estimator matches the target at the *mean* uphill
    # step, and the distribution of steps is skewed, so exact agreement is not
    # expected or claimed.
    assert 0.15 < rate < 0.75


def test_aco_pheromone_stays_within_max_min_bounds(instance):
    from qroute.algorithms.aco import AntColony

    aco = AntColony(instance, seed=0, rho=0.1)
    tau = np.full((instance.size, instance.size), 0.5)
    routes = [[1, 2, 3], list(range(4, instance.n_customers + 1))]
    aco._global_update(tau, routes, best_cost=1000.0, symmetric=True)
    tau_max = 1.0 / (0.1 * 1000.0)
    tau_min = tau_max * 0.5 / instance.n_customers
    assert tau.min() >= tau_min - 1e-12
    assert tau.max() <= tau_max + 1e-12


def test_aco_is_deterministic_despite_the_numba_kernel(instance):
    """Randomness is drawn from the seeded generator and passed into the kernel."""
    from qroute.algorithms.aco import AntColony

    stop = StopCriteria(max_iterations=4)
    a = AntColony(instance, stop=stop, seed=99).solve()
    b = AntColony(instance, stop=stop, seed=99).solve()
    assert [h.best_cost for h in a.history] == [h.best_cost for h in b.history]
