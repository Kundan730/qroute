"""Tests for the quantum rotation gate and the two QIEA engines built on it.

The gate is a small piece of numerics that everything else in this module
depends on, so it is tested on its own properties - unitarity, direction of the
probability shift, and the anti-collapse guarantee - rather than only through
the optimisers. The optimiser tests then check the two things that are easy to
get wrong in a metaheuristic and impossible to spot from the objective value:
the returned solution really is a valid permutation of all customers, and a
fixed seed really does reproduce the run.
"""

from __future__ import annotations

import numpy as np
import pytest

from qroute.algorithms.base import StopCriteria
from qroute.algorithms.qiea import QIEA, QuantumRotationKeys
from qroute.algorithms.qtypes import QubitRegister
from qroute.problems.loaders import load

PI = np.pi


# --------------------------------------------------------------------------
# QubitRegister
# --------------------------------------------------------------------------
def test_initial_state_is_equal_superposition():
    reg = QubitRegister(16)
    assert np.allclose(reg.probabilities(), 0.5)
    assert reg.entropy() == pytest.approx(1.0)
    assert np.allclose(reg.alpha, 1.0 / np.sqrt(2.0))


def test_amplitudes_stay_normalised_under_many_rotations():
    """The gate is orthogonal, so |alpha|^2 + |beta|^2 must never drift."""
    rng = np.random.default_rng(7)
    reg = QubitRegister(64)
    for _ in range(2000):
        x = reg.observe(rng)
        b = (rng.random(64) < 0.5).astype(np.uint8)
        reg.rotate(x, b, bool(rng.random() < 0.5), 0.03 * PI, rng)
        norm = reg.alpha ** 2 + reg.beta ** 2
        assert np.abs(norm - 1.0).max() < 1e-12
    assert np.isfinite(reg.probabilities()).all()


def test_rotation_moves_probability_toward_the_better_bit():
    """Deterministic, single-step check of the four-quadrant sign rule.

    A register is nudged once toward a target string; the observation
    probability must move up where the target bit is 1 and down where it is 0.
    """
    reg = QubitRegister(8)
    p0 = reg.probabilities().copy()
    x = np.zeros(8, dtype=np.uint8)
    target = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    # x disagrees with the target on the first four positions and agrees on the
    # last four, so both the disagreeing and the agreeing rows are exercised.
    reg.rotate(x, target, better=False, delta_theta=0.02 * PI)
    p1 = reg.probabilities()
    assert (p1[:4] > p0[:4] + 1e-6).all()
    assert (p1[4:] < p0[4:] - 1e-6).all()


def test_rotation_toward_target_raises_observed_frequency():
    """Statistical version of the same claim, with a fixed seed.

    Sample the register before and after a run of rotations toward a fixed
    target and compare the empirical frequency of the target bit.
    """
    rng = np.random.default_rng(2024)
    m = 32
    reg = QubitRegister(m)
    target = (np.random.default_rng(11).random(m) < 0.5).astype(np.uint8)

    def agreement(register: QubitRegister, draws: int, gen) -> float:
        hits = 0
        for _ in range(draws):
            hits += int((register.observe(gen) == target).sum())
        return hits / (draws * m)

    before = agreement(reg, 400, rng)
    for _ in range(60):
        x = reg.observe(rng)
        reg.rotate(x, target, better=False, delta_theta=0.01 * PI, rng=rng)
    after = agreement(reg, 400, rng)

    assert before == pytest.approx(0.5, abs=0.05)
    assert after > before + 0.25


def test_rotation_sign_is_correct_in_every_quadrant():
    """The sign rule must work whatever the signs of alpha and beta are.

    Rotations can carry an amplitude negative; a rule that only handles the
    first quadrant silently pushes probability the wrong way once that happens.
    """
    root = 1.0 / np.sqrt(2.0)
    for sa in (+1.0, -1.0):
        for sb in (+1.0, -1.0):
            for bit in (0, 1):
                reg = QubitRegister(1, alpha=np.array([sa * root]),
                                    beta=np.array([sb * root]))
                p0 = float(reg.probabilities()[0])
                reg.rotate(np.array([1 - bit], dtype=np.uint8),
                           np.array([bit], dtype=np.uint8),
                           better=False, delta_theta=0.02 * PI)
                p1 = float(reg.probabilities()[0])
                if bit == 1:
                    assert p1 > p0, f"alpha sign {sa}, beta sign {sb}"
                else:
                    assert p1 < p0, f"alpha sign {sa}, beta sign {sb}"


def test_repeated_rotation_pushes_a_qubit_against_the_pole():
    """Sustained one-sided pressure saturates a register that is not clamped.

    With a fixed step the state cannot land exactly on the pole - it settles
    into a small limit cycle just short of it - so the assertion is that the
    probability gets close, not that it reaches 1.
    """
    m = 16
    reg = QubitRegister(m)
    zeros = np.zeros(m, dtype=np.uint8)
    ones = np.ones(m, dtype=np.uint8)
    for _ in range(200):
        reg.rotate(zeros, ones, better=False, delta_theta=0.01 * PI)
    assert reg.probabilities().min() > 0.97


def test_h_epsilon_prevents_collapse():
    """A qubit that reaches a pole is dead; H-epsilon is what revives it.

    Exactly at ``alpha = 0`` the state is ``|1>``: observation can never return
    0, and the gate's own degenerate row gives a zero angle because the qubit
    already sits on the target bit. Nothing in the algorithm can recover that
    degree of freedom - which is precisely the failure H-epsilon exists to
    prevent.
    """
    rng = np.random.default_rng(3)
    m = 24
    dead = QubitRegister(m, alpha=np.zeros(m), beta=np.ones(m))
    assert dead.probabilities().min() == 1.0
    assert sum(int((dead.observe(rng) == 0).sum()) for _ in range(500)) == 0

    # Rotation alone cannot bring it back while the target bit stays 1.
    dead.rotate(np.zeros(m, np.uint8), np.ones(m, np.uint8), False, 0.05 * PI)
    assert dead.probabilities().min() == 1.0

    dead.h_epsilon(0.01)
    assert np.allclose(dead.probabilities(), 0.99)
    assert sum(int((dead.observe(rng) == 0).sum()) for _ in range(500)) > 0

    # And the clamp holds under continued pressure, generation after generation.
    for _ in range(500):
        dead.rotate(np.zeros(m, np.uint8), np.ones(m, np.uint8), False, 0.05 * PI, rng)
        dead.h_epsilon(0.01)
    assert np.allclose(dead.probabilities(), 0.99)


def test_h_epsilon_preserves_amplitude_signs_and_norm():
    reg = QubitRegister(4, alpha=np.array([0.6, -0.6, 0.0, -1.0]),
                        beta=np.array([0.8, 0.8, 1.0, 0.0]))
    signs_a = np.sign(reg.alpha)
    reg.h_epsilon(0.05)
    assert np.allclose(reg.alpha ** 2 + reg.beta ** 2, 1.0)
    assert reg.probabilities().min() >= 0.05 - 1e-12
    assert reg.probabilities().max() <= 0.95 + 1e-12
    # A sign that was already negative must stay negative.
    assert np.sign(reg.alpha[1]) == signs_a[1]


def test_han_kim_table_does_not_reinforce_agreeing_zeros():
    """Document the asymmetry of the literal 2002 table via a test.

    Under ``table="han_kim"`` an agreeing 0-bit gets a zero-magnitude rotation,
    which is why ``"symmetric"`` is the default for numeric genotypes.
    """
    zero = np.zeros(4, dtype=np.uint8)
    hk = QubitRegister(4, table="han_kim")
    sym = QubitRegister(4, table="symmetric")
    hk.rotate(zero, zero, False, 0.02 * PI)
    sym.rotate(zero, zero, False, 0.02 * PI)
    assert np.allclose(hk.probabilities(), 0.5)
    assert (sym.probabilities() < 0.5 - 1e-6).all()


def test_entropy_falls_as_the_register_converges():
    rng = np.random.default_rng(5)
    reg = QubitRegister(32)
    target = (rng.random(32) < 0.5).astype(np.uint8)
    start = reg.entropy()
    for _ in range(100):
        reg.rotate(reg.observe(rng), target, False, 0.02 * PI, rng)
        reg.h_epsilon(0.01)
    assert reg.entropy() < start - 0.5


def test_register_rejects_bad_arguments():
    with pytest.raises(ValueError):
        QubitRegister(0)
    with pytest.raises(ValueError):
        QubitRegister(4, table="nonsense")
    with pytest.raises(ValueError):
        QubitRegister(4).h_epsilon(0.9)
    with pytest.raises(ValueError):
        QubitRegister(4).rotate(np.zeros(3, np.uint8), np.zeros(3, np.uint8), True)


def test_copy_is_independent():
    reg = QubitRegister(8)
    clone = reg.copy()
    reg.rotate(np.zeros(8, np.uint8), np.ones(8, np.uint8), False, 0.05 * PI)
    assert np.allclose(clone.probabilities(), 0.5)
    assert not np.allclose(reg.probabilities(), 0.5)


# --------------------------------------------------------------------------
# Optimisers
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_instance():
    return load("A-n32-k5")


@pytest.mark.parametrize("cls", [QIEA, QuantumRotationKeys])
def test_optimiser_returns_a_valid_solution(cls, small_instance):
    res = cls(small_instance, StopCriteria(max_iterations=8),
              seed=3, population_size=6).solve()
    res.best.validate(small_instance.n_customers)
    assert res.best.is_feasible
    assert res.best_cost >= small_instance.meta["bks"] - 1e-6
    assert res.iterations == 8
    assert len(res.history) == 8
    # The history must be monotone in best cost: the incumbent never worsens.
    curve = [h.best_cost for h in res.history]
    assert all(b <= a + 1e-9 for a, b in zip(curve, curve[1:]))


@pytest.mark.parametrize("cls", [QIEA, QuantumRotationKeys])
def test_same_seed_reproduces_the_run(cls, small_instance):
    kw = dict(population_size=6)
    a = cls(small_instance, StopCriteria(max_iterations=10), seed=42, **kw).solve()
    b = cls(small_instance, StopCriteria(max_iterations=10), seed=42, **kw).solve()
    assert a.best_cost == b.best_cost
    assert a.best.routes == b.best.routes
    assert [h.best_cost for h in a.history] == [h.best_cost for h in b.history]
    assert [h.mean_cost for h in a.history] == [h.mean_cost for h in b.history]


@pytest.mark.parametrize("cls", [QIEA, QuantumRotationKeys])
def test_different_seeds_explore_differently(cls, small_instance):
    kw = dict(population_size=6)
    a = cls(small_instance, StopCriteria(max_iterations=6), seed=1, **kw).solve()
    b = cls(small_instance, StopCriteria(max_iterations=6), seed=2, **kw).solve()
    assert [h.mean_cost for h in a.history] != [h.mean_cost for h in b.history]


def test_qiea_construction_visits_every_customer_exactly_once(small_instance):
    """The mask-guided construction must be a permutation for any mask.

    Including the two extremes: an all-zero mask (pure nearest-neighbour
    fallback) and an all-one mask (always take the nearest neighbour that is
    still unvisited).
    """
    opt = QIEA(small_instance, StopCriteria(max_iterations=1), seed=0)
    n = small_instance.n_customers
    rng = np.random.default_rng(0)
    masks = [
        np.zeros((opt.size, opt.k), dtype=bool),
        np.ones((opt.size, opt.k), dtype=bool),
        rng.random((opt.size, opt.k)) < 0.5,
    ]
    for mask in masks:
        tour = opt._construct(mask)
        assert sorted(tour.tolist()) == list(range(1, n + 1))


def test_quantum_rotation_keys_bit_encoding_round_trips(small_instance):
    """Bits -> key -> bits must be the identity at the register's resolution."""
    opt = QuantumRotationKeys(small_instance, StopCriteria(max_iterations=1),
                              seed=0, bits=10)
    rng = np.random.default_rng(9)
    bits = (rng.random(opt._register_size()) < 0.5).astype(np.uint8)
    keys = opt._keys_from_bits(bits)
    assert keys.shape == (small_instance.n_customers,)
    assert (keys >= 0.0).all() and (keys < 1.0).all()
    back = opt._target_bits(bits, keys, [])
    assert np.array_equal(back, bits)


def test_quantum_rotation_keys_low_bit_depth_causes_key_collisions(small_instance):
    """Make the documented bit-depth trade-off visible rather than asserted.

    With 3 bits there are only 8 distinct key levels for 31 customers, so ties
    are unavoidable and the ordering is largely decided by the argsort's stable
    tie-break rather than by the search.
    """
    opt = QuantumRotationKeys(small_instance, StopCriteria(max_iterations=1),
                              seed=0, bits=3)
    rng = np.random.default_rng(4)
    bits = (rng.random(opt._register_size()) < 0.5).astype(np.uint8)
    keys = opt._keys_from_bits(bits)
    assert len(np.unique(keys)) <= 8 < small_instance.n_customers


def test_migration_periods_are_respected(small_instance):
    """A global migration must make every reference cost identical."""
    opt = QuantumRotationKeys(small_instance, StopCriteria(max_iterations=1),
                              seed=0, population_size=8)
    best_bits = [np.zeros(4, dtype=np.uint8) for _ in range(8)]
    best_bits[3][:] = 1
    best_cost = np.arange(8, dtype=float) + 10.0
    best_cost[3] = 1.0
    routes = [[[1]] for _ in range(8)]

    opt.global_period = 2
    opt.local_period = 0                              # isolate global migration
    opt._migrate(1, best_bits, best_cost, routes)     # not a global generation
    assert best_cost[0] == 10.0

    opt._migrate(2, best_bits, best_cost, routes)     # global generation
    assert np.allclose(best_cost, 1.0)
    assert all(bits.all() for bits in best_bits)


def test_local_migration_stays_inside_its_group(small_instance):
    """Local migration must not leak the population best across group borders."""
    opt = QuantumRotationKeys(small_instance, StopCriteria(max_iterations=1),
                              seed=0, population_size=10, group_size=5)
    opt.global_period = 0
    opt.local_period = 1
    best_bits = [np.zeros(4, dtype=np.uint8) for _ in range(10)]
    best_cost = np.full(10, 100.0)
    best_cost[0] = 1.0                                # best sits in group 0
    routes = [[[1]] for _ in range(10)]
    opt._migrate(1, best_bits, best_cost, routes)
    assert np.allclose(best_cost[:5], 1.0)
    assert np.allclose(best_cost[5:], 100.0)


def test_time_budget_is_honoured(small_instance):
    res = QuantumRotationKeys(small_instance,
                              StopCriteria(max_iterations=10 ** 9, max_seconds=1.0),
                              seed=0, population_size=6).solve()
    assert 0.5 < res.seconds < 4.0
    assert res.iterations > 1
    res.best.validate(small_instance.n_customers)


def test_theta_schedules(small_instance):
    stop = StopCriteria(max_iterations=100)
    fixed = QuantumRotationKeys(small_instance, stop, seed=0)
    assert fixed.theta(1) == fixed.theta(99)
    linear = QuantumRotationKeys(small_instance, stop, seed=0,
                                 theta_schedule="linear")
    linear._t0 = 0.0
    assert linear.theta(100) < linear.theta(1)
    with pytest.raises(ValueError):
        QuantumRotationKeys(small_instance, stop, seed=0, theta_schedule="bogus")


def test_solomon_instance_with_time_windows():
    """The engines must handle VRPTW, not only pure CVRP."""
    inst = load("C101")
    res = QuantumRotationKeys(inst, StopCriteria(max_iterations=5), seed=0,
                              population_size=6).solve()
    res.best.validate(inst.n_customers)
    assert np.isfinite(res.best_cost)


# --------------------------------------------------------------------------
# Adversarial verification pass: claims the docstrings make, turned into tests
# --------------------------------------------------------------------------
def _applied_angle_multiple(table: str, x_bit: int, b_bit: int, better: bool,
                            phase: float = 0.7) -> float:
    """Rotation actually applied, in multiples of ``delta_theta``.

    Reads the angle straight off the amplitudes with ``arctan2`` rather than
    inferring it from a probability, so the assertion is on the gate's own
    parameter and not on a monotone function of it.
    """
    delta = 0.01 * PI
    reg = QubitRegister(1, table=table,
                        alpha=np.array([np.cos(phase)]),
                        beta=np.array([np.sin(phase)]))
    reg.rotate(np.array([x_bit], np.uint8), np.array([b_bit], np.uint8),
               better, delta)
    return float(np.arctan2(reg.beta[0], reg.alpha[0]) - phase) / delta


def test_rotation_table_matches_the_documented_magnitudes_exactly():
    """Every one of the eight table rows, in both tables, to the multiple.

    The docstring publishes a lookup table and says the sign column is derived
    rather than copied. That makes the table a claim about behaviour, so it is
    checked as one: 2.5 for agreement, 5.0 to move onto a better reference's
    bit, 1.0 to hold a better observation's own bit, 0.0 for an agreeing zero
    under the literal 2002 table.
    """
    expected = {
        # (x_i, b_i, x better): (symmetric, han_kim)
        (0, 0, False): (-2.5, 0.0),
        (0, 0, True): (-2.5, 0.0),
        (0, 1, False): (+5.0, +5.0),
        (0, 1, True): (-1.0, -1.0),
        (1, 0, False): (-5.0, -5.0),
        (1, 0, True): (+1.0, +1.0),
        (1, 1, False): (+2.5, +2.5),
        (1, 1, True): (+2.5, +2.5),
    }
    for (x_bit, b_bit, better), (sym, hk) in expected.items():
        assert _applied_angle_multiple("symmetric", x_bit, b_bit, better) == \
            pytest.approx(sym, abs=1e-9), (x_bit, b_bit, better, "symmetric")
        assert _applied_angle_multiple("han_kim", x_bit, b_bit, better) == \
            pytest.approx(hk, abs=1e-9), (x_bit, b_bit, better, "han_kim")


def test_every_table_row_moves_probability_the_right_way_from_any_phase():
    """Exhaustive sweep of the four-quadrant sign rule.

    ``test_rotation_sign_is_correct_in_every_quadrant`` checks four amplitude
    sign combinations at one magnitude. This walks the qubit right around the
    circle, and covers all eight table rows rather than only the disagreeing
    ones.

    Phases within one step of a pole are excluded deliberately, not to make the
    test pass: a *finite* rotation applied to a qubit that is already closer to
    a pole than the step size overshoots straight past it, so the probability
    can only move the other way. That is a property of any fixed-angle gate and
    is asserted separately below, together with the two things that contain it -
    the collapsed-qubit guard and the H-epsilon clamp.
    """
    step = 5.0 * 0.001 * PI                  # the largest multiple in the table
    guard = 4.0 * step
    phases = [p for p in np.linspace(0.0, 2 * PI, 241)
              if min(abs(p - q) for q in (0.0, PI / 2, PI, 1.5 * PI, 2 * PI)) > guard]
    assert len(phases) > 150
    for phase in phases:
        for x_bit in (0, 1):
            for b_bit in (0, 1):
                for better in (False, True):
                    target = x_bit if better else b_bit
                    for table in ("symmetric", "han_kim"):
                        reg = QubitRegister(1, table=table,
                                            alpha=np.array([np.cos(phase)]),
                                            beta=np.array([np.sin(phase)]))
                        p0 = float(reg.probabilities()[0])
                        reg.rotate(np.array([x_bit], np.uint8),
                                   np.array([b_bit], np.uint8),
                                   better, 0.001 * PI)
                        dp = float(reg.probabilities()[0]) - p0
                        if table == "han_kim" and x_bit == 0 and b_bit == 0:
                            assert abs(dp) < 1e-14      # the zero-magnitude row
                        elif target == 1:
                            assert dp > 0, (phase, x_bit, b_bit, better, table)
                        else:
                            assert dp < 0, (phase, x_bit, b_bit, better, table)


def test_a_qubit_already_on_its_target_pole_is_left_alone():
    """The collapsed-qubit guard must survive a merely numerical zero.

    A qubit written as ``(cos(pi), sin(pi))`` has ``beta = 1.2e-16``, not 0. An
    exact ``alpha * beta == 0`` guard misses that state, and the gate then
    applies a full-size step to a qubit whose target probability is already 1 -
    which can only push it off its own target. Both the exactly-zero and the
    numerically-zero spellings of each pole must be no-ops.
    """
    cases = [
        # (alpha, beta, target bit) - the qubit is already certain of the target
        (1.0, 0.0, 0), (np.cos(PI), np.sin(PI), 0), (-1.0, 0.0, 0),
        (0.0, 1.0, 1), (np.cos(PI / 2), np.sin(PI / 2), 1), (0.0, -1.0, 1),
    ]
    for a, b, target in cases:
        reg = QubitRegister(1, alpha=np.array([a]), beta=np.array([b]))
        p0 = float(reg.probabilities()[0])
        bit = np.array([target], np.uint8)
        reg.rotate(bit, bit, False, 0.05 * PI)
        assert float(reg.probabilities()[0]) == pytest.approx(p0, abs=1e-15), (a, b, target)


def test_a_collapsed_qubit_pointed_at_the_wrong_bit_is_rescued():
    """The other degenerate row: on a pole, but on the *wrong* pole.

    Here either rotation direction helps, so the gate must pick one and move,
    rather than freezing the way it does when the qubit is already correct.
    """
    rng = np.random.default_rng(0)
    for a, b, target in [(1.0, 0.0, 1), (np.cos(PI), np.sin(PI), 1),
                         (0.0, 1.0, 0), (np.cos(PI / 2), np.sin(PI / 2), 0)]:
        reg = QubitRegister(1, alpha=np.array([a]), beta=np.array([b]))
        bit = np.array([target], np.uint8)
        reg.rotate(bit, bit, False, 0.05 * PI, rng)
        p = float(reg.probabilities()[0])
        assert (p > 1e-6 if target == 1 else p < 1.0 - 1e-6), (a, b, target)


def test_h_epsilon_keeps_the_register_out_of_the_overshoot_region():
    """Why the finite-step overshoot never matters in a real run.

    The largest table entry is ``5 * delta_theta``. A qubit can only be carried
    past a pole if it sits closer to that pole than the step, and H-epsilon
    holds every qubit at least ``arcsin(sqrt(eps))`` away from one. With the
    module defaults those two numbers are within a factor of two of each other,
    so the overshoot *can* happen - and when it does, the clamp simply puts the
    qubit back at the boundary on the next call, which is exactly the intended
    saturation behaviour rather than a lost degree of freedom.
    """
    eps = 0.01
    reg = QubitRegister(1, alpha=np.array([np.sqrt(1 - eps)]),
                        beta=np.array([np.sqrt(eps)]))
    for _ in range(50):
        # Sustained pressure toward 0 on a qubit already parked at the floor.
        reg.rotate(np.zeros(1, np.uint8), np.zeros(1, np.uint8), False, 0.01 * PI)
        reg.h_epsilon(eps)
    p = float(reg.probabilities()[0])
    assert p == pytest.approx(eps, abs=1e-12)
    # The degree of freedom is still alive: the other bit is still observable.
    rng = np.random.default_rng(0)
    assert sum(int(reg.observe(rng)[0]) for _ in range(2000)) > 0


def test_han_kim_table_drifts_a_register_toward_all_ones():
    """The stated reason ``symmetric`` is the default, measured.

    Under an unbiased reference string the literal 2002 table reinforces
    agreeing ones but not agreeing zeros, so a register with no fitness signal
    at all still drifts upward. ``symmetric`` must stay put.
    """
    drift = {}
    for table in ("han_kim", "symmetric"):
        rng = np.random.default_rng(0)
        reg = QubitRegister(2000, table=table)
        for _ in range(300):
            x = reg.observe(rng)
            b = (rng.random(2000) < 0.5).astype(np.uint8)   # carries no signal
            reg.rotate(x, b, False, 0.01 * PI, rng)
            reg.h_epsilon(0.01)
        drift[table] = float(reg.probabilities().mean())
    assert drift["han_kim"] > 0.60
    assert drift["symmetric"] == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("cls", [QIEA, QuantumRotationKeys])
def test_gate_off_control_never_leaves_equal_superposition(cls, small_instance):
    """``delta_theta=0`` must be a genuine no-rotation control.

    The module docstring's ablation compares against this configuration, so the
    control has to be real: with a zero angle every register must stay at
    exactly one bit of entropy for the whole run, meaning the search degenerates
    to repeated uniform sampling and nothing is learned.
    """
    res = cls(small_instance, StopCriteria(max_iterations=12), seed=5,
              population_size=6, delta_theta=0.0).solve()
    assert [h.diversity for h in res.history] == [1.0] * 12
    res.best.validate(small_instance.n_customers)


@pytest.mark.parametrize("cls", [QIEA, QuantumRotationKeys])
def test_gate_on_converges_registers_to_the_h_epsilon_floor(cls, small_instance):
    """With rotation enabled the registers do move, and they saturate.

    This is the other half of the control: entropy must fall well below one bit.
    It also documents where it stops - the binary entropy of ``eps`` itself, so
    a "converged" register is pinned against the clamp rather than sitting
    somewhere informative in between.
    """
    res = cls(small_instance, StopCriteria(max_iterations=30), seed=5,
              population_size=6).solve()
    eps = 0.01
    floor = -(eps * np.log2(eps) + (1 - eps) * np.log2(1 - eps))
    assert res.history[0].diversity > 0.9
    assert res.history[-1].diversity < 0.5
    assert res.history[-1].diversity == pytest.approx(floor, abs=0.15)


def test_history_feasibility_is_measured_not_assumed(small_instance):
    """The ``feasible`` column must be able to say False.

    A convergence log that hard-codes True is worse than no column at all. An
    impossible route-duration cap makes every solution infeasible, so every row
    must report it, and the returned solution must own up to it too.
    """
    import dataclasses

    impossible = dataclasses.replace(small_instance, max_route_duration=1.0,
                                     meta=dict(small_instance.meta))
    res = QuantumRotationKeys(impossible, StopCriteria(max_iterations=4), seed=0,
                              population_size=4).solve()
    assert [h.feasible for h in res.history] == [False] * 4
    assert all(isinstance(h.feasible, bool) for h in res.history)
    assert not res.best.is_feasible
    assert impossible.evaluate(res.best.routes).total_violation > 0.0
    res.best.validate(impossible.n_customers)

    # ... and the same optimiser on the unmodified instance says True, so the
    # column is not stuck in either position.
    ok = QuantumRotationKeys(small_instance, StopCriteria(max_iterations=4),
                             seed=0, population_size=4).solve()
    assert all(h.feasible for h in ok.history)


def test_qiea_slot_lookup_matches_a_dense_neighbour_matrix(small_instance):
    """Guard the sparse slot table against the obvious dense implementation.

    ``_slot`` is a flat dict rather than a ``(size, size)`` matrix purely to
    keep memory linear in the node count. The two must agree everywhere,
    including on the pairs that are absent from any neighbour list.
    """
    opt = QIEA(small_instance, StopCriteria(max_iterations=1), seed=0)
    dense = np.full((opt.size, opt.size), -1, dtype=np.int32)
    rows = np.repeat(np.arange(opt.size), opt.k)
    dense[rows, opt.neigh.ravel()] = np.tile(np.arange(opt.k), opt.size)
    for u in range(opt.size):
        for v in range(opt.size):
            assert opt._slot.get(u * opt.size + v, -1) == dense[u, v]
    assert len(opt._slot) <= opt.size * opt.k


def test_qiea_lamarckian_target_encodes_the_solution_not_the_draw(small_instance):
    """The Lamarckian write-back must actually rewrite the rotation target.

    A write-back that quietly returned the observation would leave the default
    configuration identical to ``lamarckian=False`` while the docstring claimed
    a large measured difference for it, so the difference is asserted directly:
    the target mask must be sparse (about one arc per node) and must mark the
    arcs of the decoded routes.
    """
    opt = QIEA(small_instance, StopCriteria(max_iterations=1), seed=0)
    rng = np.random.default_rng(1)
    bits = (rng.random(opt._register_size()) < 0.5).astype(np.uint8)
    routes, _cost, new_keys = opt.decoder.decode(opt._keys_from_bits(bits))
    target = opt._target_bits(bits, new_keys, routes)

    assert not np.array_equal(target, bits)
    assert target.sum() < bits.sum() / 2          # sparse, as documented
    mask = target.reshape(opt.size, opt.k)
    for route in routes:
        prev = 0
        for customer in route:
            slot = opt._slot.get(prev * opt.size + customer, -1)
            if slot >= 0:
                assert mask[prev, slot] == 1
            prev = customer


@pytest.mark.parametrize("cls", [QIEA, QuantumRotationKeys])
def test_result_params_record_the_decoder_configuration(cls, small_instance):
    """A benchmark row has to say what produced it.

    The penalty weights decide how far outside the feasible region the search
    may look, and this module's default (``None``, meaning the decoder scales
    them from the instance) differs from ``qpso.py``'s hard-coded 1000.0. A
    comparison between the two is only interpretable if each result carries its
    own setting, so the settings must survive into ``OptimizationResult.params``.
    """
    res = cls(small_instance, StopCriteria(max_iterations=2), seed=0,
              population_size=4, penalty_capacity=250.0).solve()
    for key in ("population_size", "delta_theta", "rotation_table", "lamarckian",
                "epsilon", "neighbours", "local_search_rounds",
                "penalty_capacity", "penalty_time_window", "penalty_duration",
                "vehicle_cost"):
        assert key in res.params, key
    assert res.params["penalty_capacity"] == 250.0
    assert res.params["penalty_time_window"] is None
