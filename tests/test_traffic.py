"""Tests for the traffic and congestion layer.

The tests are written against the *claims* made in the module docstrings, so
that documentation and behaviour cannot drift apart: if someone retunes the
hourly profile and moves the evening peak to 17:00, the test that asserts the
peak lies inside 18:00-20:00 fails and the docstring gets corrected with it.

No test contacts a live traffic API. The TomTom source is exercised only for
its no-key behaviour and, through a stubbed request method, for its parsing.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from qroute.traffic import bpr, events, profiles, sources
from qroute.traffic.simulator import TrafficSimulator, edge_arrays_from_network


# --------------------------------------------------------------- fixtures
def make_toy_graph() -> nx.MultiDiGraph:
    """A small hand-built road network with a known structure.

    Six directed edges spanning four road classes and two lane counts, with
    free-flow times chosen to be round numbers so expected values can be
    written down by hand. Used instead of the OSM extract wherever the test
    does not specifically need a real city.
    """
    g = nx.MultiDiGraph()
    spec = [
        (1, 2, "primary", 2, 500.0, 50.0),
        (2, 3, "primary", 3, 800.0, 80.0),
        (3, 4, "secondary", 2, 400.0, 40.0),
        (4, 1, "residential", 1, 200.0, 30.0),
        (2, 4, "tertiary", 2, 300.0, 30.0),
        (4, 2, "living_street", 1, 300.0, 45.0),
    ]
    for u, v, hw, lanes, length, tt in spec:
        g.add_edge(u, v, key=0, highway=hw, lanes=str(lanes), length=length, travel_time=tt)
    return g


@pytest.fixture
def toy_sim() -> TrafficSimulator:
    return TrafficSimulator(make_toy_graph(), seed=7, start_minute=10 * 60.0)


@pytest.fixture(scope="module")
def bengaluru():
    """The real OSM extract, skipped if the data file is not present."""
    osmnx = pytest.importorskip("osmnx", reason="osmnx is needed to load the OSM extract")
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "osm" / "bengaluru_koramangala.graphml"
    if not path.exists():
        pytest.skip(f"OSM extract not found at {path}")
    return osmnx.load_graphml(path)


# ============================================================ volume-delay
class TestBPR:
    def test_free_flow_at_zero_volume(self):
        """t(0) must equal t0 exactly, for both volume-delay functions."""
        assert bpr.bpr_multiplier(0.0) == pytest.approx(1.0, abs=1e-12)
        assert bpr.conical_multiplier(0.0) == pytest.approx(1.0, abs=1e-12)
        t0 = np.array([10.0, 25.0, 3.5])
        assert bpr.bpr_travel_time(t0, np.zeros(3)) == pytest.approx(t0)
        assert bpr.conical_travel_time(t0, np.zeros(3)) == pytest.approx(t0)

    def test_monotone_increasing_in_volume(self):
        x = np.linspace(0.0, 3.0, 601)
        for f in (bpr.bpr_multiplier, bpr.conical_multiplier):
            y = f(x)
            assert np.all(np.diff(y) > 0.0), f"{f.__name__} is not strictly increasing"

    def test_monotone_from_volume_and_capacity(self):
        """The textbook t(v) form, at fixed capacity, is increasing in v."""
        t = bpr.bpr_travel_time_from_volume(60.0, np.arange(0, 3000, 50.0), 1500.0)
        assert np.all(np.diff(t) > 0.0)
        assert t[0] == pytest.approx(60.0)

    def test_standard_coefficients_and_reference_values(self):
        assert (bpr.BPR_ALPHA, bpr.BPR_BETA) == (0.15, 4.0)
        # At capacity BPR adds exactly alpha.
        assert bpr.bpr_multiplier(1.0) == pytest.approx(1.15)
        # The conical function is pinned to 2.0 at capacity for every a.
        for a in (1.5, 4.0, 10.0):
            assert bpr.conical_multiplier(1.0, a) == pytest.approx(2.0)
            assert bpr.conical_multiplier(0.0, a) == pytest.approx(1.0, abs=1e-12)

    def test_conical_has_positive_derivative_at_zero_but_bpr_does_not(self):
        """The documented reason for offering the conical alternative."""
        h = 1e-6
        d_bpr = (bpr.bpr_multiplier(h) - bpr.bpr_multiplier(0.0)) / h
        d_con = (bpr.conical_multiplier(h) - bpr.conical_multiplier(0.0)) / h
        assert d_bpr < 1e-9
        assert d_con > 0.1

    def test_conical_is_linear_far_above_capacity_while_bpr_explodes(self):
        """The second documented reason for offering conical -- with the caveat.

        Conical is *steeper* than BPR through the near-capacity band and only
        overtakes it in deep oversaturation, around x = 3.2. Both halves of
        that statement are asserted here so the docstring table cannot rot.
        """
        assert bpr.conical_multiplier(1.0) > bpr.bpr_multiplier(1.0)
        assert bpr.conical_multiplier(2.0) > bpr.bpr_multiplier(2.0)
        crossover = 3.2
        assert bpr.conical_multiplier(crossover - 0.3) > bpr.bpr_multiplier(crossover - 0.3)
        assert bpr.conical_multiplier(crossover + 0.3) < bpr.bpr_multiplier(crossover + 0.3)
        # Deep oversaturation: BPR's quartic runs away, conical stays linear.
        assert bpr.bpr_multiplier(5.0) > 90.0
        assert bpr.conical_multiplier(5.0) < 40.0
        # Asymptotic slope is 2a, because both the square-root term and the
        # -a(1-x) term contribute a*x once x is large.
        slope = bpr.conical_multiplier(21.0) - bpr.conical_multiplier(20.0)
        assert slope == pytest.approx(2 * bpr.CONICAL_A, rel=1e-3)

    def test_conical_b_requires_a_above_one(self):
        with pytest.raises(ValueError):
            bpr.conical_b(1.0)

    def test_saturation_and_congestion_level(self):
        assert bpr.saturation_ratio(750.0, 1500.0) == pytest.approx(0.5)
        assert math.isinf(float(bpr.saturation_ratio(750.0, 0.0)))
        assert bpr.congestion_level(20.0, 10.0) == pytest.approx(1.0)
        assert bpr.congestion_level(10.0, 10.0) == pytest.approx(0.0)
        # Degenerate zero-length edges report 0 rather than inf.
        assert bpr.congestion_level(5.0, 0.0) == pytest.approx(0.0)

    def test_saturation_for_ratio_inverts_bpr(self):
        for target in (1.1, 1.75, 2.5, 4.0):
            x = bpr.saturation_for_ratio(target)
            assert bpr.bpr_multiplier(x) == pytest.approx(target)
        with pytest.raises(ValueError):
            bpr.saturation_for_ratio(0.5)

    def test_bands_are_ordered_and_cover_the_line(self):
        levels = np.array([0.0, 0.2, 0.5, 1.0, 5.0])
        idx = bpr.congestion_band(levels)
        assert list(idx) == [0, 1, 2, 3, 4]
        counts = bpr.band_counts(levels)
        assert sum(counts.values()) == len(levels)
        assert counts["free"] == 1 and counts["severe"] == 1

    def test_capacity_scales_with_lanes_and_class(self):
        cap = bpr.edge_capacity(["primary", "primary", "residential"], [1, 3, 1])
        assert cap[1] == pytest.approx(3 * cap[0])
        assert cap[0] > cap[2], "an arterial lane must carry more than a residential lane"


# ================================================================= profiles
class TestProfile:
    def test_peaks_are_where_the_docstring_claims(self):
        p = profiles.default_profile()
        peaks = p.peak_hours(weekend=False)
        lo, hi = profiles.MORNING_PEAK_WINDOW
        assert lo <= peaks["morning"] <= hi
        lo, hi = profiles.EVENING_PEAK_WINDOW
        assert lo <= peaks["evening"] <= hi
        lo, hi = profiles.QUIET_WINDOW
        assert lo <= peaks["quietest"] <= hi

    def test_global_maximum_is_at_one_of_the_two_peaks(self):
        p = profiles.default_profile()
        hours = np.linspace(0, 24, 24 * 60, endpoint=False)
        vals = p.multiplier(hours, weekend=False)
        peak_hour = float(hours[int(np.argmax(vals))])
        assert (9 <= peak_hour <= 11) or (18 <= peak_hour <= 20)

    def test_night_is_quieter_than_both_peaks(self):
        p = profiles.default_profile()
        night = p.multiplier(np.array([1.0, 2.0, 3.0, 4.0]))
        peak = p.multiplier(np.array([10.0, 19.0]))
        assert night.max() < 0.25 * peak.min()

    def test_peak_ratio_matches_the_calibration_target(self):
        """The headline claim: peak trips take PEAK_TRAVEL_TIME_RATIO x free flow."""
        p = profiles.default_profile()
        assert p.reference_peak_ratio() == pytest.approx(profiles.PEAK_TRAVEL_TIME_RATIO, rel=1e-6)
        # And the derived saturation really does invert BPR.
        assert bpr.bpr_multiplier(p.peak_saturation) == pytest.approx(
            profiles.PEAK_TRAVEL_TIME_RATIO
        )

    def test_recalibrating_moves_the_peak_consistently(self):
        p = profiles.DemandProfile(peak_ratio=2.5)
        assert p.reference_peak_ratio() == pytest.approx(2.5, rel=1e-6)

    def test_daily_mean_ratio_sits_between_free_flow_and_peak(self):
        p = profiles.default_profile()
        mean = p.daily_mean_ratio(weekend=False)
        assert 1.0 < mean < profiles.PEAK_TRAVEL_TIME_RATIO
        # A city spends most of the day off-peak, so the daily mean must be
        # much closer to free flow than to the peak.
        assert mean < 1.4

    def test_weekend_is_lighter_than_the_weekday(self):
        p = profiles.default_profile()
        assert p.daily_mean_ratio(True) < p.daily_mean_ratio(False)
        assert p.multiplier(10.0, weekend=True) < p.multiplier(10.0, weekend=False)

    def test_interpolation_is_smooth_and_periodic(self):
        p = profiles.default_profile()
        hours = np.linspace(0, 24, 24 * 12, endpoint=False)
        vals = p.multiplier(hours)
        # Continuous across midnight: 23:59 and 00:01 must nearly agree.
        assert p.multiplier(23.99) == pytest.approx(p.multiplier(-0.01), rel=1e-6)
        assert p.multiplier(25.5) == pytest.approx(p.multiplier(1.5))
        # No overshoot above the tabulated maximum (PCHIP is shape-preserving;
        # a plain cubic spline would invent a spike here).
        assert vals.max() <= p.weekday.max() + 1e-9
        assert vals.min() >= 0.0

    def test_interpolation_actually_interpolates(self):
        p = profiles.default_profile()
        step = profiles.DemandProfile(interpolation="step")
        # Halfway between hours 9 and 10 the smooth profile must differ from
        # the piecewise-constant one.
        assert p.multiplier(9.5) != pytest.approx(step.multiplier(9.5))
        assert min(p.weekday[9], p.weekday[10]) <= p.multiplier(9.5) <= max(
            p.weekday[9], p.weekday[10]
        )

    def test_linear_interpolation_fallback_agrees_at_the_knots(self):
        a = profiles.default_profile()
        b = profiles.DemandProfile(interpolation="linear")
        for h in range(24):
            assert a.multiplier(float(h)) == pytest.approx(b.multiplier(float(h)))

    def test_weekend_flag_follows_the_day_of_week(self):
        p = profiles.default_profile()
        _, weekend_mon = p.at_minute(10 * 60)                      # Monday
        _, weekend_sat = p.at_minute(5 * 24 * 60 + 10 * 60)        # Saturday
        assert weekend_mon is False and weekend_sat is True

    def test_class_sensitivity_ordering(self):
        assert profiles.class_sensitivity("primary") == 1.0, "primary is the reference class"
        assert (
            profiles.class_sensitivity("trunk")
            > profiles.class_sensitivity("secondary")
            > profiles.class_sensitivity("tertiary")
            > profiles.class_sensitivity("residential")
            > profiles.class_sensitivity("living_street")
        )

    def test_flat_fallback_profile_is_a_no_op_at_zero(self):
        p = profiles.flat_profile(0.0)
        assert p.daily_mean_ratio() == pytest.approx(1.0)
        assert p.reference_peak_ratio() == pytest.approx(1.0)

    def test_rejects_malformed_profiles(self):
        with pytest.raises(ValueError):
            profiles.DemandProfile(weekday=np.ones(12))
        with pytest.raises(ValueError):
            profiles.DemandProfile(weekday=-np.ones(24))


# =================================================================== events
class TestEvents:
    def test_hcm_table_values(self):
        """The published residual-capacity figures, used verbatim."""
        B = events.BlockageType
        assert events.residual_capacity(2, B.SHOULDER_DISABLEMENT) == 0.95
        assert events.residual_capacity(2, B.SHOULDER_ACCIDENT) == 0.81
        assert events.residual_capacity(2, B.ONE_LANE_BLOCKED) == 0.35
        assert events.residual_capacity(3, B.SHOULDER_DISABLEMENT) == 0.99
        assert events.residual_capacity(3, B.SHOULDER_ACCIDENT) == 0.83
        assert events.residual_capacity(3, B.ONE_LANE_BLOCKED) == 0.49
        assert events.residual_capacity(3, B.TWO_LANES_BLOCKED) == 0.17

    def test_residual_capacity_is_monotone_in_severity(self):
        B = events.BlockageType
        for lanes in (2, 3, 4, 6):
            order = [
                events.residual_capacity(lanes, b)
                for b in (
                    B.SHOULDER_DISABLEMENT,
                    B.SHOULDER_ACCIDENT,
                    B.ONE_LANE_BLOCKED,
                    B.TWO_LANES_BLOCKED,
                )
            ]
            assert order == sorted(order, reverse=True), f"{lanes} lanes: {order}"

    def test_blocking_the_only_lane_is_a_closure(self):
        assert events.residual_capacity(1, events.BlockageType.ONE_LANE_BLOCKED) == 0.0

    def test_extrapolated_rows_are_flagged_as_such(self):
        tabulated = events.lane_blockage([0], 0, 60, lanes=2)
        extrapolated = events.lane_blockage([0], 0, 60, lanes=5)
        assert tabulated.tabulated is True
        assert extrapolated.tabulated is False

    def test_activation_window_is_half_open(self):
        e = events.lane_blockage([0], start_minute=100, duration_minutes=30)
        assert not e.is_active(99.9)
        assert e.is_active(100.0)
        assert e.is_active(129.9)
        assert not e.is_active(130.0), "the end instant must not be active"

    def test_severity_tapers_the_capacity_loss(self):
        full = events.lane_blockage([0], 0, 60, lanes=2, severity=1.0)
        half = events.lane_blockage([0], 0, 60, lanes=2, severity=0.5)
        none = events.lane_blockage([0], 0, 60, lanes=2, severity=0.0)
        assert full.capacity_multiplier() == pytest.approx(0.35)
        assert half.capacity_multiplier() == pytest.approx(1.0 - 0.5 * 0.65)
        assert none.capacity_multiplier() == pytest.approx(1.0)

    def test_slowdown_changes_time_not_capacity(self):
        e = events.slowdown([0], 0, 60, speed_multiplier=0.5)
        assert e.capacity_multiplier() == pytest.approx(1.0)
        assert e.time_multiplier() == pytest.approx(2.0)

    def test_queue_add_remove_and_query(self):
        q = events.EventQueue()
        a = q.add(events.lane_blockage([0], 0, 60))
        b = q.add(events.closure([1], 30, 60))
        assert len(q) == 2
        assert q.get(a.event_id) is a
        assert {e.event_id for e in q.active_at(0)} == {a.event_id}
        assert {e.event_id for e in q.active_at(45)} == {a.event_id, b.event_id}
        assert {e.event_id for e in q.active_at(70)} == {b.event_id}
        assert q.active_at(200) == []
        assert q.remove(b) is True
        assert q.remove(b) is False
        assert len(q) == 1
        q.clear()
        assert len(q) == 0

    def test_next_change_walks_the_timeline(self):
        q = events.EventQueue()
        q.add(events.lane_blockage([0], 10, 20))
        q.add(events.closure([1], 50, 10))
        assert q.next_change(0) == 10
        assert q.next_change(10) == 30
        assert q.next_change(30) == 50
        assert q.next_change(60) is None

    def test_overlapping_events_compose_multiplicatively(self):
        q = events.EventQueue()
        q.add(events.lane_blockage([0], 0, 60, lanes=2, blockage="shoulder_accident"))
        q.add(events.lane_blockage([0], 0, 60, lanes=2, blockage="shoulder_disablement"))
        cap, tmul, closed = q.apply(10, 2)
        assert cap[0] == pytest.approx(0.81 * 0.95)
        assert cap[1] == pytest.approx(1.0)
        assert not closed.any()

    def test_closure_zeroes_capacity_and_sets_the_mask(self):
        q = events.EventQueue()
        q.add(events.closure([1], 0, 60))
        cap, _, closed = q.apply(10, 3)
        assert cap[1] == 0.0 and closed[1]
        assert not closed[0] and not closed[2]

    def test_out_of_range_edge_indices_are_ignored(self):
        q = events.EventQueue()
        q.add(events.closure([99], 0, 60))
        cap, _, closed = q.apply(10, 3)
        assert not closed.any() and np.all(cap == 1.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            events.lane_blockage([], 0, 60)
        with pytest.raises(ValueError):
            events.lane_blockage([0], 0, 0)
        with pytest.raises(ValueError):
            events.lane_blockage([0], 0, 60, severity=1.5)
        with pytest.raises(ValueError):
            events.slowdown([0], 0, 60, speed_multiplier=0.0)

    def test_serialisation_is_json_clean(self):
        import json

        q = events.EventQueue()
        q.add(events.lane_blockage([0, 1], 0, 60, description="test"))
        q.add(events.closure([2], 0, 60))
        payload = json.dumps(q.as_dict(minute=10))
        assert "lane_blockage" in payload and "closure" in payload


# ================================================================ simulator
class TestSimulator:
    def test_builds_from_a_plain_networkx_graph(self, toy_sim):
        assert toy_sim.edges.n_edges == 6
        assert toy_sim.edges.free_flow_time[0] == pytest.approx(50.0)
        assert toy_sim.edges.lanes[1] == 3.0
        assert str(toy_sim.edges.road_class[0]) == "primary"

    def test_rejects_something_that_is_not_a_network(self):
        with pytest.raises(TypeError):
            edge_arrays_from_network(object())

    def test_clock_arithmetic(self, toy_sim):
        toy_sim.set_time(0)
        assert toy_sim.hour_of_day == 0.0 and toy_sim.day_of_week == 0
        toy_sim.advance(90)
        assert toy_sim.time_minutes == 90 and toy_sim.hour_of_day == 1.5
        toy_sim.set_clock(hour=9.5, day_of_week=6)
        assert toy_sim.is_weekend and toy_sim.hour_of_day == pytest.approx(9.5)

    def test_free_flow_at_night(self, toy_sim):
        """At 03:00 the model must be indistinguishable from free flow."""
        toy_sim.set_clock(3.0)
        t = toy_sim.edge_travel_times()
        assert t == pytest.approx(toy_sim.edges.free_flow_time, rel=1e-3)

    def test_peak_is_slower_than_off_peak(self, toy_sim):
        night = toy_sim.set_clock(3.0).edge_travel_times().copy()
        peak = toy_sim.set_clock(19.0).edge_travel_times().copy()
        assert np.all(peak >= night - 1e-9)
        assert peak.sum() > night.sum() * 1.05

    def test_reproducible_given_a_seed(self):
        g = make_toy_graph()
        a = TrafficSimulator(g, seed=123).set_clock(18.5)
        b = TrafficSimulator(make_toy_graph(), seed=123).set_clock(18.5)
        c = TrafficSimulator(g, seed=124).set_clock(18.5)
        np.testing.assert_array_equal(a.edge_travel_times(), b.edge_travel_times())
        assert not np.allclose(a.edge_travel_times(), c.edge_travel_times())

    def test_time_travel_returns_identical_weights(self, toy_sim):
        """Advancing and coming back must reproduce the weights bit for bit.

        This is the property that makes replaying a scenario meaningful, and it
        is why the noise is a pre-generated field indexed by time rather than a
        stream consumed as the clock runs.
        """
        toy_sim.set_clock(9.25)
        before = toy_sim.edge_travel_times().copy()
        for step in (7.0, 33.0, 120.0, -55.0, 991.0):
            toy_sim.advance(step)
            toy_sim.edge_travel_times()
        toy_sim.set_clock(9.25)
        np.testing.assert_array_equal(before, toy_sim.edge_travel_times())

    def test_weights_are_periodic_over_a_day(self, toy_sim):
        toy_sim.set_time(600.0)
        a = toy_sim.edge_travel_times().copy()
        toy_sim.set_time(600.0 + 7 * 24 * 60.0)  # one whole week later
        np.testing.assert_allclose(a, toy_sim.edge_travel_times())

    def test_noise_varies_smoothly_in_time(self, toy_sim):
        toy_sim.set_clock(14.0)
        a = toy_sim.edge_travel_times().copy()
        toy_sim.set_clock(14.02)  # about a minute later
        b = toy_sim.edge_travel_times().copy()
        toy_sim.set_clock(20.0)
        c = toy_sim.edge_travel_times().copy()
        assert np.max(np.abs(b - a)) < np.max(np.abs(c - a))

    def test_incident_capacity_reduction_increases_travel_time(self, toy_sim):
        """The central claim of the event model."""
        toy_sim.set_clock(19.0)
        before = toy_sim.edge_travel_times().copy()
        toy_sim.add_event(
            events.lane_blockage([0], toy_sim.time_minutes - 5, 60, lanes=2)
        )
        after = toy_sim.edge_travel_times()
        assert after[0] > before[0]
        # 65 percent of capacity gone raises saturation by 1/0.35, and BPR is
        # quartic, so the delay term must grow by roughly (1/0.35)^4 = 67x.
        delay_before = before[0] - toy_sim.edges.free_flow_time[0]
        delay_after = after[0] - toy_sim.edges.free_flow_time[0]
        assert delay_after / delay_before == pytest.approx((1 / 0.35) ** 4, rel=1e-6)
        # Untouched edges must be untouched.
        np.testing.assert_allclose(after[1:], before[1:])

    def test_incident_costs_more_at_the_peak_than_at_night(self):
        """Capacity-based incident modelling, rather than a flat time penalty."""
        sim = TrafficSimulator(make_toy_graph(), seed=3, noise_sigma=0.0, demand_spread=0.0)
        blockage = events.lane_blockage([0], 0, 60 * 24 * 7, lanes=2)

        def extra(hour: float) -> float:
            sim.clear_events()
            sim.set_clock(hour)
            base = sim.edge_travel_times()[0]
            sim.add_event(blockage)
            return sim.edge_travel_times()[0] - base

        assert extra(19.0) > 20 * extra(3.0)

    def test_active_closure_makes_the_edge_unusable(self, toy_sim):
        toy_sim.set_clock(12.0)
        assert np.isfinite(toy_sim.edge_travel_times()[2])
        toy_sim.add_event(events.closure([2], toy_sim.time_minutes, 60))
        t = toy_sim.edge_travel_times()
        assert math.isinf(float(t[2]))
        assert toy_sim.closed_mask()[2]
        assert toy_sim.speed_factors()[2] == 0.0
        assert np.all(np.isfinite(np.delete(t, 2)))

    def test_closure_clears_when_its_window_ends(self, toy_sim):
        toy_sim.set_clock(12.0)
        start = toy_sim.time_minutes
        toy_sim.add_event(events.closure([2], start, 30))
        assert math.isinf(float(toy_sim.edge_travel_times()[2]))
        toy_sim.set_time(start + 31)
        assert np.isfinite(toy_sim.edge_travel_times()[2])

    def test_removing_an_event_restores_the_weights(self, toy_sim):
        toy_sim.set_clock(19.0)
        before = toy_sim.edge_travel_times().copy()
        e = toy_sim.add_event(events.closure([0], toy_sim.time_minutes, 60))
        assert not np.array_equal(before, toy_sim.edge_travel_times())
        toy_sim.remove_event(e)
        np.testing.assert_array_equal(before, toy_sim.edge_travel_times())

    def test_slowdown_scales_travel_time_directly(self, toy_sim):
        toy_sim.set_clock(3.0)  # free flow, so the effect is isolated
        before = toy_sim.edge_travel_times().copy()
        toy_sim.add_event(
            events.slowdown([3], toy_sim.time_minutes, 60, speed_multiplier=0.25)
        )
        assert toy_sim.edge_travel_times()[3] == pytest.approx(before[3] * 4.0, rel=1e-6)

    def test_conical_vdf_is_selectable_and_changes_the_answer(self):
        """Selecting conical must genuinely re-price the network.

        At a peak saturation of about 1.5 conical charges considerably *more*
        than BPR, which is why the simulator warns that switching VDF voids the
        profile calibration.

        The two agree only at *exactly* zero demand, not merely at night: at
        03:00 the profile still puts about 0.12 saturation on an arterial, and
        because conical has a non-zero derivative at the origin it already
        charges 2 percent there where BPR charges essentially nothing. That is
        the documented difference, so it is asserted rather than tolerated.
        """
        common = dict(seed=11, noise_sigma=0.0, demand_spread=0.0)
        a = TrafficSimulator(make_toy_graph(), vdf="bpr", **common)
        b = TrafficSimulator(make_toy_graph(), vdf="conical", **common)
        a.set_clock(19.0)
        b.set_clock(19.0)
        assert a.saturation()[0] > 1.0
        assert b.edge_travel_times()[0] > a.edge_travel_times()[0]

        # Small but non-zero night demand: conical already charges, BPR does not.
        a.set_clock(3.0)
        b.set_clock(3.0)
        t0 = a.edges.free_flow_time
        assert np.allclose(a.edge_travel_times(), t0, rtol=1e-4)
        assert np.all(b.edge_travel_times() > t0 * 1.005)

        # At exactly zero demand they coincide.
        flat = profiles.flat_profile(0.0)
        za = TrafficSimulator(make_toy_graph(), profile=flat, vdf="bpr", **common)
        zb = TrafficSimulator(make_toy_graph(), profile=flat, vdf="conical", **common)
        np.testing.assert_allclose(za.edge_travel_times(), zb.edge_travel_times(), rtol=1e-12)

    def test_calibration_holds_on_the_reference_class(self, bengaluru):
        """The 1.75 target must actually be met on a real network.

        Checked on primary roads, the class the calibration is defined against.
        The tolerance is loose because the per-edge log-normal spread only
        cancels in expectation over the class, not exactly.
        """
        sim = TrafficSimulator(bengaluru, seed=42)
        sim.set_clock(19.0)
        primary = sim.class_summary()["primary"]["mean_ratio"]
        assert primary == pytest.approx(profiles.PEAK_TRAVEL_TIME_RATIO, rel=0.10)

    def test_road_classes_are_ordered_by_congestion(self, bengaluru):
        sim = TrafficSimulator(bengaluru, seed=42).set_clock(19.0)
        cs = sim.class_summary()
        assert (
            cs["primary"]["mean_ratio"]
            > cs["secondary"]["mean_ratio"]
            > cs["tertiary"]["mean_ratio"]
            > cs["residential"]["mean_ratio"]
        )

    def test_state_is_json_serialisable(self, toy_sim):
        import json

        toy_sim.set_clock(19.0)
        toy_sim.add_event(events.lane_blockage([0], toy_sim.time_minutes, 30))
        payload = json.dumps(toy_sim.state(top_k=3))
        state = json.loads(payload)
        assert state["n_edges"] == 6
        assert state["n_active_events"] == 1
        assert len(state["worst_edges"]) == 3
        assert sum(state["congestion"]["bands"].values()) == 6

    def test_apply_to_writes_back_in_the_same_edge_order(self):
        g = make_toy_graph()
        sim = TrafficSimulator(g, seed=5).set_clock(19.0)
        n = sim.apply_to(g)
        assert n == 6
        written = np.array([d["travel_time"] for *_x, d in g.edges(keys=True, data=True)])
        np.testing.assert_allclose(written, sim.edge_travel_times())

    def test_apply_to_prefers_a_road_network_update_method(self):
        class FakeRoadNetwork:
            def __init__(self, graph):
                self.graph = graph
                self.received = None

            def update_travel_times(self, times):
                self.received = np.asarray(times)

        net = FakeRoadNetwork(make_toy_graph())
        sim = TrafficSimulator(net, seed=5).set_clock(19.0)
        sim.apply_to(net)
        np.testing.assert_array_equal(net.received, sim.edge_travel_times())

    def test_absolute_capacity_cancels_out_of_the_travel_time(self):
        """Documented consequence of the saturation-first demand model.

        Only *relative* capacity changes (incidents) move the answer, so the
        choice of capacity table cannot silently shift the calibration.
        """
        g = make_toy_graph()
        a = TrafficSimulator(g, seed=4).set_clock(19.0)
        b = TrafficSimulator(
            make_toy_graph(), seed=4, capacity=np.full(6, 9999.0)
        ).set_clock(19.0)
        np.testing.assert_allclose(a.edge_travel_times(), b.edge_travel_times())
        assert a.capacity_source == "traffic.bpr" and b.capacity_source == "explicit"

    def test_network_capacity_is_preferred_over_the_local_table(self):
        class ArrayNetwork:
            free_flow_time = np.array([10.0, 20.0])
            edge_class = ["primary", "residential"]
            edge_lanes = [2, 1]
            length = np.array([100.0, 200.0])
            edge_keys = [(1, 2, 0), (2, 3, 0)]
            edge_capacity = np.array([4321.0, 123.0])

        sim = TrafficSimulator(ArrayNetwork(), seed=1)
        assert sim.capacity_source == "network"
        np.testing.assert_array_equal(sim.base_capacity, ArrayNetwork.edge_capacity)

    def test_accepts_an_array_style_road_network(self):
        class ArrayNetwork:
            free_flow_time = np.array([10.0, 20.0, 30.0])
            road_class = ["primary", "residential", "tertiary"]
            lanes = [2, 1, 2]
            length = np.array([100.0, 200.0, 300.0])
            edge_keys = [(1, 2, 0), (2, 3, 0), (3, 1, 0)]

        sim = TrafficSimulator(ArrayNetwork(), seed=1).set_clock(19.0)
        assert sim.edges.n_edges == 3
        assert sim.edges.index_of((2, 3, 0)) == 1
        assert np.all(sim.edge_travel_times() >= ArrayNetwork.free_flow_time)

    def test_speed_factors_invert_travel_times(self, toy_sim):
        toy_sim.set_clock(19.0)
        f = toy_sim.speed_factors()
        t = toy_sim.edge_travel_times()
        assert np.all((f > 0) & (f <= 1.0))
        np.testing.assert_allclose(toy_sim.edges.free_flow_time / f, t, rtol=1e-9)

    def test_day_summary_restores_the_clock(self, toy_sim):
        toy_sim.set_time(555.0)
        rows = toy_sim.day_summary(step_minutes=120)
        assert toy_sim.time_minutes == 555.0
        assert len(rows) == 12
        peak = max(rows, key=lambda r: r["mean_ratio"])
        assert peak["hour"] in (8.0, 10.0, 18.0, 20.0)

    def test_zero_noise_gives_the_analytic_answer(self):
        """With the stochastic terms off, the model is a closed-form check."""
        sim = TrafficSimulator(
            make_toy_graph(), seed=0, noise_sigma=0.0, demand_spread=0.0
        ).set_clock(10.0)
        p = sim.profile
        expected_sat = p.saturation(10.0) * profiles.class_sensitivity("primary")
        assert sim.saturation()[0] == pytest.approx(float(expected_sat))
        assert sim.edge_travel_times()[0] == pytest.approx(
            50.0 * bpr.bpr_multiplier(expected_sat)
        )

    def test_find_edges_by_class_and_name(self):
        g = make_toy_graph()
        for u, v, k, d in g.edges(keys=True, data=True):
            d["name"] = f"{d['highway'].title()} Road"
        sim = TrafficSimulator(g, seed=0)
        assert sim.find_edges(road_class="primary") == [0, 1]
        with pytest.raises(ValueError):
            sim.find_edges(name="Primary")
        sim.attach_names(g)
        assert sim.find_edges(name="primary") == [0, 1]


# ================================================================== sources
class TestSources:
    def test_simulated_source_never_claims_to_be_live(self, toy_sim):
        src = sources.SimulatedSource(toy_sim)
        obs = src.fetch()
        assert obs.live is False
        assert obs.coverage == 1.0
        assert obs.speed_factor.shape == (6,)
        np.testing.assert_allclose(
            obs.travel_times(toy_sim.edges.free_flow_time), toy_sim.edge_travel_times()
        )

    def test_simulated_source_can_tick_the_clock(self, toy_sim):
        src = sources.SimulatedSource(toy_sim, advance_minutes=15.0)
        t0 = toy_sim.time_minutes
        src.fetch()
        src.fetch()
        assert toy_sim.time_minutes == t0 + 30.0

    def test_tomtom_without_a_key_raises_an_actionable_error(self, monkeypatch):
        monkeypatch.delenv(sources.TOMTOM_API_KEY_ENV, raising=False)
        assert sources.api_key_available() is False
        with pytest.raises(sources.MissingAPIKeyError) as exc:
            sources.TomTomFlowSource(probes=[(0, 12.93, 77.62)], n_edges=6)
        msg = str(exc.value)
        assert sources.TOMTOM_API_KEY_ENV in msg
        assert "developer.tomtom.com" in msg

    def test_tomtom_never_reveals_the_key(self, monkeypatch):
        monkeypatch.setenv(sources.TOMTOM_API_KEY_ENV, "secret-key-do-not-log")
        src = sources.TomTomFlowSource(probes=[(0, 12.93, 77.62)], n_edges=6)
        assert "secret-key-do-not-log" not in repr(src.describe())
        assert src.describe()["api_key_present"] is True

    def test_tomtom_parses_a_flow_response_without_network_access(self, monkeypatch, toy_sim):
        """The parsing path, exercised with the HTTP layer stubbed out."""
        monkeypatch.setenv(sources.TOMTOM_API_KEY_ENV, "dummy")
        src = sources.TomTomFlowSource(
            probes=[(1, 12.93, 77.62)],
            n_edges=6,
            baseline=sources.SimulatedSource(toy_sim),
        )
        monkeypatch.setattr(
            src,
            "_request",
            lambda lat, lon: {
                "flowSegmentData": {"currentSpeed": 12, "freeFlowSpeed": 48, "confidence": 0.9}
            },
        )
        obs = src.fetch()
        assert obs.live is True
        assert obs.speed_factor[1] == pytest.approx(0.25)
        assert obs.coverage == pytest.approx(1 / 6)
        assert obs.meta["probes_covered"] == 1

    def test_resolve_source_falls_back_visibly_without_a_key(self, monkeypatch, toy_sim):
        monkeypatch.delenv(sources.TOMTOM_API_KEY_ENV, raising=False)
        src, reason = sources.resolve_source(
            toy_sim, prefer_live=True, probes=[(0, 12.93, 77.62)]
        )
        assert isinstance(src, sources.SimulatedSource)
        assert reason is not None and sources.TOMTOM_API_KEY_ENV in reason
        assert src.fetch().live is False

    def test_resolve_source_reports_missing_probes(self, toy_sim):
        src, reason = sources.resolve_source(toy_sim, prefer_live=True, probes=[])
        assert isinstance(src, sources.SimulatedSource)
        assert reason is not None and "corridor" in reason

    def test_resolve_source_returns_live_when_configured(self, monkeypatch, toy_sim):
        monkeypatch.setenv(sources.TOMTOM_API_KEY_ENV, "dummy")
        src, reason = sources.resolve_source(
            toy_sim, prefer_live=True, probes=[(0, 12.93, 77.62)]
        )
        assert isinstance(src, sources.TomTomFlowSource)
        assert reason is None

    def test_observation_marks_closed_edges_as_infinite(self, toy_sim):
        toy_sim.set_clock(12.0)
        toy_sim.add_event(events.closure([2], toy_sim.time_minutes, 60))
        obs = sources.SimulatedSource(toy_sim).fetch()
        assert math.isinf(float(obs.travel_times(toy_sim.edges.free_flow_time)[2]))

    def test_fallback_tagging(self, toy_sim):
        obs = sources.fallback_observation(
            sources.SimulatedSource(toy_sim).fetch(), "live feed timed out"
        )
        assert obs.live is False and obs.fallback_reason == "live feed timed out"
        assert obs.as_dict()["fallback_reason"] == "live feed timed out"


# ================================================= integration with qroute.graph
def test_road_network_integration_if_available():
    """Use the graph component's RoadNetwork if it exists yet.

    ``qroute.graph`` is built by a separate component and may not be present.
    The traffic layer never imports it at module scope, so its absence is a
    skip and not a failure.
    """
    pytest.importorskip("qroute.graph", reason="qroute.graph is not implemented yet")
    import qroute.graph as qg

    factory = None
    for name in ("RoadNetwork", "load_network", "from_graphml", "load"):
        factory = getattr(qg, name, None)
        if factory is not None:
            break
    if factory is None:
        pytest.skip("qroute.graph exposes no recognised RoadNetwork constructor")

    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "osm" / "delhi_connaught.graphml"
    if not path.exists():
        pytest.skip("OSM extract not available")
    try:
        network = factory(str(path))
    except Exception as exc:  # pragma: no cover - depends on a component in flux
        pytest.skip(f"could not construct a RoadNetwork: {exc}")

    sim = TrafficSimulator(network, seed=1).set_clock(19.0)
    times = sim.edge_travel_times()
    assert times.shape[0] == sim.edges.n_edges > 0
    assert np.all(times >= sim.edges.free_flow_time - 1e-9)
    assert sim.apply_to(network) == sim.edges.n_edges


# ================================================================ performance
def test_full_network_update_is_fast(bengaluru):
    """The stated requirement: updating every edge takes milliseconds.

    The threshold is deliberately loose (25 ms for 34k edges against a measured
    figure under 1 ms) so that a slow CI machine does not produce a spurious
    failure; the point is to catch a regression that reintroduces a Python
    loop over edges, which would cost seconds.
    """
    sim = TrafficSimulator(bengaluru, seed=42)
    stats = sim.benchmark_update(repeats=15)
    assert stats["n_edges"] > 30000
    assert stats["median_ms"] < 25.0, stats
