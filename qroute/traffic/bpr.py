"""Volume-delay functions: how travel time grows with traffic volume.

A volume-delay function (VDF) maps a *saturation ratio* ``x = v / c`` -- the
traffic volume on a link divided by the link's capacity -- onto a travel-time
multiplier applied to the free-flow travel time ``t0``. This is the single
piece of physics that turns a static road graph into a traffic model, so it is
kept in its own module with no dependencies on the rest of the platform.

Two functions are provided.

**BPR** (US Bureau of Public Roads, 1964), the field standard::

    t(x) = t0 * (1 + alpha * x**beta)          alpha = 0.15, beta = 4

It is used as the default because every published congestion index, every
transport-planning textbook and every reviewer will recognise it, and because
its parameters have accepted default values. Its two well-known defects are:

1. ``dt/dx = t0 * alpha * beta * x**(beta-1)`` is **zero at x = 0**. An empty
   road is modelled as perfectly insensitive to one more vehicle, which is not
   true and which makes equilibrium assignment converge slowly.
2. For ``x`` much above 1 the fourth power explodes: at ``x = 2`` BPR already
   predicts 3.4x the free-flow time, at ``x = 3`` it predicts 13x. Real
   oversaturated urban links do not degrade that fast because queues spill back
   and metering caps the inflow.

**Conical** (Spiess, 1990), the alternative::

    t(x) = t0 * (2 + sqrt(a**2 * (1 - x)**2 + b**2) - a * (1 - x) - b)
    with  b = (2a - 1) / (2a - 2)  and  a > 1

The constant ``b`` is chosen precisely so that ``t(0) = t0`` exactly. The
conical function has a strictly positive derivative everywhere (fixing defect
1) and is asymptotically linear in ``x`` with slope ``2a`` (fixing defect 2):
as ``x`` grows, ``sqrt(a^2 (1-x)^2 + b^2) -> a x`` and ``-a(1-x) -> a x``, so
the two terms each contribute ``a x``.

Prefer conical when a solver needs a well-conditioned gradient, or when the
model runs far into oversaturation. Prefer BPR for anything that has to be
compared against published work.

The two are **not interchangeable**, and the following measured values (this
module, ``a = 4``) are the reason to be careful:

======  =======  =========
``x``   BPR      conical
======  =======  =========
0.5     1.009    1.149
1.0     1.150    2.000
1.5     1.759    5.149
3.0     13.150   16.918
5.0     94.750   32.876
======  =======  =========

Conical is *steeper* than BPR everywhere below ``x`` about 3.2, not gentler.
It is pinned to exactly ``2 * t0`` at capacity for every ``a``, against BPR's
1.15 -- a deliberate design property, since BPR is widely criticised for being
implausibly flat at capacity, but a large difference all the same. Only in deep
oversaturation does the linear asymptote win: by ``x = 5`` conical is a third
of BPR. Anything calibrated against one function must be recalibrated for the
other; the calibration in :mod:`qroute.traffic.profiles` inverts BPR
specifically and does not carry over.

Two derived quantities are exported and it matters which is which:

* ``saturation_ratio = v / c`` -- an *input* to the VDF, a demand measure.
* ``congestion_level = (t - t0) / t0`` -- an *output*, the fractional delay.
  A value of 0.75 means the trip takes 75 percent longer than in free flow.

**The user interface colours edges by congestion_level**, not by saturation,
because congestion level is what a driver experiences and it is the quantity
published congestion indices report. Saturation is exposed for diagnostics and
for the incident model (which reduces ``c``, not ``t`` directly).
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Standard BPR coefficients. The 1964 BPR report fitted alpha = 0.15,
# beta = 4 to US highway data; they remain the default in every major
# assignment package, so deviating from them would need justification.
BPR_ALPHA: Final[float] = 0.15
BPR_BETA: Final[float] = 4.0

# Spiess's conical shape parameter. Larger `a` makes the function hug BPR more
# closely below capacity at the cost of a steeper knee near x = 1.
CONICAL_A: Final[float] = 4.0

ArrayLike = np.ndarray | float


def bpr_multiplier(
    saturation: ArrayLike,
    alpha: float = BPR_ALPHA,
    beta: float = BPR_BETA,
) -> np.ndarray:
    """Return the BPR travel-time multiplier ``1 + alpha * x**beta``.

    ``saturation`` may be any shape; the computation is fully vectorised so it
    runs over every edge of a 34k-edge road network in one call. Negative
    saturations are clipped to zero rather than raising, because upstream noise
    models can occasionally produce a small negative demand and a hard failure
    there would be more disruptive than clamping.
    """
    x = np.clip(np.asarray(saturation, dtype=np.float64), 0.0, None)
    return 1.0 + alpha * np.power(x, beta)


def bpr_travel_time(
    free_flow_time: ArrayLike,
    saturation: ArrayLike,
    alpha: float = BPR_ALPHA,
    beta: float = BPR_BETA,
) -> np.ndarray:
    """Congested travel time from free-flow time and saturation ratio."""
    return np.asarray(free_flow_time, dtype=np.float64) * bpr_multiplier(saturation, alpha, beta)


def bpr_travel_time_from_volume(
    free_flow_time: ArrayLike,
    volume: ArrayLike,
    capacity: ArrayLike,
    alpha: float = BPR_ALPHA,
    beta: float = BPR_BETA,
) -> np.ndarray:
    """BPR expressed in its textbook form ``t(v) = t0 (1 + alpha (v/c)^beta)``.

    Zero or negative capacities are treated as fully closed links and produce
    ``inf``, which is the correct behaviour for a shortest-path search: an edge
    with no capacity must never be selected. Callers that prefer a large finite
    number should post-process with :func:`numpy.nan_to_num`.
    """
    return bpr_travel_time(free_flow_time, saturation_ratio(volume, capacity), alpha, beta)


def conical_b(a: float = CONICAL_A) -> float:
    """The dependent conical parameter ``b = (2a - 1) / (2a - 2)``.

    This is not a free parameter: it is the unique value that makes the conical
    function pass through ``t(0) = t0``.
    """
    if a <= 1.0:
        raise ValueError(f"conical shape parameter a must be > 1, got {a}")
    return (2.0 * a - 1.0) / (2.0 * a - 2.0)


def conical_multiplier(saturation: ArrayLike, a: float = CONICAL_A) -> np.ndarray:
    """Return Spiess's conical volume-delay multiplier.

    Exact reference values, useful when reading the tests: ``f(0) = 1``
    regardless of ``a``, and ``f(1) = 2`` regardless of ``a``. Above capacity
    the function tends to a straight line of slope ``2a``, so it overtakes
    BPR's quartic only around ``x = 3.2`` (with ``a = 4``); below that it
    charges *more* than BPR, not less. See the module docstring's table.
    """
    x = np.clip(np.asarray(saturation, dtype=np.float64), 0.0, None)
    b = conical_b(a)
    u = 1.0 - x
    return 2.0 + np.sqrt(a * a * u * u + b * b) - a * u - b


def conical_travel_time(
    free_flow_time: ArrayLike, saturation: ArrayLike, a: float = CONICAL_A
) -> np.ndarray:
    """Congested travel time under the conical VDF."""
    return np.asarray(free_flow_time, dtype=np.float64) * conical_multiplier(saturation, a)


def saturation_ratio(volume: ArrayLike, capacity: ArrayLike) -> np.ndarray:
    """Return ``v / c``, with non-positive capacity mapped to ``inf``.

    A closed or capacity-zero link is genuinely infinitely saturated; returning
    ``inf`` keeps the arithmetic honest instead of hiding a division by zero
    behind an arbitrary large constant.
    """
    v = np.asarray(volume, dtype=np.float64)
    c = np.asarray(capacity, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(c > 0.0, v / np.where(c > 0.0, c, 1.0), np.inf)
    return np.clip(x, 0.0, None)


def congestion_level(travel_time: ArrayLike, free_flow_time: ArrayLike) -> np.ndarray:
    """Fractional delay ``(t - t0) / t0``; this is what the UI colours by.

    0.0 means free flow, 1.0 means the trip takes twice as long as it would on
    an empty road. Links with a non-positive free-flow time (degenerate
    zero-length edges that OSM occasionally contains) report 0.0 rather than
    ``inf``, since a zero-length edge cannot be congested in any useful sense.
    """
    t = np.asarray(travel_time, dtype=np.float64)
    t0 = np.asarray(free_flow_time, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lvl = np.where(t0 > 0.0, (t - t0) / np.where(t0 > 0.0, t0, 1.0), 0.0)
    return np.clip(lvl, 0.0, None)


def saturation_for_ratio(
    target_ratio: float, alpha: float = BPR_ALPHA, beta: float = BPR_BETA
) -> float:
    """Invert BPR: the saturation ``x`` at which ``t / t0 == target_ratio``.

    Used by :mod:`qroute.traffic.profiles` to calibrate the peak of the
    time-of-day profile against a published congestion index, rather than
    hand-tuning the multipliers until the answer looks right.
    """
    if target_ratio < 1.0:
        raise ValueError("target travel-time ratio must be at least 1.0")
    return float(((target_ratio - 1.0) / alpha) ** (1.0 / beta))


# ------------------------------------------------------------ road capacity
# Per-lane capacity in passenger-car units per hour, by OSM highway class.
#
# Provenance and honesty note: the Highway Capacity Manual gives a base
# capacity of 2400 pc/h/ln for a 120 km/h freeway, falling to about 2250 at
# 90 km/h. Indian urban roads are far below that because of mixed traffic,
# frequent at-grade intersections, kerbside parking and pedestrian friction;
# IRC:106-1990 quotes design service volumes for urban arterials that
# correspond to roughly 1500-1800 PCU/h/lane for divided arterials and much
# less for local streets. The numbers below are round values inside those
# published ranges. They are engineering defaults, not measurements of any
# particular Bengaluru corridor, and any user with counted volumes should
# override them by passing an explicit capacity array to the simulator.
LANE_CAPACITY_PCU_PER_HOUR: Final[dict[str, float]] = {
    "motorway": 2000.0,
    "motorway_link": 1400.0,
    "trunk": 1800.0,
    "trunk_link": 1200.0,
    "primary": 1500.0,
    "primary_link": 1000.0,
    "secondary": 1200.0,
    "secondary_link": 900.0,
    "tertiary": 900.0,
    "tertiary_link": 700.0,
    "unclassified": 700.0,
    "residential": 600.0,
    "living_street": 300.0,
    "service": 400.0,
    "busway": 900.0,
    "road": 700.0,
}

DEFAULT_LANE_CAPACITY: Final[float] = 600.0

# Default lane count when OSM does not tag one. Most Indian residential streets
# are effectively two-way single-carriageway; arterials are usually tagged.
DEFAULT_LANES: Final[dict[str, float]] = {
    "motorway": 3.0,
    "trunk": 3.0,
    "primary": 2.0,
    "secondary": 2.0,
    "tertiary": 2.0,
    "unclassified": 1.0,
    "residential": 1.0,
    "living_street": 1.0,
    "service": 1.0,
}

DEFAULT_LANE_COUNT: Final[float] = 1.0


def lane_capacity(road_class: str) -> float:
    """Per-lane hourly capacity for an OSM ``highway`` value."""
    return LANE_CAPACITY_PCU_PER_HOUR.get(road_class, DEFAULT_LANE_CAPACITY)


def default_lanes(road_class: str) -> float:
    """Assumed directional lane count when the OSM ``lanes`` tag is missing."""
    return DEFAULT_LANES.get(road_class, DEFAULT_LANE_COUNT)


def edge_capacity(road_classes, lanes) -> np.ndarray:
    """Vectorised hourly capacity ``c = lanes * per-lane capacity``.

    ``road_classes`` is a sequence of OSM highway strings and ``lanes`` a
    matching numeric array. Kept as a plain loop over the *distinct* classes
    rather than per edge, so it stays O(edges) with a small constant.
    """
    lanes_arr = np.asarray(lanes, dtype=np.float64)
    classes = np.asarray(road_classes, dtype=object)
    per_lane = np.full(classes.shape, DEFAULT_LANE_CAPACITY, dtype=np.float64)
    for name in set(classes.tolist()):
        per_lane[classes == name] = lane_capacity(str(name))
    return per_lane * np.clip(lanes_arr, 1.0, None)


# ---------------------------------------------------------------- UI banding
# Thresholds on congestion_level, chosen to align with the colours used by the
# common consumer traffic maps: green up to a tenth extra, then amber, then
# red, then dark red. They are a presentation choice, not a modelling one.
CONGESTION_BANDS: Final[tuple[tuple[str, float], ...]] = (
    ("free", 0.10),
    ("light", 0.35),
    ("moderate", 0.75),
    ("heavy", 1.50),
    ("severe", float("inf")),
)

BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in CONGESTION_BANDS)


def congestion_band(level: ArrayLike) -> np.ndarray:
    """Map congestion levels onto band indices into :data:`BAND_NAMES`.

    Returns an integer array so the API can ship a compact per-edge colour
    key instead of 34k floats.
    """
    edges = np.array([hi for _, hi in CONGESTION_BANDS[:-1]], dtype=np.float64)
    lvl = np.asarray(level, dtype=np.float64)
    return np.searchsorted(edges, lvl, side="right").astype(np.int8)


def band_counts(level: ArrayLike) -> dict[str, int]:
    """Histogram of edges per congestion band, for the dashboard summary."""
    idx = congestion_band(level)
    counts = np.bincount(np.asarray(idx, dtype=np.intp), minlength=len(BAND_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(BAND_NAMES)}
