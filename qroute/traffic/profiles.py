"""Time-of-day and day-of-week demand profiles for an Indian city.

The simulator needs to know, for any instant of the week, how heavily loaded
the network is. That is expressed here as a dimensionless *demand multiplier*
which the simulator turns into a per-edge saturation ratio ``v / c`` and then
feeds to the volume-delay function in :mod:`qroute.traffic.bpr`.

Shape
-----
The weekday curve has the twin-peaked shape characteristic of a commuter city,
with the peaks placed where Indian metros actually see them rather than where
European or American profiles put them:

* morning peak 09:00-11:00 (later than the 07:00-09:00 of most Western
  profiles, because Indian office start times cluster around 09:30-10:00),
* evening peak 18:00-20:00,
* a broad, non-trivial midday plateau -- commercial and delivery traffic keeps
  Indian arterials at roughly 70 percent of peak all afternoon,
* a genuine trough 01:00-05:00.

The weekend curve has no morning commute peak, rises slowly through the
morning and peaks in the evening from shopping and leisure trips, at about
80 percent of the weekday peak level.

Calibration -- read this before quoting the numbers
---------------------------------------------------
The *shape* is synthetic. It is a hand-built curve, not a fit to counted
traffic; no open per-hour link-volume dataset for Bengaluru was available to
this build, and the module says so rather than implying otherwise.

The *scale* is calibrated, and this is the part that carries meaning. The peak
multiplier is chosen by inverting the BPR function so that a reference urban
arterial at the peak of the weekday profile takes
:data:`PEAK_TRAVEL_TIME_RATIO` = 1.75 times its free-flow travel time. That
figure is the project's calibration target and it sits in the range that
published congestion indices report for Bengaluru: TomTom's Traffic Index has
placed the city's congestion level in the high-50s to mid-60s percent in
recent editions, which is exactly the statement "a peak trip takes about
1.6-1.8 times as long as the same trip on an empty road".

Honesty caveat: an attempt to fetch the current TomTom Traffic Index page
during this build returned only the JavaScript shell, so the 1.75 target was
*not* machine-verified here. Treat it as the specified target, reproduced
exactly by the code (see :func:`DemandProfile.reference_peak_ratio`), rather
than as a measurement made by this repository.

Per-road-class sensitivity
--------------------------
The same time profile does not hit every street equally. Arterials carry the
through-commute and saturate; residential lanes carry local access trips and
barely notice the peak. The class sensitivities in
:data:`ROAD_CLASS_SENSITIVITY` multiply the demand, with ``primary`` fixed at
1.0 as the reference class against which the 1.75 calibration is defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

import numpy as np

from qroute.traffic.bpr import BPR_ALPHA, BPR_BETA, saturation_for_ratio

HOURS_PER_DAY: Final[int] = 24
MINUTES_PER_DAY: Final[int] = 24 * 60
MINUTES_PER_WEEK: Final[int] = 7 * MINUTES_PER_DAY

#: Calibration target: peak travel time divided by free-flow travel time on a
#: reference arterial. See the module docstring for provenance.
PEAK_TRAVEL_TIME_RATIO: Final[float] = 1.75

#: Hours whose maximum must lie inside them for the profile to be considered
#: correctly shaped. The tests assert against these, so the claim in the
#: docstring and the claim in the data cannot drift apart.
MORNING_PEAK_WINDOW: Final[tuple[int, int]] = (9, 11)
EVENING_PEAK_WINDOW: Final[tuple[int, int]] = (18, 20)
QUIET_WINDOW: Final[tuple[int, int]] = (1, 5)

# Weekday demand by hour, normalised so that the busiest hour is exactly 1.0.
# Index i is the level *at* hour i o'clock; values between hours are
# interpolated, so 1.00 at index 10 means the morning peak crests at 10:00.
WEEKDAY_SHAPE: Final[tuple[float, ...]] = (
    0.16, 0.10, 0.08, 0.08, 0.10, 0.18,   # 00-05  night trough
    0.32, 0.52, 0.78, 0.96, 1.00, 0.88,   # 06-11  morning build and peak
    0.74, 0.70, 0.68, 0.70, 0.76, 0.86,   # 12-17  midday plateau, evening build
    0.97, 1.00, 0.90, 0.70, 0.50, 0.30,   # 18-23  evening peak and unwind
)

# Weekend demand on the *same absolute scale* as the weekday curve, so the two
# are directly comparable: the weekend crest of 0.80 means Sunday evening is
# four fifths as loaded as Tuesday evening.
WEEKEND_SHAPE: Final[tuple[float, ...]] = (
    0.25, 0.17, 0.10, 0.08, 0.08, 0.10,   # 00-05  later, deeper night tail
    0.15, 0.22, 0.30, 0.40, 0.50, 0.57,   # 06-11  no commute peak, slow rise
    0.61, 0.61, 0.59, 0.61, 0.66, 0.72,   # 12-17  flat afternoon
    0.78, 0.80, 0.76, 0.66, 0.52, 0.37,   # 18-23  leisure evening peak
)

#: Demand sensitivity by OSM ``highway`` class. ``primary`` is the reference
#: class at 1.0; the peak-ratio calibration is defined on it. Values above 1
#: mean the class saturates harder than a primary arterial at the same hour.
ROAD_CLASS_SENSITIVITY: Final[dict[str, float]] = {
    "motorway": 1.10,
    "motorway_link": 1.00,
    "trunk": 1.10,
    "trunk_link": 1.00,
    "primary": 1.00,
    "primary_link": 0.95,
    "secondary": 0.92,
    "secondary_link": 0.88,
    "tertiary": 0.80,
    "tertiary_link": 0.76,
    "unclassified": 0.60,
    "road": 0.60,
    "busway": 0.70,
    "residential": 0.55,
    "living_street": 0.40,
    "service": 0.35,
}

DEFAULT_CLASS_SENSITIVITY: Final[float] = 0.60

#: The reference class the peak calibration is defined against.
REFERENCE_ROAD_CLASS: Final[str] = "primary"

Interpolation = Literal["linear", "pchip", "step"]


def class_sensitivity(road_class: str) -> float:
    """Demand sensitivity multiplier for an OSM highway class."""
    return ROAD_CLASS_SENSITIVITY.get(road_class, DEFAULT_CLASS_SENSITIVITY)


def class_sensitivity_array(road_classes: Sequence[str]) -> np.ndarray:
    """Vectorised sensitivity lookup, one value per edge.

    Loops over the distinct classes (a handful) rather than over the edges, so
    a 34k-edge network costs about twenty boolean masks.
    """
    classes = np.asarray(road_classes, dtype=object)
    out = np.full(classes.shape, DEFAULT_CLASS_SENSITIVITY, dtype=np.float64)
    for name in set(classes.tolist()):
        out[classes == name] = class_sensitivity(str(name))
    return out


@dataclass
class DemandProfile:
    """A weekly demand profile with smooth interpolation between hours.

    Parameters
    ----------
    weekday, weekend:
        24 hourly levels each, on a shared absolute scale.
    peak_ratio:
        Target ratio of peak to free-flow travel time on the reference road
        class. The peak saturation is derived from it by inverting BPR, so
        changing the target changes the model consistently instead of requiring
        the hourly numbers to be retuned.
    interpolation:
        ``"pchip"`` (default) gives a smooth, shape-preserving curve through the
        hourly points with no overshoot -- important because an overshooting
        cubic spline would invent a demand spike between 09:00 and 10:00 that
        is not in the data. ``"linear"`` is the cheap fallback and is used
        automatically if SciPy is unavailable. ``"step"`` holds each hour flat
        and exists mainly to make the interpolation effect visible in tests.
    alpha, beta:
        BPR coefficients used for the calibration inversion. They must match
        the ones the simulator evaluates with, or the calibration is void.

    All interpolation is *periodic*: 23:30 blends into 00:00 rather than
    falling off the end of the array.
    """

    weekday: np.ndarray = field(default_factory=lambda: np.array(WEEKDAY_SHAPE, dtype=np.float64))
    weekend: np.ndarray = field(default_factory=lambda: np.array(WEEKEND_SHAPE, dtype=np.float64))
    peak_ratio: float = PEAK_TRAVEL_TIME_RATIO
    interpolation: Interpolation = "pchip"
    alpha: float = BPR_ALPHA
    beta: float = BPR_BETA
    name: str = "india_urban_synthetic_v1"

    def __post_init__(self) -> None:
        self.weekday = np.ascontiguousarray(self.weekday, dtype=np.float64)
        self.weekend = np.ascontiguousarray(self.weekend, dtype=np.float64)
        if self.weekday.shape != (HOURS_PER_DAY,) or self.weekend.shape != (HOURS_PER_DAY,):
            raise ValueError("weekday and weekend profiles must both have 24 hourly values")
        if np.any(self.weekday < 0) or np.any(self.weekend < 0):
            raise ValueError("demand levels must be non-negative")
        # Peak saturation on the reference class, derived not guessed.
        self._peak_saturation = saturation_for_ratio(self.peak_ratio, self.alpha, self.beta)
        self._interp_cache: dict[bool, object] = {}
        if self.interpolation == "pchip":
            try:  # SciPy is a hard dependency of the project, but degrade anyway.
                from scipy.interpolate import PchipInterpolator  # noqa: F401
            except Exception:  # pragma: no cover - only on a broken environment
                self.interpolation = "linear"

    # ------------------------------------------------------------- internals
    def _shape(self, weekend: bool) -> np.ndarray:
        return self.weekend if weekend else self.weekday

    def _interpolator(self, weekend: bool):
        """Build (once) a periodic PCHIP over three tiled copies of the day.

        Tiling three days and evaluating in the middle copy is the simplest way
        to get periodicity without special-casing the wrap point, and costs
        nothing because the interpolator is built once and cached.
        """
        if weekend in self._interp_cache:
            return self._interp_cache[weekend]
        from scipy.interpolate import PchipInterpolator

        y = np.tile(self._shape(weekend), 3)
        x = np.arange(-HOURS_PER_DAY, 2 * HOURS_PER_DAY, dtype=np.float64)
        interp = PchipInterpolator(x, y, extrapolate=False)
        self._interp_cache[weekend] = interp
        return interp

    # ---------------------------------------------------------------- lookup
    def multiplier(self, hour: float | np.ndarray, weekend: bool = False) -> np.ndarray:
        """Demand level at a fractional hour of day, on the profile's own scale.

        ``hour`` is taken modulo 24, so 25.5 and 1.5 give the same answer.
        """
        h = np.asarray(hour, dtype=np.float64) % float(HOURS_PER_DAY)
        shape = self._shape(weekend)
        if self.interpolation == "step":
            return shape[np.floor(h).astype(np.intp) % HOURS_PER_DAY]
        if self.interpolation == "linear":
            xs = np.arange(-1.0, HOURS_PER_DAY + 1.0)
            ys = np.concatenate(([shape[-1]], shape, [shape[0]]))
            return np.interp(h, xs, ys)
        return np.asarray(self._interpolator(weekend)(h), dtype=np.float64)

    def saturation(self, hour: float | np.ndarray, weekend: bool = False) -> np.ndarray:
        """Saturation ratio ``v / c`` on the reference road class at ``hour``.

        This is the quantity the simulator scales by per-edge class sensitivity
        and noise before handing it to the volume-delay function.
        """
        return self._peak_saturation * self.multiplier(hour, weekend)

    def at_minute(self, minute_of_week: float) -> tuple[np.ndarray, bool]:
        """Saturation and weekend flag for a minute offset within the week.

        Minute 0 is Monday 00:00. Saturday and Sunday use the weekend curve.
        """
        m = float(minute_of_week) % MINUTES_PER_WEEK
        day = int(m // MINUTES_PER_DAY)
        weekend = day >= 5
        hour = (m % MINUTES_PER_DAY) / 60.0
        return self.saturation(hour, weekend), weekend

    # ----------------------------------------------------------- diagnostics
    @property
    def peak_saturation(self) -> float:
        """Saturation ratio at the crest of the weekday profile (reference class)."""
        return float(self._peak_saturation)

    def reference_peak_ratio(self) -> float:
        """Measured peak/free-flow travel-time ratio on the reference class.

        Recomputed from the profile and the BPR coefficients rather than
        returned from the stored target, so that a mis-specified profile shows
        up here instead of being silently trusted.
        """
        from qroute.traffic.bpr import bpr_multiplier

        peak = float(np.max(self.multiplier(np.linspace(0.0, 24.0, 24 * 60, endpoint=False))))
        return float(bpr_multiplier(self._peak_saturation * peak, self.alpha, self.beta))

    def daily_mean_ratio(self, weekend: bool = False, samples: int = 24 * 60) -> float:
        """Travel-time ratio averaged over the whole day on the reference class.

        The peak ratio alone is a poor summary because a city spends most of
        the day away from the peak; this is the number to compare against an
        annual congestion index, which is a time-weighted average.
        """
        from qroute.traffic.bpr import bpr_multiplier

        hours = np.linspace(0.0, 24.0, samples, endpoint=False)
        sat = self._peak_saturation * self.multiplier(hours, weekend)
        return float(np.mean(bpr_multiplier(sat, self.alpha, self.beta)))

    def peak_hours(self, weekend: bool = False) -> dict[str, float]:
        """Fractional hours at which the morning and evening crests occur."""
        hours = np.linspace(0.0, 24.0, 24 * 60, endpoint=False)
        vals = self.multiplier(hours, weekend)
        morning = (hours >= 5.0) & (hours < 13.0)
        evening = (hours >= 15.0) & (hours < 23.0)
        return {
            "morning": float(hours[morning][int(np.argmax(vals[morning]))]),
            "evening": float(hours[evening][int(np.argmax(vals[evening]))]),
            "quietest": float(hours[int(np.argmin(vals))]),
        }

    def as_dict(self) -> dict[str, object]:
        """Serialisable description, for the API and for run manifests."""
        return {
            "name": self.name,
            "synthetic": True,
            "calibration": {
                "peak_travel_time_ratio_target": self.peak_ratio,
                "peak_travel_time_ratio_achieved": round(self.reference_peak_ratio(), 4),
                "peak_saturation": round(self.peak_saturation, 4),
                "reference_road_class": REFERENCE_ROAD_CLASS,
                "weekday_mean_ratio": round(self.daily_mean_ratio(False), 4),
                "weekend_mean_ratio": round(self.daily_mean_ratio(True), 4),
            },
            "interpolation": self.interpolation,
            "weekday": [round(float(v), 4) for v in self.weekday],
            "weekend": [round(float(v), 4) for v in self.weekend],
            "peak_hours_weekday": self.peak_hours(False),
        }


def default_profile(**overrides) -> DemandProfile:
    """The profile the platform uses unless a caller supplies another.

    Kept as a function rather than a module-level singleton so that two
    simulators in the same process cannot share (and mutate) one object.
    """
    return DemandProfile(**overrides)


def flat_profile(level: float = 0.0) -> DemandProfile:
    """A constant-demand profile, used as the synthetic fallback and in tests.

    ``level`` is on the same scale as the shaped profiles, so ``0.0`` gives
    free-flow conditions everywhere and ``1.0`` gives permanent peak. This is
    the documented fallback when no time-of-day information is wanted at all,
    for example when reproducing a static CVRP benchmark inside the traffic
    stack: with ``level = 0`` every edge keeps its free-flow travel time and
    the traffic layer becomes a no-op.
    """
    arr = np.full(HOURS_PER_DAY, float(level), dtype=np.float64)
    return DemandProfile(weekday=arr, weekend=arr.copy(), name=f"flat_{level:g}")
