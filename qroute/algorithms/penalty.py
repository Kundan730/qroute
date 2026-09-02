"""Adaptive penalty weights for the soft constraints of the routing problem.

Why controlled infeasibility instead of rejection
-------------------------------------------------
The obvious way to handle capacity, time-window and duration constraints is to
reject any candidate that violates one. That is simple, and on loosely
constrained instances it works. It fails badly when the constraints are tight,
for two related reasons.

First, the feasible region of a tightly constrained VRP is *disconnected* under
the usual neighbourhood operators. Two good feasible solutions can differ by a
sequence of moves in which every intermediate solution is slightly overloaded.
A search that rejects infeasibility cannot cross that ridge; it has to go around
it, and often there is no way around within the move set. Second, rejection
throws away information: a solution that is one unit over capacity but ten
percent cheaper is telling the search exactly where to look, and discarding it
wastes the evaluation that produced it.

The alternative, used by the strongest VRP metaheuristics (Vidal et al.'s hybrid
genetic search, Cordeau and Laporte's unified tabu search), is to allow
infeasible solutions but charge them a penalty, and to *tune that penalty during
the run* so the search spends a controlled fraction of its time just outside the
feasible region. If almost everything the search produces is feasible, the
penalty is too high and the search is being needlessly constrained, so it is
relaxed. If almost nothing is feasible, the penalty is too low and the search is
drifting into a region whose costs are meaningless, so it is tightened.

The target fraction is the one parameter that matters. Vidal et al. report that
holding roughly 20 percent of recent solutions feasible per constraint works
well across instance families, which is the default here. The multiplicative
adjustment (times 1.2 up, times 0.85 down) is deliberately asymmetric: tightening
is faster than loosening, so a run that wanders far into infeasibility is pulled
back quickly, while a run that is comfortably feasible relaxes gently and does
not oscillate.

Honest scope note: this is a heuristic controller, not a method with a
convergence guarantee. Its only claim is that it removes the need to hand-tune
one penalty per instance family, and the ``history`` attribute exists so that
claim can be inspected rather than asserted.

Usage
-----
The manager is passive: it does not know about the search. An optimiser calls
:meth:`register` once per evaluated candidate with that candidate's violation
amounts, and reads :attr:`capacity`, :attr:`time_window` and :attr:`duration`
whenever it needs the current weights. Because the decoder caches its penalties
at construction time, an optimiser that wants live adaptation must copy the new
values onto the decoder itself; :meth:`apply_to` does that in one call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from qroute.core.types import SolutionStats
from qroute.problems.instance import Instance

#: Names of the three soft constraints this manager controls, in a fixed order
#: so that ``history`` rows and ``as_dict`` keys are stable across versions.
CONSTRAINTS: tuple[str, str, str] = ("capacity", "time_window", "duration")


@dataclass
class PenaltyRecord:
    """One row of the adjustment history, for plotting and for the report."""

    evaluations: int
    capacity: float
    time_window: float
    duration: float
    feasible_capacity: float
    feasible_time_window: float
    feasible_duration: float

    def as_dict(self) -> dict[str, float]:
        return {
            "evaluations": float(self.evaluations),
            "capacity": self.capacity,
            "time_window": self.time_window,
            "duration": self.duration,
            "feasible_capacity": self.feasible_capacity,
            "feasible_time_window": self.feasible_time_window,
            "feasible_duration": self.feasible_duration,
        }


class AdaptivePenalty:
    """Multiplicative penalty controller following the HGS scheme.

    Parameters
    ----------
    instance:
        Used only to derive sensible starting values. Passing ``None`` keeps the
        neutral defaults, which is what the unit tests do.
    target:
        Desired fraction of recently evaluated candidates that are feasible with
        respect to each constraint, considered separately.
    tolerance:
        Dead band around ``target``. Nothing is adjusted while the measured
        fraction lies within ``target +/- tolerance``, which stops the weights
        from twitching on sampling noise.
    interval:
        Number of registered candidates between adjustments.
    increase, decrease:
        Multipliers applied when the feasible fraction is respectively too low
        and too high.
    floor, ceiling:
        Hard bounds. The ceiling matters: an unbounded penalty eventually makes
        every infeasible candidate compare equal at floating-point precision,
        which silently turns the scheme back into rejection.
    """

    def __init__(
        self,
        instance: Instance | None = None,
        target: float = 0.2,
        tolerance: float = 0.05,
        interval: int = 100,
        increase: float = 1.2,
        decrease: float = 0.85,
        floor: float = 0.1,
        ceiling: float = 1e5,
        capacity: float | None = None,
        time_window: float | None = None,
        duration: float | None = None,
    ) -> None:
        if not 0.0 < target < 1.0:
            raise ValueError("target must lie strictly between 0 and 1")
        if interval < 1:
            raise ValueError("interval must be at least 1")
        if not (0.0 < floor <= ceiling):
            raise ValueError("require 0 < floor <= ceiling")
        if increase <= 1.0 or not 0.0 < decrease < 1.0:
            raise ValueError("increase must exceed 1 and decrease must lie in (0, 1)")

        self.target = float(target)
        self.tolerance = float(tolerance)
        self.interval = int(interval)
        self.increase = float(increase)
        self.decrease = float(decrease)
        self.floor = float(floor)
        self.ceiling = float(ceiling)

        base = self._initial_capacity_penalty(instance)
        self._values: dict[str, float] = {
            "capacity": self._clamp(capacity if capacity is not None else base),
            "time_window": self._clamp(time_window if time_window is not None else 1.0),
            "duration": self._clamp(duration if duration is not None else 1.0),
        }
        self._initial: dict[str, float] = dict(self._values)

        self._seen = 0                      # candidates since the last adjustment
        self._feasible = {c: 0 for c in CONSTRAINTS}
        self.total_registered = 0
        self.history: list[PenaltyRecord] = []

    # ------------------------------------------------------------- construction
    @staticmethod
    def _initial_capacity_penalty(instance: Instance | None) -> float:
        """Scale the capacity penalty to the instance's own units.

        One unit of overload should cost about as much as a long arc, otherwise
        the weight is meaningless: on an instance whose arcs cost ~1000 a
        penalty of 1.0 is no penalty at all, and on one whose arcs cost ~0.01 a
        penalty of 1000 is pure rejection. The exchange rate between the two
        units is what fixes this, and it is a property of the instance.

        The exact rule is delegated to
        :meth:`~qroute.algorithms.decoder.Decoder.default_capacity_penalty`
        rather than duplicated, so the starting point of the adaptive scheme and
        the fixed weight a decoder uses when no controller is attached can never
        drift apart. The import is deferred because it pulls in the compiled
        kernels, and this module is otherwise cheap to import.
        """
        if instance is None:
            return 1.0
        from qroute.algorithms.decoder import Decoder

        return Decoder.default_capacity_penalty(instance)

    def _clamp(self, value: float) -> float:
        return float(min(max(float(value), self.floor), self.ceiling))

    # ---------------------------------------------------------------- read-out
    @property
    def capacity(self) -> float:
        return self._values["capacity"]

    @property
    def time_window(self) -> float:
        return self._values["time_window"]

    @property
    def duration(self) -> float:
        return self._values["duration"]

    def as_dict(self) -> dict[str, float]:
        return dict(self._values)

    def feasible_fractions(self) -> dict[str, float]:
        """Feasible fraction measured so far in the current window."""
        if self._seen == 0:
            return {c: float("nan") for c in CONSTRAINTS}
        return {c: self._feasible[c] / self._seen for c in CONSTRAINTS}

    def apply_to(self, decoder: Any) -> None:
        """Copy the current weights onto a :class:`~qroute.algorithms.decoder.Decoder`.

        The decoder holds its penalties as plain floats read on the hot path, so
        an optimiser that adapts penalties mid-run has to push the new values in
        rather than expecting the decoder to pull them.
        """
        decoder.pen_cap = self.capacity
        decoder.pen_tw = self.time_window
        decoder.pen_dur = self.duration

    # ------------------------------------------------------------------ update
    def register(self, evidence: Any) -> bool:
        """Record one evaluated candidate. Returns True if the weights changed.

        ``evidence`` may be any of:

        * a :class:`~qroute.core.types.SolutionStats`;
        * a mapping with any of the keys ``capacity``, ``time_window``,
          ``duration`` (or the ``*_violation`` spellings), holding either a
          violation amount or a boolean "is feasible" flag;
        * a sequence of three violation amounts or flags in the order
          ``(capacity, time_window, duration)``.

        Booleans are read as "feasible", numbers as "violation amount", which
        covers both the kernels (which return amounts) and callers that only
        know feasibility.
        """
        flags = self._to_flags(evidence)
        self._seen += 1
        self.total_registered += 1
        for name, ok in zip(CONSTRAINTS, flags):
            if ok:
                self._feasible[name] += 1
        if self._seen < self.interval:
            return False
        return self._adjust()

    def register_many(self, evidences: Iterable[Any]) -> bool:
        changed = False
        for e in evidences:
            changed |= self.register(e)
        return changed

    def _adjust(self) -> bool:
        changed = False
        fractions = self.feasible_fractions()
        for name in CONSTRAINTS:
            frac = fractions[name]
            old = self._values[name]
            if frac < self.target - self.tolerance:
                new = self._clamp(old * self.increase)
            elif frac > self.target + self.tolerance:
                new = self._clamp(old * self.decrease)
            else:
                continue
            if new != old:
                self._values[name] = new
                changed = True
        self.history.append(
            PenaltyRecord(
                evaluations=self.total_registered,
                capacity=self._values["capacity"],
                time_window=self._values["time_window"],
                duration=self._values["duration"],
                feasible_capacity=fractions["capacity"],
                feasible_time_window=fractions["time_window"],
                feasible_duration=fractions["duration"],
            )
        )
        self._seen = 0
        for c in CONSTRAINTS:
            self._feasible[c] = 0
        return changed

    def reset(self) -> None:
        """Return to the initial weights and clear the history."""
        self._values = dict(self._initial)
        self._seen = 0
        self._feasible = {c: 0 for c in CONSTRAINTS}
        self.total_registered = 0
        self.history = []

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _flag(value: Any) -> bool:
        """Interpret one field as "this constraint is satisfied"."""
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return float(value) <= 1e-9

    @classmethod
    def _to_flags(cls, evidence: Any) -> tuple[bool, bool, bool]:
        if isinstance(evidence, SolutionStats):
            return (
                evidence.capacity_violation <= 1e-9,
                evidence.time_window_violation <= 1e-9,
                evidence.duration_violation <= 1e-9,
            )
        if isinstance(evidence, Mapping):
            out = []
            for name in CONSTRAINTS:
                if name in evidence:
                    out.append(cls._flag(evidence[name]))
                elif f"{name}_violation" in evidence:
                    out.append(cls._flag(evidence[f"{name}_violation"]))
                else:
                    out.append(True)  # constraint absent means nothing to violate
            return tuple(out)  # type: ignore[return-value]
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            vals = list(evidence)[:3]
            while len(vals) < 3:
                vals.append(True)
            return tuple(cls._flag(v) for v in vals)  # type: ignore[return-value]
        raise TypeError(f"cannot interpret {type(evidence).__name__} as violation evidence")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        v = self._values
        return (f"AdaptivePenalty(cap={v['capacity']:.3g}, tw={v['time_window']:.3g}, "
                f"dur={v['duration']:.3g}, target={self.target:.2f})")
