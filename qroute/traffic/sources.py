"""Where traffic conditions come from: simulation, or a live feed.

The rest of the platform must not care whether the congestion it is routing
around is simulated or measured, so every source returns the same thing: a
per-edge **speed factor**, the current speed as a fraction of the free-flow
speed. A factor of 1.0 is free flow, 0.5 means the link is running at half
speed and therefore takes twice as long. Travel time follows as
``t = t0 / factor``.

Speed factor rather than travel time is the common currency because it is
exactly what live traffic APIs report (``currentSpeed / freeFlowSpeed``), and
because it is dimensionless -- a live observation on a 400 m OSM edge can be
transferred to the model without knowing how the provider measured length.

Honesty about live data
-----------------------
:class:`TomTomFlowSource` needs an API key in ``TOMTOM_API_KEY``. If the key is
absent it raises :class:`MissingAPIKeyError` with an actionable message. It
never silently degrades. Falling back to simulation is a decision the caller
makes explicitly through :func:`resolve_source`, and the returned observation
then carries ``live=False`` and ``fallback_reason`` set, so the flag can be put
in front of the user. A demo that quietly presents simulated numbers as live
traffic would be dishonest, and the code is arranged so that doing it takes
deliberate effort.

No test in this repository calls the live API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

TOMTOM_API_KEY_ENV = "TOMTOM_API_KEY"
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/{style}/{zoom}/json"


class TrafficSourceError(RuntimeError):
    """Base class for every failure a traffic source can report."""


class MissingAPIKeyError(TrafficSourceError):
    """Raised when a live source is constructed without credentials."""


class TrafficFetchError(TrafficSourceError):
    """Raised when a live source is reachable but the request failed."""


@dataclass
class TrafficObservation:
    """One snapshot of network conditions.

    Attributes
    ----------
    speed_factor:
        ``(n_edges,)`` array in ``[0, 1]``. Zero means impassable.
    live:
        True only when the numbers came from a real measurement. Anything the
        user interface labels "live" must check this flag.
    source:
        Human-readable provenance string for the audit trail.
    timestamp:
        Unix time at which the observation was produced.
    coverage:
        Fraction of edges for which the source supplied a real value. A live
        source that could only price 40 of 34,000 edges has coverage 0.0012,
        and the rest were filled from the fallback -- the dashboard should say
        so rather than implying the whole network was measured.
    fallback_reason:
        Set when this observation is a substitute for a source that failed.
    """

    speed_factor: np.ndarray
    live: bool
    source: str
    timestamp: float = field(default_factory=time.time)
    coverage: float = 1.0
    fallback_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def travel_times(self, free_flow_time: np.ndarray) -> np.ndarray:
        """Convert to absolute travel times; a zero factor becomes ``inf``."""
        f = np.asarray(self.speed_factor, dtype=np.float64)
        t0 = np.asarray(free_flow_time, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0.0, t0 / np.where(f > 0.0, f, 1.0), np.inf)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable header; the per-edge array is deliberately excluded."""
        return {
            "live": self.live,
            "source": self.source,
            "timestamp": self.timestamp,
            "coverage": round(float(self.coverage), 6),
            "fallback_reason": self.fallback_reason,
            "n_edges": int(np.asarray(self.speed_factor).shape[0]),
            "mean_speed_factor": round(float(np.mean(self.speed_factor)), 4),
            **self.meta,
        }


class TrafficSource(ABC):
    """Interface every traffic source implements."""

    #: Whether observations from this source represent real measurements.
    live: bool = False
    name: str = "abstract"

    @abstractmethod
    def fetch(self) -> TrafficObservation:
        """Return the current per-edge speed factors."""

    def close(self) -> None:
        """Release any resources. The default implementation does nothing."""

    def __enter__(self) -> "TrafficSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SimulatedSource(TrafficSource):
    """Wraps a :class:`~qroute.traffic.simulator.TrafficSimulator`.

    This is the default and the fallback. It is honest about itself: every
    observation it produces has ``live=False``.

    ``advance_minutes`` lets the source act as a ticking clock -- each
    :meth:`fetch` moves the simulated time forward by that many minutes, which
    is how the API's polling loop animates a day without the caller having to
    drive the simulator directly. Leave it at 0 for a source that simply
    reports whatever time the simulator is currently at.
    """

    live = False
    name = "simulated"

    def __init__(self, simulator, advance_minutes: float = 0.0) -> None:
        self.simulator = simulator
        self.advance_minutes = float(advance_minutes)

    def fetch(self) -> TrafficObservation:
        if self.advance_minutes:
            self.simulator.advance(self.advance_minutes)
        return TrafficObservation(
            speed_factor=self.simulator.speed_factors(),
            live=False,
            source=f"simulated:{self.simulator.profile.name}",
            coverage=1.0,
            meta={
                "time_minutes": self.simulator.time_minutes,
                "hour_of_day": round(self.simulator.hour_of_day, 4),
                "n_active_events": len(self.simulator.events.active_at(self.simulator.time_minutes)),
            },
        )


class TomTomFlowSource(TrafficSource):
    """Live speeds from the TomTom Traffic Flow Segment Data API.

    The API is point-based: one request returns the flow state of the road
    segment nearest a given coordinate. There is no bulk endpoint, so a network
    of 34,000 edges cannot be covered -- and should not be, both because of the
    request budget and because most of those edges are residential stubs no
    provider measures. The intended use is to price a **small set of monitored
    corridors** (the arterials the routes actually use) and let the simulator
    supply the rest; :attr:`TrafficObservation.coverage` reports what fraction
    that was.

    Parameters
    ----------
    probes:
        Sequence of ``(edge_index, latitude, longitude)``. One HTTP request is
        made per probe.
    n_edges:
        Size of the output array.
    api_key:
        Read from ``TOMTOM_API_KEY`` when omitted. Never hard-code a key, and
        never log the value -- :meth:`describe` deliberately reports only
        whether a key is present.
    baseline:
        Optional source (normally a :class:`SimulatedSource`) used to fill the
        edges no probe covers. When omitted, uncovered edges report 1.0, i.e.
        free flow, which is optimistic; pass a baseline in production.
    zoom, style:
        TomTom request parameters. Zoom 10 is roughly the city-arterial level
        of detail; ``absolute`` returns raw speeds rather than speeds relative
        to a reference.
    timeout, max_retries:
        Per-request network settings. Failures raise
        :class:`TrafficFetchError`; the caller decides whether to fall back.
    """

    live = True
    name = "tomtom_flow"

    def __init__(
        self,
        probes: Sequence[tuple[int, float, float]],
        n_edges: int,
        api_key: str | None = None,
        baseline: TrafficSource | None = None,
        zoom: int = 10,
        style: str = "absolute",
        timeout: float = 5.0,
        max_retries: int = 2,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(TOMTOM_API_KEY_ENV)
        if not key:
            raise MissingAPIKeyError(
                "TomTom live traffic requires an API key. Set the environment "
                f"variable {TOMTOM_API_KEY_ENV} to a key from "
                "https://developer.tomtom.com/ (the free tier allows 2,500 "
                "requests per day), then restart the service. To run without "
                "live data, use qroute.traffic.sources.SimulatedSource, or call "
                "resolve_source(prefer_live=True, ...) which falls back to "
                "simulation and marks the observation live=False."
            )
        self._key = key
        self.probes = [(int(i), float(lat), float(lon)) for i, lat, lon in probes]
        self.n_edges = int(n_edges)
        self.baseline = baseline
        self.zoom = int(zoom)
        self.style = str(style)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    # ------------------------------------------------------------- internals
    def _request(self, lat: float, lon: float) -> dict[str, Any]:
        """One flow-segment request. Separated so tests can monkeypatch it."""
        url = TOMTOM_FLOW_URL.format(style=self.style, zoom=self.zoom)
        query = urllib.parse.urlencode({"point": f"{lat},{lon}", "unit": "KMPH", "key": self._key})
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(f"{url}?{query}", timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last = exc
                if attempt < self.max_retries:
                    # Linear back-off: the free tier rate-limits rather than
                    # failing hard, so a short pause is usually enough.
                    time.sleep(0.5 * (attempt + 1))
        raise TrafficFetchError(f"TomTom flow request failed for ({lat}, {lon}): {last}")

    # ------------------------------------------------------------------ fetch
    def fetch(self) -> TrafficObservation:
        if self.baseline is not None:
            base = np.asarray(self.baseline.fetch().speed_factor, dtype=np.float64).copy()
            if base.shape != (self.n_edges,):
                raise TrafficFetchError("baseline source returned the wrong edge count")
        else:
            base = np.ones(self.n_edges, dtype=np.float64)

        covered = 0
        confidences: list[float] = []
        for idx, lat, lon in self.probes:
            payload = self._request(lat, lon)
            seg = payload.get("flowSegmentData") or {}
            cur = seg.get("currentSpeed")
            free = seg.get("freeFlowSpeed")
            if not cur or not free:
                continue
            if 0 <= idx < self.n_edges:
                base[idx] = float(np.clip(float(cur) / float(free), 0.0, 1.0))
                covered += 1
                confidences.append(float(seg.get("confidence", 1.0)))

        return TrafficObservation(
            speed_factor=base,
            live=covered > 0,
            source="tomtom:flowSegmentData",
            coverage=covered / max(self.n_edges, 1),
            meta={
                "probes_requested": len(self.probes),
                "probes_covered": covered,
                "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
                "baseline": getattr(self.baseline, "name", None),
            },
        )

    def describe(self) -> dict[str, Any]:
        """Configuration summary. Never includes the key itself."""
        return {
            "name": self.name,
            "live": True,
            "n_probes": len(self.probes),
            "n_edges": self.n_edges,
            "zoom": self.zoom,
            "style": self.style,
            "api_key_present": True,
        }


def api_key_available(env_var: str = TOMTOM_API_KEY_ENV) -> bool:
    """Whether a live-traffic key is configured, without revealing it."""
    return bool(os.environ.get(env_var))


def resolve_source(
    simulator,
    prefer_live: bool = False,
    probes: Sequence[tuple[int, float, float]] | None = None,
    api_key: str | None = None,
    **tomtom_kwargs,
) -> tuple[TrafficSource, str | None]:
    """Return ``(source, fallback_reason)``, choosing live data when possible.

    This is the single place in the platform where the live/simulated decision
    is made. It returns the reason for any fallback rather than swallowing it,
    so the caller can surface a visible flag. ``fallback_reason`` is ``None``
    only when a genuinely live source was constructed.

    Falling back is never silent in the sense that matters: the observation the
    returned source produces carries ``live=False``, and the reason string is
    handed straight back to the caller.
    """
    if not prefer_live:
        return SimulatedSource(simulator), None
    if not probes:
        return SimulatedSource(simulator), "no monitored corridors were configured for live probing"
    try:
        baseline = SimulatedSource(simulator)
        src = TomTomFlowSource(
            probes=probes,
            n_edges=simulator.edges.n_edges,
            api_key=api_key,
            baseline=baseline,
            **tomtom_kwargs,
        )
        return src, None
    except MissingAPIKeyError as exc:
        # The whole message is passed through, including the environment
        # variable name and the sign-up URL: the point of the fallback flag is
        # to tell the operator how to fix it, not merely that it happened.
        return SimulatedSource(simulator), str(exc)


def fallback_observation(observation: TrafficObservation, reason: str) -> TrafficObservation:
    """Tag an observation as a substitute for a failed live source."""
    observation.fallback_reason = reason
    observation.live = False
    return observation
