"""Incidents, lane blockages and road closures on a scheduled timeline.

An incident is modelled the way traffic engineering models it: not by inventing
a delay, but by **reducing the capacity of the affected links** and letting the
volume-delay function produce the delay. That is the reason the numbers here
are capacity multipliers rather than time penalties -- a lane blockage on an
empty street at 03:00 should cost almost nothing, while the same blockage on a
saturated arterial at 19:00 should be catastrophic, and a capacity-based model
gets both right for free.

Three event kinds are supported:

``LANE_BLOCKAGE``
    A partial obstruction. The residual-capacity multiplier comes from the
    Highway Capacity Manual's incident table (see
    :data:`HCM_RESIDUAL_CAPACITY`), keyed by the number of lanes in the
    direction and by what is blocked.

``CLOSURE``
    The link is removed from the network for the duration. Implemented as
    capacity zero plus an explicit boolean mask, because a router must be able
    to distinguish "very slow" from "not traversable".

``SLOWDOWN``
    A direct speed reduction with no capacity change -- waterlogging, a
    procession, a badly resurfaced stretch. Modelled as a travel-time
    multiplier, since the mechanism is a lower free speed rather than a lost
    lane.

Times are in minutes on the simulator's own clock, so an event scheduled at
minute 1140 fires at 19:00 of simulated day 0.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Iterable, Iterator, Sequence

import numpy as np


class EventKind(str, Enum):
    """What physically happened. Inherits ``str`` so it serialises to JSON."""

    LANE_BLOCKAGE = "lane_blockage"
    CLOSURE = "closure"
    SLOWDOWN = "slowdown"


class BlockageType(str, Enum):
    """The HCM incident categories, in increasing order of severity."""

    SHOULDER_DISABLEMENT = "shoulder_disablement"
    SHOULDER_ACCIDENT = "shoulder_accident"
    ONE_LANE_BLOCKED = "one_lane_blocked"
    TWO_LANES_BLOCKED = "two_lanes_blocked"


# Proportion of the original capacity that survives an incident, from the
# Highway Capacity Manual's freeway-incident residual-capacity table. The two
# rows below are the ones the platform specification pins down; they are used
# verbatim.
#
# Note the shape of the table: on a 2-lane carriageway losing one lane leaves
# only 35 percent of capacity, not the 50 percent a naive "half the lanes"
# argument would give, because the merge and the rubbernecking cost more than
# the lane itself. That non-linearity is the whole reason for using a lookup
# table instead of a formula.
HCM_RESIDUAL_CAPACITY: Final[dict[int, dict[BlockageType, float]]] = {
    2: {
        BlockageType.SHOULDER_DISABLEMENT: 0.95,
        BlockageType.SHOULDER_ACCIDENT: 0.81,
        BlockageType.ONE_LANE_BLOCKED: 0.35,
    },
    3: {
        BlockageType.SHOULDER_DISABLEMENT: 0.99,
        BlockageType.SHOULDER_ACCIDENT: 0.83,
        BlockageType.ONE_LANE_BLOCKED: 0.49,
        BlockageType.TWO_LANES_BLOCKED: 0.17,
    },
}

# Fraction of a lane's own capacity that is lost *in addition* to the blocked
# lane, used only when extrapolating outside the tabulated rows.
_EXTRAPOLATION_FRICTION: Final[float] = 0.30


def residual_capacity(lanes: int, blockage: BlockageType) -> float:
    """Surviving capacity fraction for ``blockage`` on a ``lanes``-lane link.

    For 2 and 3 lanes this is a direct table lookup and is exact. Outside those
    rows the platform must still answer, so it extrapolates: the surviving
    fraction is taken as the share of unblocked lanes reduced by a fixed
    friction term. This extrapolation is **not** from the HCM and is flagged as
    such by :attr:`TrafficEvent.tabulated`; it is a defensible interpolation of
    the tabulated behaviour, not a published figure.

    A single-lane link with any lane blocked is a closure, and returns 0.0.
    """
    lanes = max(1, int(lanes))
    row = HCM_RESIDUAL_CAPACITY.get(lanes)
    if row is not None and blockage in row:
        return row[blockage]

    if blockage is BlockageType.SHOULDER_DISABLEMENT:
        # Shoulder events barely touch capacity and the tabulated values rise
        # towards 1.0 with lane count; 0.99 for 3+ lanes is the safe reading.
        return 0.95 if lanes <= 2 else 0.99
    if blockage is BlockageType.SHOULDER_ACCIDENT:
        return 0.81 if lanes <= 2 else min(0.87, 0.83 + 0.02 * (lanes - 3))

    blocked = 1 if blockage is BlockageType.ONE_LANE_BLOCKED else 2
    if blocked >= lanes:
        return 0.0
    open_share = (lanes - blocked) / lanes
    return float(max(0.0, open_share * (1.0 - _EXTRAPOLATION_FRICTION)))


_event_ids = itertools.count(1)


@dataclass
class TrafficEvent:
    """A scheduled disruption affecting a set of edges.

    Attributes
    ----------
    edges:
        Indices into the simulator's edge array. Integer indices rather than
        ``(u, v, key)`` tuples because the simulator applies events with
        vectorised fancy indexing; :meth:`TrafficSimulator.add_event` accepts
        edge keys and resolves them for you.
    start_minute, duration_minutes:
        Activation window on the simulator clock. A duration of ``inf`` means
        the event never clears, which is how a permanent closure (a flyover
        under construction) is expressed.
    severity:
        Scales the *reduction*, not the multiplier: ``severity = 0`` is a
        no-op and ``severity = 1`` is the full tabulated effect. It exists so
        that a partially-cleared incident can be tapered rather than switched
        off abruptly.
    lanes, blockage:
        Only meaningful for ``LANE_BLOCKAGE``; they select the HCM row.
    speed_multiplier:
        Only meaningful for ``SLOWDOWN``; the factor applied to free speed,
        so 0.5 halves the speed and doubles the travel time.
    """

    kind: EventKind
    edges: tuple[int, ...]
    start_minute: float = 0.0
    duration_minutes: float = 60.0
    severity: float = 1.0
    lanes: int = 2
    blockage: BlockageType = BlockageType.ONE_LANE_BLOCKED
    speed_multiplier: float = 0.5
    description: str = ""
    event_id: int = field(default_factory=lambda: next(_event_ids))

    def __post_init__(self) -> None:
        self.kind = EventKind(self.kind)
        if not isinstance(self.blockage, BlockageType):
            self.blockage = BlockageType(self.blockage)
        self.edges = tuple(int(e) for e in self.edges)
        if not self.edges:
            raise ValueError("an event must affect at least one edge")
        if self.duration_minutes <= 0:
            raise ValueError("event duration must be positive")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must lie in [0, 1]")
        if not 0.0 < self.speed_multiplier <= 1.0:
            raise ValueError("speed_multiplier must lie in (0, 1]")

    # ----------------------------------------------------------------- timing
    @property
    def end_minute(self) -> float:
        return self.start_minute + self.duration_minutes

    def is_active(self, minute: float) -> bool:
        """Half-open interval ``[start, end)``.

        Half-open so that two back-to-back events on the same edge never both
        apply for one instant, which would double-count their capacity loss.
        """
        return self.start_minute <= minute < self.end_minute

    # ------------------------------------------------------------- magnitudes
    @property
    def tabulated(self) -> bool:
        """True when the capacity figure comes straight from the HCM table."""
        if self.kind is not EventKind.LANE_BLOCKAGE:
            return False
        row = HCM_RESIDUAL_CAPACITY.get(int(self.lanes))
        return row is not None and self.blockage in row

    def capacity_multiplier(self) -> float:
        """Factor applied to the edge capacity while the event is active."""
        if self.kind is EventKind.CLOSURE:
            return 0.0
        if self.kind is EventKind.SLOWDOWN:
            return 1.0
        base = residual_capacity(self.lanes, self.blockage)
        # Taper by severity on the *loss*, so severity 0 leaves capacity intact.
        return float(1.0 - self.severity * (1.0 - base))

    def time_multiplier(self) -> float:
        """Factor applied to free-flow travel time while the event is active."""
        if self.kind is not EventKind.SLOWDOWN:
            return 1.0
        loss = 1.0 / self.speed_multiplier - 1.0
        return float(1.0 + self.severity * loss)

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "edges": list(self.edges),
            "n_edges": len(self.edges),
            "start_minute": self.start_minute,
            "duration_minutes": self.duration_minutes,
            "end_minute": self.end_minute,
            "severity": self.severity,
            "lanes": self.lanes,
            "blockage": self.blockage.value if self.kind is EventKind.LANE_BLOCKAGE else None,
            "capacity_multiplier": round(self.capacity_multiplier(), 4),
            "time_multiplier": round(self.time_multiplier(), 4),
            "hcm_tabulated": self.tabulated,
            "description": self.description,
        }


# ------------------------------------------------------------- constructors
def lane_blockage(
    edges: Sequence[int],
    start_minute: float,
    duration_minutes: float,
    lanes: int = 2,
    blockage: BlockageType | str = BlockageType.ONE_LANE_BLOCKED,
    severity: float = 1.0,
    description: str = "",
) -> TrafficEvent:
    """A partial obstruction priced from the HCM residual-capacity table."""
    return TrafficEvent(
        kind=EventKind.LANE_BLOCKAGE,
        edges=tuple(edges),
        start_minute=start_minute,
        duration_minutes=duration_minutes,
        lanes=lanes,
        blockage=BlockageType(blockage),
        severity=severity,
        description=description,
    )


def closure(
    edges: Sequence[int],
    start_minute: float,
    duration_minutes: float = float("inf"),
    description: str = "",
) -> TrafficEvent:
    """A full road closure: the edges become untraversable for the duration."""
    return TrafficEvent(
        kind=EventKind.CLOSURE,
        edges=tuple(edges),
        start_minute=start_minute,
        duration_minutes=duration_minutes,
        description=description,
    )


def slowdown(
    edges: Sequence[int],
    start_minute: float,
    duration_minutes: float,
    speed_multiplier: float = 0.5,
    severity: float = 1.0,
    description: str = "",
) -> TrafficEvent:
    """A direct speed reduction with no capacity loss (flooding, a procession)."""
    return TrafficEvent(
        kind=EventKind.SLOWDOWN,
        edges=tuple(edges),
        start_minute=start_minute,
        duration_minutes=duration_minutes,
        speed_multiplier=speed_multiplier,
        severity=severity,
        description=description,
    )


class EventQueue:
    """The set of scheduled events, with vectorised evaluation at a given time.

    The container is deliberately a flat list. A heap or interval tree would be
    asymptotically better, but the realistic number of concurrent incidents in
    a city sector is tens, not thousands, and a linear scan over tens of events
    costs far less than the per-edge arithmetic that follows it. If that ever
    stops being true the place to fix it is :meth:`active_at`, which is the only
    method that scans.
    """

    def __init__(self, events: Iterable[TrafficEvent] | None = None) -> None:
        self._events: list[TrafficEvent] = list(events or ())

    # ------------------------------------------------------------ container
    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[TrafficEvent]:
        return iter(self._events)

    def __contains__(self, event: object) -> bool:
        return event in self._events

    @property
    def events(self) -> list[TrafficEvent]:
        return list(self._events)

    def add(self, event: TrafficEvent) -> TrafficEvent:
        """Schedule ``event`` and return it, so its id can be kept."""
        if not isinstance(event, TrafficEvent):
            raise TypeError(f"expected TrafficEvent, got {type(event).__name__}")
        self._events.append(event)
        return event

    def extend(self, events: Iterable[TrafficEvent]) -> None:
        for e in events:
            self.add(e)

    def remove(self, event: TrafficEvent | int) -> bool:
        """Remove by object or by ``event_id``; returns whether anything went."""
        target = event.event_id if isinstance(event, TrafficEvent) else int(event)
        before = len(self._events)
        self._events = [e for e in self._events if e.event_id != target]
        return len(self._events) != before

    def clear(self) -> None:
        self._events.clear()

    def get(self, event_id: int) -> TrafficEvent | None:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    # ------------------------------------------------------------ evaluation
    def active_at(self, minute: float) -> list[TrafficEvent]:
        """Events live at ``minute``, in insertion order."""
        return [e for e in self._events if e.is_active(minute)]

    def next_change(self, minute: float) -> float | None:
        """The next time the active set changes, or ``None`` if it never does.

        Lets a caller step the clock event-by-event instead of by fixed
        increments when replaying a scenario.
        """
        times = [
            t
            for e in self._events
            for t in (e.start_minute, e.end_minute)
            if t > minute and np.isfinite(t)
        ]
        return min(times) if times else None

    def apply(
        self, minute: float, n_edges: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(capacity_multiplier, time_multiplier, closed_mask)``.

        All three arrays are length ``n_edges``. Overlapping events *compose
        multiplicatively* on capacity and on time: two independent obstructions
        on one link each take their own share of what is left. That is the
        conservative reading and it keeps the result order-independent, which
        an additive rule would not.
        """
        cap = np.ones(n_edges, dtype=np.float64)
        tmul = np.ones(n_edges, dtype=np.float64)
        closed = np.zeros(n_edges, dtype=bool)
        for e in self._events:
            if not e.is_active(minute):
                continue
            idx = np.asarray(e.edges, dtype=np.intp)
            idx = idx[(idx >= 0) & (idx < n_edges)]
            if idx.size == 0:
                continue
            if e.kind is EventKind.CLOSURE:
                closed[idx] = True
                cap[idx] = 0.0
            elif e.kind is EventKind.LANE_BLOCKAGE:
                m = e.capacity_multiplier()
                cap[idx] *= m
                if m <= 0.0:
                    closed[idx] = True
            else:  # SLOWDOWN
                tmul[idx] *= e.time_multiplier()
        return cap, tmul, closed

    def as_dict(self, minute: float | None = None) -> list[dict[str, object]]:
        """Serialisable listing; annotated with liveness when ``minute`` given."""
        out = []
        for e in self._events:
            d = e.as_dict()
            if minute is not None:
                d["active"] = e.is_active(minute)
            out.append(d)
        return out
