"""The traffic simulator: a clock, a demand model, and per-edge travel times.

This is the "dynamic weight update mechanism" of the platform. It owns a
simulated clock, and at any instant it can produce the current travel time of
every edge in the road network. Feeding those times back into the router is
what turns a static VRP solver into a traffic-aware one.

Pipeline, per edge, per call
----------------------------
1. **Demand.** The time-of-day profile (:mod:`qroute.traffic.profiles`) gives a
   saturation ratio for the reference road class. It is multiplied by the
   edge's road-class sensitivity, by a static per-edge heterogeneity factor,
   and by a smoothly time-varying noise field. All three are drawn once from
   the seed, so the result is a deterministic function of ``(edge, time)``.
2. **Incidents.** The event queue (:mod:`qroute.traffic.events`) supplies a
   capacity multiplier, a direct time multiplier and a closed mask. Capacity
   loss enters as ``x = v / (c * m)``, i.e. it raises saturation rather than
   adding a delay, which is what makes an incident cheap at night and ruinous
   at the peak.
3. **Volume-delay.** BPR by default, conical optionally
   (:mod:`qroute.traffic.bpr`), turns saturation into a travel-time multiplier.
4. **Output.** ``t = t0 * vdf(x) * time_multiplier``, with closed edges set to
   ``inf`` so no router can select them.

Everything above is one pass of NumPy arithmetic over arrays of length
``n_edges``; there is no Python loop over edges anywhere on this path.

Known limitation, measured
--------------------------
Combining capacity-based incidents with BPR's quartic produces very large
multipliers in deep oversaturation. Measured on the Bengaluru extract: a
one-lane blockage on the 3-lane Hosur Road corridor at 19:00 leaves 49 percent
of capacity, which raises saturation from about 1.5 to about 3.05 and takes the
corridor from 1.80x free-flow travel time to 14.9x. The arithmetic is correct
and the qualitative behaviour -- a blocked lane is nearly free at 03:00 and
ruinous at 19:00 -- is right, but 14.9x overstates a real corridor, because BPR
has no queue spillback or inflow metering to cap the degradation. Two honest
mitigations: the router avoids such an edge long before the number matters (it
reroutes at +21.7 percent, not +727 percent), and ``vdf="conical"`` bounds the
growth linearly for anyone who needs the oversaturated regime to be credible in
its own right.

Network coupling
----------------
``qroute.graph.RoadNetwork`` is written by a different component and is
imported lazily and defensively. The simulator only needs, per directed edge:
a free-flow travel time, a road class, a lane count and an edge key. It will
take those from a ``RoadNetwork``-like object if it exposes them, from a raw
NetworkX ``MultiDiGraph`` (as produced by OSMnx) otherwise, or from arrays
handed in directly. Writing results back is equally defensive: see
:meth:`TrafficSimulator.apply_to`.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from qroute.core.rng import make_rng
from qroute.traffic import bpr as _bpr
from qroute.traffic.events import EventQueue, TrafficEvent
from qroute.traffic.profiles import (
    MINUTES_PER_DAY,
    DemandProfile,
    class_sensitivity_array,
    default_profile,
)

VDFName = Literal["bpr", "conical"]

# Attribute names the simulator will accept on a RoadNetwork-like object when
# harvesting edge data. Listed most specific first. Kept as data so that the
# graph component can add a name without this module changing.
_FREE_FLOW_ATTRS = ("free_flow_time", "free_flow_times", "base_travel_time", "edge_travel_time")
_CLASS_ATTRS = ("road_class", "road_classes", "highway", "edge_class")
_LANES_ATTRS = ("lanes", "edge_lanes", "n_lanes")
_LENGTH_ATTRS = ("length", "lengths", "edge_length")
_KEYS_ATTRS = ("edge_keys", "edges", "edge_index")
# A RoadNetwork that publishes its own capacity table is the authority on it:
# two components disagreeing about how many vehicles an arterial carries would
# be a silent, hard-to-find inconsistency. See TrafficSimulator's `capacity`.
_CAPACITY_ATTRS = ("edge_capacity", "capacity", "edge_capacities")
# Methods tried, in order, when pushing new weights back onto a network.
_WRITEBACK_METHODS = (
    "update_travel_times",
    "set_travel_times",
    "update_edge_weights",
    "set_edge_weights",
    "update_weights",
    "apply_travel_times",
)


@dataclass
class EdgeArrays:
    """Array-of-structs view of a road network, in a fixed edge order.

    The simulator works exclusively on this. Building it once and reusing it is
    what keeps the per-tick cost to a handful of vectorised operations.
    """

    free_flow_time: np.ndarray      # seconds
    road_class: np.ndarray          # object array of OSM highway strings
    lanes: np.ndarray               # directional lane count, float
    length: np.ndarray              # metres
    keys: list                      # opaque per-edge identity, e.g. (u, v, k)
    capacity: np.ndarray | None = None   # veh/h, when the network publishes one

    @property
    def n_edges(self) -> int:
        return int(self.free_flow_time.shape[0])

    def index_of(self, key) -> int:
        """Position of an edge key; ``-1`` when absent."""
        if not hasattr(self, "_key_index"):
            self._key_index = {k: i for i, k in enumerate(self.keys)}  # type: ignore[attr-defined]
        return self._key_index.get(key, -1)  # type: ignore[attr-defined]


def _first_attr(obj: Any, names: Sequence[str]):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None and not callable(v):
            return v
    return None


def _parse_lanes(raw, road_class: str) -> float:
    """Turn an OSM ``lanes`` tag into a directional lane count.

    OSM lane tags are messy: they can be missing, a string, or a list when the
    edge was simplified from several ways. The list case takes the maximum,
    which is the conservative choice for capacity (an under-counted lane makes
    the model pessimistic, and a pessimistic traffic model is safer than an
    optimistic one for a routing demo).
    """
    if raw is None:
        return _bpr.default_lanes(road_class)
    if isinstance(raw, (list, tuple)):
        vals = [_parse_lanes(x, road_class) for x in raw]
        return max(vals) if vals else _bpr.default_lanes(road_class)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _bpr.default_lanes(road_class)
    return v if v >= 1.0 else _bpr.default_lanes(road_class)


def _normalise_class(raw) -> str:
    """OSM ``highway`` values are sometimes lists after simplification."""
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if raw else "unclassified"
    if raw is None:
        return "unclassified"
    s = str(raw)
    if s.startswith("["):  # a list that survived a GraphML round trip as text
        s = s.strip("[]").split(",")[0].strip().strip("'\"")
    return s or "unclassified"


def edge_arrays_from_graph(graph) -> EdgeArrays:
    """Build an :class:`EdgeArrays` from a NetworkX (Multi)DiGraph.

    Free-flow time is taken from the ``travel_time`` attribute when OSMnx has
    added one, else derived from ``length`` and ``speed_kph``, else from
    ``length`` at a 30 km/h default. The edge order is
    ``graph.edges(keys=True)``, which NetworkX guarantees is stable for an
    unmodified graph -- the simulator's arrays and any write-back therefore
    line up.
    """
    is_multi = getattr(graph, "is_multigraph", lambda: False)()
    edges = list(graph.edges(keys=True, data=True)) if is_multi else [
        (u, v, 0, d) for u, v, d in graph.edges(data=True)
    ]
    n = len(edges)
    t0 = np.empty(n, dtype=np.float64)
    length = np.empty(n, dtype=np.float64)
    lanes = np.empty(n, dtype=np.float64)
    klass = np.empty(n, dtype=object)
    keys: list = []
    for i, (u, v, k, d) in enumerate(edges):
        cls = _normalise_class(d.get("highway"))
        klass[i] = cls
        ln = float(d.get("length", 0.0) or 0.0)
        length[i] = ln
        tt = d.get("travel_time")
        if tt is None:
            speed = float(d.get("speed_kph", 30.0) or 30.0)
            tt = ln / max(speed, 1.0) * 3.6
        t0[i] = float(tt)
        lanes[i] = _parse_lanes(d.get("lanes"), cls)
        keys.append((u, v, k) if is_multi else (u, v))
    return EdgeArrays(free_flow_time=t0, road_class=klass, lanes=lanes, length=length, keys=keys)


def edge_arrays_from_network(network) -> EdgeArrays:
    """Adapt whatever the graph component hands us into :class:`EdgeArrays`.

    Accepted, in order of preference:

    1. an object exposing free-flow times, classes and lanes as arrays
       (the expected ``RoadNetwork`` shape),
    2. an object wrapping a NetworkX graph in ``.graph`` / ``.G`` / ``.nx``,
    3. a NetworkX graph itself,
    4. an :class:`EdgeArrays` (returned unchanged).

    The probing is deliberate: the graph component is being written in
    parallel with this one, and hard-coding one attribute name would couple the
    two builds together for no benefit.
    """
    if isinstance(network, EdgeArrays):
        return network

    t0 = _first_attr(network, _FREE_FLOW_ATTRS)
    if t0 is not None:
        t0 = np.asarray(t0, dtype=np.float64)
        n = t0.shape[0]
        raw_cls = _first_attr(network, _CLASS_ATTRS)
        klass = (
            np.asarray([_normalise_class(c) for c in raw_cls], dtype=object)
            if raw_cls is not None
            else np.full(n, "residential", dtype=object)
        )
        raw_lanes = _first_attr(network, _LANES_ATTRS)
        lanes = (
            np.asarray(
                [_parse_lanes(l, str(klass[i])) for i, l in enumerate(raw_lanes)],
                dtype=np.float64,
            )
            if raw_lanes is not None
            else np.asarray([_bpr.default_lanes(str(c)) for c in klass], dtype=np.float64)
        )
        raw_len = _first_attr(network, _LENGTH_ATTRS)
        length = (
            np.asarray(raw_len, dtype=np.float64) if raw_len is not None
            else np.zeros(n, dtype=np.float64)
        )
        raw_keys = _first_attr(network, _KEYS_ATTRS)
        keys = list(raw_keys) if raw_keys is not None else list(range(n))
        raw_cap = _first_attr(network, _CAPACITY_ATTRS)
        cap = None
        if raw_cap is not None:
            cap = np.asarray(raw_cap, dtype=np.float64)
            if cap.shape != (n,):
                cap = None
        return EdgeArrays(t0, klass, lanes, length, keys, cap)

    for attr in ("graph", "G", "nx", "_graph"):
        inner = getattr(network, attr, None)
        if inner is not None and hasattr(inner, "edges") and hasattr(inner, "nodes"):
            return edge_arrays_from_graph(inner)

    if hasattr(network, "edges") and hasattr(network, "nodes"):
        return edge_arrays_from_graph(network)

    raise TypeError(
        "cannot read edges from "
        f"{type(network).__name__}; expected a RoadNetwork with free-flow time / "
        "road class / lane arrays, a NetworkX graph, or an EdgeArrays"
    )


class TrafficSimulator:
    """Simulated-time traffic state over a road network.

    Parameters
    ----------
    network:
        Anything :func:`edge_arrays_from_network` accepts.
    profile:
        Demand profile; :func:`qroute.traffic.profiles.default_profile` if None.
    seed:
        Seeds the static per-edge heterogeneity and the time-varying noise.
        Two simulators with the same seed and network are bit-identical.
    start_minute:
        Initial clock reading, in minutes from Monday 00:00.
    vdf:
        ``"bpr"`` (default) or ``"conical"``; see :mod:`qroute.traffic.bpr`.
        Note that the demand profile's peak calibration inverts BPR
        specifically. Switching to conical leaves the demand unchanged but
        makes every congested edge markedly slower -- conical charges 2x free
        flow at capacity where BPR charges 1.15x -- so the 1.75 peak target no
        longer holds and the profile must be recalibrated to use it seriously.
    noise_sigma:
        Log-normal spread of the *time-varying* demand noise. 0 disables it.
    demand_spread:
        Log-normal spread of the *static* per-edge demand heterogeneity, which
        represents the fact that two residential streets in the same class do
        not carry the same traffic. Constant in time, so it does not wash out.
    noise_slots_per_day:
        Resolution of the pre-generated noise field. 48 slots means the noise
        is redrawn every 30 simulated minutes and linearly interpolated in
        between, which is smooth enough to look like drifting conditions and
        cheap enough to hold in memory (48 x n_edges float32).
    moment:
        Which moment of the log-normal spreads is pinned to the calibration --
        ``"delay"`` (default, mean travel time) or ``"demand"`` (mean volume).
        See the comment at the point of use; the choice moves the network mean
        by a factor of nearly 1.4 and is not cosmetic.
    capacity:
        Optional explicit per-edge hourly capacity. When omitted, the network's
        own ``edge_capacity`` is used if it publishes one, and otherwise the
        road-class table in :mod:`qroute.traffic.bpr`.

        Be clear about what this does and does not affect. The demand model is
        *saturation-first*: the profile emits ``v / c`` directly rather than an
        absolute volume, so the absolute capacity cancels out of the travel
        time and only *relative* changes to it -- an incident's capacity
        multiplier -- move the answer. The absolute array is therefore used
        only for the vehicle-kilometre weighting behind the reported network
        averages. Two consequences worth knowing: a six-lane and a two-lane
        primary road are equally congested at the same hour under this model,
        and the 1.75 peak calibration is unaffected by which capacity table is
        chosen. Making lane count move congestion would need per-link counted
        volumes, which no open dataset supplied here provides.

    Reproducibility is by construction, not by convention: the noise is a
    pre-generated field indexed by time rather than a stream consumed as the
    clock advances, so stepping forward and stepping back returns *exactly* the
    same weights. That property is asserted by the tests.
    """

    def __init__(
        self,
        network,
        profile: DemandProfile | None = None,
        seed: int = 0,
        start_minute: float = 8 * 60.0,
        vdf: VDFName = "bpr",
        noise_sigma: float = 0.18,
        demand_spread: float = 0.25,
        noise_slots_per_day: int = 48,
        capacity: np.ndarray | None = None,
        conical_a: float = _bpr.CONICAL_A,
        moment: Literal["delay", "demand"] = "delay",
    ) -> None:
        self.edges = edge_arrays_from_network(network)
        self.profile = profile if profile is not None else default_profile()
        self.seed = int(seed)
        self.vdf: VDFName = vdf
        self.conical_a = float(conical_a)
        self.noise_sigma = float(noise_sigma)
        self.demand_spread = float(demand_spread)
        self.noise_slots_per_day = int(noise_slots_per_day)
        if moment not in ("delay", "demand"):
            raise ValueError("moment must be 'delay' or 'demand'")
        self.moment = moment
        self.events = EventQueue()
        self._time = float(start_minute)
        self._cache: tuple[Any, np.ndarray] | None = None

        n = self.edges.n_edges
        # Precedence: an explicit argument, then the network's own published
        # capacity, then this module's road-class table. Deferring to the
        # network keeps a single source of truth when qroute.graph supplies one.
        if capacity is not None:
            self.base_capacity = np.asarray(capacity, dtype=np.float64)
            self.capacity_source = "explicit"
        elif self.edges.capacity is not None:
            self.base_capacity = np.asarray(self.edges.capacity, dtype=np.float64)
            self.capacity_source = "network"
        else:
            self.base_capacity = _bpr.edge_capacity(self.edges.road_class, self.edges.lanes)
            self.capacity_source = "traffic.bpr"
        if self.base_capacity.shape != (n,):
            raise ValueError(f"capacity must have shape ({n},)")
        self.sensitivity = class_sensitivity_array(self.edges.road_class)
        # Weight used whenever a *network* average is quoted. A published
        # congestion index averages over trips, and trips are concentrated on
        # arterials, so averaging over OSM edges (four fifths of which are
        # residential stubs a few tens of metres long) would understate
        # congestion by a wide margin. length x capacity x class sensitivity is
        # proportional to vehicle-kilometres travelled on the link, which is
        # the weighting an index actually uses.
        self.vkt_weight = self.edges.length * self.base_capacity * self.sensitivity

        rng = make_rng(self.seed)
        # Log-normal spreads must be re-centred, or the calibration in
        # `profiles` silently drifts with whatever sigma happens to be chosen.
        # *Which* moment to fix is a real decision, and it matters:
        #
        #   "demand"  offsets by sigma^2 / 2, giving E[demand] = 1. Physically
        #             natural, but BPR is convex, so by Jensen's inequality the
        #             mean *travel time* then lands well above the calibrated
        #             value -- with sigma = 0.31 and beta = 4 the mean arterial
        #             ratio comes out at 2.38 against a 1.75 target. Measured,
        #             not estimated; see the module's report.
        #   "delay"   offsets by beta * sigma^2 / 2, giving E[L^beta] = 1 and
        #             hence a class-mean travel-time multiplier exactly equal
        #             to the noiseless calibrated one. The typical edge is a
        #             little quieter and the tail a little worse, which is also
        #             the more realistic shape: most links are fine and a few
        #             are badly stuck.
        #
        # "delay" is the default because the calibration target is a statement
        # about mean travel time. The correction is derived for BPR; under the
        # conical VDF (less convex) it slightly over-corrects, which is
        # conservative rather than wrong.
        exponent = self.profile.beta if self.moment == "delay" else 1.0
        if self.demand_spread > 0:
            s = self.demand_spread
            self.edge_demand = np.exp(rng.normal(0.0, s, n) - 0.5 * exponent * s * s)
        else:
            self.edge_demand = np.ones(n, dtype=np.float64)
        # Time-varying field: one row per slot, wrapping at the end of the day.
        if self.noise_sigma > 0:
            s = self.noise_sigma
            z = rng.normal(0.0, s, (self.noise_slots_per_day, n)) - 0.5 * exponent * s * s
            self._noise = np.exp(z).astype(np.float32)
        else:
            self._noise = None

    # ------------------------------------------------------------------ clock
    @property
    def time_minutes(self) -> float:
        """Current simulated time, minutes since Monday 00:00."""
        return self._time

    @property
    def hour_of_day(self) -> float:
        return (self._time % MINUTES_PER_DAY) / 60.0

    @property
    def day_of_week(self) -> int:
        """0 = Monday .. 6 = Sunday."""
        return int((self._time // MINUTES_PER_DAY) % 7)

    @property
    def is_weekend(self) -> bool:
        return self.day_of_week >= 5

    def set_time(self, minute: float) -> "TrafficSimulator":
        """Jump the clock to ``minute``. Returns self so calls can chain."""
        self._time = float(minute)
        self._cache = None
        return self

    def advance(self, minutes: float) -> "TrafficSimulator":
        """Move the clock forward (or backward, for a negative argument)."""
        return self.set_time(self._time + float(minutes))

    def set_clock(self, hour: float, day_of_week: int = 0) -> "TrafficSimulator":
        """Convenience: set the clock from a human day/hour instead of minutes."""
        return self.set_time(day_of_week * MINUTES_PER_DAY + hour * 60.0)

    def invalidate(self) -> None:
        """Drop the cached weights.

        Needed only if an event object is mutated in place after being added;
        add/remove already invalidate.
        """
        self._cache = None

    # ----------------------------------------------------------------- events
    def add_event(self, event: TrafficEvent) -> TrafficEvent:
        self._cache = None
        return self.events.add(event)

    def remove_event(self, event: TrafficEvent | int) -> bool:
        self._cache = None
        return self.events.remove(event)

    def clear_events(self) -> None:
        self._cache = None
        self.events.clear()

    def edge_indices(self, keys: Iterable) -> list[int]:
        """Resolve edge keys such as ``(u, v, k)`` into array indices."""
        out = []
        for k in keys:
            i = self.edges.index_of(k)
            if i >= 0:
                out.append(i)
        return out

    def find_edges(self, name: str | None = None, road_class: str | None = None) -> list[int]:
        """Indices of edges matching a street name and/or road class.

        Street name matching needs the original graph attributes, which the
        array view does not keep, so this only filters on road class unless the
        simulator was built with :meth:`attach_names`.
        """
        mask = np.ones(self.edges.n_edges, dtype=bool)
        if road_class is not None:
            mask &= self.edges.road_class == road_class
        if name is not None:
            names = getattr(self, "_edge_names", None)
            if names is None:
                raise ValueError("edge names were not attached; call attach_names(graph) first")
            lowered = name.lower()
            mask &= np.array([lowered in (s or "").lower() for s in names], dtype=bool)
        return np.flatnonzero(mask).tolist()

    def attach_names(self, graph) -> None:
        """Record the ``name`` attribute of each edge, for :meth:`find_edges`.

        Optional and kept out of :class:`EdgeArrays` because street names are a
        presentation concern and cost memory the simulation loop never touches.
        """
        is_multi = getattr(graph, "is_multigraph", lambda: False)()
        it = graph.edges(keys=True, data=True) if is_multi else graph.edges(data=True)
        names = []
        for rec in it:
            d = rec[-1]
            raw = d.get("name")
            names.append(", ".join(raw) if isinstance(raw, (list, tuple)) else (raw or ""))
        if len(names) != self.edges.n_edges:
            raise ValueError("graph edge count does not match the simulator's network")
        self._edge_names = names

    # ------------------------------------------------------------- simulation
    def _noise_at(self, minute: float) -> np.ndarray:
        """Linear interpolation between the two nearest pre-generated slots."""
        if self._noise is None:
            return np.ones(self.edges.n_edges, dtype=np.float64)
        slots = self.noise_slots_per_day
        pos = (minute % MINUTES_PER_DAY) / MINUTES_PER_DAY * slots
        i0 = int(np.floor(pos)) % slots
        i1 = (i0 + 1) % slots
        w = pos - np.floor(pos)
        return (1.0 - w) * self._noise[i0] + w * self._noise[i1]

    def _demand(self) -> np.ndarray:
        """Per-edge saturation *before* incidents, i.e. ``v / c`` at full capacity."""
        base, _weekend = self.profile.at_minute(self._time)
        return float(base) * self.sensitivity * self.edge_demand * self._noise_at(self._time)

    @staticmethod
    def _saturate(demand: np.ndarray, capacity_multiplier: np.ndarray) -> np.ndarray:
        """Apply an incident's capacity loss to a demand array.

        The demand produced by the profile is already a saturation ratio on
        full capacity, so a capacity multiplier ``m`` enters as a divisor:
        ``v / (c * m) = (v / c) / m``. Doing it this way avoids reconstructing
        an absolute volume and dividing again, which would be identical
        arithmetic with one more array allocation.
        """
        ok = capacity_multiplier > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ok, demand / np.where(ok, capacity_multiplier, 1.0), np.inf)

    def saturation(self) -> np.ndarray:
        """Current per-edge saturation ratio ``v / c``, incidents included."""
        cap_mult, _, _ = self.events.apply(self._time, self.edges.n_edges)
        return self._saturate(self._demand(), cap_mult)

    def _vdf(self, saturation: np.ndarray) -> np.ndarray:
        if self.vdf == "conical":
            return _bpr.conical_multiplier(saturation, self.conical_a)
        return _bpr.bpr_multiplier(saturation)

    def edge_travel_times(self) -> np.ndarray:
        """Travel time of every edge, in seconds, at the current clock reading.

        Closed edges are ``inf``. The result is cached against the clock and the
        active event set, so calling this repeatedly at one instant is free.
        """
        key = (self._time, tuple(e.event_id for e in self.events.active_at(self._time)))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        cap_mult, time_mult, closed = self.events.apply(self._time, self.edges.n_edges)
        sat = self._saturate(self._demand(), cap_mult)
        times = self.edges.free_flow_time * self._vdf(sat) * time_mult
        times = np.where(closed, np.inf, times)
        self._cache = (key, times)
        return times

    def congestion_levels(self) -> np.ndarray:
        """Per-edge ``(t - t0) / t0``. This is the quantity the map colours by."""
        return _bpr.congestion_level(self.edge_travel_times(), self.edges.free_flow_time)

    def speed_factors(self) -> np.ndarray:
        """Current speed as a fraction of free-flow speed, i.e. ``t0 / t``.

        The same quantity a live traffic API reports, which is why
        :class:`qroute.traffic.sources.SimulatedSource` returns this and not
        the raw times: it makes simulated and live sources interchangeable.
        """
        t = self.edge_travel_times()
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.isfinite(t) & (t > 0), self.edges.free_flow_time / t, 0.0)
        return np.clip(f, 0.0, 1.0)

    def closed_mask(self) -> np.ndarray:
        _, _, closed = self.events.apply(self._time, self.edges.n_edges)
        return closed

    # ------------------------------------------------------------- write-back
    def apply_to(self, network, attribute: str = "travel_time") -> int:
        """Push the current travel times onto ``network``; returns edge count.

        Tries, in order: a documented update method on a ``RoadNetwork``
        (see :data:`_WRITEBACK_METHODS`); a writable ``free_flow_time``-style
        array; and finally direct NetworkX edge-attribute assignment in the
        same edge order the arrays were built in.

        The NetworkX path is a Python loop, so it is the slow one -- still
        O(edges), but roughly an order of magnitude slower than computing the
        times in the first place. Prefer a ``RoadNetwork`` that takes an array.
        """
        times = self.edge_travel_times()
        for meth in _WRITEBACK_METHODS:
            fn = getattr(network, meth, None)
            if callable(fn):
                fn(times)
                return times.shape[0]

        graph = None
        for attr in ("graph", "G", "nx", "_graph"):
            inner = getattr(network, attr, None)
            if inner is not None and hasattr(inner, "edges"):
                graph = inner
                break
        if graph is None and hasattr(network, "edges") and hasattr(network, "nodes"):
            graph = network
        if graph is None:
            raise TypeError(
                f"do not know how to write travel times back to {type(network).__name__}"
            )

        is_multi = getattr(graph, "is_multigraph", lambda: False)()
        it = graph.edges(keys=True, data=True) if is_multi else graph.edges(data=True)
        for i, rec in enumerate(it):
            rec[-1][attribute] = float(times[i])
        return times.shape[0]

    # ------------------------------------------------------------------ state
    def _weighted_ratio(self, level: np.ndarray, finite: np.ndarray) -> float:
        """Vehicle-km-weighted travel-time ratio over the traversable edges."""
        w = self.vkt_weight[finite]
        wsum = float(np.sum(w))
        if wsum <= 0:
            return float(np.mean(1.0 + level[finite]))
        return float(np.sum((1.0 + level[finite]) * w) / wsum)

    def class_summary(self) -> dict[str, dict[str, float]]:
        """Current mean travel-time ratio and congestion, broken down by class.

        The single most useful diagnostic when a network-wide average looks
        surprising: it shows immediately whether the arterials are loaded and
        the average is being diluted by residential edges, or whether the
        demand model itself is off.
        """
        t = self.edge_travel_times()
        t0 = self.edges.free_flow_time
        lvl = _bpr.congestion_level(t, t0)
        finite = np.isfinite(t)
        out: dict[str, dict[str, float]] = {}
        for name in sorted(set(self.edges.road_class.tolist())):
            m = (self.edges.road_class == name) & finite
            if not np.any(m):
                continue
            out[str(name)] = {
                "n_edges": int(np.count_nonzero(m)),
                "km": round(float(np.sum(self.edges.length[m]) / 1000.0), 2),
                "mean_ratio": round(float(np.mean(1.0 + lvl[m])), 4),
                "p95_ratio": round(float(np.percentile(1.0 + lvl[m], 95)), 4),
            }
        return out

    def state(self, include_edges: bool = False, top_k: int = 0) -> dict[str, Any]:
        """A JSON-serialisable snapshot of the current traffic state.

        By default only aggregates and the event list are returned: 34k
        per-edge floats do not belong in an API response body. Pass
        ``include_edges=True`` for the full arrays, or ``top_k`` for the worst
        ``k`` edges, which is what the dashboard's "worst corridors" panel uses.
        """
        t = self.edge_travel_times()
        t0 = self.edges.free_flow_time
        lvl = _bpr.congestion_level(t, t0)
        finite = np.isfinite(t)
        # Length-weighted mean is the honest network figure: an unweighted mean
        # over OSM edges is dominated by very short residential stubs.
        w = self.edges.length
        wsum = float(np.sum(w[finite]))
        mean_lvl = float(np.sum(lvl[finite] * w[finite]) / wsum) if wsum > 0 else float(np.mean(lvl[finite]))
        snap: dict[str, Any] = {
            "time_minutes": self._time,
            "hour_of_day": round(self.hour_of_day, 4),
            "day_of_week": self.day_of_week,
            "weekend": self.is_weekend,
            "vdf": self.vdf,
            "seed": self.seed,
            "profile": self.profile.name,
            "n_edges": int(self.edges.n_edges),
            "n_closed": int(np.count_nonzero(~finite)),
            "reference_saturation": round(float(self.profile.at_minute(self._time)[0]), 4),
            "congestion": {
                "mean_level_length_weighted": round(mean_lvl, 4),
                "vkt_weighted_ratio": round(self._weighted_ratio(lvl, finite), 4),
                "mean_level": round(float(np.mean(lvl[finite])), 4),
                "median_level": round(float(np.median(lvl[finite])), 4),
                "p95_level": round(float(np.percentile(lvl[finite], 95)), 4),
                "max_level": round(float(np.max(lvl[finite])), 4),
                "bands": _bpr.band_counts(lvl[finite]),
            },
            "travel_time_seconds": {
                "total_free_flow": round(float(np.sum(t0)), 2),
                "total_current": round(float(np.sum(t[finite])), 2),
                "network_ratio": round(float(np.sum(t[finite]) / max(np.sum(t0[finite]), 1e-9)), 4),
            },
            "events": self.events.as_dict(self._time),
            "n_active_events": len(self.events.active_at(self._time)),
        }
        if top_k > 0:
            order = np.argsort(-np.where(finite, lvl, -1.0))[:top_k]
            snap["worst_edges"] = [
                {
                    "index": int(i),
                    "key": list(self.edges.keys[i]) if isinstance(self.edges.keys[i], tuple)
                    else self.edges.keys[i],
                    "road_class": str(self.edges.road_class[i]),
                    "congestion_level": round(float(lvl[i]), 4),
                    "travel_time_s": round(float(t[i]), 2),
                    "free_flow_s": round(float(t0[i]), 2),
                }
                for i in order
            ]
        if include_edges:
            snap["edge_travel_time_s"] = t.tolist()
            snap["edge_congestion_level"] = lvl.tolist()
        return snap

    # ------------------------------------------------------------- reporting
    def day_summary(self, day_of_week: int = 0, step_minutes: int = 60) -> list[dict[str, float]]:
        """Sweep a whole simulated day and report per-step network statistics.

        Restores the clock afterwards, so it is safe to call mid-run.
        """
        saved = self._time
        rows: list[dict[str, float]] = []
        try:
            for m in range(0, MINUTES_PER_DAY, step_minutes):
                self.set_time(day_of_week * MINUTES_PER_DAY + m)
                t = self.edge_travel_times()
                t0 = self.edges.free_flow_time
                finite = np.isfinite(t)
                lvl = _bpr.congestion_level(t, t0)
                w = self.edges.length[finite]
                wsum = max(float(np.sum(w)), 1e-9)
                rows.append(
                    {
                        "hour": m / 60.0,
                        # Unweighted mean over edges: dominated by short
                        # residential stubs, so it understates the network.
                        "mean_ratio": float(np.mean((1.0 + lvl)[finite])),
                        "length_weighted_ratio": float(np.sum((1.0 + lvl[finite]) * w) / wsum),
                        # Travel-weighted mean: the figure comparable with a
                        # published congestion index, which averages over trips.
                        "vkt_weighted_ratio": self._weighted_ratio(lvl, finite),
                        "mean_congestion": float(np.mean(lvl[finite])),
                        "p95_congestion": float(np.percentile(lvl[finite], 95)),
                        "mean_travel_time_s": float(np.mean(t[finite])),
                        "total_travel_time_s": float(np.sum(t[finite])),
                        "n_closed": int(np.count_nonzero(~finite)),
                    }
                )
        finally:
            self.set_time(saved)
        return rows

    def benchmark_update(self, repeats: int = 20) -> dict[str, float]:
        """Measure the cost of one full edge-weight update, in milliseconds.

        Each repeat moves the clock, so the cache never short-circuits the work.
        """
        saved = self._time
        try:
            self.set_time(saved)
            self.edge_travel_times()  # warm any lazy allocation
            samples = []
            for i in range(repeats):
                self.set_time(saved + i * 7.0)
                t = _time.perf_counter()
                self.edge_travel_times()
                samples.append((_time.perf_counter() - t) * 1e3)
        finally:
            self.set_time(saved)
        arr = np.array(samples)
        return {
            "n_edges": float(self.edges.n_edges),
            "repeats": float(repeats),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TrafficSimulator(n_edges={self.edges.n_edges}, "
            f"t={self._time:.0f}min, hour={self.hour_of_day:.2f}, "
            f"vdf={self.vdf!r}, events={len(self.events)})"
        )
