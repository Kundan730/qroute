"""Traffic and congestion layer: the platform's dynamic weight-update engine.

The four pieces, in the order data flows through them:

* :mod:`qroute.traffic.bpr` -- volume-delay functions (BPR and conical) and
  road-capacity handling. Converts a saturation ratio ``v / c`` into a
  travel-time multiplier.
* :mod:`qroute.traffic.profiles` -- time-of-day and day-of-week demand curves
  for an Indian city, calibrated so a peak trip on an arterial takes about
  1.75 times its free-flow time.
* :mod:`qroute.traffic.events` -- incidents, lane blockages and closures on a
  timeline, priced by capacity reduction from the HCM residual-capacity table.
* :mod:`qroute.traffic.simulator` -- the clock that combines the three and
  produces a travel time for every edge of the road network, in milliseconds.
* :mod:`qroute.traffic.sources` -- the interface separating simulated
  conditions from a live traffic feed, with an explicit and visible fallback.

Nothing here imports :mod:`qroute.graph`; the simulator adapts to whatever
network representation it is handed.
"""

from __future__ import annotations

from qroute.traffic.bpr import (
    BPR_ALPHA,
    BPR_BETA,
    band_counts,
    bpr_multiplier,
    bpr_travel_time,
    bpr_travel_time_from_volume,
    congestion_band,
    congestion_level,
    conical_multiplier,
    conical_travel_time,
    edge_capacity,
    saturation_for_ratio,
    saturation_ratio,
)
from qroute.traffic.events import (
    HCM_RESIDUAL_CAPACITY,
    BlockageType,
    EventKind,
    EventQueue,
    TrafficEvent,
    closure,
    lane_blockage,
    residual_capacity,
    slowdown,
)
from qroute.traffic.profiles import (
    PEAK_TRAVEL_TIME_RATIO,
    DemandProfile,
    class_sensitivity,
    default_profile,
    flat_profile,
)
from qroute.traffic.simulator import EdgeArrays, TrafficSimulator, edge_arrays_from_network
from qroute.traffic.sources import (
    MissingAPIKeyError,
    SimulatedSource,
    TomTomFlowSource,
    TrafficFetchError,
    TrafficObservation,
    TrafficSource,
    TrafficSourceError,
    api_key_available,
    resolve_source,
)

__all__ = [
    "BPR_ALPHA",
    "BPR_BETA",
    "BlockageType",
    "DemandProfile",
    "EdgeArrays",
    "EventKind",
    "EventQueue",
    "HCM_RESIDUAL_CAPACITY",
    "MissingAPIKeyError",
    "PEAK_TRAVEL_TIME_RATIO",
    "SimulatedSource",
    "TomTomFlowSource",
    "TrafficEvent",
    "TrafficFetchError",
    "TrafficObservation",
    "TrafficSimulator",
    "TrafficSource",
    "TrafficSourceError",
    "api_key_available",
    "band_counts",
    "bpr_multiplier",
    "bpr_travel_time",
    "bpr_travel_time_from_volume",
    "class_sensitivity",
    "closure",
    "congestion_band",
    "congestion_level",
    "conical_multiplier",
    "conical_travel_time",
    "default_profile",
    "edge_arrays_from_network",
    "edge_capacity",
    "flat_profile",
    "lane_blockage",
    "residual_capacity",
    "resolve_source",
    "saturation_for_ratio",
    "saturation_ratio",
    "slowdown",
]
