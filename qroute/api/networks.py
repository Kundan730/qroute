"""Road-network, traffic-simulator and shortest-path endpoints.

The three groups of routes here are what turn the platform from a benchmark
harness into something a judge can click on:

``/api/networks/...``
    The bundled OpenStreetMap extracts, as GeoJSON with a level-of-detail knob,
    and the generator that turns a graph into a routing instance.
``/api/traffic/...``
    The simulated clock and the incident queue. Every mutation is followed by
    the O(edges) dynamic weight update, so the map, the shortest paths and any
    instance built afterwards all describe the same moment.
``/api/route/exact``
    A single exact shortest path, which is the control the metaheuristics are
    judged against on the small end and the thing that makes "the congestion
    actually changes the route" visible on the map.

Every handler is a synchronous ``def``. FastAPI runs those in a worker thread,
which is what we want: they take a per-network lock and then do tens of
milliseconds of NumPy and compiled-code work, and doing that on the event loop
would stall every other request including the SSE streams.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from qroute.api.schemas import (
    EventRequest,
    InstanceDetail,
    InstanceRequest,
    NetworkSummary,
    TimeRequest,
)
from qroute.api.state import STATE, NetworkBundle, StoredInstance, instance_detail

log = logging.getLogger("qroute.api.networks")

router = APIRouter(tags=["networks"])

#: Cap on the number of edge features one request may return. The Bengaluru
#: extract has 34266 edges and their true polylines serialise to about 12 MB,
#: which is not a payload a map should be asked to swallow in one go. The
#: default level of detail returns roughly 2000.
MAX_EDGE_FEATURES: int = 40_000


def _bundle(network_id: str) -> NetworkBundle:
    """Resolve a network id or fail with a 404 that says what does exist."""
    try:
        return STATE.get_network(network_id)
    except KeyError:
        available = STATE.available_network_ids()
        raise HTTPException(
            status_code=404,
            detail=f"unknown network {network_id!r}; available: {', '.join(available) or 'none'}",
        ) from None


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------


@router.get("/api/networks", response_model=list[NetworkSummary])
def list_networks() -> list[dict[str, Any]]:
    """Every bundled road graph, with node and edge counts and a bounding box.

    Networks that have not been loaded yet are listed with ``loaded=false`` and
    whatever the on-disk index knows about them, so the UI can show the choice
    without triggering a nine-second load of all three.
    """
    return STATE.network_summaries()


@router.get("/api/networks/{network_id}", response_model=NetworkSummary)
def get_network(network_id: str) -> dict[str, Any]:
    """Detail for one network. Loads it if it is not in memory yet."""
    return _bundle(network_id).summary()


@router.get("/api/networks/{network_id}/edges")
def network_edges(
    network_id: str,
    min_importance: int = Query(
        3,
        ge=0,
        le=6,
        description="Level of detail. 0 draws every road including service "
        "alleys; 3 draws secondary roads and above, which is about 2000 of the "
        "34266 edges of the Bengaluru extract and is what pans smoothly.",
    ),
    max_edges: Optional[int] = Query(None, ge=1, le=MAX_EDGE_FEATURES),
    geometry: bool = Query(True, description="Follow the true OSM polyline of each edge."),
    min_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    min_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    max_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    max_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
) -> dict[str, Any]:
    """GeoJSON of the road edges, coloured by the current congestion level.

    Each feature carries ``edge``, the index into the network's edge array. That
    index is the handle the traffic endpoints take, so clicking a road on the map
    and blocking it needs no extra lookup.
    """
    bundle = _bundle(network_id)
    bbox = None
    corners = (min_lat, min_lon, max_lat, max_lon)
    if any(c is not None for c in corners):
        if any(c is None for c in corners):
            raise HTTPException(
                status_code=422,
                detail="a bounding box needs all four of min_lat, min_lon, max_lat, max_lon",
            )
        if min_lat >= max_lat or min_lon >= max_lon:  # type: ignore[operator]
            raise HTTPException(status_code=422, detail="the bounding box is empty or inverted")
        bbox = (min_lat, min_lon, max_lat, max_lon)

    started = time.perf_counter()
    with bundle.lock:
        payload = bundle.network.edge_geojson(
            min_importance=min_importance,
            max_edges=max_edges,
            bbox=bbox,
            include_geometry=geometry,
        )
        payload["properties"]["time_minutes"] = bundle.simulator.time_minutes
        payload["properties"]["hour_of_day"] = round(bundle.simulator.hour_of_day, 3)
    payload["properties"]["build_seconds"] = round(time.perf_counter() - started, 4)
    return payload


@router.post("/api/networks/{network_id}/instance", response_model=InstanceDetail)
def build_network_instance(network_id: str, request: InstanceRequest) -> dict[str, Any]:
    """Turn a road graph into a routing instance at the current traffic state.

    The stops are chosen by farthest-point sampling over real junctions and the
    matrices are genuine shortest paths through the network, so the instance the
    optimiser sees is the city, not a scatter of points. Demands are sampled from
    the seed and recorded as such in the metadata: real delivery volumes for
    Koramangala are not public data and this does not pretend otherwise.
    """
    bundle = _bundle(network_id)
    from qroute.graph.builder import build_instance

    depot_latlon = None
    if (request.depot_lat is None) != (request.depot_lon is None):
        raise HTTPException(
            status_code=422, detail="give both depot_lat and depot_lon, or neither"
        )
    if request.depot_lat is not None and request.depot_lon is not None:
        depot_latlon = (request.depot_lat, request.depot_lon)

    started = time.perf_counter()
    with bundle.lock:
        if request.minute is not None:
            bundle.set_minute(request.minute)
        minute = bundle.simulator.time_minutes
        name = f"{network_id}-n{request.n_customers}-s{request.seed}"
        try:
            instance, matrices = build_instance(
                bundle.network,
                request.n_customers,
                seed=request.seed,
                name=name,
                depot_latlon=depot_latlon,
                sampling=request.sampling,
                capacity=request.capacity,
                n_vehicles=request.n_vehicles,
                service_time_s=request.service_time_s,
                return_matrices=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    instance.meta["traffic_minute"] = float(minute)
    instance.meta["traffic_hour"] = round(float(minute % (24 * 60)) / 60.0, 3)
    instance.meta["built_seconds"] = round(time.perf_counter() - started, 4)
    stored = STATE.store_instance(
        StoredInstance(
            name=name,
            instance=instance,
            family="network",
            network_id=network_id,
            matrices=matrices,
            stop_nodes=np.asarray(matrices.nodes),
            request=request.model_dump(),
        )
    )
    log.info(
        "built instance %s (%d customers) in %.2fs",
        name, instance.n_customers, time.perf_counter() - started,
    )
    return instance_detail(stored)


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------


@router.get("/api/traffic/{network_id}/state")
def traffic_state(
    network_id: str,
    top_k: int = Query(8, ge=0, le=200, description="How many worst edges to include."),
) -> dict[str, Any]:
    """The simulated clock, the congestion summary and the active incidents.

    The congestion figures are reported three ways on purpose. The unweighted
    mean over OSM edges is dominated by short residential stubs and understates
    what a driver experiences; the length-weighted mean and the
    vehicle-kilometre-weighted ratio are the figures a published congestion index
    would use. All three are returned rather than one being chosen silently.
    """
    bundle = _bundle(network_id)
    with bundle.lock:
        state = bundle.simulator.state(top_k=top_k)
        state["network"] = network_id
        state["classes"] = bundle.simulator.class_summary()
    return state


@router.post("/api/traffic/{network_id}/time")
def set_traffic_time(network_id: str, request: TimeRequest) -> dict[str, Any]:
    """Move the simulated clock and recompute every edge weight."""
    if request.minute is not None and request.hour is not None:
        raise HTTPException(
            status_code=422, detail="give either minute or hour, not both"
        )
    if request.minute is None and request.hour is None:
        raise HTTPException(status_code=422, detail="give a minute or an hour")
    minute = (
        request.minute
        if request.minute is not None
        else request.day_of_week * 24 * 60.0 + request.hour * 60.0  # type: ignore[operator]
    )
    bundle = _bundle(network_id)
    started = time.perf_counter()
    with bundle.lock:
        bundle.set_minute(minute)
        state = bundle.simulator.state(top_k=8)
    state["network"] = network_id
    state["update_seconds"] = round(time.perf_counter() - started, 5)
    return state


@router.post("/api/traffic/{network_id}/events", status_code=201)
def add_traffic_event(network_id: str, request: EventRequest) -> dict[str, Any]:
    """Inject an incident on a set of edges and re-price the network.

    ``lane_blockage`` is priced from the Highway Capacity Manual residual
    capacity table, ``closure`` removes the edge, and ``slowdown`` scales the
    free-flow speed directly. The distinction matters: a capacity loss is nearly
    free at three in the morning and ruinous at the evening peak, whereas a
    speed reduction costs the same at both.
    """
    bundle = _bundle(network_id)
    from qroute.traffic import events as ev

    n_edges = bundle.simulator.edges.n_edges
    bad = [int(e) for e in request.edges if not 0 <= int(e) < n_edges]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"edge indices out of range for network {network_id!r} "
            f"(0..{n_edges - 1}): {bad[:10]}",
        )

    with bundle.lock:
        start_minute = (
            request.start_minute
            if request.start_minute is not None
            else bundle.simulator.time_minutes
        )
        try:
            if request.kind == "closure":
                event = ev.closure(
                    request.edges,
                    start_minute,
                    request.duration_minutes,
                    description=request.description,
                )
            elif request.kind == "slowdown":
                event = ev.slowdown(
                    request.edges,
                    start_minute,
                    request.duration_minutes,
                    speed_multiplier=request.speed_multiplier,
                    severity=request.severity,
                    description=request.description,
                )
            else:
                event = ev.lane_blockage(
                    request.edges,
                    start_minute,
                    request.duration_minutes,
                    lanes=request.lanes,
                    blockage=request.blockage or ev.BlockageType.ONE_LANE_BLOCKED,
                    severity=request.severity,
                    description=request.description,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        bundle.simulator.add_event(event)
        bundle.apply_traffic()
        state = bundle.simulator.state(top_k=8)
    state["network"] = network_id
    state["event"] = event.as_dict()
    log.info("network %s: added %s on %d edges", network_id, request.kind, len(request.edges))
    return state


@router.delete("/api/traffic/{network_id}/events/{event_id}")
def clear_traffic_event(network_id: str, event_id: int) -> dict[str, Any]:
    """Clear one incident and re-price the network."""
    bundle = _bundle(network_id)
    with bundle.lock:
        removed = bundle.simulator.remove_event(int(event_id))
        if not removed:
            known = [e.event_id for e in bundle.simulator.events]
            raise HTTPException(
                status_code=404,
                detail=f"no event {event_id} on network {network_id!r}; "
                f"active event ids: {known or 'none'}",
            )
        bundle.apply_traffic()
        state = bundle.simulator.state(top_k=8)
    state["network"] = network_id
    state["removed_event_id"] = int(event_id)
    return state


@router.delete("/api/traffic/{network_id}/events")
def clear_all_traffic_events(network_id: str) -> dict[str, Any]:
    """Clear every incident. The reset button behind the demonstration."""
    bundle = _bundle(network_id)
    with bundle.lock:
        bundle.simulator.clear_events()
        bundle.apply_traffic()
        state = bundle.simulator.state(top_k=8)
    state["network"] = network_id
    return state


# --------------------------------------------------------------------------
# Exact shortest path
# --------------------------------------------------------------------------


def _resolve_node(bundle: NetworkBundle, node: Optional[int],
                  lat: Optional[float], lon: Optional[float], label: str) -> int:
    """Internal node index from either an OSM node id or a coordinate."""
    if node is not None:
        try:
            return bundle.network.index_of(int(node))
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"{label} node {node} is not in network {bundle.id!r}",
            ) from None
    if lat is None or lon is None:
        raise HTTPException(
            status_code=422,
            detail=f"give {label}_node, or both {label}_lat and {label}_lon",
        )
    return int(bundle.network.nearest_node(lat, lon))


@router.get("/api/route/exact")
def exact_route(
    network: str = Query(..., description="Road network id."),
    from_node: Optional[int] = Query(None, description="OSM node id of the origin."),
    to_node: Optional[int] = Query(None, description="OSM node id of the destination."),
    from_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    from_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    to_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    to_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    depart_minute: Optional[float] = Query(
        None,
        ge=0.0,
        le=7 * 24 * 60.0,
        description="Departure time on the simulator clock, in minutes since "
        "Monday 00:00. Defaults to the current clock. The clock is restored "
        "afterwards, so asking for a route does not move the demonstration.",
    ),
    weight: str = Query("travel_time", pattern="^(travel_time|length)$"),
) -> dict[str, Any]:
    """Exact shortest path between two nodes, as a GeoJSON ``Feature``.

    The search is A* with a great-circle lower bound, which is exact rather than
    heuristic: the bound never over-estimates the remaining cost, so the first
    time the destination is settled its label is final. The response reports how
    many nodes were expanded, which is the honest way to show that goal direction
    helps without claiming an approximation is an optimum.

    The properties also carry the free-flow duration of the same path, so the
    map can say "this trip costs 4.7 minutes now against 3.1 at free flow"
    instead of showing an uncalibrated colour.
    """
    bundle = _bundle(network)
    from qroute.graph.paths import astar

    with bundle.lock:
        source = _resolve_node(bundle, from_node, from_lat, from_lon, "from")
        target = _resolve_node(bundle, to_node, to_lat, to_lon, "to")
        if source == target:
            raise HTTPException(status_code=422, detail="origin and destination are the same node")

        restore_to: Optional[float] = None
        if depart_minute is not None and abs(depart_minute - bundle.simulator.time_minutes) > 1e-9:
            restore_to = bundle.simulator.time_minutes
            bundle.set_minute(depart_minute)
        minute = bundle.simulator.time_minutes
        try:
            started = time.perf_counter()
            path = astar(bundle.network, source, target, weight=weight)
            seconds = time.perf_counter() - started
            if path.is_empty():
                raise HTTPException(
                    status_code=422,
                    detail="no path exists between those two nodes in this network",
                )
            free_flow_s = float(
                sum(
                    bundle.network.edge_free_flow_time[
                        bundle.network._fastest_edge_between(a, b)
                    ]
                    for a, b in zip(path.nodes[:-1], path.nodes[1:])
                )
            )
            feature = bundle.network.route_geojson(path.nodes)
        finally:
            if restore_to is not None:
                bundle.set_minute(restore_to)

    feature["properties"].update(
        {
            "network": network,
            "weight": weight,
            "depart_minute": float(minute),
            "depart_hour": round(float(minute % (24 * 60)) / 60.0, 3),
            "from_node": int(bundle.network.node_id_of(source)),
            "to_node": int(bundle.network.node_id_of(target)),
            "free_flow_s": round(free_flow_s, 2),
            "delay_ratio": round(path.duration_s / free_flow_s, 4) if free_flow_s > 0 else 1.0,
            "nodes_expanded": int(path.expanded),
            "search_seconds": round(seconds, 4),
            "algorithm": "a-star (exact, great-circle lower bound)",
        }
    )
    return feature
