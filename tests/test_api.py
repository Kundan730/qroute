"""End-to-end tests for the HTTP API.

These run the real application against the real data: real benchmark instances,
the real bundled road graph, the real traffic simulator and real solver
processes. Nothing is mocked, because the point of the API is that the browser
sees exactly what the library produces, and a mock would test the mock.

The whole file is budgeted at under a minute. Two things dominate that budget
and are therefore paid once, in module-scoped fixtures: the JIT warm-up at
application startup, and loading the Bengaluru road graph. Preloading is turned
off through the environment so the graph is loaded lazily by the first test that
needs it rather than for the tests that do not.

Solver time limits are deliberately short. A two-second QPSO run on a
32-customer instance is not a meaningful optimisation experiment, and these
tests do not assert anything about solution quality beyond feasibility and
"a cost was produced" - quality is the benchmark suite's job, not the API's.
"""

from __future__ import annotations

import json
import os
import time

import pytest

# Must be set before the application is created: it is read when the startup
# thread runs, and loading three road graphs would cost half the test budget.
os.environ.setdefault("QROUTE_API_PRELOAD", "none")

from fastapi.testclient import TestClient  # noqa: E402

from qroute.api.app import create_app  # noqa: E402

#: The instance every quick solver test uses. 31 customers, best known 784, and
#: it loads in milliseconds.
SMALL_INSTANCE = "A-n32-k5"

#: Wall-clock limit handed to solvers started by the tests.
RUN_SECONDS = 2.0


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def network_id(client) -> str:
    """The first bundled road network, or a skip when none is on disk."""
    networks = client.get("/api/networks").json()
    if not networks:
        pytest.skip("no road networks under data/osm")
    return networks[0]["id"]


def _wait_for_run(client, run_id: str, timeout: float = 60.0) -> dict:
    """Poll a run until it leaves the running state."""
    deadline = time.time() + timeout
    status = client.get(f"/api/runs/{run_id}").json()
    while status["state"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.2)
        status = client.get(f"/api/runs/{run_id}").json()
    return status


def _read_sse(response) -> list[tuple[str, dict]]:
    """Collect ``(event, payload)`` pairs from a streaming response."""
    events: list[tuple[str, dict]] = []
    name = "message"
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line.split(":", 1)[1])))
            if name in ("done", "error", "cancelled"):
                break
    return events


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


def test_health_reports_capabilities(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["algorithms"] >= 5
    assert body["instances"] > 100
    assert set(body["network_ids"]) >= set(body["networks_loaded"])
    # OR-Tools is a hard dependency; PyVRP is optional and may honestly be absent.
    assert body["ortools_available"] is True
    assert isinstance(body["pyvrp_available"], bool)


def test_health_reports_the_warm_up(client):
    """The startup warm-up must have compiled the kernels before tests solve."""
    deadline = time.time() + 90.0
    warmup = client.get("/api/health").json()["warmup"]
    while not warmup["done"] and time.time() < deadline:
        time.sleep(0.5)
        warmup = client.get("/api/health").json()["warmup"]
    assert warmup["done"], "the warm-up never finished"
    assert warmup["seconds"] > 0.0
    assert "failed" not in warmup["detail"]


def test_algorithm_catalogue_has_parameter_schemas(client):
    catalogue = client.get("/api/algorithms").json()
    by_name = {entry["name"]: entry for entry in catalogue}
    assert {"qpso", "pso", "ga", "sa", "aco"} <= set(by_name)
    qpso = by_name["qpso"]
    assert qpso["description"]
    assert qpso["supports_warm_start"] is True
    params = {p["name"]: p for p in qpso["params"]}
    assert params["swarm_size"]["kind"] == "int"
    assert params["swarm_size"]["default"] == 30
    assert params["beta_schedule"]["kind"] == "choice"
    assert "linear" in params["beta_schedule"]["choices"]
    assert params["local_search"]["kind"] == "bool"
    # Constructor plumbing must never be offered as a form field.
    assert "decoder" not in params and "initial_keys" not in params


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------


def test_instance_listing_and_detail(client):
    rows = client.get("/api/instances").json()
    assert len(rows) > 100
    families = {row["family"] for row in rows}
    assert {"cvrp", "vrptw"} <= families

    detail = client.get(f"/api/instances/{SMALL_INSTANCE}").json()
    assert detail["name"] == SMALL_INSTANCE
    assert detail["n_customers"] == 31
    assert detail["bks"] == 784.0
    assert len(detail["coords"]) == 32
    assert len(detail["demand"]) == 32
    assert detail["geographic"] is False

    windows = client.get("/api/instances/C101").json()
    assert windows["family"] == "vrptw"
    assert windows["has_time_windows"] is True
    assert len(windows["time_windows"]) == windows["n_customers"] + 1


def test_unknown_instance_is_a_helpful_404(client):
    response = client.get("/api/instances/definitely-not-an-instance")
    assert response.status_code == 404
    assert "unknown instance" in response.json()["detail"]
    assert "/api/instances" in response.json()["detail"]


# --------------------------------------------------------------------------
# Networks and traffic
# --------------------------------------------------------------------------


def test_network_listing_and_edges(client, network_id):
    summary = client.get(f"/api/networks/{network_id}").json()
    assert summary["n_nodes"] > 1000
    assert summary["n_edges"] > summary["n_nodes"]
    min_lat, min_lon, max_lat, max_lon = summary["bbox"]
    assert min_lat < summary["center"][0] < max_lat
    assert min_lon < summary["center"][1] < max_lon

    detailed = client.get(
        f"/api/networks/{network_id}/edges", params={"min_importance": 0, "geometry": False}
    ).json()
    arterials = client.get(
        f"/api/networks/{network_id}/edges", params={"min_importance": 4, "geometry": False}
    ).json()
    # The level-of-detail filter has to actually reduce the payload, otherwise
    # the map is asked to draw 34k polylines.
    assert len(arterials["features"]) < len(detailed["features"]) / 4
    props = arterials["features"][0]["properties"]
    assert {"edge", "u", "v", "highway", "length_m", "travel_time_s", "congestion"} <= set(props)
    assert props["travel_time_s"] >= props["free_flow_s"] - 1e-6


def test_setting_the_clock_changes_congestion(client, network_id):
    night = client.post(f"/api/traffic/{network_id}/time", json={"hour": 3.0}).json()
    peak = client.post(f"/api/traffic/{network_id}/time", json={"hour": 9.0}).json()
    assert night["hour_of_day"] == 3.0
    assert peak["hour_of_day"] == 9.0
    assert (
        peak["travel_time_seconds"]["network_ratio"]
        > night["travel_time_seconds"]["network_ratio"]
    ), "the morning peak must be slower than three in the morning"
    assert peak["congestion"]["mean_level_length_weighted"] > 0.0
    assert night["congestion"]["mean_level_length_weighted"] == pytest.approx(0.0, abs=1e-3)


def test_injecting_and_clearing_an_incident(client, network_id):
    client.post(f"/api/traffic/{network_id}/time", json={"hour": 9.0})
    before = client.get(f"/api/traffic/{network_id}/state", params={"top_k": 20}).json()
    assert before["n_active_events"] == 0
    assert before["n_closed"] == 0
    edges = [edge["index"] for edge in before["worst_edges"][:10]]

    created = client.post(
        f"/api/traffic/{network_id}/events",
        json={
            "kind": "closure",
            "edges": edges,
            "duration_minutes": 90.0,
            "description": "test closure",
        },
    )
    assert created.status_code == 201
    state = created.json()
    assert state["n_active_events"] == 1
    assert state["n_closed"] == len(edges)
    event_id = state["event"]["event_id"]
    assert state["event"]["kind"] == "closure"

    # A lane blockage is priced from the HCM table rather than invented.
    blockage = client.post(
        f"/api/traffic/{network_id}/events",
        json={"kind": "lane_blockage", "edges": edges[:3], "lanes": 3, "duration_minutes": 30.0},
    ).json()
    assert blockage["n_active_events"] == 2
    assert 0.0 < blockage["event"]["capacity_multiplier"] < 1.0

    cleared = client.delete(f"/api/traffic/{network_id}/events/{event_id}").json()
    assert cleared["n_closed"] == 0
    assert cleared["n_active_events"] == 1

    reset = client.delete(f"/api/traffic/{network_id}/events").json()
    assert reset["n_active_events"] == 0


def test_unknown_network_and_unknown_event(client, network_id):
    response = client.get("/api/networks/atlantis/edges")
    assert response.status_code == 404
    assert "unknown network" in response.json()["detail"]

    response = client.delete(f"/api/traffic/{network_id}/events/999999")
    assert response.status_code == 404

    response = client.post(
        f"/api/traffic/{network_id}/events",
        json={"kind": "closure", "edges": [10**9]},
    )
    assert response.status_code == 422
    assert "out of range" in response.json()["detail"]


# --------------------------------------------------------------------------
# Exact shortest path
# --------------------------------------------------------------------------


def test_exact_route_between_two_nodes(client, network_id):
    summary = client.get(f"/api/networks/{network_id}").json()
    min_lat, min_lon, max_lat, max_lon = summary["bbox"]
    # Two corners of the extract, snapped to the nearest junctions: a long
    # enough trip that congestion has somewhere to show up.
    origin = {"from_lat": min_lat + 0.01, "from_lon": min_lon + 0.01}
    destination = {"to_lat": max_lat - 0.01, "to_lon": max_lon - 0.01}

    client.post(f"/api/traffic/{network_id}/time", json={"hour": 3.0})
    night = client.get(
        "/api/route/exact",
        params={"network": network_id, **origin, **destination, "depart_minute": 3 * 60},
    )
    assert night.status_code == 200
    night_feature = night.json()
    assert night_feature["type"] == "Feature"
    assert night_feature["geometry"]["type"] == "LineString"
    assert len(night_feature["geometry"]["coordinates"]) > 10
    night_props = night_feature["properties"]
    assert night_props["distance_m"] > 1000.0
    assert night_props["duration_s"] > 0.0
    assert night_props["nodes_expanded"] > 0
    assert night_props["delay_ratio"] == pytest.approx(1.0, abs=0.05)

    peak = client.get(
        "/api/route/exact",
        params={"network": network_id, **origin, **destination, "depart_minute": 9 * 60},
    ).json()
    assert peak["properties"]["duration_s"] > night_props["duration_s"], (
        "the same trip must take longer in the morning peak"
    )
    assert peak["properties"]["delay_ratio"] > 1.0
    # Asking for a route must not move the demonstration's clock.
    assert client.get(f"/api/traffic/{network_id}/state").json()["hour_of_day"] == 3.0

    incomplete = client.get("/api/route/exact", params={"network": network_id})
    assert incomplete.status_code == 422
    assert "from_node" in incomplete.json()["detail"]

    missing_node = client.get(
        "/api/route/exact", params={"network": network_id, "from_node": 1, "to_node": 2}
    )
    assert missing_node.status_code == 404


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def test_run_streams_to_completion(client):
    started = client.post(
        "/api/runs",
        json={
            "algorithm": "qpso",
            "instance": SMALL_INSTANCE,
            "seed": 1,
            "max_seconds": RUN_SECONDS,
            "params": {"swarm_size": 12},
        },
    )
    assert started.status_code == 201
    run_id = started.json()["run_id"]

    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        events = _read_sse(response)

    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    ticks = [payload for name, payload in events if name == "tick"]
    assert ticks, "the stream carried no iteration ticks"
    # The throttle must hold: at most ten ticks a second plus a little slack for
    # the first iteration, which is always emitted.
    assert len(ticks) <= RUN_SECONDS * 10 + 5
    for tick in ticks:
        assert {"iteration", "best_cost", "mean_cost", "diversity", "elapsed",
                "evaluations"} <= set(tick)
    assert any("routes" in tick for tick in ticks), "no tick carried the incumbent routes"
    # Best cost is monotone non-increasing: it is a running incumbent.
    costs = [tick["best_cost"] for tick in ticks]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))

    final = events[-1][1]
    assert final["state"] == "done"
    assert final["best_cost"] > 0
    assert final["feasible"] is True
    assert final["bks"] == 784.0
    assert final["best_cost"] >= final["bks"] - 1e-6, "a run cannot beat the proven optimum"
    assert sorted(c for route in final["routes"] for c in route) == list(range(1, 32))
    assert final["history"], "the finished run reported no convergence history"
    assert final["geojson"] is None, "a CVRPLIB instance has no road geometry"


def test_reconnecting_to_a_finished_run_replays_it(client):
    run_id = client.post(
        "/api/runs",
        json={"algorithm": "sa", "instance": SMALL_INSTANCE, "seed": 4,
              "max_seconds": RUN_SECONDS},
    ).json()["run_id"]
    _wait_for_run(client, run_id)
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        events = _read_sse(response)
    assert [name for name, _ in events][-1] == "done"
    assert any(name == "tick" for name, _ in events)


def test_cancelling_a_run(client):
    run_id = client.post(
        "/api/runs",
        json={"algorithm": "qpso", "instance": SMALL_INSTANCE, "seed": 7, "max_seconds": 120.0},
    ).json()["run_id"]

    # Let the search get far enough to have reported something.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if client.get(f"/api/runs/{run_id}").json()["iterations"] > 0:
            break
        time.sleep(0.2)

    cancelled = client.post(f"/api/runs/{run_id}/cancel").json()
    assert cancelled["state"] == "cancelled"
    assert cancelled["best_cost"] is not None, "the work done before the cancel was lost"
    assert client.get(f"/api/runs/{run_id}").json()["state"] == "cancelled"
    # Cancelling a finished run is idempotent rather than an error.
    assert client.post(f"/api/runs/{run_id}/cancel").json()["state"] == "cancelled"


def test_run_rejects_unknown_algorithm_and_instance(client):
    bad_algorithm = client.post(
        "/api/runs", json={"algorithm": "quantum-magic", "instance": SMALL_INSTANCE}
    )
    assert bad_algorithm.status_code == 422
    assert "unknown algorithm" in bad_algorithm.json()["detail"]

    bad_instance = client.post("/api/runs", json={"algorithm": "qpso", "instance": "nope"})
    assert bad_instance.status_code == 404

    bad_field = client.post(
        "/api/runs",
        json={"algorithm": "qpso", "instance": SMALL_INSTANCE, "max_seconds": -3.0},
    )
    assert bad_field.status_code == 422

    misspelt = client.post(
        "/api/runs",
        json={"algorithm": "qpso", "instance": SMALL_INSTANCE, "max_second": 3.0},
    )
    assert misspelt.status_code == 422, "a misspelt field must not be silently ignored"


def test_unknown_run_is_a_404(client):
    assert client.get("/api/runs/0123456789ab").status_code == 404
    assert client.post("/api/runs/0123456789ab/cancel").status_code == 404


# --------------------------------------------------------------------------
# The road-network round trip: build, solve, re-optimise
# --------------------------------------------------------------------------


def test_network_instance_solve_and_reoptimize(client, network_id):
    client.delete(f"/api/traffic/{network_id}/events")
    client.post(f"/api/traffic/{network_id}/time", json={"hour": 9.0})

    built = client.post(
        f"/api/networks/{network_id}/instance", json={"n_customers": 20, "seed": 11}
    )
    assert built.status_code == 201 or built.status_code == 200
    instance = built.json()
    assert instance["family"] == "network"
    assert instance["geographic"] is True
    assert instance["n_customers"] == 20
    assert len(instance["node_ids"]) == 21
    assert instance["meta"]["traffic_hour"] == 9.0
    # Coordinates must be real geography, inside the network's bounding box.
    bbox = client.get(f"/api/networks/{network_id}").json()["bbox"]
    for lat, lon in instance["coords"]:
        assert bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]

    first = client.post(
        "/api/runs",
        json={"algorithm": "qpso", "instance": instance["name"], "seed": 2,
              "max_seconds": RUN_SECONDS},
    ).json()["run_id"]
    status = _wait_for_run(client, first)
    assert status["state"] == "done"
    assert status["best_cost"] > 0
    assert status["geojson"] is not None, "a road instance must come back with road geometry"
    features = status["geojson"]["features"]
    assert len(features) == status["n_routes"]
    for feature in features:
        assert feature["geometry"]["type"] == "LineString"
        assert len(feature["geometry"]["coordinates"]) > 2
        assert feature["properties"]["distance_m"] > 0

    # Close the worst corridors, then re-optimise from the previous plan.
    worst = client.get(f"/api/traffic/{network_id}/state", params={"top_k": 12}).json()
    client.post(
        f"/api/traffic/{network_id}/events",
        json={"kind": "closure", "edges": [e["index"] for e in worst["worst_edges"]],
              "duration_minutes": 120.0, "description": "test incident"},
    )
    second = client.post(
        f"/api/runs/{first}/reoptimize", json={"max_seconds": RUN_SECONDS}
    )
    assert second.status_code == 201
    reopt = _wait_for_run(client, second.json()["run_id"])
    assert reopt["state"] == "done"
    assert reopt["parent_run_id"] == first
    assert reopt["warm_started"] is True
    # baseline_cost prices the *previous* plan under the *new* travel times, so
    # the pair is the honest "keep the old plan" against "re-optimise" comparison.
    assert reopt["baseline_cost"] is not None
    assert reopt["best_cost"] <= reopt["baseline_cost"] + 1e-6

    client.delete(f"/api/traffic/{network_id}/events")


def test_reoptimize_needs_a_finished_parent(client):
    run_id = client.post(
        "/api/runs",
        json={"algorithm": "qpso", "instance": SMALL_INSTANCE, "seed": 3, "max_seconds": 60.0},
    ).json()["run_id"]
    response = client.post(f"/api/runs/{run_id}/reoptimize", json={"max_seconds": 1.0})
    assert response.status_code == 409
    client.post(f"/api/runs/{run_id}/cancel")


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------


def test_benchmark_listing_and_detail(client):
    listing = client.get("/api/benchmarks").json()
    if not listing:
        pytest.skip("no benchmark result sets under results/runs")
    name = listing[0]["name"]
    for row in listing:
        assert row["n_runs"] >= 0

    detail = client.get(f"/api/benchmarks/{name}").json()
    assert detail["name"] == name
    assert detail["algorithms"]
    assert detail["instances"]
    assert detail["cells"]
    key = f"{detail['instances'][0]}|{detail['algorithms'][0]}"
    cell = detail["cells"][key]
    assert cell["runs"] >= 1
    assert cell["cost"]["best"] <= cell["cost"]["worst"]
    if detail["curves"]:
        curve = next(iter(detail["curves"].values()))
        costs = [point[1] for point in curve]
        assert all(b <= a + 1e-6 for a, b in zip(costs, costs[1:])), (
            "a mean best-cost curve must be non-increasing"
        )
    if detail["omnibus"] is not None:
        omnibus = detail["omnibus"]
        assert set(omnibus["mean_ranks"]) == set(omnibus["algorithms"])
        assert 0.0 <= omnibus["p_value"] <= 1.0
        for comparison in omnibus["post_hoc"]:
            assert comparison["p_adjusted"] >= comparison["p_value"] - 1e-12

    assert client.get("/api/benchmarks/no-such-sweep").status_code == 404
