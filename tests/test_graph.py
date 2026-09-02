"""Ground-truth tests for the road-network layer.

Every test here compares against something independently trustworthy rather
than against the implementation's own output: NetworkX for shortest paths, a
brute-force haversine scan for nearest-node, a hand-built graph for the
parallel-edge rule, and the mathematical definition of FIFO for the
time-dependent traversal.

The heavier tests run on ``delhi_connaught`` (the smallest bundled extract) so
the suite stays quick; the reported performance figures in the module docstrings
were measured on ``bengaluru_koramangala``.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from qroute.graph import builder, matrix, osm, paths
from qroute.graph.network import RoadNetwork
from qroute.problems.instance import Instance

GRAPH_DIR = Path("data/osm")
SMALL_GRAPH = GRAPH_DIR / "delhi_connaught.graphml"

pytestmark = pytest.mark.skipif(
    not SMALL_GRAPH.exists(), reason="bundled OSM graphs are not present"
)


@pytest.fixture(scope="module")
def network() -> RoadNetwork:
    return RoadNetwork.from_graphml(SMALL_GRAPH)


@pytest.fixture(scope="module")
def reference_graph(network: RoadNetwork) -> nx.DiGraph:
    """A plain DiGraph built independently of the CSR, for cross-checking.

    Parallel edges are collapsed by minimum travel time, which is the rule the
    RoadNetwork claims to implement; if it in fact summed or took the last
    edge, this graph would disagree and the shortest-path tests would fail.
    """
    tail, head = network.edge_endpoints()
    g = nx.DiGraph()
    g.add_nodes_from(range(network.n_nodes))
    for e in range(network.n_edges):
        u, v = int(tail[e]), int(head[e])
        w = float(network.edge_travel_time[e])
        existing = g.get_edge_data(u, v)
        if existing is None or w < existing["weight"]:
            g.add_edge(u, v, weight=w, length=float(network.edge_length[e]))
    return g


# ---------------------------------------------------------------- OSM loading

def test_speed_imputation_uses_the_class_table():
    """A residential edge with no maxspeed must get the tabled speed, not OSMnx's."""
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=77.0, y=28.0, street_count=3)
    g.add_node(2, x=77.001, y=28.0, street_count=3)
    g.add_edge(1, 2, 0, highway="residential", length=100.0)
    g.add_edge(2, 1, 0, highway="primary", length=100.0, maxspeed="60")
    osm.impute_speeds(g)

    assert g[1][2][0]["speed_kph"] == osm.FREE_FLOW_SPEED_KPH["residential"]
    expected = 100.0 / (osm.FREE_FLOW_SPEED_KPH["residential"] / 3.6)
    assert g[1][2][0]["travel_time"] == pytest.approx(expected)
    # A genuine maxspeed tag wins over the table.
    assert g[2][1][0]["speed_kph"] == pytest.approx(60.0)


def test_maxspeed_and_lane_parsing():
    assert osm.parse_maxspeed("40") == pytest.approx(40.0)
    assert osm.parse_maxspeed("40 km/h") == pytest.approx(40.0)
    assert osm.parse_maxspeed("30 mph") == pytest.approx(48.28, abs=0.01)
    assert osm.parse_maxspeed(["50", "30"]) == pytest.approx(30.0)   # conservative
    assert osm.parse_maxspeed("IN:urban") is None
    assert osm.parse_maxspeed(None) is None
    assert osm.parse_lanes(None) == 1
    assert osm.parse_lanes("3") == 3
    assert osm.parse_lanes(["2", "4"]) == 4


def test_highway_class_resolves_lists_to_the_most_important():
    assert osm.highway_class("tertiary") == "tertiary"
    assert osm.highway_class(["living_street", "residential"]) == "residential"
    assert osm.highway_class(["residential", "secondary"]) == "secondary"


def test_capacity_annotation_multiplies_by_lanes():
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.001, y=0.0)
    g.add_edge(1, 2, 0, highway="primary", length=50.0, lanes="3")
    osm.impute_speeds(g)
    osm.annotate_capacity(g)
    per_lane = osm.EDGE_CAPACITY_VEH_PER_HOUR_PER_LANE["primary"]
    assert g[1][2][0]["capacity"] == pytest.approx(3 * per_lane)
    assert g[1][2][0]["lanes_used"] == 3


def test_strongly_connected_component_is_actually_strongly_connected():
    raw = osm.load_graph(SMALL_GRAPH, strongly_connected=False)
    trimmed = osm.largest_strongly_connected(raw)
    assert trimmed.number_of_nodes() <= raw.number_of_nodes()
    assert nx.is_strongly_connected(trimmed)


# -------------------------------------------------------------- CSR structure

def test_parallel_edges_collapse_by_minimum_not_sum():
    """Two roads between the same junctions: the driver takes the faster one."""
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=0.0, y=0.0, street_count=3)
    g.add_node(2, x=0.001, y=0.0, street_count=3)
    # A fast but long main road and a slow but short lane between the same
    # junctions: 300 m at 45 km/h is 24 s, 200 m at 15 km/h is 48 s.
    g.add_edge(1, 2, 0, highway="primary", length=300.0)
    g.add_edge(1, 2, 1, highway="living_street", length=200.0)
    g.add_edge(2, 1, 0, highway="primary", length=300.0)
    osm.impute_speeds(g)
    osm.annotate_capacity(g)
    net = RoadNetwork(g, name="toy")

    fast = 300.0 / (osm.FREE_FLOW_SPEED_KPH["primary"] / 3.6)
    slow = 200.0 / (osm.FREE_FLOW_SPEED_KPH["living_street"] / 3.6)
    assert fast < slow
    assert net.n_edges == 3
    assert net.n_arcs == 2                       # two ordered pairs
    csr = net.csr_travel_time.toarray()
    assert csr[0, 1] == pytest.approx(fast)      # minimum, not fast + slow
    # Distance must come from the edge actually driven, i.e. the fast one.
    assert net.csr_length.toarray()[0, 1] == pytest.approx(300.0)


def test_csr_matches_the_edge_arrays(network: RoadNetwork):
    """Every stored CSR entry is the minimum travel time of its parallel edges."""
    tail, head = network.edge_endpoints()
    dense_min: dict[tuple[int, int], float] = {}
    for e in range(network.n_edges):
        key = (int(tail[e]), int(head[e]))
        w = float(network.edge_travel_time[e])
        dense_min[key] = min(dense_min.get(key, math.inf), w)
    csr = network.csr_travel_time.tocoo()
    assert csr.nnz == len(dense_min) == network.n_arcs
    for u, v, w in zip(csr.row, csr.col, csr.data):
        assert w == pytest.approx(dense_min[(int(u), int(v))])


# ------------------------------------------------------------ shortest paths

def test_dijkstra_matches_networkx_on_sampled_pairs(network, reference_graph):
    rng = np.random.default_rng(20260902)
    sources = rng.choice(network.n_nodes, 5, replace=False)
    targets = rng.choice(network.n_nodes, 20, replace=False)
    for s in sources:
        dist, _pred = paths.dijkstra(network, int(s))
        ref = nx.single_source_dijkstra_path_length(reference_graph, int(s), weight="weight")
        for t in targets:
            assert dist[t] == pytest.approx(ref[int(t)], rel=1e-12, abs=1e-9)


def test_scipy_matrix_matches_networkx(network, reference_graph):
    rng = np.random.default_rng(7)
    sources = rng.choice(network.n_nodes, 4, replace=False)
    mat, _ = matrix.travel_time_matrix(
        network, sources, targets=np.arange(network.n_nodes)
    )
    for r, s in enumerate(sources):
        ref = nx.single_source_dijkstra_path_length(reference_graph, int(s), weight="weight")
        got = np.array([ref[i] for i in range(network.n_nodes)])
        assert np.allclose(mat[r], got, rtol=1e-12, atol=1e-9)


def test_astar_equals_dijkstra(network):
    rng = np.random.default_rng(11)
    pairs = rng.choice(network.n_nodes, (12, 2), replace=False)
    for s, t in pairs:
        s, t = int(s), int(t)
        if s == t:
            continue
        d = paths.dijkstra(network, s, t)
        a = paths.astar(network, s, t)
        assert a.cost == pytest.approx(d.cost, rel=1e-12, abs=1e-9)
        assert a.nodes[0] == s and a.nodes[-1] == t
        # The paths may differ when several are optimal, but the cost may not.
        assert a.duration_s == pytest.approx(d.duration_s, rel=1e-9)


def test_astar_expands_no_more_nodes_than_dijkstra(network):
    """Goal direction should pay for itself; a regression here means a broken heuristic."""
    rng = np.random.default_rng(3)
    total_d = total_a = 0
    for s, t in rng.choice(network.n_nodes, (8, 2), replace=False):
        if s == t:
            continue
        total_d += paths.dijkstra(network, int(s), int(t)).expanded
        total_a += paths.astar(network, int(s), int(t)).expanded
    assert total_a <= total_d


def test_astar_on_length_weight_matches_dijkstra(network):
    rng = np.random.default_rng(5)
    s, t = (int(x) for x in rng.choice(network.n_nodes, 2, replace=False))
    d = paths.dijkstra(network, s, t, weight="length")
    a = paths.astar(network, s, t, weight="length")
    assert a.cost == pytest.approx(d.cost, rel=1e-12, abs=1e-9)


# ------------------------------------------------------- time-dependent paths

def test_time_dependent_equals_static_when_factors_are_one(network):
    """The control case: a profile of all-1.0 factors must reproduce free flow."""
    rng = np.random.default_rng(1234)
    source = int(rng.integers(network.n_nodes))
    flat = paths.SpeedProfile(
        np.array([0.0, 21600.0, 43200.0, 64800.0]), np.ones(4)
    )
    arrive, _ = paths.time_dependent_dijkstra(network, source, 0.0, flat)
    static, _ = paths.dijkstra(network, source)
    assert np.allclose(arrive, static, rtol=1e-9, atol=1e-6)


def test_time_dependent_ignores_departure_time_under_a_flat_profile(network):
    flat = paths.SpeedProfile.constant(1.0)
    rng = np.random.default_rng(99)
    s, t = (int(x) for x in rng.choice(network.n_nodes, 2, replace=False))
    a = paths.time_dependent_dijkstra(network, s, 0.0, flat, target=t)
    b = paths.time_dependent_dijkstra(network, s, 47_000.0, flat, target=t)
    assert a.duration_s == pytest.approx(b.duration_s, rel=1e-12)


def test_constant_slowdown_scales_travel_time_exactly():
    """At a uniform factor f, every edge must take exactly 1/f times as long."""
    profile = paths.SpeedProfile.constant(0.4)
    for free_flow in (1.0, 37.5, 900.0):
        arrival = paths.traverse_edge(1000.0, free_flow, profile)
        assert arrival - 1000.0 == pytest.approx(free_flow / 0.4)


def test_step_speed_traversal_splits_at_the_period_boundary():
    """The defining property of the Ichoua-Gendreau-Potvin traversal.

    Free-flow time 100 s, departing at t=50 with a period boundary at t=100.
    In the first period the factor is 1.0, so the 50 s available cover half the
    edge; the remaining half is driven at factor 0.5 and takes another 100 s.
    Arrival is therefore t=200, a duration of 150 s. The naive "multiply the
    free-flow time by the departure period's factor" rule would answer 100 s.
    """
    profile = paths.SpeedProfile(np.array([0.0, 100.0]), np.array([1.0, 0.5]))
    arrival = paths.traverse_edge(50.0, 100.0, profile)
    assert arrival == pytest.approx(200.0)
    assert arrival - 50.0 == pytest.approx(150.0)
    # Departing inside the slow period costs the full 200 s.
    assert paths.traverse_edge(100.0, 100.0, profile) - 100.0 == pytest.approx(200.0)


def test_fifo_holds_on_random_edges_and_departures(network):
    """Departing later must never arrive earlier. This is what makes Dijkstra valid."""
    profile = paths.SpeedProfile(
        np.array([0.0, 25200.0, 36000.0, 61200.0, 72000.0]),
        np.array([1.0, 0.35, 0.7, 0.4, 0.9]),
    )
    rng = np.random.default_rng(2026)
    edges = rng.choice(network.n_edges, 400, replace=False)
    departures = np.sort(rng.uniform(0.0, 2 * 86400.0, 40))
    for e in edges:
        free_flow = float(network.edge_free_flow_time[e])
        arrivals = np.array(
            [paths.traverse_edge(d, free_flow, profile) for d in departures]
        )
        assert np.all(np.diff(arrivals) >= -1e-9), f"FIFO violated on edge {e}"
        # And a later departure never arrives before an earlier one departed.
        assert np.all(arrivals >= departures)


def test_congested_departure_is_never_faster_than_free_flow(network):
    """Sanity: slowing the network down cannot shorten a journey."""
    peak = paths.SpeedProfile(
        np.array([0.0, 25200.0, 36000.0]), np.array([1.0, 0.3, 1.0])
    )
    rng = np.random.default_rng(41)
    s, t = (int(x) for x in rng.choice(network.n_nodes, 2, replace=False))
    free = paths.time_dependent_dijkstra(network, s, 0.0, peak, target=t)
    jam = paths.time_dependent_dijkstra(network, s, 26_000.0, peak, target=t)
    assert jam.duration_s >= free.duration_s - 1e-9


def test_speed_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        paths.SpeedProfile(np.array([100.0, 200.0]), np.ones(2))      # must start at 0
    with pytest.raises(ValueError):
        paths.SpeedProfile(np.array([0.0, 0.0]), np.ones(2))          # not increasing
    with pytest.raises(ValueError):
        paths.SpeedProfile(np.array([0.0, 100.0]), np.array([1.0, 0.0]))   # zero speed


# ------------------------------------------------------- dynamic weight update

def test_update_weights_changes_shortest_path_costs(network):
    rng = np.random.default_rng(8)
    s, t = (int(x) for x in rng.choice(network.n_nodes, 2, replace=False))
    before = paths.dijkstra(network, s, t).cost
    try:
        network.update_weights(factors=2.5)
        after = paths.dijkstra(network, s, t).cost
        assert after == pytest.approx(2.5 * before, rel=1e-9)
        assert np.allclose(network.edge_congestion, 1.0 - 1.0 / 2.5)
        assert network.csr_congestion.data.max() == pytest.approx(1.0 - 1.0 / 2.5)
    finally:
        network.reset_weights()
    assert paths.dijkstra(network, s, t).cost == pytest.approx(before, rel=1e-12)
    assert np.all(network.edge_congestion == 0.0)


def test_update_weights_is_idempotent_and_structure_preserving(network):
    rng = np.random.default_rng(17)
    factors = 1.0 + 2.0 * rng.random(network.n_edges)
    indptr_before = network._indptr.copy()
    indices_before = network._indices.copy()
    try:
        network.update_weights(factors=factors)
        first = network.csr_travel_time.data.copy()
        network.update_weights(factors=factors)
        # Factors apply to free flow, not to the current state, so re-applying
        # the same traffic snapshot must give the same network.
        assert np.array_equal(first, network.csr_travel_time.data)
        assert np.array_equal(indptr_before, network._indptr)
        assert np.array_equal(indices_before, network._indices)
    finally:
        network.reset_weights()


def test_update_weights_selects_the_new_fastest_parallel_edge():
    """When traffic reverses which parallel road is faster, the CSR must follow."""
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=0.0, y=0.0, street_count=3)
    g.add_node(2, x=0.001, y=0.0, street_count=3)
    g.add_edge(1, 2, 0, highway="primary", length=300.0)
    g.add_edge(1, 2, 1, highway="living_street", length=200.0)
    g.add_edge(2, 1, 0, highway="primary", length=300.0)
    osm.impute_speeds(g)
    osm.annotate_capacity(g)
    net = RoadNetwork(g, name="toy")
    assert net.csr_length.toarray()[0, 1] == pytest.approx(300.0)   # primary wins

    # Jam the primary road tenfold; the 200 m lane becomes the fastest route,
    # so both the travel time and the reported distance must switch to it.
    factors = np.ones(net.n_edges)
    factors[np.array([c == "primary" for c in net.edge_classes])] = 10.0
    net.update_weights(factors=factors)
    assert net.csr_length.toarray()[0, 1] == pytest.approx(200.0)
    assert net.csr_travel_time.toarray()[0, 1] == pytest.approx(
        200.0 / (osm.FREE_FLOW_SPEED_KPH["living_street"] / 3.6)
    )


def test_update_weights_validates_shape(network):
    with pytest.raises(ValueError):
        network.update_weights(np.ones(3))
    with pytest.raises(ValueError):
        network.update_weights(factors=np.zeros(network.n_edges))
    with pytest.raises(ValueError):
        network.update_weights()


# --------------------------------------------------------------- matrix build

def test_matrix_has_no_infinities_on_the_strongly_connected_component(network):
    rng = np.random.default_rng(64)
    nodes = rng.choice(network.n_nodes, 40, replace=False)
    result = matrix.build_matrices(network, nodes)
    assert result.is_finite
    assert np.all(np.isfinite(result.distance))
    assert np.all(np.isfinite(result.congestion))
    assert np.all(np.diag(result.duration) == 0.0)
    assert np.all(result.duration >= 0.0)


def test_matrix_distance_is_measured_along_the_fastest_path(network):
    """The distance matrix must describe the same path as the duration matrix."""
    rng = np.random.default_rng(31)
    nodes = rng.choice(network.n_nodes, 12, replace=False)
    result = matrix.build_matrices(network, nodes)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            path = matrix.route_node_path(network, result, i, j)
            assert path[0] == nodes[i] and path[-1] == nodes[j]
            duration = distance = 0.0
            for a, b in zip(path[:-1], path[1:]):
                e = network._fastest_edge_between(int(a), int(b))
                assert e >= 0, "reconstructed path uses a non-existent edge"
                duration += float(network.edge_travel_time[e])
                distance += float(network.edge_length[e])
            assert duration == pytest.approx(result.duration[i, j], rel=1e-9)
            assert distance == pytest.approx(result.distance[i, j], rel=1e-9)


def test_matrix_congestion_is_a_time_weighted_mean(network):
    rng = np.random.default_rng(77)
    nodes = rng.choice(network.n_nodes, 10, replace=False)
    try:
        network.update_weights(factors=1.0 + 2.0 * rng.random(network.n_edges))
        result = matrix.build_matrices(network, nodes)
        assert np.all(result.congestion >= 0.0) and np.all(result.congestion <= 1.0)
        i, j = 0, 3
        path = matrix.route_node_path(network, result, i, j)
        num = den = 0.0
        for a, b in zip(path[:-1], path[1:]):
            e = network._fastest_edge_between(int(a), int(b))
            tt = float(network.edge_travel_time[e])
            num += float(network.edge_congestion[e]) * tt
            den += tt
        assert result.congestion[i, j] == pytest.approx(num / den, rel=1e-9)
    finally:
        network.reset_weights()


def test_matrix_triangle_inequality(network):
    """Shortest-path durations must satisfy d(i,k) <= d(i,j) + d(j,k)."""
    rng = np.random.default_rng(101)
    nodes = rng.choice(network.n_nodes, 25, replace=False)
    d = matrix.build_matrices(network, nodes).duration
    # violations[i, j, k] = d(i,k) - (d(i,j) + d(j,k)); all must be <= 0.
    violations = d[:, None, :] - (d[:, :, None] + d[None, :, :])
    assert violations.max() <= 1e-6


# ------------------------------------------------------------- geo and export

def test_nearest_node_matches_brute_force_haversine(network):
    rng = np.random.default_rng(55)
    lats = network.coords[:, 0]
    lons = network.coords[:, 1]
    query_lat = rng.uniform(lats.min(), lats.max(), 25)
    query_lon = rng.uniform(lons.min(), lons.max(), 25)
    got = network.nearest_node(query_lat, query_lon)
    for q, (la, lo) in enumerate(zip(query_lat, query_lon)):
        # Exact great-circle distance from the query point to every node.
        p1, p2 = np.deg2rad(la), np.deg2rad(lats)
        dl = np.deg2rad(lons - lo)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        exact = np.argmin(a)
        assert got[q] == exact or a[got[q]] == pytest.approx(a[exact], rel=1e-9)


def test_nearest_node_returns_scalar_for_scalar_input(network):
    idx = network.nearest_node(float(network.coords[10, 0]), float(network.coords[10, 1]))
    assert isinstance(idx, int)
    assert idx == 10


def test_edge_geojson_level_of_detail_is_monotone(network):
    everything = network.edge_geojson(min_importance=0)
    arterials = network.edge_geojson(min_importance=3)
    assert len(everything["features"]) == network.n_edges
    assert 0 < len(arterials["features"]) < len(everything["features"])
    for feature in arterials["features"]:
        cls = feature["properties"]["highway"]
        assert osm.CLASS_IMPORTANCE.get(cls, osm.DEFAULT_IMPORTANCE) >= 3
        assert feature["geometry"]["type"] == "LineString"
        lon, lat = feature["geometry"]["coordinates"][0]
        assert 60.0 < lon < 100.0 and 5.0 < lat < 40.0   # GeoJSON is lon, lat


def test_edge_geojson_respects_max_edges(network):
    small = network.edge_geojson(max_edges=100)
    assert len(small["features"]) == 100


def test_route_geojson_traces_a_real_path(network):
    rng = np.random.default_rng(19)
    s, t = (int(x) for x in rng.choice(network.n_nodes, 2, replace=False))
    result = paths.dijkstra(network, s, t)
    feature = network.route_geojson(result.nodes)
    assert feature["geometry"]["type"] == "LineString"
    assert len(feature["geometry"]["coordinates"]) >= len(result.nodes)
    # GeoJSON properties are rounded to two decimals for payload size.
    assert feature["properties"]["duration_s"] == pytest.approx(result.duration_s, abs=0.01)
    assert feature["properties"]["distance_m"] == pytest.approx(result.distance_m, abs=0.01)
    assert feature["properties"]["node_ids"][0] == network.node_id_of(s)


def test_route_geojson_rejects_non_adjacent_nodes(network):
    with pytest.raises(ValueError):
        network.route_geojson([0, network.n_nodes - 1])


# ------------------------------------------------------------ instance builder

def test_build_instance_produces_a_valid_instance(network):
    inst = builder.build_instance(network, 30, seed=3)
    assert isinstance(inst, Instance)
    assert inst.size == 31
    assert inst.n_customers == 30
    assert inst.demand[0] == 0.0
    assert np.all(inst.demand[1:] >= 1)
    assert inst.duration.shape == (31, 31)
    assert inst.distance.shape == (31, 31)
    assert inst.congestion.shape == (31, 31)
    assert inst.coords.shape == (31, 2)
    assert inst.node_ids is not None and len(inst.node_ids) == 31
    assert np.all(np.isfinite(inst.duration))
    assert inst.meta["network"] == network.name
    assert inst.meta["units"]["duration"] == "seconds"
    # Default objective on a road network is travel time, not distance.
    assert inst.weights.time == 1.0
    # Every stop id must be a real node of the network.
    for node_id in inst.node_ids:
        assert network.index_of(node_id) >= 0


def test_build_instance_is_reproducible(network):
    a = builder.build_instance(network, 20, seed=42)
    b = builder.build_instance(network, 20, seed=42)
    c = builder.build_instance(network, 20, seed=43)
    assert a.node_ids == b.node_ids
    assert np.array_equal(a.demand, b.demand)
    assert np.allclose(a.duration, b.duration)
    assert a.node_ids != c.node_ids


def test_build_instance_depot_snaps_to_coordinates(network):
    lat, lon = float(network.coords[100, 0]), float(network.coords[100, 1])
    inst = builder.build_instance(network, 10, seed=1, depot_latlon=(lat, lon))
    assert inst.node_ids[0] == network.node_id_of(100)
    assert inst.coords[0, 0] == pytest.approx(lat)


def test_spread_sampling_covers_more_ground_than_random(network):
    """Farthest-point sampling should genuinely spread the stops out."""
    spread = builder.select_stops(network, 40, seed=5, sampling="spread")
    uniform = builder.select_stops(network, 40, seed=5, sampling="random")

    def mean_nearest_neighbour(coords: np.ndarray) -> float:
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).mean())

    assert mean_nearest_neighbour(spread.coords) > mean_nearest_neighbour(uniform.coords)


def test_instance_evaluation_agrees_with_the_matrices(network):
    """A route's reported duration must equal the sum of the matrix entries."""
    inst, mats = builder.build_instance(network, 15, seed=9, return_matrices=True)
    routes = [[1, 2, 3], [4, 5, 6, 7]]
    stats = inst.evaluate(routes)
    expected = 0.0
    for route in routes:
        prev = 0
        for c in route:
            expected += inst.duration[prev, c]
            prev = c
        expected += inst.duration[prev, 0]
    assert stats.duration == pytest.approx(expected, rel=1e-12)
    assert mats.size == inst.size


def test_routes_geojson_matches_the_instance_durations(network):
    inst, mats = builder.build_instance(network, 12, seed=13, return_matrices=True)
    routes = [[1, 2], [3, 4, 5]]
    collection = builder.routes_geojson(network, mats, routes)
    assert len(collection["features"]) == 2
    for feature, route in zip(collection["features"], routes):
        legs = [0, *route, 0]
        expected = sum(inst.duration[a, b] for a, b in zip(legs[:-1], legs[1:]))
        assert feature["properties"]["duration_s"] == pytest.approx(expected, rel=1e-4)
