"""Road-network layer: the graph-based network model of the problem statement.

Nodes are intersections and depots, edges are road segments carrying distance,
free-flow travel time, a congestion level and a capacity. The layer is split so
that each concern can be tested against ground truth on its own:

``osm``      loading, cleaning and speed/capacity imputation for OSM extracts
``network``  :class:`~qroute.graph.network.RoadNetwork` - CSR adjacency,
             O(edges) dynamic weight update, nearest-node lookup, GeoJSON export
``paths``    exact Dijkstra, A*, and FIFO-correct time-dependent Dijkstra
``matrix``   many-to-many travel-time / distance / congestion matrices
``builder``  assembling a routing :class:`~qroute.problems.instance.Instance`
             from a network

Everything works offline from the GraphML files under ``data/osm``.
"""

from __future__ import annotations

from qroute.graph.builder import (
    StopSelection,
    build_instance,
    leg_node_paths,
    routes_geojson,
    select_stops,
)
from qroute.graph.matrix import (
    MatrixResult,
    build_matrices,
    reconstruct_path,
    route_node_path,
    travel_time_matrix,
)
from qroute.graph.network import RoadNetwork
from qroute.graph.osm import (
    EDGE_CAPACITY_VEH_PER_HOUR_PER_LANE,
    FREE_FLOW_SPEED_KPH,
    graph_summary,
    impute_speeds,
    largest_strongly_connected,
    list_graphs,
    load_graph,
    save_graph,
)
from qroute.graph.paths import (
    PathResult,
    SpeedProfile,
    astar,
    dijkstra,
    time_dependent_dijkstra,
    traverse_edge,
)

__all__ = [
    "EDGE_CAPACITY_VEH_PER_HOUR_PER_LANE",
    "FREE_FLOW_SPEED_KPH",
    "MatrixResult",
    "PathResult",
    "RoadNetwork",
    "SpeedProfile",
    "StopSelection",
    "astar",
    "build_instance",
    "build_matrices",
    "dijkstra",
    "graph_summary",
    "impute_speeds",
    "largest_strongly_connected",
    "leg_node_paths",
    "list_graphs",
    "load_graph",
    "reconstruct_path",
    "route_node_path",
    "routes_geojson",
    "save_graph",
    "select_stops",
    "time_dependent_dijkstra",
    "travel_time_matrix",
    "traverse_edge",
]
