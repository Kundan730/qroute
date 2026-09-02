"""Process-wide state shared by the HTTP endpoints.

Three things have to outlive a single request, and this module owns all three.

**Road networks.** Loading, cleaning and CSR-indexing the bundled Bengaluru
extract takes about nine seconds and roughly a hundred megabytes; building the
traffic simulator's edge arrays on top of it takes another two. Doing that per
request is out of the question, so a network is loaded at most once and then
shared. Loading is lazy, guarded by a lock so two simultaneous first requests do
not both pay for it, and by default the first bundled network is loaded in a
background thread at startup so that a demonstration does not begin with a
nine-second pause.

**Generated instances.** ``POST /api/networks/{id}/instance`` produces a routing
instance from a road graph. The instance itself is small, but the
:class:`~qroute.graph.matrix.MatrixResult` that goes with it holds the Dijkstra
predecessor trees needed to draw a route as a real polyline, and rebuilding
those to render one run would cost as much as the run. The store keeps a bounded
number of recent instances with their matrices.

**Runs.** The registry lives in :mod:`qroute.api.runs`; the attribute is here so
that everything mutable has a single owner. It is injected at application
startup rather than imported, which keeps this module free of any dependency on
the run machinery.

Concurrency
-----------
The simulator and the network hold mutable state (a clock, an event list, live
edge weights). Every endpoint that reads or writes them does so while holding
``NetworkBundle.lock``, and every such operation is short - setting the clock and
recomputing 34k edge weights takes about two milliseconds. Endpoints run in
FastAPI's thread pool (they are ``def``, not ``async def``), so a blocking lock
there does not stall the event loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qroute.api.runs import RunRegistry
    from qroute.graph.matrix import MatrixResult
    from qroute.graph.network import RoadNetwork
    from qroute.problems.instance import Instance
    from qroute.traffic.simulator import TrafficSimulator

log = logging.getLogger("qroute.api")

#: Travel time given to an edge that an incident has closed. The simulator marks
#: closures with ``inf``, but a CSR matrix cannot store one (SciPy's shortest
#: path treats a stored infinity as a hard error and a stored zero as an absent
#: edge), and :meth:`RoadNetwork.update_weights` rejects it explicitly. A finite
#: but enormous time has the same routing effect - no shortest path will ever
#: use the edge unless there is genuinely no alternative - while keeping the
#: matrices finite so the optimiser's cost comparisons stay meaningful.
CLOSED_EDGE_TRAVEL_TIME_S: float = 1e7

#: How many generated instances to keep. Each carries a ``(k, n_nodes)``
#: predecessor matrix; at 40 stops on the Bengaluru extract that is about 2 MB,
#: so a couple of dozen is a sensible ceiling for a demonstration server.
MAX_STORED_INSTANCES: int = 32

#: Default simulator seed. Fixed so that two runs of the platform show the same
#: congestion pattern and a screenshot can be reproduced.
DEFAULT_SIMULATOR_SEED: int = 26137

#: Default clock: Monday 09:00, in the middle of the morning peak.
DEFAULT_START_MINUTE: float = 9 * 60.0


# --------------------------------------------------------------------------
# Road networks
# --------------------------------------------------------------------------


class NetworkBundle:
    """A loaded road network together with its traffic simulator.

    The two are kept side by side because they must never disagree: the
    simulator computes the current travel time of every edge, and the network is
    what shortest paths and origin-destination matrices are read from. Every
    change to the simulator (a new clock reading, a new incident) is followed by
    :meth:`apply_traffic`, which is the dynamic weight update.

    The edge order is shared. :func:`qroute.traffic.simulator.edge_arrays_from_graph`
    iterates ``graph.edges(keys=True)``, which is exactly the order
    :class:`~qroute.graph.network.RoadNetwork` uses when it builds its own edge
    arrays, so index *i* means the same physical road segment on both sides.
    That is what makes an integer edge index a usable handle for the frontend:
    the map draws feature ``edge=i`` and posts an incident on edge ``i``.
    """

    def __init__(self, network_id: str, network: "RoadNetwork", simulator: "TrafficSimulator"):
        self.id = network_id
        self.network = network
        self.simulator = simulator
        self.lock = threading.RLock()
        self.loaded_at = time.time()
        coords = network.coords
        self.bbox = [
            float(coords[:, 0].min()),
            float(coords[:, 1].min()),
            float(coords[:, 0].max()),
            float(coords[:, 1].max()),
        ]
        self.center = [float(coords[:, 0].mean()), float(coords[:, 1].mean())]

    # ------------------------------------------------------------ weights
    def apply_traffic(self) -> None:
        """Push the simulator's current edge travel times onto the network.

        This is the O(edges) dynamic weight update: the CSR sparsity structure
        is untouched and only the three data arrays are recomputed. Closed edges
        are given :data:`CLOSED_EDGE_TRAVEL_TIME_S` rather than infinity, for the
        reason documented on that constant.
        """
        times = self.simulator.edge_travel_times()
        closed = ~np.isfinite(times)
        if closed.any():
            times = np.where(closed, CLOSED_EDGE_TRAVEL_TIME_S, times)
        congestion = np.clip(self.simulator.congestion_levels(), 0.0, 1.0)
        congestion = np.where(closed, 1.0, congestion)
        self.network.update_weights(travel_times=times, congestion=congestion)

    def set_minute(self, minute: float) -> None:
        """Move the simulated clock and re-price the network."""
        self.simulator.set_time(float(minute))
        self.apply_traffic()

    # ------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        s = self.network.summary()
        return {
            "id": self.id,
            "name": self.network.name,
            "n_nodes": int(self.network.n_nodes),
            "n_edges": int(self.network.n_edges),
            "center": self.center,
            "bbox": self.bbox,
            "loaded": True,
            "total_length_km": s["total_length_km"],
        }


# --------------------------------------------------------------------------
# Generated instances
# --------------------------------------------------------------------------


@dataclass
class StoredInstance:
    """A routing instance the API produced, plus what is needed to redraw it.

    ``matrices`` is present only for instances built from a road network. It
    carries the shortest-path predecessor trees, which turn a solver's answer
    (a sequence of stop indices) into a polyline that follows real roads.
    ``stop_nodes`` lets the same stops be re-measured after traffic changes,
    which is what re-optimisation needs.
    """

    name: str
    instance: "Instance"
    family: str = "network"
    network_id: Optional[str] = None
    matrices: Optional["MatrixResult"] = None
    stop_nodes: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def geographic(self) -> bool:
        return self.network_id is not None


# --------------------------------------------------------------------------
# The state object
# --------------------------------------------------------------------------


class ApiState:
    """Everything the API keeps between requests."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.runs: Optional["RunRegistry"] = None

        self._networks: dict[str, NetworkBundle] = {}
        self._network_locks: dict[str, threading.Lock] = {}
        self._loading: set[str] = set()
        self._registry_lock = threading.Lock()

        self._instances: "OrderedDict[str, StoredInstance]" = OrderedDict()
        self._instance_lock = threading.Lock()

        self._benchmark_index: Optional[list[dict[str, Any]]] = None
        self._benchmark_index_lock = threading.Lock()

        self.warmup: dict[str, Any] = {"done": False, "seconds": 0.0, "detail": "not started"}
        self.solvers: dict[str, Any] = {
            "ortools_available": False,
            "pyvrp_available": False,
            "pyvrp_version": None,
            "probed": False,
        }

    # ------------------------------------------------------- capabilities
    def probe_solvers(self) -> dict[str, Any]:
        """Find out once which optional solvers are installed.

        ``/api/health`` reports this, and the browser polls health while the
        server is warming up. Probing on demand would mean the first poll pays
        for importing PyVRP - measured at 3.7 seconds - inside a request that is
        supposed to answer instantly, so the probe happens here, at startup, and
        the endpoint only reads the answer.
        """
        ortools_ok = True
        try:
            import ortools  # noqa: F401
        except Exception:  # pragma: no cover - ortools is a hard dependency
            ortools_ok = False
        try:
            from qroute.baselines import pyvrp_hgs

            pyvrp_ok = pyvrp_hgs.available()
            pyvrp_version = pyvrp_hgs.version()
        except Exception:  # pragma: no cover - defensive
            pyvrp_ok, pyvrp_version = False, None
        self.solvers = {
            "ortools_available": ortools_ok,
            "pyvrp_available": bool(pyvrp_ok),
            "pyvrp_version": pyvrp_version,
            "probed": True,
        }
        return self.solvers

    # ---------------------------------------------------------- networks
    @staticmethod
    def available_network_ids() -> list[str]:
        """Names of the GraphML road networks bundled under ``data/osm``."""
        from qroute.graph import osm as osm_mod

        return osm_mod.list_graphs()

    def loaded_network_ids(self) -> list[str]:
        with self._registry_lock:
            return sorted(self._networks)

    def loading_network_ids(self) -> list[str]:
        with self._registry_lock:
            return sorted(self._loading)

    def is_loaded(self, network_id: str) -> bool:
        with self._registry_lock:
            return network_id in self._networks

    def get_network(self, network_id: str) -> NetworkBundle:
        """Return the bundle for ``network_id``, loading it if necessary.

        Raises ``KeyError`` when no such graph is bundled; the endpoint turns
        that into a 404 with the list of names that do exist.
        """
        with self._registry_lock:
            bundle = self._networks.get(network_id)
            if bundle is not None:
                return bundle
            if network_id not in self.available_network_ids():
                raise KeyError(network_id)
            lock = self._network_locks.setdefault(network_id, threading.Lock())

        # Loading happens outside the registry lock so that a nine-second load
        # of one network does not block a request for another one.
        with lock:
            with self._registry_lock:
                bundle = self._networks.get(network_id)
                if bundle is not None:
                    return bundle
                self._loading.add(network_id)
            try:
                bundle = self._build_bundle(network_id)
            finally:
                with self._registry_lock:
                    self._loading.discard(network_id)
            with self._registry_lock:
                self._networks[network_id] = bundle
            return bundle

    @staticmethod
    def _build_bundle(network_id: str) -> NetworkBundle:
        from qroute.graph.network import RoadNetwork
        from qroute.traffic.simulator import TrafficSimulator, edge_arrays_from_graph

        started = time.perf_counter()
        network = RoadNetwork.from_graphml(network_id)
        # The edge arrays are read from the NetworkX graph rather than from the
        # RoadNetwork's public properties on purpose: the graph carries the
        # highway class and lane count that the congestion model needs, and the
        # generic adapter in the simulator would fall back to calling every edge
        # residential with one lane. The edge order is identical either way.
        arrays = edge_arrays_from_graph(network.graph)
        simulator = TrafficSimulator(
            arrays, seed=DEFAULT_SIMULATOR_SEED, start_minute=DEFAULT_START_MINUTE
        )
        bundle = NetworkBundle(network_id, network, simulator)
        bundle.apply_traffic()
        log.info(
            "loaded network %s (%d nodes, %d edges) in %.2fs",
            network_id,
            network.n_nodes,
            network.n_edges,
            time.perf_counter() - started,
        )
        return bundle

    def network_summaries(self) -> list[dict[str, Any]]:
        """Summaries for every bundled network, without forcing a load.

        Node and edge counts for a network that is not loaded yet come from
        ``data/osm/index.json`` when it exists, and are ``0`` otherwise. The
        ``loaded`` flag says which case a row is in, so the frontend can show
        "not loaded yet" rather than "empty graph".
        """
        index = self._osm_index()
        out: list[dict[str, Any]] = []
        for nid in self.available_network_ids():
            if self.is_loaded(nid):
                out.append(self.get_network(nid).summary())
                continue
            meta = index.get(nid, {})
            bbox = meta.get("bbox")
            center = meta.get("center")
            if center is None and bbox is not None:
                center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
            out.append(
                {
                    "id": nid,
                    "name": meta.get("name", nid),
                    # The bundled index uses the short key names; accept both so
                    # a regenerated index in either shape still gives real counts.
                    "n_nodes": int(meta.get("n_nodes", meta.get("nodes", 0)) or 0),
                    "n_edges": int(meta.get("n_edges", meta.get("edges", 0)) or 0),
                    "center": [float(c) for c in center] if center else [0.0, 0.0],
                    "bbox": [float(b) for b in bbox] if bbox else [0.0, 0.0, 0.0, 0.0],
                    "loaded": False,
                    "total_length_km": meta.get("total_length_km"),
                }
            )
        return out

    @staticmethod
    def _osm_index() -> dict[str, dict[str, Any]]:
        """Read ``data/osm/index.json`` if the data component wrote one."""
        import json

        from qroute.graph.osm import OSM_DIR

        path = Path(OSM_DIR) / "index.json"
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        if isinstance(raw, dict) and "networks" in raw:
            raw = raw["networks"]
        if isinstance(raw, list):
            return {str(r.get("id") or r.get("name")): r for r in raw if isinstance(r, dict)}
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        return {}

    def preload(self, spec: Optional[str] = None) -> None:
        """Load networks named by ``spec`` in the calling thread.

        ``spec`` is read from ``QROUTE_API_PRELOAD`` when omitted: ``none``
        loads nothing, ``first`` loads only the first bundled graph, an empty or
        missing value loads them all, and anything else is treated as a
        comma-separated list of names.

        Loading all three bundled extracts takes about 28 seconds and settles at
        roughly 550 MB resident, measured on the development machine. That is
        worth paying in the background at startup: it means no click during a
        demonstration ever waits ten seconds for a graph to load, and the
        network list carries real bounding boxes rather than placeholders.
        """
        if spec is None:
            spec = os.environ.get("QROUTE_API_PRELOAD", "")
        spec = spec.strip()
        available = self.available_network_ids()
        if spec.lower() == "none" or not available:
            return
        if spec.lower() == "first":
            wanted = available[:1]
        elif spec.lower() == "all" or not spec:
            wanted = available
        else:
            wanted = [s.strip() for s in spec.split(",") if s.strip()]
        for nid in wanted:
            try:
                self.get_network(nid)
            except Exception:  # a bad graph must not stop the server booting
                log.exception("preloading network %s failed", nid)

    # --------------------------------------------------------- instances
    def store_instance(self, stored: StoredInstance) -> StoredInstance:
        with self._instance_lock:
            self._instances[stored.name] = stored
            self._instances.move_to_end(stored.name)
            while len(self._instances) > MAX_STORED_INSTANCES:
                self._instances.popitem(last=False)
        return stored

    def get_stored_instance(self, name: str) -> Optional[StoredInstance]:
        with self._instance_lock:
            stored = self._instances.get(name)
            if stored is not None:
                self._instances.move_to_end(stored.name)
            return stored

    def stored_instances(self) -> list[StoredInstance]:
        with self._instance_lock:
            return list(self._instances.values())

    def resolve_instance(self, name: str) -> StoredInstance:
        """Look a name up in the generated store, then in the benchmark sets.

        Raises ``KeyError`` when neither knows it.
        """
        stored = self.get_stored_instance(name)
        if stored is not None:
            return stored

        from qroute.problems.loaders import load

        try:
            instance = load(name)
        except FileNotFoundError as exc:
            raise KeyError(name) from exc
        family = str(instance.meta.get("family", ""))
        family = "vrptw" if family == "solomon" else "cvrp"
        return StoredInstance(name=instance.name, instance=instance, family=family)

    # -------------------------------------------------------- benchmarks
    def benchmark_instances(self) -> list[dict[str, Any]]:
        """Summaries of every benchmark instance on disk, cached after the first call.

        Reading all 138 files takes about 0.4 s, which is tolerable once and not
        per request.
        """
        # The build happens *inside* the lock rather than beside it. Health is
        # polled every couple of hundred milliseconds while the server warms up,
        # and with the check and the build separated every poll that arrived
        # before the first one finished would start its own build. One caller
        # pays 0.4 s and the rest wait for that same answer.
        with self._benchmark_index_lock:
            if self._benchmark_index is not None:
                return self._benchmark_index

            from qroute.problems.loaders import list_instances, load

            rows: list[dict[str, Any]] = []
            for family, names in list_instances().items():
                for name in names:
                    try:
                        inst = load(name)
                    except Exception:
                        log.exception("could not read benchmark instance %s", name)
                        continue
                    rows.append(instance_summary(inst, family))
            rows.sort(key=lambda r: (r["family"], r["n_customers"], r["name"]))
            self._benchmark_index = rows
            return rows

    # ----------------------------------------------------------- warm-up
    def warm_up(self) -> dict[str, Any]:
        """Compile the JIT kernels with one tiny solve.

        The numba kernels behind the decoder and the local search are compiled
        on first call. On a cold cache that costs tens of seconds; on a warm one
        it is a fraction of a second. Either way it should be paid at startup and
        not by whoever presses the first button, and either way the figure
        belongs in ``/api/health`` rather than in a comment.
        """
        started = time.perf_counter()
        detail = ""
        try:
            from qroute.algorithms.base import StopCriteria
            from qroute.algorithms.registry import build
            from qroute.core.rng import make_rng
            from qroute.problems.instance import Instance

            rng = make_rng(0)
            coords = rng.uniform(0.0, 100.0, size=(9, 2))
            diff = coords[:, None, :] - coords[None, :, :]
            distance = np.sqrt((diff**2).sum(-1))
            demand = np.zeros(9)
            demand[1:] = 5.0
            inst = Instance(name="warmup", distance=distance, demand=demand, capacity=20.0)
            stop = StopCriteria(max_iterations=2, max_seconds=30.0)
            result = build("qpso", inst, stop=stop, seed=0, swarm_size=6).solve()
            detail = f"qpso on a 8-customer instance, best cost {result.best.cost:.2f}"
        except Exception as exc:  # never let a warm-up failure stop the server
            log.exception("warm-up failed")
            detail = f"failed: {type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - started
        self.warmup = {"done": True, "seconds": round(seconds, 3), "detail": detail}
        log.info("warm-up finished in %.2fs (%s)", seconds, detail)
        return self.warmup

    def start_background_startup(self) -> threading.Thread:
        """Warm up and preload without blocking the server from accepting requests."""

        def _work() -> None:
            self.probe_solvers()
            # The benchmark index is built before the road networks, not after:
            # it costs 0.4 s against their 28 s, and listing instances is what
            # the UI asks for first.
            try:
                self.benchmark_instances()
            except Exception:
                log.exception("benchmark index build failed")
            self.warm_up()
            self.preload()

        thread = threading.Thread(target=_work, name="qroute-api-startup", daemon=True)
        thread.start()
        return thread


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def instance_summary(instance: "Instance", family: str) -> dict[str, Any]:
    """The short form of an instance, as ``GET /api/instances`` lists it."""
    family = {"solomon": "vrptw", "cvrplib": "cvrp"}.get(family, family)
    return {
        "name": instance.name,
        "family": family,
        "n_customers": int(instance.n_customers),
        "capacity": float(instance.capacity),
        "n_vehicles": int(instance.n_vehicles) if instance.n_vehicles else None,
        "bks": float(instance.meta["bks"]) if instance.meta.get("bks") else None,
        "has_time_windows": bool(instance.has_time_windows),
    }


def instance_detail(stored: StoredInstance) -> dict[str, Any]:
    """The full form, including the coordinates the map and the plot need."""
    inst = stored.instance
    detail = instance_summary(inst, stored.family)
    coords = inst.coords
    detail.update(
        {
            "coords": [[float(a), float(b)] for a, b in coords] if coords is not None else [],
            "demand": [float(d) for d in inst.demand],
            "geographic": stored.geographic,
            "node_ids": [int(n) for n in inst.node_ids] if inst.node_ids else None,
            "meta": _jsonable(dict(inst.meta)),
        }
    )
    if inst.time_windows is not None:
        detail["time_windows"] = [[float(a), float(b)] for a, b in inst.time_windows]
    if inst.service_time is not None:
        detail["service_time"] = [float(s) for s in inst.service_time]
    return detail


def _jsonable(value: Any) -> Any:
    """Make numpy scalars and arrays safe for ``json.dumps``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


#: The single instance used by the application. A module-level object rather
#: than a FastAPI dependency because the run registry's reader threads and the
#: startup thread need it too, and none of those are inside a request scope.
STATE = ApiState()
