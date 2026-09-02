"""Request and response models for the HTTP API.

These models are the server side of a contract whose client side is declared in
``frontend/src/api/types.ts``. Field names and types are chosen to match that
file exactly, so that a change on either side shows up immediately rather than
as an ``undefined`` deep inside a chart.

Two conventions are worth stating because they recur:

* **Optional means "the backend may honestly not know this"**, not "the backend
  was too lazy to fill it in". ``bks`` is ``None`` when no reference solution
  ships with the instance; ``geojson`` is ``None`` when the instance is not
  built on a road network and therefore has no road geometry to draw.
* **Responses carry units in their field names** (``_s``, ``_m``, ``_km``,
  ``minute``) wherever a bare number would be ambiguous. The library's internal
  units are metres and seconds for road networks, and the instance's own
  arbitrary units for the CVRPLIB and Solomon benchmark families.

Request models use ``extra="forbid"`` so that a misspelt field is a 422 with a
useful message rather than a silently ignored setting - a setting the user
believes they changed but did not is worse than an error.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


class WarmUpInfo(BaseModel):
    """What the startup warm-up did and how long it cost.

    The compiled kernels in :mod:`qroute.algorithms.kernels` are JIT-compiled on
    first use. Without a warm-up the first request a judge makes would pay that
    cost and look like a slow solver; reporting the figure here is the honest
    alternative to hiding it.
    """

    done: bool = False
    seconds: float = 0.0
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    networks: int
    instances: int
    algorithms: int
    network_ids: list[str] = Field(default_factory=list)
    networks_loaded: list[str] = Field(default_factory=list)
    networks_loading: list[str] = Field(default_factory=list)
    ortools_available: bool = False
    pyvrp_available: bool = False
    pyvrp_version: Optional[str] = None
    warmup: WarmUpInfo = Field(default_factory=WarmUpInfo)
    active_runs: int = 0
    worker_start_method: Optional[str] = None
    workers_primed: bool = False
    worker_prime_seconds: Optional[float] = None


ParamKind = Literal["int", "float", "bool", "choice", "text"]


class ParamSpec(BaseModel):
    """One tunable parameter of a solver, as the frontend's form needs it."""

    name: str
    kind: ParamKind
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list[str]] = None
    description: Optional[str] = None


class AlgorithmInfo(BaseModel):
    name: str
    description: str
    kind: Literal["metaheuristic", "baseline", "exact", "reference"] = "metaheuristic"
    supports_warm_start: bool = False
    #: ``None`` when installation is not in question; ``False`` for an optional
    #: dependency that is not present, so the UI can grey the entry out instead
    #: of offering a run that will fail.
    available: Optional[bool] = None
    params: list[ParamSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------

InstanceFamily = Literal["cvrp", "vrptw", "network"]


class InstanceSummary(BaseModel):
    name: str
    family: InstanceFamily
    n_customers: int
    capacity: float
    n_vehicles: Optional[int] = None
    bks: Optional[float] = None
    has_time_windows: bool = False


class InstanceDetail(InstanceSummary):
    coords: list[list[float]] = Field(default_factory=list)
    demand: list[float] = Field(default_factory=list)
    time_windows: Optional[list[list[float]]] = None
    service_time: Optional[list[float]] = None
    node_ids: Optional[list[int]] = None
    geographic: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class InstanceRequest(BaseModel):
    """Body of ``POST /api/networks/{id}/instance``."""

    model_config = ConfigDict(extra="forbid")

    n_customers: int = Field(default=30, ge=2, le=400)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    minute: Optional[float] = Field(
        default=None,
        description="Simulator clock in minutes since Monday 00:00. When given, "
        "the traffic state is moved there before the matrices are built, so the "
        "instance and the map agree.",
    )
    capacity: Optional[float] = Field(default=None, gt=0)
    n_vehicles: Optional[int] = Field(default=None, ge=1, le=200)
    service_time_s: float = Field(default=0.0, ge=0.0)
    sampling: Literal["spread", "random"] = "spread"
    depot_lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    depot_lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)


# --------------------------------------------------------------------------
# Networks and traffic
# --------------------------------------------------------------------------


class NetworkSummary(BaseModel):
    id: str
    name: str
    n_nodes: int
    n_edges: int
    center: list[float]
    bbox: list[float]
    loaded: bool = False
    total_length_km: Optional[float] = None


class TimeRequest(BaseModel):
    """Body of ``POST /api/traffic/{id}/time``.

    Either give ``minute`` directly, or give an ``hour`` (and optionally a
    ``day_of_week``) and let the server convert. Giving both is rejected rather
    than silently resolved, because the two would disagree the moment a caller
    forgot to update one of them.
    """

    model_config = ConfigDict(extra="forbid")

    minute: Optional[float] = Field(default=None, ge=0.0, le=7 * 24 * 60.0)
    hour: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    day_of_week: int = Field(default=0, ge=0, le=6)


class EventRequest(BaseModel):
    """Body of ``POST /api/traffic/{id}/events``.

    ``edges`` are indices into the network's edge array, which is exactly the
    ``edge`` property carried by every feature of ``/api/networks/{id}/edges``.
    That is what lets the user click a road on the map and block it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["lane_blockage", "closure", "slowdown"]
    edges: list[int] = Field(min_length=1, max_length=5000)
    start_minute: Optional[float] = Field(
        default=None, description="Defaults to the simulator's current clock."
    )
    duration_minutes: float = Field(default=60.0, gt=0.0)
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    lanes: int = Field(default=2, ge=1, le=8)
    blockage: Optional[str] = None
    speed_multiplier: float = Field(default=0.5, gt=0.0, le=1.0)
    description: str = ""


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

RunState = Literal["queued", "running", "done", "cancelled", "failed"]


class RunRequest(BaseModel):
    """Body of ``POST /api/runs``."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "qpso"
    instance: str = Field(
        description="A benchmark instance name such as 'A-n32-k5', or the name "
        "of an instance previously generated from a road network."
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    max_seconds: float = Field(default=10.0, gt=0.0, le=600.0)
    max_iterations: int = Field(default=1_000_000, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ReoptimizeRequest(BaseModel):
    """Body of ``POST /api/runs/{id}/reoptimize``.

    The point of this endpoint is the "dynamic re-optimisation" half of the
    problem statement: traffic changed, the previous plan is now priced wrongly,
    and we want a new plan quickly rather than a new plan from scratch. The
    previous incumbent is fed to the new search as its starting point.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    max_seconds: float = Field(default=5.0, gt=0.0, le=600.0)
    max_iterations: int = Field(default=1_000_000, ge=1)
    refresh_traffic: bool = Field(
        default=True,
        description="Rebuild the travel-time matrices from the current traffic "
        "state before re-optimising. Only meaningful for instances that were "
        "generated from a road network.",
    )
    warm_start: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class RunHandle(BaseModel):
    run_id: str


class RunTick(BaseModel):
    """One convergence sample, as delivered by the SSE stream."""

    iteration: int
    best_cost: float
    mean_cost: float
    diversity: float
    elapsed: float
    evaluations: int
    feasible: Optional[bool] = None
    routes: Optional[list[list[int]]] = None


class RunStatus(BaseModel):
    run_id: str
    state: RunState
    algorithm: str
    instance: str
    seed: int
    bks: Optional[float] = None
    best_cost: Optional[float] = None
    n_routes: Optional[int] = None
    feasible: Optional[bool] = None
    routes: Optional[list[list[int]]] = None
    stats: Optional[dict[str, float]] = None
    iterations: int = 0
    evaluations: int = 0
    seconds: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)
    geojson: Optional[dict[str, Any]] = None
    coords: Optional[list[list[float]]] = None
    error: Optional[str] = None
    history: list[RunTick] = Field(default_factory=list)
    # Extra context that the reference UI shows but the TypeScript contract
    # treats as optional.
    warm_started: bool = False
    parent_run_id: Optional[str] = None
    baseline_cost: Optional[float] = None


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------


class BenchmarkSummary(BaseModel):
    name: str
    n_instances: int
    n_algorithms: int
    n_runs: int
    max_seconds: float
    timestamp: Optional[float] = None


class StatSummary(BaseModel):
    n: int
    best: float
    mean: float
    median: float
    std: float
    iqr: float
    worst: float


class BenchmarkCell(BaseModel):
    instance: str
    algorithm: str
    cost: StatSummary
    gap: Optional[StatSummary] = None
    feasible_runs: int
    runs: int
    mean_seconds: float
    mean_iterations: float
    hit_bks: int
    median_time_to_1pct: Optional[float] = None


class BenchmarkRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance: str
    algorithm: str
    seed: int
    cost: Optional[float] = None
    gap: Optional[float] = None
    bks: Optional[float] = None
    n_routes: Optional[int] = None
    feasible: Optional[bool] = None
    iterations: Optional[int] = None
    evaluations: Optional[int] = None
    seconds: float = 0.0
    status: str = "ok"


class PairwiseResultModel(BaseModel):
    a: str
    b: str
    n: int
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    p_adjusted: Optional[float] = None
    effect_size: float = 0.0
    winner: Optional[str] = None
    text: str = ""


class FriedmanResultModel(BaseModel):
    algorithms: list[str]
    mean_ranks: dict[str, float]
    statistic: float
    p_value: float
    n_instances: int
    post_hoc: list[PairwiseResultModel] = Field(default_factory=list)
    control: Optional[str] = None


class BenchmarkDetail(BaseModel):
    name: str
    algorithms: list[str]
    instances: list[str]
    cells: dict[str, BenchmarkCell]
    rows: list[BenchmarkRow]
    curves: dict[str, list[list[float]]] = Field(
        default_factory=dict,
        description="Mean best-cost convergence curve per 'instance|algorithm' "
        "key, as [elapsed_seconds, best_cost] pairs.",
    )
    omnibus: Optional[FriedmanResultModel] = None
    n_ok: int = 0
    n_failed: int = 0
    max_seconds: float = 0.0
    environment: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """The only error shape the API emits. Never contains a traceback."""

    detail: str
    error_id: Optional[str] = None
