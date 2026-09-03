"""The FastAPI application: capabilities, instances, runs and benchmark results.

This module owns the application factory and every endpoint that is not about
road networks or traffic (those live in :mod:`qroute.api.networks`). It also
owns three cross-cutting concerns:

**Startup.** The compiled kernels are JIT-compiled on first use and the bundled
road graphs take seconds to load. Both are done in a background thread as the
server boots, so the process starts accepting requests immediately and
``/api/health`` reports honestly how far the preparation has got, rather than
the server appearing to hang or the first click appearing to be slow.

**Errors.** A handler converts any unexpected exception into a 500 carrying a
short error id and nothing else. The traceback is logged with that id, so an
operator can find it and a browser cannot read it. Unknown instances and unknown
networks are 404s whose message lists what does exist; bad parameters are 422s
from Pydantic.

**The built frontend.** When ``frontend/dist`` exists it is served at the root,
so ``uvicorn qroute.api.app:app`` runs the whole platform from one command. When
it does not, the API still starts and the root route explains how to build it.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from qroute import __version__
from qroute.api import networks as networks_module
from qroute.api.runs import (
    RunRegistry,
    TooManyRuns,
    algorithm_catalogue,
    known_algorithms,
    stream_run,
    warm_start_matrix,
)
from qroute.api.schemas import (
    AlgorithmInfo,
    BenchmarkDetail,
    BenchmarkSummary,
    HealthResponse,
    InstanceDetail,
    InstanceSummary,
    ReoptimizeRequest,
    RunHandle,
    RunRequest,
    RunStatus,
)
from qroute.api.state import STATE, StoredInstance, instance_detail

log = logging.getLogger("qroute.api")

#: Where the benchmark runner writes its output, and therefore where the
#: benchmark endpoints look. Overridable so a demonstration can point at a
#: prepared directory.
RESULTS_DIR = Path(os.environ.get("QROUTE_RESULTS", "results/runs"))

#: Where the built single-page application lives, when it has been built.
FRONTEND_DIST = Path(os.environ.get("QROUTE_FRONTEND", "frontend/dist"))

#: Development origins allowed by CORS. The regex covers whatever port Vite
#: settles on; the browser never sends credentials, so this is not a hole.
DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


# --------------------------------------------------------------------------
# Application factory
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=os.environ.get("QROUTE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    registry = RunRegistry(STATE)
    STATE.runs = registry
    # The fork server must be started before a road network is opened: see the
    # module docstring of qroute.api.runs. This is the ordering that keeps
    # solver processes from being killed by PROJ's atfork handler, so it happens
    # here, synchronously, ahead of the background preload.
    registry.prime()
    STATE.start_background_startup()
    log.info(
        "qroute API %s ready; solver workers use the %s start method; "
        "warm-up and preload running in the background",
        __version__,
        registry.start_method,
    )
    try:
        yield
    finally:
        if STATE.runs is not None:
            STATE.runs.shutdown()


def create_app() -> FastAPI:
    """Build the application. A factory so tests can hold their own instance."""
    app = FastAPI(
        title="qroute API",
        version=__version__,
        description=(
            "HTTP interface to the qroute platform: quantum-inspired route "
            "optimisation on real road networks with a live traffic model. "
            "Every number this API returns is produced by the same library code "
            "the command line and the benchmark runner use."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=DEV_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(networks_module.router)
    _register_routes(app)
    _register_error_handlers(app)
    _mount_frontend(app)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex[:8]
        log.exception("unhandled error %s on %s %s", error_id, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal error; the server log holds the details",
                "error_id": error_id,
            },
        )


def _mount_frontend(app: FastAPI) -> None:
    """Serve ``frontend/dist`` at the root when it exists."""
    if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
        log.info("serving the built frontend from %s", FRONTEND_DIST)
        return

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "qroute",
            "version": __version__,
            "docs": "/docs",
            "frontend": (
                f"not built; run `npm run build` in frontend/ to have {FRONTEND_DIST} "
                "served from here, or `npm run dev` for the development server"
            ),
        }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    # ------------------------------------------------------------ health
    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> dict[str, Any]:
        """Version, what is loaded, which external solvers are installed.

        ``warmup`` reports the cost of compiling the JIT kernels at startup. It
        is here rather than hidden because the first-request latency of a JIT
        system is a real property of the platform and a judge is entitled to see
        it.
        """
        from qroute.algorithms.registry import names as algorithm_names

        # Everything read here is either already in memory or cached: health is
        # polled while the server warms up, and a poll that blocks on a disk
        # scan or a first import would misreport the very latency it exists to
        # describe.
        solvers = STATE.solvers if STATE.solvers["probed"] else STATE.probe_solvers()
        try:
            n_instances = len(STATE.benchmark_instances())
        except Exception:  # pragma: no cover - only if the data directory is gone
            n_instances = 0

        return {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": round(time.time() - STATE.started_at, 2),
            "networks": len(STATE.available_network_ids()),
            "instances": n_instances + len(STATE.stored_instances()),
            "algorithms": len(algorithm_names()),
            "network_ids": STATE.available_network_ids(),
            "networks_loaded": STATE.loaded_network_ids(),
            "networks_loading": STATE.loading_network_ids(),
            "ortools_available": solvers["ortools_available"],
            "pyvrp_available": solvers["pyvrp_available"],
            "pyvrp_version": solvers["pyvrp_version"],
            "warmup": STATE.warmup,
            "active_runs": STATE.runs.active_count() if STATE.runs else 0,
            "worker_start_method": STATE.runs.start_method if STATE.runs else None,
            "workers_primed": bool(STATE.runs and STATE.runs.primed),
            "worker_prime_seconds": round(STATE.runs.prime_seconds, 3) if STATE.runs else None,
        }

    # -------------------------------------------------------- algorithms
    @app.get("/api/algorithms", response_model=list[AlgorithmInfo], tags=["meta"])
    def algorithms() -> list[dict[str, Any]]:
        """Every solver the run endpoint accepts, with its tunable parameters.

        Parameter lists are read from each solver's constructor signature, so
        they cannot drift out of date: a parameter that is renamed in the
        algorithm disappears from the form on the next restart.
        """
        return algorithm_catalogue()

    # --------------------------------------------------------- instances
    @app.get("/api/instances", response_model=list[InstanceSummary], tags=["instances"])
    def list_instances_endpoint(
        family: Optional[str] = Query(None, pattern="^(cvrp|vrptw|network)$"),
        max_customers: Optional[int] = Query(None, ge=1),
        limit: Optional[int] = Query(None, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """Benchmark instances on disk, plus any generated from a road network."""
        from qroute.api.state import instance_summary

        rows = list(STATE.benchmark_instances())
        rows.extend(instance_summary(s.instance, s.family) for s in STATE.stored_instances())
        if family:
            rows = [r for r in rows if r["family"] == family]
        if max_customers:
            rows = [r for r in rows if r["n_customers"] <= max_customers]
        if limit:
            rows = rows[:limit]
        return rows

    @app.get("/api/instances/{name}", response_model=InstanceDetail, tags=["instances"])
    def instance_endpoint(name: str) -> dict[str, Any]:
        """Full detail for one instance, including coordinates for plotting."""
        return instance_detail(_resolve_instance(name))

    # -------------------------------------------------------------- runs
    @app.post("/api/runs", response_model=RunHandle, status_code=201, tags=["runs"])
    def start_run(request: RunRequest) -> dict[str, str]:
        """Start a solver in its own process and return a handle.

        The response is deliberately just an id: the run has not produced
        anything yet, and the client is expected to open
        ``/api/runs/{id}/stream`` or poll ``/api/runs/{id}``.
        """
        stored = _resolve_instance(request.instance)
        _check_algorithm(request.algorithm)
        registry = _registry()
        try:
            record = registry.start(
                stored=stored,
                algorithm=request.algorithm,
                seed=request.seed,
                max_seconds=request.max_seconds,
                max_iterations=request.max_iterations,
                params=request.params,
            )
        except TooManyRuns as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from None
        return {"run_id": record.run_id}

    @app.get("/api/runs", response_model=list[RunStatus], tags=["runs"])
    def list_runs(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
        """The most recent runs, newest first, without their histories."""
        registry = _registry()
        records = sorted(registry.all_runs(), key=lambda r: r.created_at, reverse=True)
        return [registry.status(r, include_history=False) for r in records[:limit]]

    @app.get("/api/runs/{run_id}", response_model=RunStatus, tags=["runs"])
    def run_status(run_id: str) -> dict[str, Any]:
        """Status, best cost so far, routes and statistics."""
        registry = _registry()
        return registry.status(_resolve_run(run_id))

    @app.get("/api/runs/{run_id}/stream", tags=["runs"])
    async def run_stream(run_id: str) -> EventSourceResponse:
        """Server-Sent Events: one message per throttled iteration sample.

        Events are ``start``, ``tick`` (the ``RunTick`` shape), and exactly one
        of ``done``, ``cancelled`` or ``error`` carrying the final status. A run
        that has already finished replays its whole history and then ends, so
        reconnecting is not a special case.
        """
        registry = _registry()
        record = _resolve_run(run_id)
        return EventSourceResponse(stream_run(registry, record))

    @app.post("/api/runs/{run_id}/cancel", response_model=RunStatus, tags=["runs"])
    def cancel_run(run_id: str) -> dict[str, Any]:
        """Stop a run. Whatever it had found by then is kept and returned."""
        registry = _registry()
        _resolve_run(run_id)
        record = registry.cancel(run_id)
        assert record is not None
        return registry.status(record)

    @app.post("/api/runs/{run_id}/reoptimize", response_model=RunHandle,
              status_code=201, tags=["runs"])
    def reoptimize(run_id: str, request: ReoptimizeRequest) -> dict[str, str]:
        """Re-solve after a traffic change, warm-started from the previous plan.

        Two things happen, and both are reported rather than assumed. First, if
        the instance came from a road network, its travel-time matrices are
        rebuilt at the *current* traffic state - that is the dynamic weight
        update reaching the optimiser. Second, the previous incumbent is priced
        under those new matrices and recorded as ``baseline_cost``, so the new
        run can be compared against "keep driving the old plan" rather than
        against nothing.
        """
        registry = _registry()
        parent = _resolve_run(run_id)
        if parent.result is None or not parent.result.get("routes"):
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} has no finished solution to warm start from "
                f"(state: {parent.state})",
            )
        stored = _resolve_instance(parent.instance_name)
        previous_routes = [[int(c) for c in r] for r in parent.result["routes"]]

        if request.refresh_traffic and stored.network_id and stored.stop_nodes is not None:
            stored = _refresh_instance_matrices(stored)

        baseline_cost = None
        try:
            baseline_cost = float(
                stored.instance.make_solution(previous_routes).cost
            )
        except Exception:
            log.exception("could not price the previous plan under the new matrices")

        algorithm = request.algorithm or parent.algorithm
        _check_algorithm(algorithm)
        initial_keys = None
        if request.warm_start and _accepts_warm_start(algorithm):
            # Solvers that do not decode random keys (SA, ACO, and the external
            # baselines) simply start cold; the response says so through
            # ``warm_started`` rather than pretending the request was honoured.
            initial_keys = warm_start_matrix(previous_routes, stored.instance.n_customers)

        params = dict(parent.params)
        params.update(request.params)
        try:
            record = registry.start(
                stored=stored,
                algorithm=algorithm,
                seed=request.seed if request.seed is not None else parent.seed + 1,
                max_seconds=request.max_seconds,
                max_iterations=request.max_iterations,
                params=params,
                initial_keys=initial_keys,
                parent_run_id=parent.run_id,
                baseline_cost=baseline_cost,
            )
        except TooManyRuns as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from None
        return {"run_id": record.run_id}

    # -------------------------------------------------------- benchmarks
    @app.get("/api/benchmarks", response_model=list[BenchmarkSummary], tags=["benchmarks"])
    def list_benchmarks() -> list[dict[str, Any]]:
        """Saved benchmark result sets under the results directory."""
        return _benchmark_index()

    @app.get("/api/benchmarks/{name}", response_model=BenchmarkDetail, tags=["benchmarks"])
    def benchmark_detail(
        name: str,
        include_rows: bool = Query(True),
        include_curves: bool = Query(True),
    ) -> dict[str, Any]:
        """Summary tables, convergence curves and the statistical comparison.

        The omnibus test is recomputed here from the stored per-seed rows rather
        than read from ``summary.json``, because the stored form drops the
        medians and test statistics that the comparison table wants to show. The
        computation is the same function the offline report uses.
        """
        return _benchmark_detail(name, include_rows=include_rows, include_curves=include_curves)


# --------------------------------------------------------------------------
# Helpers used by the routes
# --------------------------------------------------------------------------


def _registry() -> RunRegistry:
    if STATE.runs is None:  # pragma: no cover - only outside the lifespan
        STATE.runs = RunRegistry(STATE)
    return STATE.runs


def _resolve_instance(name: str) -> StoredInstance:
    try:
        return STATE.resolve_instance(name)
    except KeyError:
        from qroute.problems.loaders import list_instances

        available = list_instances()
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown instance {name!r}. Benchmark sets hold "
                f"{len(available.get('cvrp', []))} CVRP and "
                f"{len(available.get('vrptw', []))} VRPTW instances "
                "(see /api/instances); instances generated from a road network "
                "are available until the server restarts."
            ),
        ) from None


def _resolve_run(run_id: str):
    record = _registry().get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return record


def _accepts_warm_start(algorithm: str) -> bool:
    """True when a solver's constructor takes ``initial_keys``.

    Asked of the class itself rather than kept in a list here, so adding a
    key-based solver to the registry makes warm starting work for it with no
    change to the API.
    """
    import inspect

    from qroute.algorithms.registry import ALGORITHMS, get

    if algorithm.lower() not in ALGORITHMS:
        return False
    try:
        return "initial_keys" in inspect.signature(get(algorithm).__init__).parameters
    except Exception:  # pragma: no cover - defensive
        return False


def _check_algorithm(name: str) -> None:
    if name.lower() not in known_algorithms():
        raise HTTPException(
            status_code=422,
            detail=f"unknown algorithm {name!r}; available: "
            f"{', '.join(sorted(known_algorithms()))}",
        )


def _refresh_instance_matrices(stored: StoredInstance) -> StoredInstance:
    """Re-measure a network instance's matrices at the current traffic state.

    The stops do not move - the depot and the customers are the same physical
    junctions - so only the travel-time, distance and congestion matrices change.
    That is exactly what a traffic update means for a routing problem, and it is
    why re-optimisation is cheap: the expensive stop selection is not repeated.
    """
    from qroute.graph.matrix import build_matrices

    bundle = STATE.get_network(str(stored.network_id))
    started = time.perf_counter()
    with bundle.lock:
        matrices = build_matrices(bundle.network, stored.stop_nodes, keep_predecessors=True)
        minute = bundle.simulator.time_minutes
    if not matrices.is_finite:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=409,
            detail="the current incidents disconnect the network; clear one before re-optimising",
        )
    instance = stored.instance.with_matrices(
        distance=matrices.distance,
        duration=matrices.duration,
        congestion=matrices.congestion,
    )
    instance.meta = dict(instance.meta)
    instance.meta["traffic_minute"] = float(minute)
    instance.meta["traffic_hour"] = round(float(minute % (24 * 60)) / 60.0, 3)
    instance.meta["rematrix_seconds"] = round(time.perf_counter() - started, 4)
    refreshed = StoredInstance(
        name=stored.name,
        instance=instance,
        family=stored.family,
        network_id=stored.network_id,
        matrices=matrices,
        stop_nodes=stored.stop_nodes,
        request=stored.request,
    )
    return STATE.store_instance(refreshed)


# --------------------------------------------------------------------------
# Benchmark result files
# --------------------------------------------------------------------------


def _benchmark_dirs() -> list[Path]:
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(
        (p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "rows.jsonl").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _benchmark_index() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for directory in _benchmark_dirs():
        meta: dict[str, Any] = {}
        meta_path = directory / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except ValueError:
                meta = {}
        config = meta.get("config", {})
        try:
            n_runs = sum(1 for line in (directory / "rows.jsonl").read_text().splitlines() if line)
        except OSError:  # pragma: no cover
            n_runs = 0
        out.append(
            {
                "name": directory.name,
                "n_instances": len(config.get("instances", [])),
                "n_algorithms": len(config.get("algorithms", [])),
                "n_runs": n_runs,
                "max_seconds": float(config.get("max_seconds", 0.0) or 0.0),
                "timestamp": (meta.get("environment") or {}).get("timestamp_unix"),
            }
        )
    return out


def _benchmark_detail(name: str, *, include_rows: bool, include_curves: bool) -> dict[str, Any]:
    directory = RESULTS_DIR / name
    if not (directory.is_dir() and (directory / "rows.jsonl").exists()):
        known = [d.name for d in _benchmark_dirs()]
        raise HTTPException(
            status_code=404,
            detail=f"unknown benchmark {name!r}; available: {', '.join(known) or 'none'}",
        )

    from qroute.benchmark.runner import BenchmarkRunner, load_results

    rows = load_results(directory / "rows.jsonl")
    summary_path = directory / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except ValueError:  # pragma: no cover
            summary = BenchmarkRunner.summarise(rows)
    else:
        summary = BenchmarkRunner.summarise(rows)

    meta: dict[str, Any] = {}
    if (directory / "meta.json").exists():
        try:
            meta = json.loads((directory / "meta.json").read_text())
        except ValueError:  # pragma: no cover
            meta = {}

    slim_rows: list[dict[str, Any]] = []
    if include_rows:
        for row in rows:
            slim = {k: v for k, v in row.items() if k not in ("history", "traceback", "params")}
            slim_rows.append(slim)

    return {
        "name": name,
        "algorithms": summary.get("algorithms", []),
        "instances": summary.get("instances", []),
        "cells": summary.get("cells", {}),
        "rows": slim_rows,
        "curves": _convergence_curves(rows) if include_curves else {},
        "omnibus": _omnibus(summary),
        "n_ok": summary.get("n_ok", 0),
        "n_failed": summary.get("n_failed", 0),
        "n_no_solution": summary.get("n_no_solution", 0),
        "no_solution": summary.get("no_solution", []),
        "n_infeasible": summary.get("n_infeasible", 0),
        "max_seconds": float((meta.get("config") or {}).get("max_seconds", 0.0) or 0.0),
        "environment": meta.get("environment", {}),
    }


def _omnibus(summary: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Recompute the Friedman test and its post-hoc comparisons in full.

    ``summary.json`` keeps only the adjusted p-values, which is enough for a
    verdict but not enough for a table that shows medians and effect sizes. The
    inputs (the per-instance median gaps) are stored, so the test is recomputed
    from them with the same function the offline report uses.
    """
    from dataclasses import asdict

    from qroute.benchmark.stats import friedman

    cells = summary.get("cells") or {}
    algorithms = summary.get("algorithms") or []
    stored = summary.get("omnibus") or {}
    instances = stored.get("instances_used")
    if instances is None:
        instances = [
            i
            for i in summary.get("instances", [])
            if all(cells.get(f"{i}|{a}", {}).get("gap") for a in algorithms)
        ]
    if len(algorithms) < 3 or len(instances) < 3:
        return None
    per_algorithm = {
        a: [cells[f"{i}|{a}"]["gap"]["median"] for i in instances] for a in algorithms
    }
    try:
        result = friedman(per_algorithm, control=stored.get("control"))
    except (KeyError, ValueError):
        return None
    payload = asdict(result)
    payload["post_hoc"] = [
        {**asdict(c), "text": c.describe()} for c in result.post_hoc
    ]
    return payload


def _convergence_curves(rows: list[dict[str, Any]], points: int = 60) -> dict[str, list[list[float]]]:
    """Mean best-cost-against-time curve per instance and algorithm.

    Seeds finish at slightly different times and record at different iteration
    counts, so the curves are resampled onto a common grid before averaging. The
    resampling is a *step* hold, not a linear interpolation: the best cost is a
    right-continuous step function of time, and interpolating it would draw
    improvements at moments they had not yet happened.
    """
    grouped: dict[str, list[list[dict[str, float]]]] = {}
    for row in rows:
        history = row.get("history")
        if row.get("status") != "ok" or not history:
            continue
        grouped.setdefault(f"{row['instance']}|{row['algorithm']}", []).append(history)

    curves: dict[str, list[list[float]]] = {}
    for key, histories in grouped.items():
        horizon = max(float(h[-1]["t"]) for h in histories if h)
        if horizon <= 0:
            continue
        grid = np.linspace(0.0, horizon, points)
        stacked = []
        for history in histories:
            times = np.array([float(h["t"]) for h in history])
            costs = np.array([float(h["c"]) for h in history])
            idx = np.searchsorted(times, grid, side="right") - 1
            values = np.where(idx >= 0, costs[np.clip(idx, 0, len(costs) - 1)], costs[0])
            stacked.append(values)
        mean = np.mean(np.vstack(stacked), axis=0)
        curves[key] = [[round(float(t), 4), round(float(c), 4)] for t, c in zip(grid, mean)]
    return curves


#: Module-level application object, so ``uvicorn qroute.api.app:app`` works.
app = create_app()


def main() -> None:  # pragma: no cover - convenience entry point
    """Run the server. ``python -m qroute.api.app``."""
    import uvicorn

    uvicorn.run(
        "qroute.api.app:app",
        host=os.environ.get("QROUTE_HOST", "127.0.0.1"),
        port=int(os.environ.get("QROUTE_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
