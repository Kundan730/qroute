"""Running solvers out of process and streaming their convergence to the browser.

Why a separate process
----------------------
A solver is a tight numeric loop that holds the GIL for seconds at a time. Run
inside the web server it would freeze every other request, including the one
asking it to stop. Each run therefore gets its own operating-system process:
the server stays responsive, a run can be cancelled by terminating it, and a
solver that segfaults in a compiled kernel takes only itself down.

Everything handed to a worker is plain data or a picklable object, and the
worker entry point is a module-level function: on macOS a child does not inherit
the parent's memory, it re-imports the package and unpickles its arguments, so a
closure or a bound method would not survive the trip.

Why ``forkserver`` and not ``spawn``
-----------------------------------
The obvious choice on macOS is ``spawn``. It crashes here, and the reason is
worth recording because it is invisible from Python.

Loading a road network imports OSMnx, which imports pyproj, which loads the
PROJ shared library. PROJ registers a ``pthread_atfork`` *child* handler that
tears down its SQLite connection cache. ``spawn`` is implemented as fork
followed by exec, so that handler runs in the forked child, in the window
between fork and exec where only async-signal-safe calls are legal. It closes
SQLite handles and logs while doing it, and the child dies with SIGSEGV before
it ever reaches ``exec``. The parent sees only "exit code -11". Confirmed from
the macOS crash report: the faulting stack is
``_pthread_atfork_child_handlers`` -> ``SQLiteHandleCache`` -> ``sqlite3Close``
-> ``os_log``.

``forkserver`` avoids it. A small server process is started once, and every
worker is forked from *that* process rather than from the web server. As long as
the fork server is started before any road network is loaded - which
:meth:`RunRegistry.prime` guarantees by starting it during application startup -
it never has PROJ mapped, so there is no atfork handler to run and nothing to
crash.

The fork server also gets a warm-up for free. Its preload list is a set of
module names it imports once, and every worker is a fork of that process, so
importing the solver modules and materialising the compiled kernels there is
paid once at startup rather than in each worker. Measured on a 35-stop road
instance: the first iteration reached the browser after 3.64 s without it and
after 0.18 s with it, which on a five-second run is the difference between four
iterations and twenty-one.

A separate hazard, shared by both start methods, is that the child re-executes
the parent's ``__main__`` before it runs anything of ours. That is harmless
under a guarded entry point and fatal under a heredoc or a REPL.
:func:`_neutral_main_module` suppresses it, so how the server was launched stops
mattering.

How progress gets back
----------------------
Every optimiser accepts ``callback=fn(IterationRecord)``. The worker installs a
callback that pushes a small dictionary onto a ``multiprocessing.Queue`` created
from the spawn context, one queue per run. On the server side one daemon thread
per run drains that queue into the run record. Endpoints and the SSE generator
only ever read the record, so they never block on the queue.

A ``Manager().Queue()`` would be the textbook choice here, and was the first
implementation, but it does not survive every way this server is started: the
manager child re-imports the parent's main module, which fails under
``python -m uvicorn``. See :meth:`RunRegistry._queue` for the measurement and for
how the robustness a manager would have given is recovered without one.

Throttling
----------
An interior-point iteration of QPSO on a 100-customer instance takes a few
milliseconds, so an unthrottled stream would emit hundreds of events a second -
far more than a browser can draw and enough to make the SSE connection itself
the bottleneck. The worker therefore emits at most :data:`MAX_EVENTS_PER_SECOND`
ticks, always including the first and the last, and attaches the full route
list only when the incumbent actually improved. The *complete* history is still
recorded by the optimiser and returned when the run finishes, so throttling
costs the live view some frames but costs the reported convergence curve
nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from qroute.api.state import STATE, ApiState, StoredInstance

log = logging.getLogger("qroute.api.runs")

#: Upper bound on live SSE ticks per second, per run.
MAX_EVENTS_PER_SECOND: float = 10.0

#: How many runs may execute at once. Each pins one core; the machine also has
#: to serve the API and, during a demonstration, a browser.
MAX_ACTIVE_RUNS: int = 4

#: Algorithms that are not :class:`~qroute.algorithms.base.Optimizer` subclasses
#: and therefore cannot report per-iteration progress. They still run, and still
#: return a full result; the stream simply carries no ticks for them.
NON_STREAMING = {"ortools", "ortools_gls", "pyvrp", "hgs", "cpsat", "exact"}

#: Start method for solver processes, in order of preference. See the module
#: docstring for why ``forkserver`` comes first: ``spawn`` segfaults on macOS
#: once PROJ has been loaded, which happens as soon as a road network is opened.
#: ``spawn`` remains the fallback for platforms without a fork server.
START_METHODS: tuple[str, ...] = ("forkserver", "spawn")

#: Modules the fork server imports once, so every worker starts with them ready.
#: ``__main__`` is deliberately absent: including it is what makes ``spawn``
#: re-execute the parent's entry point in the child. Importing the solver
#: modules here costs 2.2 seconds once, in the fork server, instead of 2.2
#: seconds in every worker.
FORKSERVER_PRELOAD: list[str] = [
    "qroute.api.runs",
    "qroute.algorithms.qpso",
    "qroute.algorithms.pso",
    "qroute.algorithms.ga",
    "qroute.algorithms.sa",
    "qroute.algorithms.aco",
]

#: Set in the fork server's environment, and read at the bottom of this module.
#: It is how a warm-up gets executed inside the fork server, whose only hook is
#: a list of module names to import.
WORKER_PRELOAD_ENV: str = "QROUTE_WORKER_PRELOAD"


# --------------------------------------------------------------------------
# Algorithm catalogue
# --------------------------------------------------------------------------

#: Bounds, steps and one-line explanations for the parameters the UI exposes.
#: Only presentation metadata lives here - the defaults and the set of
#: parameters themselves are read from each solver's signature, so this table
#: can never claim a parameter that does not exist.
_PARAM_META: dict[str, dict[str, Any]] = {
    "swarm_size": {"min": 5, "max": 200, "step": 1, "description": "Number of particles."},
    "population": {"min": 10, "max": 300, "step": 1, "description": "Number of individuals."},
    "n_ants": {"min": 2, "max": 100, "step": 1, "description": "Ants released per iteration."},
    "beta_start": {"min": 0.2, "max": 1.8, "step": 0.05,
                   "description": "Initial contraction-expansion coefficient."},
    "beta_end": {"min": 0.1, "max": 1.8, "step": 0.05,
                 "description": "Final contraction-expansion coefficient."},
    "beta_schedule": {"choices": ["linear", "exponential", "fixed"],
                      "description": "How beta moves from start to end."},
    "weighted_mbest": {"description": "Weight the mean best position by fitness rank."},
    "mutation": {"choices": ["none", "gaussian", "cauchy"],
                 "description": "Perturbation applied to the random keys."},
    "mutation_rate": {"min": 0.0, "max": 1.0, "step": 0.01},
    "mutation_scale": {"min": 0.0, "max": 1.0, "step": 0.01},
    "elite_fraction": {"min": 0.0, "max": 0.5, "step": 0.01,
                       "description": "Share of the population protected from restarts."},
    "restart_after": {"min": 0, "max": 500, "step": 1,
                      "description": "Stagnant iterations before a partial restart; 0 disables."},
    "restart_fraction": {"min": 0.0, "max": 1.0, "step": 0.05},
    "local_search": {"description": "Refine decoded routes with 2-opt / Or-opt."},
    "ls_policy": {"choices": ["all", "sample"],
                  "description": "Refine every particle, or a sample plus the incumbent."},
    "ls_fraction": {"min": 0.0, "max": 1.0, "step": 0.05},
    "neighbours": {"min": 5, "max": 40, "step": 1,
                   "description": "Granular neighbourhood size for local search."},
    "local_search_rounds": {"min": 1, "max": 200, "step": 1},
    "inertia": {"choices": ["constriction", "linear", "fixed"]},
    "topology": {"choices": ["ring", "gbest"]},
    "crossover": {"choices": ["ox", "blend", "uniform", "mixed"]},
    "w": {"min": 0.0, "max": 1.5, "step": 0.01},
    "c1": {"min": 0.0, "max": 4.0, "step": 0.01},
    "c2": {"min": 0.0, "max": 4.0, "step": 0.01},
    "alpha": {"min": 0.0, "max": 5.0, "step": 0.001},
    "beta": {"min": 0.0, "max": 10.0, "step": 0.1},
    "q0": {"min": 0.0, "max": 1.0, "step": 0.01},
    "rho": {"min": 0.0, "max": 1.0, "step": 0.01},
    "xi": {"min": 0.0, "max": 1.0, "step": 0.01},
    "tournament_size": {"min": 2, "max": 10, "step": 1},
    "penalty_capacity": {"min": 0.0, "max": 1e6, "step": 10.0},
    "penalty_time_window": {"min": 0.0, "max": 1e6, "step": 10.0},
    "penalty_duration": {"min": 0.0, "max": 1e6, "step": 10.0},
    "vehicle_cost": {"min": 0.0, "max": 1e6, "step": 1.0,
                     "description": "Fixed cost charged per route used."},
    "seconds": {"min": 0.1, "max": 600.0, "step": 0.5},
}

#: Constructor arguments that are plumbing rather than tuning, and must never be
#: offered to the browser: they are objects, not values.
_HIDDEN_PARAMS = {"self", "instance", "stop", "seed", "callback", "decoder", "initial_keys", "kw"}

#: Solvers that are reached through the benchmark runner's dispatcher rather
#: than through the algorithm registry.
_EXTRA_ALGORITHMS: list[tuple[str, str, str]] = [
    ("random", "Multi-start local search: the control the search rules are judged against",
     "reference"),
    ("ortools", "OR-Tools routing with guided local search (industrial baseline)", "baseline"),
    ("pyvrp", "PyVRP hybrid genetic search (state of the art for CVRP/VRPTW)", "baseline"),
    ("cpsat", "CP-SAT exact model; proves optimality on small instances", "exact"),
]


def _param_specs(cls: type) -> list[dict[str, Any]]:
    """Derive the tunable-parameter list of a solver from its signature."""
    specs: list[dict[str, Any]] = []
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return specs
    for name, param in signature.parameters.items():
        if name in _HIDDEN_PARAMS or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = param.default
        if default is inspect.Parameter.empty:
            continue
        meta = dict(_PARAM_META.get(name, {}))
        if isinstance(default, bool):
            kind = "bool"
        elif isinstance(default, int):
            kind = "int"
        elif isinstance(default, float):
            kind = "float"
        elif isinstance(default, str):
            kind = "choice" if meta.get("choices") else "text"
        else:
            # Tuples and anything else are not representable in a simple form;
            # showing a control that cannot set them would be worse than
            # omitting them.
            continue
        specs.append(
            {
                "name": name,
                "kind": kind,
                "default": default,
                "min": meta.get("min"),
                "max": meta.get("max"),
                "step": meta.get("step"),
                "choices": meta.get("choices"),
                "description": meta.get("description"),
            }
        )
    return specs


def algorithm_catalogue() -> list[dict[str, Any]]:
    """Every solver the run endpoint accepts, with its parameter schema.

    Resolving a registry name imports its module, which pulls in the compiled
    kernels. That is the same import the warm-up already paid for, so listing
    the catalogue is cheap after startup and merely slow-ish before it.
    """
    from qroute.algorithms.registry import DESCRIPTIONS, get, names

    out: list[dict[str, Any]] = []
    for name in names():
        try:
            cls = get(name)
            params = _param_specs(cls)
            warm = "initial_keys" in inspect.signature(cls.__init__).parameters
        except Exception:  # pragma: no cover - a broken solver must still list
            log.exception("could not introspect algorithm %s", name)
            params, warm = [], False
        out.append(
            {
                "name": name,
                "description": DESCRIPTIONS.get(name, ""),
                "kind": "metaheuristic",
                "supports_warm_start": warm,
                "params": params,
            }
        )

    for name, description, kind in _EXTRA_ALGORITHMS:
        entry: dict[str, Any] = {
            "name": name,
            "description": description,
            "kind": kind,
            "supports_warm_start": False,
            "params": [],
        }
        if name == "random":
            try:
                from qroute.benchmark.reference import RandomRestart

                entry["params"] = _param_specs(RandomRestart)
            except Exception:  # pragma: no cover
                log.exception("could not introspect the random-restart control")
        if name == "pyvrp":
            from qroute.baselines import pyvrp_hgs

            entry["available"] = pyvrp_hgs.available()
        out.append(entry)
    return out


def known_algorithms() -> set[str]:
    from qroute.algorithms.registry import names

    return set(names()) | {n for n, _d, _k in _EXTRA_ALGORITHMS} | NON_STREAMING | {"restart"}


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def _worker_context():
    """The multiprocessing context solver processes are launched from.

    Prefers a fork server for the reasons in the module docstring, and falls
    back to ``spawn`` on any platform that does not offer one.
    """
    available = mp.get_all_start_methods()
    for method in START_METHODS:
        if method not in available:
            continue
        context = mp.get_context(method)
        if method == "forkserver":
            try:
                context.set_forkserver_preload(FORKSERVER_PRELOAD)
                # The fork server is exec'd as a fresh interpreter and inherits
                # this environment, so the flag reaches it and nothing else: the
                # web server process has already imported this module, and the
                # workers are forks of the fork server rather than new imports.
                os.environ[WORKER_PRELOAD_ENV] = "1"
            except Exception:  # pragma: no cover - defensive
                log.exception("could not set the fork server preload list")
        return context
    return mp.get_context()  # pragma: no cover - no supported method


#: Serialises the brief window in which ``__main__`` is disguised, below.
_START_LOCK = threading.Lock()


@contextlib.contextmanager
def _neutral_main_module():
    """Stop a worker from re-executing the parent's entry point.

    Both ``spawn`` and ``forkserver`` run ``multiprocessing.spawn.prepare`` in
    the child, which re-imports or re-executes whatever the parent's ``__main__``
    is. That is fine when the parent was started from a guarded script, and fatal
    otherwise: launched from a heredoc the child dies with
    ``FileNotFoundError: .../<stdin>``, and launched from a module with side
    effects it would run them a second time.

    ``multiprocessing`` skips the fixup entirely when the main module's spec is
    named ``__main__``, so for the moment it takes to hand the child its
    preparation data we present a spec that says exactly that. The child then
    starts with a bare ``__main__``, which is all a worker needs: everything it
    runs is imported from :mod:`qroute`, and nothing in the payload refers to a
    class defined in the parent's entry point.

    The disguise is process-wide while it lasts, so it is held under a lock and
    restored in a ``finally``.
    """
    main = sys.modules.get("__main__")
    if main is None:  # pragma: no cover - no interpreter has no __main__
        yield
        return

    class _MainSpec:
        name = "__main__"

    with _START_LOCK:
        had_spec = hasattr(main, "__spec__")
        previous = getattr(main, "__spec__", None)
        try:
            main.__spec__ = _MainSpec()
            yield
        finally:
            if had_spec:
                main.__spec__ = previous
            else:  # pragma: no cover - every module normally has __spec__
                del main.__spec__


def _noop() -> None:
    """Body of the priming child. Exists only to force the fork server up."""


def warm_kernels() -> float:
    """Compile and load the JIT kernels by solving a tiny instance.

    Called at the bottom of this module when it is imported by the fork server.
    Every worker is then a fork of a process in which the compiled kernels are
    already resident, and its first iteration lands in milliseconds instead of
    after four seconds - measured: 2.2 s to import the solver modules plus 1.9 s
    to materialise the cached kernels, which on a five-second run would have
    been most of the budget.

    Returns the seconds it took. Failures are logged and swallowed: a worker
    that has to compile its own kernels is slow, not broken.
    """
    started = time.perf_counter()
    try:
        from qroute.algorithms.base import StopCriteria
        from qroute.algorithms.qpso import QPSO
        from qroute.core.rng import make_rng
        from qroute.problems.instance import Instance

        rng = make_rng(0)
        coords = rng.uniform(0.0, 100.0, size=(9, 2))
        diff = coords[:, None, :] - coords[None, :, :]
        demand = np.zeros(9)
        demand[1:] = 5.0
        instance = Instance(
            name="worker-warmup",
            distance=np.sqrt((diff**2).sum(-1)),
            demand=demand,
            capacity=20.0,
        )
        QPSO(instance, StopCriteria(max_iterations=2, max_seconds=60.0),
             seed=0, swarm_size=6).solve()
    except Exception:  # pragma: no cover - defensive
        log.exception("worker kernel warm-up failed")
    return time.perf_counter() - started


def _emit(progress_queue, message: dict[str, Any]) -> None:
    """Push one message, tolerating a queue whose other end has gone away."""
    try:
        progress_queue.put(message)
    except Exception:  # pragma: no cover - only on an already-dead manager
        pass


def solver_worker(payload: dict[str, Any], progress_queue) -> None:
    """Entry point of a solver process. Must stay importable and picklable.

    ``payload`` carries the instance itself (a small object of NumPy arrays),
    the algorithm name, its parameters, the seed and the stopping rule. The
    function never returns a value; everything travels back over the queue.
    """
    # One thread per solver, so a wall-clock budget means the same thing here as
    # it does in the benchmark runner.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    started = time.perf_counter()
    try:
        from qroute.algorithms.base import StopCriteria

        instance = payload["instance"]
        algorithm = str(payload["algorithm"]).lower()
        params = dict(payload.get("params") or {})
        seed = int(payload["seed"])
        stop = StopCriteria(
            max_iterations=int(payload.get("max_iterations", 1_000_000)),
            max_seconds=float(payload.get("max_seconds", 10.0)),
        )
        initial_keys = payload.get("initial_keys")

        interval = 1.0 / MAX_EVENTS_PER_SECOND
        holder: dict[str, Any] = {"optimizer": None, "last_emit": 0.0, "last_routes_cost": None}

        def callback(record) -> None:
            now = time.perf_counter()
            due = (record.iteration <= 1) or (now - holder["last_emit"] >= interval)
            optimizer = holder["optimizer"]
            # The incumbent is the optimiser's own ``_best``; reading it is the
            # only way to attach geometry to a tick, because IterationRecord
            # carries costs but no routes.
            best = getattr(optimizer, "_best", None) if optimizer is not None else None
            improved = (
                best is not None
                and best.routes
                and (
                    holder["last_routes_cost"] is None
                    or best.cost < holder["last_routes_cost"] - 1e-9
                )
            )
            if not due and not improved:
                return
            holder["last_emit"] = now
            message: dict[str, Any] = {
                "type": "tick",
                "iteration": int(record.iteration),
                "elapsed": float(record.elapsed),
                "evaluations": int(record.evaluations),
                "best_cost": float(record.best_cost),
                "mean_cost": float(record.mean_cost),
                "diversity": float(record.diversity),
                "feasible": bool(record.feasible),
            }
            if improved:
                message["routes"] = [[int(c) for c in r] for r in best.routes]
                holder["last_routes_cost"] = float(best.cost)
            _emit(progress_queue, message)

        result = _build_and_solve(
            algorithm, instance, stop, seed, params, callback, holder, initial_keys
        )
        payload_out = result.to_json()
        payload_out["bks"] = instance.meta.get("bks")
        _emit(progress_queue, {"type": "done", "result": payload_out})
    except Exception as exc:
        import traceback

        _emit(
            progress_queue,
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
                "seconds": time.perf_counter() - started,
            },
        )


def _build_and_solve(algorithm, instance, stop, seed, params, callback, holder, initial_keys):
    """Dispatch a name to a solver call returning an ``OptimizationResult``.

    Mirrors :func:`qroute.benchmark.runner._dispatch` so that a run started from
    the web UI and the same run started from the benchmark runner execute
    identical code paths, which is the only way the live demonstration and the
    reported benchmark numbers can be claimed to measure the same thing.
    """
    if algorithm in ("ortools", "ortools_gls"):
        from qroute.baselines.ortools_gls import solve_ortools

        return solve_ortools(instance, seconds=stop.max_seconds, **params)
    if algorithm in ("pyvrp", "hgs"):
        from qroute.baselines.pyvrp_hgs import solve_pyvrp

        return solve_pyvrp(instance, seconds=stop.max_seconds, seed=seed, **params)
    if algorithm in ("cpsat", "exact"):
        from qroute.exact.cpsat import solve_cpsat

        # CP-SAT names its budget ``time_limit``, not ``seconds``; also confine
        # it to one worker so an exact run does not take every core of the
        # machine that is simultaneously serving the browser.
        params = {"workers": 1, "seed": seed, **params}
        return solve_cpsat(instance, time_limit=stop.max_seconds, **params)
    if algorithm in ("random", "restart"):
        from qroute.benchmark.reference import RandomRestart

        optimizer = RandomRestart(instance, stop, seed, callback, **params)
    else:
        from qroute.algorithms.registry import build

        if initial_keys is not None:
            params = dict(params)
            params["initial_keys"] = np.asarray(initial_keys, dtype=np.float64)
        optimizer = build(algorithm, instance, stop=stop, seed=seed, callback=callback, **params)
    holder["optimizer"] = optimizer
    return optimizer.solve()


# --------------------------------------------------------------------------
# Server-side run records
# --------------------------------------------------------------------------


class TooManyRuns(RuntimeError):
    """Raised when the concurrency limit is reached; the endpoint returns 429.

    A distinct type rather than a bare ``RuntimeError`` so that a failure to
    launch a worker - which is a server fault, not a busy server - cannot be
    reported to the browser as "wait your turn".
    """


@dataclass
class RunRecord:
    """Everything the API knows about one run."""

    run_id: str
    algorithm: str
    instance_name: str
    seed: int
    max_seconds: float
    max_iterations: int
    params: dict[str, Any] = field(default_factory=dict)
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    ticks: list[dict[str, Any]] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    bks: Optional[float] = None
    parent_run_id: Optional[str] = None
    warm_started: bool = False
    baseline_cost: Optional[float] = None
    network_id: Optional[str] = None
    _geojson: Optional[dict[str, Any]] = None
    _geojson_built: bool = False

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.process: Optional[mp.process.BaseProcess] = None
        self.reader: Optional[threading.Thread] = None

    # -------------------------------------------------------------- reading
    @property
    def is_terminal(self) -> bool:
        return self.state in ("done", "cancelled", "failed")

    def ticks_from(self, index: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.ticks[index:])

    def best_tick(self) -> Optional[dict[str, Any]]:
        with self.lock:
            return self.ticks[-1] if self.ticks else None


class RunRegistry:
    """Owns the run records, the worker processes and their reader threads."""

    def __init__(self, state: ApiState | None = None, max_active: int = MAX_ACTIVE_RUNS):
        self.state = state or STATE
        self.max_active = max_active
        self._runs: "dict[str, RunRecord]" = {}
        self._lock = threading.Lock()
        self._context = _worker_context()
        self.start_method = self._context.get_start_method()
        self.primed = False
        self.prime_seconds = 0.0

    # -------------------------------------------------------------- plumbing
    def prime(self) -> bool:
        """Start the fork server, before anything that would poison a fork.

        Called from the application's startup, ahead of the road-network
        preload. Running one trivial child both forces the fork server up while
        the process is still PROJ-free and proves that launching a solver works
        at all - a failure here is worth knowing about at boot rather than when
        a judge presses "solve".

        Returns whether priming succeeded. The cost is the fork server's own
        start-up plus its preload, which is reported as ``prime_seconds`` in
        ``/api/health``.
        """
        started = time.perf_counter()
        try:
            process = self._context.Process(target=_noop, name="qroute-prime", daemon=True)
            with _neutral_main_module():
                process.start()
            process.join(timeout=30.0)
            self.primed = process.exitcode == 0
            if not self.primed:  # pragma: no cover - platform dependent
                log.error("worker priming exited with code %s", process.exitcode)
        except Exception:  # pragma: no cover - platform dependent
            log.exception("could not start the solver worker launcher")
            self.primed = False
        self.prime_seconds = time.perf_counter() - started
        log.info(
            "solver worker launcher (%s) ready in %.2fs",
            self.start_method, self.prime_seconds,
        )
        return self.primed

    def _queue(self):
        """A fresh progress queue for one run.

        A plain queue rather than a ``Manager().Queue()``. The manager was the
        first choice, because a queue living in a third process cannot be left
        half-written by terminating a cancelled worker. It turned out not to
        survive every way the server is launched: starting a manager re-imports
        the parent's ``__main__`` module in the manager process, which under
        ``python -m uvicorn`` fails with an ``EOFError`` in the parent as the
        manager child dies during bootstrap.

        The robustness the manager would have bought is recovered by
        construction instead: each run gets its own queue, so a queue left
        inconsistent by a terminated worker is discarded with that run and can
        never affect another, and the reader loop polls with a timeout and
        watches the child's liveness rather than blocking forever on a ``get``.
        """
        return self._context.Queue()

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._runs.values() if not r.is_terminal)

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def all_runs(self) -> list[RunRecord]:
        with self._lock:
            return list(self._runs.values())

    def shutdown(self) -> None:
        """Terminate every live worker; called from the application's lifespan."""
        for record in self.all_runs():
            self.cancel(record.run_id)

    # --------------------------------------------------------------- start
    def start(
        self,
        *,
        stored: StoredInstance,
        algorithm: str,
        seed: int,
        max_seconds: float,
        max_iterations: int,
        params: dict[str, Any] | None = None,
        initial_keys: Optional[np.ndarray] = None,
        parent_run_id: Optional[str] = None,
        baseline_cost: Optional[float] = None,
    ) -> RunRecord:
        """Launch a solver process and return its record.

        Raises :class:`TooManyRuns` when the concurrency limit is reached, which
        the endpoint turns into a 429.
        """
        instance = stored.instance
        record = RunRecord(
            run_id=uuid.uuid4().hex[:12],
            algorithm=algorithm,
            instance_name=stored.name,
            seed=int(seed),
            max_seconds=float(max_seconds),
            max_iterations=int(max_iterations),
            params=dict(params or {}),
            bks=float(instance.meta["bks"]) if instance.meta.get("bks") else None,
            parent_run_id=parent_run_id,
            warm_started=initial_keys is not None,
            baseline_cost=baseline_cost,
            network_id=stored.network_id,
        )

        # Claiming the slot and registering the record happen in one critical
        # section. Counting first and registering afterwards looks equivalent
        # but is not: launching a worker takes tens of milliseconds, and four
        # browser tabs pressing "solve" together all counted zero active runs
        # and all started, so the machine ran six solvers against a limit of
        # four. The record starts life in the "queued" state, which
        # ``is_terminal`` excludes, so it occupies its slot from here on.
        with self._lock:
            active = sum(1 for r in self._runs.values() if not r.is_terminal)
            if active >= self.max_active:
                raise TooManyRuns(
                    f"{self.max_active} runs are already in flight; cancel one or wait"
                )
            self._runs[record.run_id] = record

        progress_queue = self._queue()
        payload = {
            "instance": instance,
            "algorithm": algorithm,
            "params": dict(params or {}),
            "seed": int(seed),
            "max_seconds": float(max_seconds),
            "max_iterations": int(max_iterations),
            "initial_keys": None if initial_keys is None else np.asarray(initial_keys),
        }
        process = self._context.Process(
            target=solver_worker,
            args=(payload, progress_queue),
            name=f"qroute-run-{record.run_id}",
            daemon=True,
        )
        record.process = process
        record.state = "running"
        record.started_at = time.time()
        try:
            with _neutral_main_module():
                process.start()
        except Exception as exc:  # pragma: no cover - only if the OS refuses a fork
            # The slot was claimed above, so it has to be given back here or a
            # launcher failure would permanently consume one of the four.
            with record.lock:
                record.state = "failed"
                record.error = f"could not start the solver process: {type(exc).__name__}: {exc}"
                record.finished_at = time.time()
            log.exception("run %s could not be launched", record.run_id)
            raise

        reader = threading.Thread(
            target=self._drain,
            args=(record, progress_queue),
            name=f"qroute-reader-{record.run_id}",
            daemon=True,
        )
        record.reader = reader
        reader.start()

        log.info(
            "run %s started: %s on %s (seed %d, %.1fs)",
            record.run_id, algorithm, stored.name, seed, max_seconds,
        )
        return record

    # --------------------------------------------------------------- drain
    def _drain(self, record: RunRecord, progress_queue) -> None:
        """Move messages from the worker's queue into the record, until it ends.

        The loop is written so that a worker which dies without sending anything
        - terminated by a cancel, or killed by the operating system - still ends
        the run rather than leaving it "running" forever.
        """
        process = record.process
        deadline = time.time() + record.max_seconds + 300.0
        while True:
            try:
                message = progress_queue.get(timeout=0.2)
            except queue_mod.Empty:
                if process is not None and not process.is_alive():
                    # Give the queue one last chance: the child may have written
                    # its result microseconds before exiting.
                    try:
                        message = progress_queue.get(timeout=0.2)
                    except queue_mod.Empty:
                        with record.lock:
                            if not record.is_terminal:
                                record.state = "cancelled" if record.state == "cancelling" else "failed"
                                if record.state == "failed" and record.error is None:
                                    record.error = (
                                        f"solver process exited with code {process.exitcode} "
                                        "without returning a result"
                                    )
                                record.finished_at = time.time()
                        break
                if time.time() > deadline:  # pragma: no cover - safety valve
                    with record.lock:
                        record.state = "failed"
                        record.error = "the solver process stopped responding"
                        record.finished_at = time.time()
                    break
                continue
            except (EOFError, OSError, BrokenPipeError):  # pragma: no cover
                with record.lock:
                    if not record.is_terminal:
                        record.state = "failed"
                        record.error = "lost contact with the solver process"
                        record.finished_at = time.time()
                break

            kind = message.get("type")
            if kind == "tick":
                with record.lock:
                    record.ticks.append(message)
            elif kind == "done":
                with record.lock:
                    record.result = message["result"]
                    if record.bks is None and message["result"].get("bks"):
                        record.bks = float(message["result"]["bks"])
                    record.state = "done" if record.state != "cancelling" else "cancelled"
                    record.finished_at = time.time()
                break
            elif kind == "error":
                log.error(
                    "run %s failed: %s\n%s",
                    record.run_id, message.get("error"), message.get("traceback", ""),
                )
                with record.lock:
                    record.error = str(message.get("error"))
                    record.state = "failed"
                    record.finished_at = time.time()
                break

        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():  # pragma: no cover - defensive
                process.terminate()

    # -------------------------------------------------------------- cancel
    def cancel(self, run_id: str) -> Optional[RunRecord]:
        record = self.get(run_id)
        if record is None:
            return None
        with record.lock:
            if record.is_terminal:
                return record
            record.state = "cancelling"
        process = record.process
        if process is not None and process.is_alive():
            process.terminate()
        # The reader thread notices the dead process and settles the state; wait
        # briefly so that the response already reflects the final state.
        if record.reader is not None:
            record.reader.join(timeout=3.0)
        with record.lock:
            if record.state == "cancelling":
                record.state = "cancelled"
                record.finished_at = time.time()
        log.info("run %s cancelled", run_id)
        return record

    # ------------------------------------------------------------- reading
    def status(self, record: RunRecord, include_history: bool = True) -> dict[str, Any]:
        """Serialise a run for ``GET /api/runs/{id}``."""
        with record.lock:
            state = record.state
            result = record.result
            ticks = list(record.ticks)
            error = record.error
        latest = ticks[-1] if ticks else None

        best_cost: Optional[float] = None
        routes: Optional[list[list[int]]] = None
        stats: Optional[dict[str, float]] = None
        n_routes: Optional[int] = None
        feasible: Optional[bool] = None
        iterations = 0
        evaluations = 0
        seconds = 0.0
        params: dict[str, Any] = dict(record.params)
        history: list[dict[str, Any]] = []

        if result is not None:
            best_cost = result.get("best_cost")
            routes = result.get("routes")
            stats = result.get("stats")
            n_routes = result.get("n_routes")
            feasible = result.get("feasible")
            iterations = int(result.get("iterations", 0))
            evaluations = int(result.get("evaluations", 0))
            seconds = float(result.get("seconds", 0.0))
            params = result.get("params") or params
            history = [
                {
                    "iteration": h["iteration"],
                    "elapsed": h["elapsed"],
                    "evaluations": h["evaluations"],
                    "best_cost": h["best_cost"],
                    "mean_cost": h["mean_cost"],
                    "diversity": h["diversity"],
                }
                for h in result.get("history", [])
            ]
        elif latest is not None:
            best_cost = latest["best_cost"]
            iterations = latest["iteration"]
            evaluations = latest["evaluations"]
            seconds = latest["elapsed"]
            feasible = latest.get("feasible")
            # The most recent tick that carried geometry is the best plan the
            # browser can draw while the search is still going.
            for tick in reversed(ticks):
                if tick.get("routes"):
                    routes = tick["routes"]
                    break
            history = [
                {k: t[k] for k in
                 ("iteration", "elapsed", "evaluations", "best_cost", "mean_cost", "diversity")}
                for t in ticks
            ]

        stored = self.state.get_stored_instance(record.instance_name)
        coords = None
        if stored is not None and stored.instance.coords is not None:
            coords = [[float(a), float(b)] for a, b in stored.instance.coords]

        out: dict[str, Any] = {
            "run_id": record.run_id,
            "state": state,
            "algorithm": record.algorithm,
            "instance": record.instance_name,
            "seed": record.seed,
            "bks": record.bks,
            "best_cost": best_cost,
            "n_routes": n_routes,
            "feasible": feasible,
            "routes": routes,
            "stats": stats,
            "iterations": iterations,
            "evaluations": evaluations,
            "seconds": seconds,
            "params": params,
            "geojson": self.route_geojson(record, routes),
            "coords": coords,
            "error": error,
            "history": history if include_history else [],
            "warm_started": record.warm_started,
            "parent_run_id": record.parent_run_id,
            "baseline_cost": record.baseline_cost,
        }
        return out

    def route_geojson(
        self, record: RunRecord, routes: Optional[list[list[int]]]
    ) -> Optional[dict[str, Any]]:
        """Road-following polylines for a finished run on a road network.

        Returns ``None`` for benchmark instances, which have coordinates but no
        road graph behind them: drawing a straight line between two CVRPLIB
        points and calling it a route would be a lie the map tells convincingly.
        """
        if not routes or record.network_id is None:
            return None
        if record._geojson_built and record.state in ("done", "cancelled", "failed"):
            return record._geojson
        stored = self.state.get_stored_instance(record.instance_name)
        if stored is None or stored.matrices is None or stored.network_id is None:
            return None
        try:
            from qroute.graph.builder import routes_geojson

            bundle = self.state.get_network(stored.network_id)
            with bundle.lock:
                geojson = routes_geojson(bundle.network, stored.matrices, routes)
        except Exception:
            log.exception("could not build route geometry for run %s", record.run_id)
            return None
        if record.state in ("done", "cancelled", "failed"):
            record._geojson = geojson
            record._geojson_built = True
        return geojson


# --------------------------------------------------------------------------
# Warm starting
# --------------------------------------------------------------------------


def canonical_keys(routes: list[list[int]], n_customers: int) -> Optional[np.ndarray]:
    """Random-key vector encoding a set of routes, for warm-starting a run.

    The key-based solvers decode a vector by sorting it: the customer with the
    smallest key is visited first. The inverse is therefore to give the customer
    visited *i*-th the key ``(i + 0.5) / n``, which is exactly the canonical
    write-back :meth:`qroute.algorithms.decoder.Decoder.tour_to_keys` performs
    after a local search. Route boundaries are not encoded - the decoder re-splits
    the giant tour under the capacity constraint - so what is preserved is the
    visiting order, which is what the search actually explores.

    Returns ``None`` when the routes do not form a permutation of the customers,
    rather than handing a solver a corrupt starting point.
    """
    tour = [int(c) for route in routes for c in route]
    if len(tour) != n_customers or sorted(tour) != list(range(1, n_customers + 1)):
        return None
    keys = np.empty(n_customers, dtype=np.float64)
    positions = np.arange(n_customers, dtype=np.float64)
    keys[np.asarray(tour, dtype=np.int64) - 1] = (positions + 0.5) / n_customers
    return keys


def warm_start_matrix(
    routes: list[list[int]], n_customers: int, rows: int = 4
) -> Optional[np.ndarray]:
    """Stack the canonical encoding into the first few population slots.

    Seeding the *whole* population with one point would collapse the search to a
    local refinement of the previous plan. Seeding a handful of slots keeps the
    incumbent in the swarm while the remaining particles still start at random,
    which is the usual compromise for re-optimisation under a changed cost
    matrix.
    """
    keys = canonical_keys(routes, n_customers)
    if keys is None:
        return None
    return np.tile(keys, (max(1, rows), 1))


# --------------------------------------------------------------------------
# Server-Sent Events
# --------------------------------------------------------------------------


async def stream_run(registry: RunRegistry, record: RunRecord, poll: float = 0.05):
    """Yield SSE messages for one run until it reaches a terminal state.

    Events emitted:

    ``start``    once, carrying the run's identity and its stopping rule
    ``tick``     one per throttled iteration sample, matching ``RunTick``
    ``done``     once, carrying the full ``RunStatus``
    ``error``    instead of ``done`` when the solver failed
    ``cancelled`` instead of ``done`` when the run was stopped

    A run that has already finished when the client connects still gets its
    whole history replayed followed by the terminal event, so reconnecting after
    a dropped connection is not a special case for the frontend.
    """
    yield {
        "event": "start",
        "data": json.dumps(
            {
                "run_id": record.run_id,
                "algorithm": record.algorithm,
                "instance": record.instance_name,
                "seed": record.seed,
                "max_seconds": record.max_seconds,
                "max_iterations": record.max_iterations,
                "bks": record.bks,
            }
        ),
    }
    index = 0
    while True:
        pending = record.ticks_from(index)
        for tick in pending:
            payload = {k: v for k, v in tick.items() if k != "type"}
            yield {"event": "tick", "data": json.dumps(payload)}
        index += len(pending)

        if record.is_terminal:
            # One last sweep: the reader thread may have appended between the
            # snapshot above and the state check.
            for tick in record.ticks_from(index):
                yield {"event": "tick", "data": json.dumps({k: v for k, v in tick.items()
                                                            if k != "type"})}
            status = registry.status(record)
            event = {"done": "done", "cancelled": "cancelled", "failed": "error"}[record.state]
            yield {"event": event, "data": json.dumps(status)}
            return
        await asyncio.sleep(poll)



# --------------------------------------------------------------------------
# Fork-server warm-up
# --------------------------------------------------------------------------
#
# Reached only inside the fork server process, which is exec'd with
# WORKER_PRELOAD_ENV set (see :func:`_worker_context`) and whose only extension
# point is the list of modules it imports. Importing this module there therefore
# also compiles the kernels, and every worker forked from it inherits them.
if os.environ.get(WORKER_PRELOAD_ENV) == "1":  # pragma: no cover - subprocess only
    os.environ.pop(WORKER_PRELOAD_ENV, None)
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    _WARM_SECONDS = warm_kernels()
