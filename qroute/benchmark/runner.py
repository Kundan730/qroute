"""Reproducible benchmark execution.

The runner exists so that a benchmark table can be regenerated from a single
command months later and come out identical. Everything that could make a run
irreproducible is pinned or recorded:

* Seeds are derived from one master seed, so run *k* of an algorithm is the same
  run regardless of execution order or how many workers are used.
* Runs execute in separate processes, each restricted to a single thread, so a
  wall-clock budget means the same thing for every algorithm. Without this an
  algorithm whose library happens to be multi-threaded would silently receive
  several times the CPU of its competitors.
* The environment is captured: library versions, the git commit, the host CPU
  count and the resolved configuration are written next to the results.

The unit of work is one ``(instance, algorithm, seed)`` triple. Results are
written incrementally, so a long run that is interrupted keeps everything it had
already finished.

Three rules protect a sweep that is already running, because a sweep of the size
this project reports costs about an hour and losing it to a small accident is
the expensive failure:

* A directory that already holds rows is never overwritten without ``force``,
  and even then the previous ``rows.jsonl`` is renamed rather than truncated.
* A worker process that dies takes its own task down, not the sweep: the tasks
  that were outstanding are re-run in fresh pools, halving until the one that
  killed the worker is identified, and only that one is recorded as failed.
* An interrupt stops scheduling and summarises what finished, rather than
  raising through the results that are already on disk.
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional, Sequence

import numpy as np

from qroute.core.rng import spawn_seeds


@dataclass
class BenchmarkConfig:
    """Everything needed to reproduce a benchmark run."""

    name: str = "benchmark"
    instances: list[str] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=lambda: ["qpso"])
    seeds: int = 10
    master_seed: int = 20260920
    max_seconds: float = 10.0
    max_iterations: int = 1_000_000
    workers: int = 0                    # 0 = os.cpu_count() - 2
    params: dict[str, dict] = field(default_factory=dict)   # per-algorithm overrides
    output_dir: str = "results/runs"
    save_history: bool = True
    history_stride: int = 1

    @staticmethod
    def from_yaml(path: str | Path) -> "BenchmarkConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        known = {f for f in BenchmarkConfig.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return BenchmarkConfig(**data)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Host resources
# ---------------------------------------------------------------------------
class WorkerPlan(NamedTuple):
    """How many worker processes a sweep will use, and why."""

    workers: int
    cpus: int
    #: Set when the operator asked for more processes than the host has cores.
    #: The sweep still runs - oversubscribing is occasionally deliberate - but
    #: the caller is expected to show this, because every result in the sweep is
    #: measured by wall clock and time-slicing silently shortens every budget.
    warning: Optional[str] = None


def host_cpus() -> int:
    """Cores this process may actually use, which is not always ``cpu_count``.

    ``os.process_cpu_count`` (3.13+) honours CPU affinity and container quotas,
    so a sweep inside a two-core CI container is told two rather than the
    sixty-four the host advertises. Older interpreters fall back to the affinity
    mask and finally to ``cpu_count``.
    """
    getter = getattr(os, "process_cpu_count", None)
    if getter is not None:
        n = getter()
        if n:
            return int(n)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:                       # Linux, and not on macOS
        try:
            return max(1, len(affinity(0)))
        except OSError:                            # pragma: no cover - platform specific
            pass
    return max(1, os.cpu_count() or 1)


def resolve_workers(requested: int | None, cpus: int | None = None) -> WorkerPlan:
    """Decide the worker count for a sweep from the request and the host.

    A benchmark configuration that names a fixed worker count is only correct on
    the machine it was written on: ``workers: 9`` on a four-core laptop puts nine
    time-budgeted runs on four cores, so every one of them completes a fraction
    of the iterations it should have and the whole table is quietly wrong. Zero,
    a negative number or ``None`` therefore mean "ask this host", and the answer
    leaves two cores for the operating system, the writer process and whatever
    else is running.
    """
    cpus = int(cpus if cpus is not None else host_cpus())
    cpus = max(1, cpus)
    if not requested or int(requested) <= 0:
        return WorkerPlan(workers=max(1, cpus - 2), cpus=cpus)
    workers = int(requested)
    warning = None
    if workers > cpus:
        warning = (
            f"{workers} workers were requested but this host offers {cpus} core"
            f"{'s' if cpus != 1 else ''}. The runs share the cores, so each one "
            f"gets roughly {cpus / workers:.0%} of a core for a budget that is "
            f"measured in wall-clock seconds; the results will understate every "
            f"algorithm. Use --workers {max(1, cpus - 2)} unless you know why "
            f"you want to oversubscribe."
        )
    return WorkerPlan(workers=workers, cpus=cpus, warning=warning)


# ---------------------------------------------------------------------------
# Kernel warm-up
# ---------------------------------------------------------------------------
#: Solvers whose inner loops are numba-compiled and therefore need warming.
#: The external solvers (OR-Tools, PyVRP, CP-SAT) are compiled C++ and have
#: nothing to warm; running them on a throwaway instance would only burn their
#: time limit, since a guided local search does not stop when it is optimal.
_JIT_ALGORITHMS = frozenset({
    "qpso", "qiea", "rotation", "qrk", "rotation_keys", "quantum_rotation_keys",
    "pso", "ga", "sa", "aco", "random", "restart",
})


class WarmUp(NamedTuple):
    """What a call to :func:`warm_kernels` achieved."""

    algorithm: str
    #: Wall-clock seconds spent compiling or loading kernels.
    seconds: float
    #: False when the algorithm has no compiled kernels, so nothing was done.
    applicable: bool
    #: Set when the warm-up itself failed. The run may proceed, but its first
    #: seconds will be spent compiling and its budget is no longer honest.
    error: Optional[str] = None


def _warmup_instance():
    """A seven-node instance that exercises every kernel and solves instantly."""
    from qroute.problems.instance import Instance

    # Fixed coordinates rather than a random draw: the warm-up must not consume
    # a random stream, and a compiled kernel does not care about the geometry.
    coords = np.array([[0.0, 0.0], [1.0, 3.0], [4.0, 1.0], [5.0, 5.0],
                       [2.0, 6.0], [6.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((delta ** 2).sum(axis=2))
    demand = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return Instance(name="warmup", distance=distance, demand=demand, capacity=3.0,
                    coords=coords)


def warm_kernels(algorithm: str) -> WarmUp:
    """Compile the JIT kernels before a timed run starts its clock.

    numba compiles on first call. With a cold on-disk cache that costs about
    twelve seconds on this project's kernels, which is longer than a typical
    demonstration budget: measured here, ``qroute solve A-n32-k5 -t 5`` against
    an empty cache completed *zero* iterations, ran for 11.7 seconds and printed
    the construction heuristic's tour as its answer, with nothing on screen to
    say that no search had happened. Even a warm cache costs about a second to
    load, which is a fifth of a five-second budget.

    Running two iterations on a throwaway instance first moves that cost outside
    the measured interval, where it belongs. The caller is expected to report
    :attr:`WarmUp.error` if it is set, because a run that could not be warmed is
    a run whose budget went on compilation.
    """
    key = str(algorithm).strip().lower()
    if key not in _JIT_ALGORITHMS:
        return WarmUp(algorithm=key, seconds=0.0, applicable=False)

    from qroute.algorithms.base import StopCriteria

    started = time.perf_counter()
    try:
        # No wall-clock limit: cutting the warm-up short would defeat it, and
        # the instance is small enough that two iterations are instantaneous
        # once the kernels exist.
        _dispatch(key, _warmup_instance(),
                  StopCriteria(max_iterations=2, max_seconds=float("inf")), 0, {})
    except Exception as exc:      # a failed warm-up must never fail the run
        return WarmUp(algorithm=key, seconds=time.perf_counter() - started,
                      applicable=True, error=f"{type(exc).__name__}: {exc}")
    return WarmUp(algorithm=key, seconds=time.perf_counter() - started, applicable=True)


#: Algorithms already warmed *in this process*. numba caches compiled code in
#: memory as well as on disk, so warming twice costs nothing useful; a worker
#: that handles two hundred tasks should pay for it once.
_WARMED: dict[str, WarmUp] = {}


def _warm_once(algorithm: str) -> Optional[WarmUp]:
    """Warm an algorithm's kernels the first time this process meets it."""
    key = str(algorithm).strip().lower()
    if key not in _WARMED:
        _WARMED[key] = warm_kernels(key)
    warm = _WARMED[key]
    return warm if warm.applicable else None


def environment_fingerprint() -> dict[str, Any]:
    """Record what the results depend on, so they can be reproduced or explained."""
    from importlib.metadata import PackageNotFoundError, version

    pkgs = {}
    for p in ("numpy", "scipy", "numba", "ortools", "networkx", "osmnx", "vrplib", "pyvrp"):
        try:
            pkgs[p] = version(p)
        except PackageNotFoundError:
            pkgs[p] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=5).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp_unix": time.time(),
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(task: dict) -> dict:
    """Execute one (instance, algorithm, seed) triple in this process.

    Defined at module level and taking only plain data so it can be pickled for
    a spawn-based process pool, which is the only start method available on
    macOS.
    """
    # Confine every solver to one thread so the wall-clock budget is comparable.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from qroute.problems.loaders import load
    from qroute.algorithms.base import StopCriteria

    name = task["instance"]
    algo = task["algorithm"]
    seed = task["seed"]
    params = dict(task.get("params") or {})

    # Compile before the clock starts. Each worker is a fresh process, so each
    # one pays this once; without it the first task in every worker spends its
    # whole wall-clock budget inside numba and reports whatever the construction
    # heuristic produced, which is a wrong number rather than a slow one.
    warm = _warm_once(algo)

    started = time.time()
    try:
        inst = load(name)
        stop = StopCriteria(max_iterations=task.get("max_iterations", 1_000_000),
                            max_seconds=task.get("max_seconds", 10.0))
        result = _dispatch(algo, inst, stop, seed, params)

        # A solver that returns nothing within its budget has not produced a
        # result, and must not be recorded as a feasible one. OR-Tools does
        # exactly this on large instances: it returns an empty assignment with
        # an infinite cost, which was being written down as a feasible row and
        # then poisoning every average it entered. "No solution found" is the
        # honest outcome and is reported as such.
        served = sum(len(r) for r in result.best.routes)
        if not result.best.routes or served < inst.n_customers or not np.isfinite(result.best.cost):
            return {
                "instance": name, "algorithm": algo, "seed": seed,
                "status": "no_solution",
                "error": (f"returned no complete solution within {stop.max_seconds:g}s "
                          f"({served} of {inst.n_customers} customers served)"),
                "seconds": float(result.seconds),
                "iterations": int(result.iterations),
                "evaluations": int(result.evaluations),
            }
        result.best.validate(inst.n_customers)
        bks = inst.meta.get("bks")
        row = {
            "instance": name,
            "algorithm": algo,
            "seed": seed,
            "cost": float(result.best.cost),
            "gap": float(result.gap_to(bks)) if bks else None,
            "bks": bks,
            "n_routes": int(result.best.n_routes),
            "feasible": bool(result.best.is_feasible),
            "violation": float(result.best.stats.total_violation),
            "iterations": int(result.iterations),
            "evaluations": int(result.evaluations),
            "seconds": float(result.seconds),
            "params": result.params,
            "status": "ok",
        }
        if warm is not None and warm.error:
            # Recorded per row rather than raised: the run still happened, but a
            # reader comparing budgets deserves to know that this one paid for
            # compilation out of its own clock.
            row["warmup_error"] = warm.error
        if bks:
            row["time_to_1pct"] = result.time_to_within(bks, 1.0)
            row["time_to_2pct"] = result.time_to_within(bks, 2.0)
            row["iters_to_1pct"] = result.iterations_to_within(bks, 1.0)
        if task.get("save_history"):
            stride = max(1, int(task.get("history_stride", 1)))
            row["history"] = [
                {"t": round(h.elapsed, 4), "i": h.iteration, "c": h.best_cost,
                 "m": h.mean_cost, "d": h.diversity}
                for h in result.history[::stride]
            ]
        return row
    except Exception as exc:   # a failing solver must not abort the whole sweep
        import traceback
        return {
            "instance": name, "algorithm": algo, "seed": seed,
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
            "seconds": time.time() - started,
        }


#: Every solver names its time budget differently, and several take no random
#: seed at all. Rather than force a uniform signature - which would misrepresent
#: what each solver actually accepts - the dispatcher translates.
_BUDGET_ALIASES = ("seconds", "time_limit", "max_seconds", "time_limit_seconds",
                   "runtime", "max_runtime")
_SEED_ALIASES = ("seed", "random_seed")


def _call_with_supported(fn, inst, **kwargs):
    """Call ``fn``, translating budget and seed names and dropping what it cannot take.

    This has one job that is easy to get subtly and catastrophically wrong. The
    external wrappers are declared ``(instance, **kwargs)`` and forward to an
    inner function that holds the real signature. An earlier version of this
    routine inspected only the wrapper, found that neither ``seconds`` nor
    ``seed`` appeared among its parameter *names*, and therefore forwarded
    neither - so every external solver silently ran at its own default budget
    with its own default seed. The benchmark's central fairness claim, that
    every solver received the same wall clock, was false for three solvers and
    nobody noticed, because dropping a keyword raises nothing.

    The rule now is the opposite way round: when a target accepts ``**kwargs``,
    pass everything and let it complain, rather than filtering against a
    signature that does not describe what it really accepts. Anything it rejects
    is removed one keyword at a time, and a rejected budget is retried under
    each of its other spellings before being given up.
    """
    import inspect

    def parameters_of(target):
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return None, False
        names = set(sig.parameters)
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
        return names, var_kw

    names, var_kw = parameters_of(fn)
    target = fn
    while var_kw:
        inner = getattr(target, "__wrapped__", None)
        if inner is None:
            break
        target = inner
        names, var_kw = parameters_of(target)

    call = dict(kwargs)
    budget = next((call.pop(a) for a in _BUDGET_ALIASES if a in call), None)
    seed = next((call.pop(a) for a in _SEED_ALIASES if a in call), None)

    if names is not None and not var_kw:
        # The signature is trustworthy: name the budget and seed the way this
        # target spells them, and drop anything it does not declare.
        if budget is not None:
            for alias in _BUDGET_ALIASES:
                if alias in names:
                    call[alias] = budget
                    break
        if seed is not None:
            for alias in _SEED_ALIASES:
                if alias in names:
                    call[alias] = seed
                    break
        call = {k: v for k, v in call.items() if k in names}
        return fn(inst, **call)

    # The target hides its real signature behind **kwargs. Try each spelling of
    # the budget and the seed in turn, keeping whatever is accepted.
    budget_spellings = [a for a in _BUDGET_ALIASES] if budget is not None else [None]
    seed_spellings = [a for a in _SEED_ALIASES] if seed is not None else [None]
    last_error: Exception | None = None
    for b_alias in budget_spellings:
        for s_alias in seed_spellings:
            attempt = dict(call)
            if b_alias is not None:
                attempt[b_alias] = budget
            if s_alias is not None:
                attempt[s_alias] = seed
            for _ in range(len(attempt) + 1):
                try:
                    return fn(inst, **attempt)
                except TypeError as exc:
                    message = str(exc)
                    if "unexpected keyword argument" not in message:
                        raise
                    bad = message.rsplit("'", 2)
                    key = bad[-2] if len(bad) >= 2 else None
                    if key is None or key not in attempt:
                        last_error = exc
                        break
                    if key in (b_alias, s_alias):
                        # This spelling is wrong; try the next one rather than
                        # silently continuing without a budget or a seed.
                        last_error = exc
                        break
                    attempt.pop(key)
            else:
                continue
    if last_error is not None:
        raise last_error
    return fn(inst)


def _dispatch(algo: str, inst, stop, seed: int, params: dict):
    """Map an algorithm name to a call that returns an OptimizationResult."""
    algo = algo.lower()
    # External solvers are wrapped as plain functions, not Optimizer subclasses,
    # because they own their own search loop.
    if algo in ("ortools", "ortools_gls"):
        from qroute.baselines.ortools_gls import solve_ortools
        return _call_with_supported(solve_ortools, inst, seconds=stop.max_seconds, **params)
    if algo in ("pyvrp", "hgs"):
        from qroute.baselines.pyvrp_hgs import solve_pyvrp
        return _call_with_supported(solve_pyvrp, inst, seconds=stop.max_seconds,
                                    seed=seed, **params)
    if algo in ("cpsat", "exact"):
        from qroute.exact.cpsat import solve_cpsat
        return _call_with_supported(solve_cpsat, inst, seconds=stop.max_seconds, **params)
    if algo in ("random", "restart"):
        from qroute.benchmark.reference import RandomRestart
        return RandomRestart(inst, stop, seed, **params).solve()
    # The two quantum rotation-gate engines are addressed directly rather than
    # through the registry, so a benchmark can include them whether or not the
    # registry has been extended.
    if algo in ("qiea", "rotation"):
        from qroute.algorithms.qiea import QIEA
        return QIEA(inst, stop, seed, **params).solve()
    if algo in ("qrk", "rotation_keys", "quantum_rotation_keys"):
        from qroute.algorithms.qiea import QuantumRotationKeys
        return QuantumRotationKeys(inst, stop, seed, **params).solve()

    from qroute.algorithms.registry import build
    return build(algo, inst, stop=stop, seed=seed, **params).solve()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _terminate_workers(pool: ProcessPoolExecutor) -> None:
    """Stop the worker processes at once after an interrupt.

    ``Executor`` offers no public way to abandon work that has already started:
    ``shutdown(wait=False, cancel_futures=True)`` un-queues what has not begun,
    but the running tasks are left to finish and the interpreter's own exit hook
    then joins them. Measured on an eight-run sweep with a two-second budget,
    Ctrl-C took twenty-two seconds to return while results nobody would ever
    read were computed; with the twenty-second budget the real sweep uses it
    would be minutes. Killing the workers first brought the same case down to
    well under a second.

    The executor's private process mapping is the only handle there is, so it is
    used deliberately and defensively: if a future version of the standard
    library renames it, the interrupt is merely slow again. It must be read
    before ``shutdown``, which sets the attribute to ``None``.
    """
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            proc.terminate()
        except Exception:                      # already gone, or not ours to kill
            pass


class ExistingResults(RuntimeError):
    """Raised rather than overwrite results a previous sweep already wrote.

    Carries the numbers a person needs in order to decide what to do, so the
    command line can format them without re-reading the directory.
    """

    def __init__(self, path: Path, rows: int, finished: bool, modified: float):
        self.path = Path(path)
        self.rows = int(rows)
        self.finished = bool(finished)
        self.modified = float(modified)
        when = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        state = "a completed sweep" if finished else "a partial sweep"
        super().__init__(
            f"{self.path} already holds {state} of {rows} runs, last written {when}"
        )


class BenchmarkRunner:
    """Runs a configuration and writes results incrementally."""

    #: How many times a task that is alone in its own pool may kill that pool
    #: before it is written off. One retry distinguishes an unlucky process (a
    #: transient out-of-memory kill, an external solver segfaulting once) from a
    #: task that reliably destroys whatever runs it; a further retry would only
    #: cost the sweep the same minutes again for the same answer.
    MAX_ATTEMPTS = 2

    def __init__(self, config: BenchmarkConfig, progress: Callable[[dict], None] | None = None,
                 force: bool = False):
        self.config = config
        self.progress = progress
        # ``force`` is deliberately a runner argument and not a configuration
        # field: it describes this invocation, not the experiment, and it must
        # not end up in the meta.json that documents how the results were made.
        self.force = bool(force)

    def tasks(self) -> list[dict]:
        cfg = self.config
        seeds = spawn_seeds(cfg.master_seed, cfg.seeds)
        out = []
        for inst in cfg.instances:
            for algo in cfg.algorithms:
                for k, seed in enumerate(seeds):
                    out.append({
                        "instance": inst,
                        "algorithm": algo,
                        "seed": int(seed),
                        "seed_index": k,
                        "params": cfg.params.get(algo, {}),
                        "max_seconds": cfg.max_seconds,
                        "max_iterations": cfg.max_iterations,
                        "save_history": cfg.save_history,
                        "history_stride": cfg.history_stride,
                    })
        return out

    def output_dir(self) -> Path:
        """Directory this sweep will write to. Nothing is created."""
        return Path(self.config.output_dir) / self.config.name

    def check_output_dir(self) -> Path:
        """Raise :class:`ExistingResults` if the target already holds a sweep.

        Separate from :meth:`run` so a caller can find out before it puts a
        progress bar on the screen, and safe to call twice: it inspects the
        directory and changes nothing.

        What is protected is *results*, not the file that holds them. A sweep
        interrupted before its first run finished leaves a ``rows.jsonl`` of
        zero rows, and refusing to start again over an empty file would demand
        ``--force`` to protect nothing - and would then file a zero-byte
        ``rows.superseded`` as though something had been rescued.
        """
        out_dir = self.output_dir()
        rows_path = out_dir / "rows.jsonl"
        if rows_path.exists() and not self.force:
            rows = count_rows(rows_path)
            if rows:
                raise ExistingResults(
                    path=out_dir, rows=rows,
                    finished=(out_dir / "summary.json").exists(),
                    modified=rows_path.stat().st_mtime,
                )
        return out_dir

    def _guard_output(self, out_dir: Path) -> Optional[Path]:
        """Refuse to destroy results already in ``out_dir``; return what was moved.

        The old behaviour truncated ``rows.jsonl`` and rewrote ``meta.json``
        before the first task had finished, so re-running a sweep by habit -
        interrupting it a minute later on realising the mistake - left the
        directory holding neither the old results nor any new ones. An hour of
        compute is not something a tool should be able to delete as a side
        effect of a command that has not produced anything yet.

        The default is therefore to refuse. ``force`` still does not delete: the
        previous rows are renamed with a timestamp, because the operator who
        typed ``--force`` wanted the new run, not the destruction of the old one.
        """
        self.check_output_dir()
        rows_path = out_dir / "rows.jsonl"
        # An empty rows file is not a result, so it is simply reused rather than
        # filed away: naming a zero-byte file "superseded" would suggest the
        # directory still holds a previous sweep somewhere.
        if not rows_path.exists() or not count_rows(rows_path):
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        superseded = out_dir / f"rows.superseded-{stamp}.jsonl"
        rows_path.rename(superseded)
        return superseded

    def run(self) -> dict:
        cfg = self.config
        out_dir = Path(cfg.output_dir) / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Before anything is written: an overwrite check that happens after
        # meta.json has been replaced is not a check at all.
        superseded = self._guard_output(out_dir)

        tasks = self.tasks()
        plan = resolve_workers(cfg.workers)

        meta = {"config": cfg.to_dict(), "environment": environment_fingerprint(),
                "n_tasks": len(tasks), "workers": plan.workers,
                "usable_cpus": plan.cpus}
        if plan.warning:
            meta["workers_warning"] = plan.warning
        if superseded is not None:
            meta["superseded_rows"] = superseded.name
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

        rows: list[dict] = []
        started = time.time()
        rows_path = out_dir / "rows.jsonl"
        interrupted = False
        with rows_path.open("w") as fh:
            def record(row: dict) -> None:
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()     # a sweep that is killed keeps every finished run
                if self.progress:
                    self.progress({"done": len(rows), "total": len(tasks), "row": row,
                                   "elapsed": time.time() - started})

            try:
                self._execute(tasks, plan.workers, record)
            except KeyboardInterrupt:
                # Everything finished so far is already on disk and in ``rows``.
                # Summarising it is worth more to the operator than a traceback,
                # and the caller is told so it can exit with the right code.
                interrupted = True

        summary = self.summarise(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return {"meta": meta, "rows": rows, "summary": summary,
                "output_dir": str(out_dir), "interrupted": interrupted,
                "n_tasks": len(tasks), "workers": plan.workers,
                "workers_warning": plan.warning,
                "superseded_rows": str(superseded) if superseded else None}

    def _execute(self, tasks: Sequence[dict], workers: int,
                 record: Callable[[dict], None]) -> None:
        """Run every task, surviving the death of a worker process.

        ``ProcessPoolExecutor`` has no notion of a single failed worker: when a
        child dies without returning - killed by the out-of-memory reaper, or
        crashed inside a native solver - the executor declares itself broken and
        every outstanding future raises ``BrokenProcessPool``, including the ones
        whose work had not started. Calling ``future.result()`` unguarded threw
        away an entire sweep because one process died, which on a long run is a
        loss of hours.

        What the pool tells us is only *which tasks did not come back*, not which
        one killed the worker, and blaming all of them would write off work that
        never ran. So the outstanding tasks are halved and each half is given its
        own pool. A half that runs cleanly is done; a half that breaks is halved
        again. When a single task is all that is left outstanding, it is the one
        that was running, and it is retried alone once before being recorded as
        a failed run - a task that reliably kills a fresh process on its own is
        not going to succeed on the third attempt, and the rest of the sweep is
        worth more than the wait.

        This costs about ``2 log2(n)`` extra process pools for one fatal task,
        against the alternatives of losing the sweep or losing every task queued
        behind the bad one.
        """
        queue: list[list[tuple[int, dict]]] = [list(enumerate(tasks))]
        solo_attempts: dict[int, int] = {}

        while queue:
            batch = queue.pop(0)
            if not batch:
                continue
            outstanding = self._run_batch(batch, workers if len(batch) > 1 else 1, record)
            if not outstanding:
                continue
            if len(outstanding) == 1:
                index, task = outstanding[0]
                solo_attempts[index] = solo_attempts.get(index, 0) + 1
                if solo_attempts[index] >= self.MAX_ATTEMPTS:
                    record(self._dead_worker_row(task, solo_attempts[index]))
                else:
                    queue.insert(0, outstanding)
            else:
                middle = len(outstanding) // 2
                # Depth first, so the culprit is found and recorded before the
                # innocent half is re-run: it keeps the peak number of live
                # pools at one and the progress display monotonic.
                queue.insert(0, outstanding[middle:])
                queue.insert(0, outstanding[:middle])

    def _run_batch(self, batch: Sequence[tuple[int, dict]], workers: int,
                   record: Callable[[dict], None]) -> list[tuple[int, dict]]:
        """Run one batch in a fresh pool; return the tasks that never came back."""
        outstanding: list[tuple[int, dict]] = []
        # Not a ``with`` block: its ``__exit__`` waits for the runs still in
        # flight, which is the opposite of what an interrupt asks for.
        pool = ProcessPoolExecutor(max_workers=max(1, workers))
        interrupted = False
        try:
            futures = {pool.submit(_run_one, task): (index, task)
                       for index, task in batch}
            for fut in as_completed(futures):
                index, task = futures[fut]
                try:
                    record(fut.result())
                    continue
                except BrokenProcessPool:
                    outstanding.append((index, task))
                    continue
                except Exception as exc:   # an unpicklable result, and the like
                    record({
                        "instance": task["instance"], "algorithm": task["algorithm"],
                        "seed": task["seed"], "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "seconds": 0.0,
                    })
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            # Order matters: ``shutdown`` clears the executor's process mapping,
            # so killing the workers afterwards silently kills nothing and the
            # interpreter waits for them at exit instead.
            if interrupted:
                _terminate_workers(pool)
            pool.shutdown(wait=not interrupted, cancel_futures=True)
        return outstanding

    @staticmethod
    def _dead_worker_row(task: dict, attempts: int) -> dict:
        """A row standing in for a task whose worker process never came back.

        Recorded rather than dropped: a missing row would look like a run that
        was never scheduled, and the difference between "not attempted" and
        "attempted and lost the process" matters when reading the table.
        """
        return {
            "instance": task["instance"], "algorithm": task["algorithm"],
            "seed": task["seed"], "status": "worker_died",
            "error": (f"the worker process died on this run, alone in its own pool, "
                      f"on {attempts} attempt{'s' if attempts != 1 else ''}; the run "
                      f"produced no result and the rest of the sweep continued"),
            "seconds": 0.0,
        }

    # ------------------------------------------------------------------ report
    @staticmethod
    def summarise(rows: Sequence[dict]) -> dict:
        """Per (instance, algorithm) statistics plus the omnibus comparison."""
        from qroute.benchmark.stats import friedman, summarise as summarise_values

        ok = [r for r in rows if r.get("status") == "ok"]
        failed = [r for r in rows if r.get("status") not in ("ok", "no_solution")]
        no_solution = [r for r in rows if r.get("status") == "no_solution"]
        by: dict[tuple[str, str], list[dict]] = {}
        for r in ok:
            by.setdefault((r["instance"], r["algorithm"]), []).append(r)

        cells = {}
        for (inst, algo), rs in by.items():
            # Gap statistics are computed over FEASIBLE runs only. The gap of an
            # infeasible solution is not a result: a plan that overloads a
            # vehicle can be arbitrarily cheap, and averaging it in makes an
            # algorithm look better precisely when it has failed. Before this
            # rule was applied, three algorithms appeared to beat a published
            # best-known cost on B-n64-k9, which is impossible; every one of
            # them had returned an overloaded plan. Infeasible runs are counted
            # and reported separately so nothing is hidden.
            feasible = [r for r in rs if r.get("feasible")]
            gaps = [r["gap"] for r in feasible if r.get("gap") is not None]
            gaps_all = [r["gap"] for r in rs if r.get("gap") is not None]
            costs = [r["cost"] for r in feasible] or [r["cost"] for r in rs]
            cell = {
                "instance": inst,
                "algorithm": algo,
                "cost": summarise_values(costs),
                "gap": summarise_values(gaps) if gaps else None,
                "gap_including_infeasible": summarise_values(gaps_all) if gaps_all else None,
                "feasible_runs": len(feasible),
                "infeasible_runs": len(rs) - len(feasible),
                "runs": len(rs),
                "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
                "mean_iterations": float(np.mean([r["iterations"] for r in rs])),
                "hit_bks": sum(1 for r in feasible if r.get("gap") is not None and r["gap"] <= 1e-9),
                "median_time_to_1pct": _median_or_none([r.get("time_to_1pct") for r in feasible]),
            }
            cells[f"{inst}|{algo}"] = cell

        algorithms = sorted({r["algorithm"] for r in ok})
        instances = sorted({r["instance"] for r in ok})
        omnibus = None
        # The omnibus test needs a complete matrix, so only instances solved by
        # every algorithm take part; anything else would compare different sets.
        # A cell only counts as scored when its gap summary actually has values.
        # summarise() returns {"n": 0} for an empty sample, which is truthy, so
        # testing the dict alone let an unscored cell into the matrix and the
        # omnibus test then failed on the missing key.
        def _scored(instance: str, algorithm: str) -> bool:
            cell = cells.get(f"{instance}|{algorithm}")
            gap = cell.get("gap") if cell else None
            return bool(gap) and gap.get("n", 0) > 0 and "median" in gap

        common = [i for i in instances if all(_scored(i, a) for a in algorithms)]
        if len(algorithms) >= 3 and len(common) >= 3:
            per_algo = {a: [cells[f"{i}|{a}"]["gap"]["median"] for i in common] for a in algorithms}
            try:
                fr = friedman(per_algo)
                omnibus = {
                    "instances_used": common,
                    "mean_ranks": fr.mean_ranks,
                    "statistic": fr.statistic,
                    "p_value": fr.p_value,
                    "control": fr.control,
                    "post_hoc": [
                        {"a": c.a, "b": c.b, "p": c.p_value, "p_holm": c.p_adjusted,
                         "effect": c.effect_size, "winner": c.winner, "text": c.describe()}
                        for c in fr.post_hoc
                    ],
                }
            except ValueError:
                omnibus = None

        infeasible_total = sum(c["infeasible_runs"] for c in cells.values())
        return {
            "cells": cells,
            "algorithms": algorithms,
            "instances": instances,
            "n_ok": len(ok),
            "n_failed": len(failed),
            "n_infeasible": infeasible_total,
            "n_no_solution": len(no_solution),
            "no_solution": [{"instance": r["instance"], "algorithm": r["algorithm"],
                             "seed": r["seed"], "reason": r.get("error")}
                            for r in no_solution],
            "gap_policy": ("Gap statistics cover feasible runs only. Infeasible runs are "
                           "counted in infeasible_runs and their gaps, which are not "
                           "meaningful, are kept separately in gap_including_infeasible."),
            "failures": [{"instance": r["instance"], "algorithm": r["algorithm"],
                          "error": r.get("error")} for r in failed[:50]],
            "omnibus": omnibus,
        }


def _median_or_none(values: Iterable[float | None]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


class RowsFile(NamedTuple):
    """The result of reading a ``rows.jsonl``, including what could not be read."""

    rows: list[dict]
    #: 1-based line numbers that were not valid JSON, in file order.
    unreadable: list[int]
    #: The file actually opened, which may be the gzipped sibling.
    path: Path

    @property
    def complete(self) -> bool:
        return not self.unreadable


def resolve_rows_path(path: str | Path) -> Path:
    """Return the rows file to read, accepting the gzipped form.

    The definitive sweep's raw log is 45 MB, so what is committed is
    ``rows.jsonl.gz`` and ``rows.jsonl`` is in ``.gitignore``. Someone who
    clones the repository and runs ``qroute report results/runs/main`` has only
    the gzipped file, and being told "rows.jsonl does not exist" about results
    that are sitting right there would be a poor introduction to the project.
    """
    path = Path(path)
    if path.exists():
        return path
    gzipped = path.with_name(path.name + ".gz")
    if gzipped.exists():
        return gzipped
    return path


def read_rows(path: str | Path) -> RowsFile:
    """Read a ``rows.jsonl`` as far as it goes, reporting the lines that failed.

    A sweep writes one JSON object per line and flushes after each, so a run
    killed mid-write leaves a final truncated line - and a full disk or a
    ``SIGKILL`` can leave a mangled one anywhere. The previous reader raised
    ``JSONDecodeError`` from the first bad line, which meant an interrupted
    sweep of a thousand good rows reported nothing at all. Every intact row is
    worth reading; the caller is given the count of the ones that were not so it
    can say so rather than quietly presenting a partial file as whole.
    """
    target = resolve_rows_path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    rows: list[dict] = []
    unreadable: list[int] = []
    number = 0
    # ``errors="replace"`` because undecodable bytes surface while the file is
    # being iterated, not from ``json.loads``: a rows file that a crash filled
    # with nulls, or simply the wrong file, otherwise raised UnicodeDecodeError
    # out of this loop and lost every intact row before it. Replacing the bytes
    # lets the damage fail one line at a time, like any other damage.
    with opener(target, "rt", errors="replace") as fh:
        try:
            for line in fh:
                number += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    unreadable.append(number)
        except (OSError, EOFError, zlib.error):
            # A corrupt or truncated gzip member raises from the decompressor
            # rather than from any one line, so there is no telling how much was
            # lost. What was read is still returned, and the line the reader
            # stopped at is marked unreadable so the caller reports damage
            # rather than presenting a partial file as a whole one.
            unreadable.append(number + 1)
    return RowsFile(rows=rows, unreadable=unreadable, path=target)


def load_results(path: str | Path) -> list[dict]:
    """Read a ``rows.jsonl`` file produced by a previous run.

    Kept as the simple form used across the package; :func:`read_rows` is the
    one to call when the caller wants to report damaged lines.
    """
    return read_rows(path).rows


def count_rows(path: str | Path) -> int:
    """Number of non-empty lines in a rows file, without parsing them.

    Damage is counted, not raised. This is what the overwrite guard asks before
    it refuses to start a sweep, and a corrupt file in the output directory is
    all the more reason to protect it: failing here would replace the refusal
    with a traceback.
    """
    target = resolve_rows_path(path)
    if not target.exists():
        return 0
    opener = gzip.open if target.suffix == ".gz" else open
    lines = 0
    with opener(target, "rt", errors="replace") as fh:
        try:
            for line in fh:
                if line.strip():
                    lines += 1
        except (OSError, EOFError, zlib.error):
            pass
    return lines
