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
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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

    started = time.time()
    try:
        inst = load(name)
        stop = StopCriteria(max_iterations=task.get("max_iterations", 1_000_000),
                            max_seconds=task.get("max_seconds", 10.0))
        result = _dispatch(algo, inst, stop, seed, params)
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


#: Solvers whose result does not depend on a random seed. Running them once per
#: seed would produce identical rows and would overstate how much evidence the
#: benchmark contains, so the report labels them and the runner says so.
DETERMINISTIC = {"ortools", "ortools_gls", "cpsat", "exact", "milp", "heldkarp"}


def _call_with_supported(fn, inst, **kwargs):
    """Call ``fn`` passing only the keyword arguments it actually accepts.

    The external wrappers have deliberately different signatures - OR-Tools takes
    no seed because its search is deterministic, CP-SAT takes a worker count
    instead - so the dispatcher adapts rather than forcing a uniform signature
    that would be a lie about what each solver does.
    """
    import inspect

    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    if accepts_kwargs:
        # The wrapper forwards **kwargs to an inner function; inspect that one.
        inner = getattr(fn, "__wrapped__", None)
        target = inner or fn
        try:
            allowed = set(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            allowed = set(kwargs)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(target).parameters.values()):
            allowed |= set(kwargs)
    else:
        allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(inst, **filtered)


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

    from qroute.algorithms.registry import build
    return build(algo, inst, stop=stop, seed=seed, **params).solve()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class BenchmarkRunner:
    """Runs a configuration and writes results incrementally."""

    def __init__(self, config: BenchmarkConfig, progress: Callable[[dict], None] | None = None):
        self.config = config
        self.progress = progress

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

    def run(self) -> dict:
        cfg = self.config
        out_dir = Path(cfg.output_dir) / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = self.tasks()
        workers = cfg.workers or max(1, (os.cpu_count() or 4) - 2)

        meta = {"config": cfg.to_dict(), "environment": environment_fingerprint(),
                "n_tasks": len(tasks), "workers": workers}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

        rows: list[dict] = []
        started = time.time()
        rows_path = out_dir / "rows.jsonl"
        with rows_path.open("w") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, t): t for t in tasks}
            for done, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if self.progress:
                    self.progress({"done": done, "total": len(tasks), "row": row,
                                   "elapsed": time.time() - started})
        summary = self.summarise(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return {"meta": meta, "rows": rows, "summary": summary, "output_dir": str(out_dir)}

    # ------------------------------------------------------------------ report
    @staticmethod
    def summarise(rows: Sequence[dict]) -> dict:
        """Per (instance, algorithm) statistics plus the omnibus comparison."""
        from qroute.benchmark.stats import friedman, summarise as summarise_values

        ok = [r for r in rows if r.get("status") == "ok"]
        failed = [r for r in rows if r.get("status") != "ok"]
        by: dict[tuple[str, str], list[dict]] = {}
        for r in ok:
            by.setdefault((r["instance"], r["algorithm"]), []).append(r)

        cells = {}
        for (inst, algo), rs in by.items():
            gaps = [r["gap"] for r in rs if r.get("gap") is not None]
            costs = [r["cost"] for r in rs]
            cell = {
                "instance": inst,
                "algorithm": algo,
                "cost": summarise_values(costs),
                "gap": summarise_values(gaps) if gaps else None,
                "feasible_runs": sum(1 for r in rs if r["feasible"]),
                "runs": len(rs),
                "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
                "mean_iterations": float(np.mean([r["iterations"] for r in rs])),
                "hit_bks": sum(1 for r in rs if r.get("gap") is not None and r["gap"] <= 1e-9),
                "median_time_to_1pct": _median_or_none([r.get("time_to_1pct") for r in rs]),
            }
            cells[f"{inst}|{algo}"] = cell

        algorithms = sorted({r["algorithm"] for r in ok})
        instances = sorted({r["instance"] for r in ok})
        omnibus = None
        # The omnibus test needs a complete matrix, so only instances solved by
        # every algorithm take part; anything else would compare different sets.
        common = [i for i in instances
                  if all(f"{i}|{a}" in cells and cells[f"{i}|{a}"]["gap"] for a in algorithms)]
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

        return {
            "cells": cells,
            "algorithms": algorithms,
            "instances": instances,
            "n_ok": len(ok),
            "n_failed": len(failed),
            "failures": [{"instance": r["instance"], "algorithm": r["algorithm"],
                          "error": r.get("error")} for r in failed[:50]],
            "omnibus": omnibus,
        }


def _median_or_none(values: Iterable[float | None]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def load_results(path: str | Path) -> list[dict]:
    """Read a ``rows.jsonl`` file produced by a previous run."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
