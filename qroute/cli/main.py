"""The ``qroute`` command line interface.

This is the entry point declared by ``[project.scripts]`` in ``pyproject.toml``,
and it is the surface a reviewer is most likely to try first. The commands are
grouped by what a user actually wants to do:

``solve`` / ``compare``
    Answer "how good is it, and is it better than the alternatives?" on a single
    benchmark instance, with the gap to the published best-known solution and a
    paired significance test rather than a single lucky run.
``bench`` / ``report``
    Run and re-read a full reproducible sweep through
    :class:`~qroute.benchmark.runner.BenchmarkRunner`. The written report and
    the figures are delegated to :mod:`qroute.benchmark.report` and
    :mod:`qroute.benchmark.plots` when those modules are installed, so a number
    quoted in the terminal is the number in the submitted document.
``exact``
    Answer "what is the true optimum?" on instances small enough to close, so
    the heuristic's gap is measured against proof rather than against folklore.
``instances``
    Say what data is actually on this machine.
``osm build`` / ``osm demo``
    Take the platform off the benchmark set and onto a real Indian city road
    network, including the incident-and-reoptimise story.
``serve`` / ``version``
    Start the HTTP API, and report exactly which versions produced a result.

Two conventions run through the file. Heavy dependencies are imported inside the
command that needs them, so ``qroute --help`` does not pay for numba, OR-Tools,
osmnx or matplotlib. And every command that involves randomness takes a
``--seed`` whose default is fixed, so two invocations of the same command line
produce the same numbers.
"""

from __future__ import annotations

import ast
import json
import math
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Optional

import typer
from rich.text import Text

from qroute.cli import render

app = typer.Typer(
    name="qroute",
    help="Quantum-inspired intelligent traffic route optimisation (SIH 2026, PS 26137).",
    no_args_is_help=True,
    add_completion=False,
)
osm_app = typer.Typer(
    name="osm",
    help="Build and solve routing instances on real road networks.",
    no_args_is_help=True,
)
app.add_typer(osm_app)

con = render.console()
err = render.console(stderr=True)

#: Solvers reachable from the CLI that are not in the algorithm registry,
#: because they own their own search loop rather than subclassing ``Optimizer``.
EXTERNAL_SOLVERS = {
    "ortools": "OR-Tools guided local search (reference solver)",
    "pyvrp": "PyVRP hybrid genetic search (state of the art)",
    "cpsat": "CP-SAT exact solver, truncated at the time limit",
    "random": "random restart control baseline",
}

#: Solvers that take no seed, so repeating them under different seeds repeats
#: the same configuration. ``solve_ortools`` has no seed argument at all: its
#: guided local search is deterministic given the instance and the budget, and
#: what little spread appears across "seeds" is the wall clock deciding how many
#: improvement passes fit in the time limit, not a different sample. Anywhere
#: the CLI reports dispersion over seeds it has to say so, or a zero standard
#: deviation reads as a remarkably stable algorithm rather than one run
#: repeated.
SEED_BLIND_SOLVERS = frozenset({"ortools", "ortools_gls"})

#: Where the API application object is expected to live. The API module is
#: developed alongside this one, so ``serve`` probes rather than assuming.
API_CANDIDATES = ("qroute.api.app:app", "qroute.api.main:app", "qroute.api:app")

#: Exit code for a run stopped by the operator. 128 + SIGINT is the shell
#: convention, and it lets a script tell "the user pressed Ctrl-C" apart from
#: "the command failed", which exit code 1 would not.
EXIT_INTERRUPTED = 130

#: Largest seed accepted. NumPy's ``SeedSequence`` takes far larger integers,
#: but a seed is also written into result files and quoted in the report, and a
#: value that does not survive a round trip through JSON's 64-bit integers is
#: not reproducible. Refusing above this is a promise the tool can keep.
MAX_SEED = 2 ** 63 - 1

#: Budgets below this are accepted but flagged. Constructing the first solution
#: on a medium instance takes tens of milliseconds, so a budget of a few
#: milliseconds measures the constructor, not the search.
SHORT_BUDGET_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _fail(message: str, hint: str = "") -> "typer.Exit":
    """Print an error the same way everywhere and exit with a non-zero code."""
    err.print(f"[bold red]error[/bold red] {message}")
    if hint:
        err.print(f"[dim]{hint}[/dim]")
    return typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Argument validation
#
# The library raises perfectly good exceptions for a nonsensical seed or budget,
# but they surface as a rich traceback through five frames of NumPy, which tells
# a user that the program broke rather than that they mistyped. Everything the
# command line accepts from a person is therefore checked at the boundary, where
# the option's own name can be quoted back.
# ---------------------------------------------------------------------------
def _check_seed(seed: int, option: str = "--seed") -> int:
    """Validate a random seed."""
    if seed < 0:
        raise _fail(f"{option} must not be negative, got {seed}",
                    "seeds index a reproducible stream, so any integer from 0 upwards")
    if seed > MAX_SEED:
        raise _fail(f"{option} must be at most {MAX_SEED}, got {seed}",
                    "larger seeds do not survive being written to the result files")
    return int(seed)


def _check_seconds(seconds: float, option: str = "--seconds") -> float:
    """Validate a wall-clock budget, warning about one too short to mean much."""
    if seconds != seconds or math.isinf(seconds):        # NaN or infinity
        raise _fail(f"{option} must be a finite number of seconds, got {seconds!r}")
    if seconds <= 0:
        raise _fail(f"{option} must be greater than zero, got {seconds:g}",
                    "a budget of zero leaves no time to search, and the answer "
                    "would be whatever the construction heuristic produced")
    if seconds < SHORT_BUDGET_SECONDS:
        err.print(f"[yellow]{option} {seconds:g} s is shorter than the time it takes to "
                  f"build a first solution on most instances; expect the result to be "
                  f"the construction heuristic rather than a search.[/yellow]")
    return float(seconds)


def _check_count(value: int, option: str, minimum: int = 1) -> int:
    """Validate a count option such as ``--seeds`` or ``--customers``."""
    if value < minimum:
        raise _fail(f"{option} must be at least {minimum}, got {value}")
    return int(value)


# ---------------------------------------------------------------------------
# Interrupts
# ---------------------------------------------------------------------------
#: Set by the handler installed in :func:`interruptible`. It exists because an
#: interrupt that lands inside a numba kernel does not arrive as a
#: ``KeyboardInterrupt``: numba's dispatcher notices that an exception is set
#: when the compiled function returns and raises ``SystemError: CPUDispatcher
#: (local_search) returned a result with an exception set`` instead. That is
#: indistinguishable from a genuine numba bug unless we remember that a signal
#: arrived, so this flag is the evidence.
_SIGINT_SEEN = False


def _interrupt_handler(signum: int, frame: Any) -> None:
    global _SIGINT_SEEN
    _SIGINT_SEEN = True
    raise KeyboardInterrupt


@contextmanager
def interruptible(activity: str, exit_code: int = EXIT_INTERRUPTED) -> Iterator[None]:
    """Turn Ctrl-C during ``activity`` into a clean message and a real exit code.

    Without this, interrupting a solve produced a full rich traceback ending in
    ``SystemError: CPUDispatcher(<function local_search ...>) returned a result
    with an exception set`` and exit code 1 - an alarming compiler-level message
    for what is simply someone changing their mind about a long run.

    The handler is installed only for the duration of the block and only in the
    main thread, so importing this module or using it from a server thread
    changes nothing about how signals behave elsewhere.
    """
    global _SIGINT_SEEN
    previous = None
    installed = False
    _SIGINT_SEEN = False
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _interrupt_handler)
        installed = True
    except ValueError:            # not the main thread; leave signals alone
        pass

    try:
        yield
    except KeyboardInterrupt:
        raise _interrupted(activity, exit_code)
    except SystemError:
        # Only a SystemError that followed a signal is ours to interpret. A
        # SystemError without one is a real defect and must keep its traceback.
        if not _SIGINT_SEEN:
            raise
        raise _interrupted(activity, exit_code)
    else:
        if _SIGINT_SEEN:
            # The interrupt arrived but never reached us. numba turns it into a
            # SystemError, and if that lands inside the optimiser's final repair
            # step - which is guarded by ``except Exception`` so a failed repair
            # cannot lose the solution - it is recorded as a note instead of
            # being re-raised. Observed once in five interrupts of a large
            # instance: a full result table and exit code 0 for a run the
            # operator had cancelled. What is on screen came from a search that
            # was stopped early, so it is not reported as a finished run.
            raise _interrupted(activity, exit_code)
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)


def _interrupted(activity: str, exit_code: int) -> "typer.Exit":
    err.print(f"\n[bold yellow]interrupted[/bold yellow] {activity} was stopped before "
              f"it finished; no result is reported.")
    return typer.Exit(code=exit_code)


def _parse_params(items: Optional[list[str]]) -> dict[str, Any]:
    """Turn repeated ``--params key=value`` options into a keyword dictionary.

    Values are parsed as Python literals when possible, so ``swarm_size=60``
    arrives as an integer and ``local_search=false`` as a boolean, and are kept
    as strings otherwise (``beta_schedule=linear``).
    """
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise _fail(f"malformed parameter {item!r}", "expected key=value, for example swarm_size=60")
        key, _, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        lowered = raw.lower()
        if lowered in ("true", "false"):
            out[key] = lowered == "true"
            continue
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def _anchor_data_root() -> None:
    """Point the data lookup at the installation rather than the shell's cwd.

    :mod:`qroute.problems.loaders` and :mod:`qroute.graph.osm` read
    ``QROUTE_DATA`` when they are imported and default it to the relative path
    ``"data"``, so every command that reads an instance used to work only when
    it was run from inside the checkout. :mod:`qroute.config` resolves an
    absolute root and publishes it, which is what the HTTP service already does;
    calling it here, before those modules are imported, gives the command line
    the same answer.

    It has to be the *first* statement of a command that reads data, before that
    command's own imports: ``qroute osm build`` pulls in the graph package on its
    first line, and by then the directory constant has already been computed.
    Calling it per command rather than from a Typer callback keeps that ordering
    visible at each call site, and keeps the commands that read no data - and so
    need no data root - from failing on a misconfigured one.
    """
    try:
        from qroute.config import ConfigError, settings
    except ImportError:            # the settings module is optional to the CLI
        return
    try:
        settings().export_to_environment()
    except ConfigError as exc:
        raise _fail(str(exc))


def _load_instance(name: str):
    """Load a benchmark instance by name or path, with a useful error message."""
    from qroute.problems.loaders import DATA_ROOT, list_instances, load

    try:
        return load(name)
    except FileNotFoundError:
        available = list_instances()
        pool = available["cvrp"] + available["vrptw"]
        if not pool:
            # Nothing at all was found, which is a different mistake from
            # mistyping a name and deserves a different answer: the data
            # directory itself is missing or is not where it is being looked for.
            raise _fail(
                f"no benchmark instances were found under {Path(DATA_ROOT).resolve()}",
                "that directory should contain benchmarks/cvrplib; set QROUTE_DATA "
                "to the data directory of a qroute checkout if it lives elsewhere",
            )
        close = [n for n in pool if n.lower().startswith(name.lower()[:3])][:8]
        raise _fail(
            f"instance {name!r} was not found",
            ("did you mean: " + ", ".join(close)) if close else
            "run `qroute instances` to see what is available locally",
        )


def _solve_once(algorithm: str, instance, seconds: float, seed: int,
                params: dict[str, Any], callback: Optional[Callable] = None,
                max_iterations: int = 1_000_000):
    """Run one solver on one instance and return an ``OptimizationResult``.

    The registry covers the metaheuristics; the external solvers each have a
    different signature (OR-Tools is deterministic and takes no seed, CP-SAT
    takes a time limit rather than an iteration budget) and are adapted here
    rather than forced into a uniform interface that would misrepresent them.
    """
    from qroute.algorithms.base import StopCriteria
    from qroute.algorithms.registry import ALGORITHMS, build

    key = algorithm.strip().lower()
    stop = StopCriteria(max_iterations=max_iterations, max_seconds=seconds)

    if key in ALGORITHMS:
        return build(key, instance, stop=stop, seed=seed, callback=callback, **params).solve()
    if key in ("ortools", "ortools_gls"):
        from qroute.baselines.ortools_gls import solve_ortools
        return solve_ortools(instance, seconds=seconds, **params)
    if key in ("pyvrp", "hgs"):
        from qroute.baselines.pyvrp_hgs import solve_pyvrp
        return solve_pyvrp(instance, seconds=seconds, seed=seed, **params)
    if key in ("cpsat", "exact"):
        from qroute.exact.cpsat import solve_cpsat
        return solve_cpsat(instance, time_limit=seconds, seed=seed, **params)
    if key in ("random", "restart"):
        from qroute.benchmark.reference import RandomRestart
        return RandomRestart(instance, stop, seed, callback=callback, **params).solve()

    raise _fail(
        f"unknown algorithm {algorithm!r}",
        "available: " + ", ".join(list(ALGORITHMS) + list(EXTERNAL_SOLVERS)),
    )


def _warm(algorithm: str) -> None:
    """Compile the solver's kernels before any clock starts, and say what happened.

    numba compiles on first call, so on a machine whose cache is cold the first
    timed run spends its entire budget inside the compiler and then reports the
    construction heuristic's tour as its answer. Measured on this project:
    ``qroute solve A-n32-k5 -t 5`` with an empty cache finished zero iterations
    and ran for 11.7 wall-clock seconds. Warming first costs the same time but
    spends it honestly, outside the budget, and once per machine.
    """
    from qroute.benchmark.runner import warm_kernels

    with con.status(f"preparing the compiled kernels for {algorithm}"):
        warm = warm_kernels(algorithm)
    if warm.error:
        err.print(f"[yellow]the compiled kernels could not be warmed "
                  f"({warm.error}). The run below pays for compilation out of its "
                  f"own time budget, so its iteration count and its wall clock do "
                  f"not mean what they usually mean.[/yellow]")
    elif warm.seconds >= 1.0:
        con.print(f"[dim]compiled the search kernels in "
                  f"{render.format_seconds(warm.seconds)}; this happens once per "
                  f"machine and is not counted against the budget[/dim]")


def _check_budget_was_spent_searching(result: Any, seconds: float, elapsed: float) -> None:
    """Say so when a run did not actually search, instead of quietly reporting it.

    Zero iterations means the answer on screen is whatever the constructor
    built. That can happen after a failed warm-up, on a budget of a few
    milliseconds, or on a machine so loaded that nothing got a turn. Whatever
    the cause, presenting it as a result of the optimiser would be a false claim.
    """
    if getattr(result, "iterations", 0) <= 0:
        err.print("[bold yellow]no search iteration completed within the budget[/bold yellow]"
                  " - the solution above is the construction heuristic's, not the "
                  "optimiser's. Give it more time with --seconds.")
    elif elapsed > max(seconds * 1.5, seconds + 2.0):
        err.print(f"[yellow]the run took {render.format_seconds(elapsed)} for a "
                  f"{render.format_seconds(seconds)} budget; the excess was spent "
                  f"outside the search loop (compilation or start-up), so compare "
                  f"the iteration count rather than the wall clock.[/yellow]")


def _progress(transient: bool = True):
    """A progress display with a spinner, a bar and an elapsed clock."""
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn)

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.completed:>4.0f}/{task.total:.0f}"),
        TimeElapsedColumn(),
        console=con,
        transient=transient,
    )


def _summarise(rows: list[dict]) -> dict:
    """Summarise benchmark rows, tolerating runs that returned no solution.

    :meth:`BenchmarkRunner.summarise` is the authority and is tried first. It
    raises ``KeyError: 'median'`` when a run reports an infinite cost, which
    happens for real: OR-Tools on a short budget returns no feasible solution at
    all for some Solomon instances, the gap is then infinity, and the summary's
    gap block collapses to ``{"n": 0}``. Losing a finished sweep to that would
    be worse than reporting it, so this falls back to the same tables minus the
    omnibus test, and the caller says loudly which runs found nothing.
    """
    from qroute.benchmark.runner import BenchmarkRunner
    from qroute.benchmark.stats import summarise as summarise_values

    try:
        return BenchmarkRunner.summarise(rows)
    except KeyError:
        pass

    import numpy as np

    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        grouped.setdefault((r["instance"], r["algorithm"]), []).append(r)

    cells: dict[str, dict] = {}
    for (inst, algo), rs in grouped.items():
        gaps = [r["gap"] for r in rs
                if r.get("gap") is not None and math.isfinite(r["gap"])]
        gap_block = summarise_values(gaps) if gaps else None
        if gap_block is not None and "median" not in gap_block:
            gap_block = None
        cells[f"{inst}|{algo}"] = {
            "instance": inst, "algorithm": algo,
            "cost": summarise_values([r["cost"] for r in rs]),
            "gap": gap_block,
            "feasible_runs": sum(1 for r in rs if r.get("feasible")),
            "runs": len(rs),
            "no_solution_runs": len(rs) - len(gaps) if rs and rs[0].get("bks") else 0,
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
            "mean_iterations": float(np.mean([r["iterations"] for r in rs])),
            "hit_bks": sum(1 for r in rs if r.get("gap") is not None and r["gap"] <= 1e-9),
            "median_time_to_1pct": None,
        }
    return {
        "cells": cells,
        "algorithms": sorted({r["algorithm"] for r in ok}),
        "instances": sorted({r["instance"] for r in ok}),
        "n_ok": len(ok), "n_failed": len(failed),
        "failures": [{"instance": r["instance"], "algorithm": r["algorithm"],
                      "error": r.get("error")} for r in failed[:50]],
        "omnibus": None,
        "degraded": True,
    }


def _report_missing_solutions(rows: list[dict]) -> None:
    """Say plainly which runs produced no solution at all, if any did."""
    lost = [r for r in rows
            if r.get("status") == "ok" and not math.isfinite(float(r.get("cost", 0.0)))]
    if not lost:
        return
    con.print(f"[bold yellow]{len(lost)} runs returned no solution "
              f"(infinite cost) and are excluded from the gap statistics:[/bold yellow]")
    for r in lost[:10]:
        con.print(f"  [yellow]{r['instance']} / {r['algorithm']} (seed {r['seed']})[/yellow]")


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=float))
    con.print(f"[dim]wrote[/dim] {path}")


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
@app.command()
def solve(
    instance: Annotated[str, typer.Argument(help="Instance name (A-n32-k5) or path to a file.")],
    algorithm: Annotated[str, typer.Option(
        "--algorithm", "-a",
        help="qpso, pso, ga, sa, aco, ortools, pyvrp, cpsat or random.")] = "qpso",
    seconds: Annotated[float, typer.Option("--seconds", "-t", help="Wall-clock budget.")] = 10.0,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    params: Annotated[Optional[list[str]], typer.Option("--params", "-p",
                      help="Algorithm parameter as key=value; repeatable.")] = None,
    iterations: Annotated[int, typer.Option("--iterations",
                          help="Iteration cap; the time budget usually binds first.")] = 1_000_000,
    json_out: Annotated[Optional[Path], typer.Option("--json",
                        help="Write the solution and convergence history here.")] = None,
    show_routes: Annotated[bool, typer.Option("--routes/--no-routes",
                           help="Print the per-route breakdown.")] = True,
) -> None:
    """Solve one instance and report cost, gap, feasibility and throughput."""
    _anchor_data_root()
    seed = _check_seed(seed)
    seconds = _check_seconds(seconds)
    iterations = _check_count(iterations, "--iterations")
    inst = _load_instance(instance)
    kwargs = _parse_params(params)
    _warm(algorithm)

    state = {"best": float("inf"), "iteration": 0}
    with _progress() as progress:
        task = progress.add_task(f"{algorithm} on {inst.name}", total=seconds)

        def on_iteration(record) -> None:
            state["best"] = record.best_cost
            state["iteration"] = record.iteration
            progress.update(task, completed=min(record.elapsed, seconds),
                            description=f"{algorithm} on {inst.name}  best {record.best_cost:,.2f}")

        started = time.perf_counter()
        with interruptible(f"{algorithm} on {inst.name}"):
            result = _solve_once(algorithm, inst, seconds, seed, kwargs, on_iteration, iterations)
        elapsed = time.perf_counter() - started
        progress.update(task, completed=seconds)

    con.print(render.solve_table(result, inst, elapsed))
    _check_budget_was_spent_searching(result, seconds, elapsed)
    if result.history:
        line, first, last = render.convergence_line(result.history)
        con.print(f"\n[bold]convergence[/bold] {line}")
        con.print(f"[dim]{first:,.2f} -> {last:,.2f} over {len(result.history)} recorded "
                  f"iterations[/dim]")
    if show_routes and result.best.routes:
        con.print()
        con.print(render.routes_table(inst, result.best))

    if json_out is not None:
        payload = result.to_json()
        payload["bks"] = inst.meta.get("bks")
        payload["gap"] = result.gap_to(inst.meta["bks"]) if inst.meta.get("bks") else None
        _write_json(json_out, payload)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
@app.command()
def compare(
    instance: Annotated[str, typer.Argument(help="Instance name or path.")],
    algorithms: Annotated[str, typer.Option("--algorithms", "-a",
                          help="Comma-separated list; the first is the control.")] = "qpso,pso,ga,sa,ortools",
    seeds: Annotated[int, typer.Option("--seeds", "-n", help="Independent runs per algorithm.")] = 5,
    seconds: Annotated[float, typer.Option("--seconds", "-t", help="Budget per run.")] = 10.0,
    master_seed: Annotated[int, typer.Option("--master-seed",
                           help="Seed the per-run seeds are derived from.")] = 20260920,
    json_out: Annotated[Optional[Path], typer.Option("--json", help="Write the raw runs here.")] = None,
) -> None:
    """Run several algorithms on one instance and test whether they differ.

    Every algorithm gets the same seeds and the same wall-clock budget, so the
    runs are paired and the Wilcoxon signed-rank test against the first
    algorithm is the appropriate one. With the small number of seeds a
    command-line comparison can afford, treat the test as an indication and the
    full ``bench`` sweep as the evidence.
    """
    _anchor_data_root()

    import numpy as np

    from qroute.benchmark.stats import wilcoxon
    from qroute.core.rng import spawn_seeds

    seeds = _check_count(seeds, "--seeds")
    seconds = _check_seconds(seconds)
    master_seed = _check_seed(master_seed, "--master-seed")
    inst = _load_instance(instance)
    names = [a.strip() for a in algorithms.split(",") if a.strip()]
    if not names:
        raise _fail("no algorithms given")
    run_seeds = spawn_seeds(master_seed, seeds)
    bks = inst.meta.get("bks")
    for name in names:
        _warm(name)

    runs: dict[str, list[dict]] = {name: [] for name in names}
    with _progress() as progress:
        task = progress.add_task("comparing", total=len(names) * seeds)
        for name in names:
            for k, seed in enumerate(run_seeds):
                progress.update(task, description=f"{name} seed {k + 1}/{seeds}")
                with interruptible(f"the comparison on {inst.name}"):
                    result = _solve_once(name, inst, seconds, seed, {})
                runs[name].append({
                    "algorithm": name, "seed": int(seed), "seed_index": k,
                    "cost": float(result.best.cost),
                    "gap": float(result.gap_to(bks)) if bks else None,
                    "n_routes": int(result.best.n_routes),
                    "feasible": bool(result.best.is_feasible),
                    "iterations": int(result.iterations),
                    "evaluations": int(result.evaluations),
                    "seconds": float(result.seconds),
                })
                progress.advance(task)

    control = names[0]
    control_scores = [r["gap"] if bks else r["cost"] for r in runs[control]]
    entries = []
    for name in names:
        rs = runs[name]
        costs = np.array([r["cost"] for r in rs], dtype=float)
        gaps = np.array([r["gap"] for r in rs], dtype=float) if bks else np.array([])
        scores = list(gaps) if bks else list(costs)
        if name == control:
            test = None
            detail = None
        else:
            w = wilcoxon(control_scores, scores, (control, name))
            # A compact cell, not w.describe(): the full sentence is about fifty
            # characters and the table has nine other columns, so off a wide
            # terminal rich folds it one character to a line and the row becomes
            # unreadable. The sentence is kept in the JSON output instead.
            verdict = "=" if w.p_value > 0.05 else (
                "control" if w.winner == control else name)
            test = Text(f"p {render.format_p(w.p_value)}  {verdict}")
            if w.winner and w.p_value <= 0.05:
                test.stylize("green" if w.winner == control else "yellow")
            detail = {"p_value": float(w.p_value), "n": int(w.n),
                      "winner": w.winner, "effect_size": float(w.effect_size),
                      "text": w.describe()}
        entries.append({
            "algorithm": name,
            "runs": len(rs),
            "cost_best": float(costs.min()),
            "cost_mean": float(costs.mean()),
            "cost_std": float(costs.std(ddof=1)) if costs.size > 1 else 0.0,
            "gap_best": float(gaps.min()) if gaps.size else None,
            "gap_mean": float(gaps.mean()) if gaps.size else None,
            "gap_std": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
            "routes_mean": float(np.mean([r["n_routes"] for r in rs])),
            "feasible_runs": sum(1 for r in rs if r["feasible"]),
            "iterations_per_second": float(np.mean([r["iterations"] / max(r["seconds"], 1e-9)
                                                    for r in rs])),
            "test": test,
            "test_detail": detail,
        })

    con.print(render.compare_table(entries, inst.name, bks, control))
    con.print(f"[dim]{seconds:g} s per run, seeds derived from master seed {master_seed}; "
              f"lower is better.[/dim]")
    # The exact two-sided signed-rank test on n pairs cannot report a p-value
    # below 2^(1-n), whatever the data: with five seeds the floor is 0.0625, so
    # every comparison is doomed to read "no significant difference" and a
    # reader could mistake that for evidence of equivalence. Say plainly that
    # the test had no power rather than letting the column imply a finding.
    blind = [n for n in names if n.strip().lower() in SEED_BLIND_SOLVERS]
    if blind and seeds > 1:
        con.print(f"[yellow]{', '.join(blind)} take no seed, so the {seeds} runs behind that row "
                  f"are the same configuration repeated; its spread is wall-clock noise in how "
                  f"many passes fit the budget, not variation over seeds.[/yellow]")
    if len(names) > 1 and seeds >= 1:
        floor = 2.0 ** (1 - seeds)
        if floor > 0.05:
            con.print(f"[yellow]With {seeds} seeds the signed-rank test cannot return a p-value "
                      f"below {floor:.4g}, so no comparison here can reach significance at the "
                      f"0.05 level. Use --seeds 6 or more, or read the full `qroute bench` "
                      f"sweep, before concluding that two algorithms are equivalent.[/yellow]")
    if json_out is not None:
        _write_json(json_out, {
            "instance": inst.name, "bks": bks, "seconds": seconds,
            "master_seed": master_seed, "seeds": [int(s) for s in run_seeds],
            "runs": {k: v for k, v in runs.items()},
            "summary": [{k: v for k, v in e.items() if k != "test"} for e in entries],
        })


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
@app.command()
def bench(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML benchmark configuration.")],
    seeds: Annotated[Optional[int], typer.Option("--seeds", help="Override the seed count.")] = None,
    seconds: Annotated[Optional[float], typer.Option("--seconds",
                       help="Override the per-run budget.")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", help="Override the output directory.")] = None,
    workers: Annotated[Optional[int], typer.Option("--workers", help="Override the worker count.")] = None,
    name: Annotated[Optional[str], typer.Option("--name", help="Override the run name.")] = None,
    force: Annotated[bool, typer.Option("--force/--no-force",
                     help="Overwrite results already in the output directory.")] = False,
) -> None:
    """Run a full benchmark sweep and write reproducible results to disk.

    A sweep refuses to write into a directory that already holds results.
    Re-running a configuration out of habit used to empty the previous rows file
    before the first new run had finished, so interrupting the mistake left
    neither set of results. Give the new run its own --name, or pass --force,
    which renames the old rows aside rather than deleting them.
    """
    _anchor_data_root()
    from qroute.benchmark.runner import (BenchmarkConfig, BenchmarkRunner,
                                         ExistingResults, resolve_workers)

    if not Path(config).exists():
        raise _fail(f"configuration {config} does not exist",
                    "the bundled configurations are in configs/")
    try:
        cfg = BenchmarkConfig.from_yaml(config)
    except ValueError as exc:
        raise _fail(str(exc))
    if seeds is not None:
        cfg.seeds = _check_count(seeds, "--seeds")
    if seconds is not None:
        cfg.max_seconds = _check_seconds(seconds)
    if out is not None:
        cfg.output_dir = str(out)
    if workers is not None:
        cfg.workers = _check_count(workers, "--workers")
    if name is not None:
        cfg.name = name
    cfg.master_seed = _check_seed(cfg.master_seed, "master_seed")

    plan = resolve_workers(cfg.workers)
    n_tasks = len(cfg.instances) * len(cfg.algorithms) * cfg.seeds
    con.print(render.kv_table("benchmark", [
        ("name", cfg.name),
        ("instances", f"{len(cfg.instances)}"),
        ("algorithms", ", ".join(cfg.algorithms)),
        ("seeds", str(cfg.seeds)),
        ("budget per run", render.format_seconds(cfg.max_seconds)),
        ("runs", str(n_tasks)),
        ("serial cost", render.format_seconds(n_tasks * cfg.max_seconds)),
        ("workers", f"{plan.workers} on {plan.cpus} usable cores"),
        # Absolute, because output_dir in the configuration file is relative to
        # wherever the command was run from and "results/runs/main" on screen
        # would not say which one.
        ("output", str((Path(cfg.output_dir) / cfg.name).resolve())),
    ]))
    if plan.warning:
        err.print(f"[yellow]{plan.warning}[/yellow]")

    runner = BenchmarkRunner(cfg, force=force)
    try:
        # Asked before the progress bar is on screen, so the refusal is the last
        # thing printed rather than being followed by an empty "0/8 running".
        runner.check_output_dir()
    except ExistingResults as exc:
        raise _fail(
            str(exc),
            f"pick another run with --name, another directory with --out, or pass "
            f"--force to move {exc.path / 'rows.jsonl'} aside and start again",
        )

    started = time.perf_counter()
    with _progress(transient=False) as progress:
        task = progress.add_task("running", total=n_tasks)

        def on_progress(event: dict) -> None:
            row = event["row"]
            gap = row.get("gap")
            tail = f"gap {gap:+.2f}%" if gap is not None else row.get("status", "")
            progress.update(task, completed=event["done"],
                            description=f"{row['instance']} / {row['algorithm']}  {tail}")

        interrupted = False
        runner.progress = on_progress
        try:
            result = runner.run()
            rows, summary = result["rows"], result["summary"]
            out_dir = Path(result["output_dir"])
            interrupted = bool(result.get("interrupted"))
        except KeyError:
            # See _summarise: an infinite cost breaks the runner's own summary
            # after every run has already been written to disk. Recover the
            # sweep rather than throwing away the compute it cost.
            from qroute.benchmark.runner import load_results
            out_dir = Path(cfg.output_dir) / cfg.name
            rows = load_results(out_dir / "rows.jsonl")
            summary = _summarise(rows)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    con.print()
    if interrupted:
        err.print(f"[bold yellow]interrupted[/bold yellow] the sweep was stopped after "
                  f"{len(rows)} of {n_tasks} runs. Everything below covers only those "
                  f"runs, and they are on disk in {out_dir}.")
        if not rows:
            # Nothing finished, so there is nothing to tabulate; empty tables
            # would only look like a result.
            err.print("[yellow]no run finished, so there is nothing to report.[/yellow]")
            raise typer.Exit(code=EXIT_INTERRUPTED)
    dead = [r for r in rows if r.get("status") == "worker_died"]
    if dead:
        err.print(f"[bold yellow]{len(dead)} runs lost their worker process[/bold yellow] "
                  f"and produced no result; the rest of the sweep continued. They are "
                  f"listed in the failures table below.")
    if summary.get("degraded"):
        con.print("[yellow]The omnibus test was skipped: at least one run found no "
                  "solution, so the algorithms are not scored on a common set of "
                  "instances.[/yellow]")
    _report_missing_solutions(rows)
    con.print()
    con.print(render.summary_table(summary))
    con.print()
    con.print(render.cell_table(summary))
    if summary.get("omnibus"):
        con.print()
        con.print(render.omnibus_table(summary["omnibus"]))
    failures = render.failure_table(summary)
    if failures is not None:
        con.print()
        con.print(failures)
    con.print(f"\n[bold]{summary['n_ok']}[/bold] runs completed in "
              f"{render.format_seconds(time.perf_counter() - started)}; "
              f"results in [bold]{out_dir}[/bold]")
    con.print(f"[dim]qroute report {out_dir} --plots[/dim]")
    if interrupted:
        # A partial sweep is a useful artefact but it is not the run that was
        # asked for, and a script driving this must be able to tell.
        raise typer.Exit(code=EXIT_INTERRUPTED)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@app.command()
def report(
    result_dir: Annotated[Path, typer.Argument(help="Directory written by `qroute bench`.")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="table, markdown or csv.")] = "table",
    plots: Annotated[bool, typer.Option("--plots/--no-plots",
                     help="Also write convergence and gap figures.")] = False,
    out: Annotated[Optional[Path], typer.Option("--out",
                   help="Write the markdown or csv output to this file.")] = None,
) -> None:
    """Turn a saved benchmark run into tables, text and figures.

    A sweep that was killed leaves a rows file whose last line is half-written.
    Every intact row is still evidence, so this reads as far as it can and says
    how many lines it could not parse, rather than refusing to report a thousand
    good runs because of one bad one. The gzipped rows file that ships with the
    repository is read directly when the uncompressed one is absent.
    """
    from qroute.benchmark.runner import read_rows, resolve_rows_path

    result_dir = Path(result_dir)
    rows_path = resolve_rows_path(result_dir / "rows.jsonl")
    if not rows_path.exists():
        raise _fail(f"{result_dir / 'rows.jsonl'} does not exist",
                    "point this at the directory a `qroute bench` run wrote")
    rows_file = read_rows(rows_path)
    rows = rows_file.rows
    if rows_file.unreadable:
        first = ", ".join(str(n) for n in rows_file.unreadable[:5])
        more = "" if len(rows_file.unreadable) <= 5 else ", ..."
        err.print(f"[yellow]{len(rows_file.unreadable)} of "
                  f"{len(rows) + len(rows_file.unreadable)} lines in {rows_path.name} "
                  f"could not be parsed (line{'s' if len(rows_file.unreadable) > 1 else ''} "
                  f"{first}{more}) and are not included below. This is what an "
                  f"interrupted sweep leaves behind; the {len(rows)} intact runs are "
                  f"reported.[/yellow]")
    if not rows:
        raise _fail(f"{rows_path} contains no readable runs")
    meta_path = result_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    summary_path = result_dir / "summary.json"
    # A stored summary is only quoted when it describes the rows actually read.
    # Printing a summary computed over runs that are no longer readable, beside
    # tables built from the ones that are, would put two different populations
    # on the same screen without saying so.
    if summary_path.exists() and rows_file.complete:
        summary = json.loads(summary_path.read_text())
    else:
        summary = _summarise(rows)

    fmt = fmt.lower()
    if fmt == "table":
        if meta:
            cfg = meta.get("config", {})
            env = meta.get("environment", {})
            con.print(render.kv_table(f"run: {cfg.get('name')}", [
                ("instances", str(len(cfg.get("instances", [])))),
                ("algorithms", ", ".join(cfg.get("algorithms", []))),
                ("seeds", str(cfg.get("seeds"))),
                ("budget per run", render.format_seconds(cfg.get("max_seconds"))),
                ("master seed", str(cfg.get("master_seed"))),
                ("python", str(env.get("python"))),
                ("platform", str(env.get("platform"))),
                ("commit", (env.get("git_commit") or "unknown")[:12]
                 + (" (dirty)" if env.get("git_dirty") else "")),
            ]))
            con.print()
        con.print(render.summary_table(summary))
        con.print()
        con.print(render.cell_table(summary))
        if summary.get("omnibus"):
            con.print()
            con.print(render.omnibus_table(summary["omnibus"]))
        failures = render.failure_table(summary)
        if failures is not None:
            con.print()
            con.print(failures)
        _report_missing_solutions(rows)
    elif fmt in ("markdown", "md"):
        # The benchmark package owns the canonical report tables, so the written
        # submission and the terminal quote the same numbers. The CLI keeps its
        # own markdown only as a fallback for a checkout without that module.
        try:
            from qroute.benchmark.report import build_report
        except ImportError:
            build_report = None
        if build_report is not None and out is None:
            built = build_report(rows, result_dir, summary=summary, meta=meta)
            written = built.get("files") or built.get("paths") or []
            # This notice goes to stderr, not stdout: the whole point of the
            # markdown format is that `qroute report --format markdown > x.md`
            # produces a file that can be pasted into the submission, and a
            # status line printed ahead of the document leaves a first line that
            # is not markdown at all.
            err.print(f"[dim]wrote {len(written) or 'the'} report files under[/dim] {result_dir}")
            report_md = Path(result_dir) / "report.md"
            if report_md.exists():
                print(report_md.read_text())
        else:
            text = render.markdown_report(summary, meta)
            if out is not None:
                Path(out).write_text(text)
                err.print(f"[dim]wrote[/dim] {out}")
            else:
                print(text)
    elif fmt == "csv":
        text = render.csv_rows(rows)
        if out is not None:
            Path(out).write_text(text)
            con.print(f"[dim]wrote[/dim] {out}")
        else:
            sys.stdout.write(text)
    else:
        raise _fail(f"unknown format {fmt!r}", "use table, markdown or csv")

    if plots:
        # Same reasoning as for the markdown: the figures in the submission come
        # from qroute.benchmark.plots when it is present.
        try:
            from qroute.benchmark.plots import all_plots
        except ImportError:
            all_plots = None
        if all_plots is not None:
            paths = all_plots(rows, result_dir / "figures")
        else:
            paths = render.write_plots(rows, summary, result_dir / "figures")
        if paths:
            for p in paths:
                con.print(f"[dim]wrote[/dim] {p}")
        else:
            con.print("[yellow]no figures written: the run has no history and no gaps[/yellow]")


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------
@app.command()
def exact(
    instance: Annotated[str, typer.Argument(help="Instance name or path.")],
    seconds: Annotated[float, typer.Option("--seconds", "-t", help="Time limit.")] = 60.0,
    method: Annotated[str, typer.Option("--method", "-m", help="cpsat, milp or heldkarp.")] = "cpsat",
    bounds: Annotated[bool, typer.Option("--bounds/--no-bounds",
                      help="Also compute the combinatorial lower bounds.")] = True,
    workers: Annotated[int, typer.Option("--workers", help="Solver threads (CP-SAT and MILP).")] = 8,
    json_out: Annotated[Optional[Path], typer.Option("--json", help="Write the outcome here.")] = None,
) -> None:
    """Try to solve an instance to proven optimality and report the bracket.

    "Proved" here means the solver closed the gap between its incumbent and its
    own dual bound. When the time limit intervenes first the command says so and
    prints the surviving gap rather than presenting the incumbent as optimal.
    """
    _anchor_data_root()
    seconds = _check_seconds(seconds)
    workers = _check_count(workers, "--workers")
    inst = _load_instance(instance)
    key = method.strip().lower()
    # The subset dynamic program has no notion of a time limit: it either fits
    # in memory and runs to completion, or it refuses. Saying "limit 60 s" there
    # would be a claim about a knob that does not exist.
    limit = "no time limit (runs to completion)" if key in ("heldkarp", "dp") else f"limit {seconds:g} s"
    con.print(f"[dim]{inst.name}: {inst.n_customers} customers, capacity "
              f"{inst.capacity:g}, {limit}[/dim]")

    with con.status(f"{key} on {inst.name}"), interruptible(f"{key} on {inst.name}"):
        started = time.perf_counter()
        if key == "cpsat":
            from qroute.exact.cpsat import solve_cvrp_cpsat
            raw = solve_cvrp_cpsat(inst, time_limit=seconds, workers=workers)
        elif key == "milp":
            from qroute.exact.milp import solve_cvrp_milp
            raw = solve_cvrp_milp(inst, time_limit=seconds, threads=workers)
        elif key in ("heldkarp", "dp"):
            from qroute.exact.heldkarp import held_karp_cvrp
            raw = held_karp_cvrp(inst)
        else:
            raise _fail(f"unknown method {method!r}", "use cpsat, milp or heldkarp")
        wall = time.perf_counter() - started

    bks = inst.meta.get("bks")
    # Only CP-SAT publishes a gap of its own; for the others it is the distance
    # between the incumbent and the dual bound the method itself produced.
    if hasattr(raw, "gap"):
        remaining = float(raw.gap)
    elif raw.cost > 0 and math.isfinite(raw.cost):
        remaining = 100.0 * (raw.cost - raw.lower_bound) / raw.cost
    else:
        remaining = None
    outcome = {
        "instance": inst.name,
        "method": key,
        "status": raw.status,
        "cost": float(raw.cost),
        "lower_bound": float(raw.lower_bound),
        "proven_optimal": bool(raw.proven_optimal),
        "gap": remaining,
        "n_vehicles": int(getattr(raw, "n_vehicles", 0)) or len(raw.routes),
        "seconds": float(raw.seconds) if raw.seconds else wall,
        "bks": float(bks) if bks else None,
        "gap_to_bks": (100.0 * (raw.cost - bks) / bks) if bks and raw.cost < float("inf") else None,
        "routes": [list(r) for r in raw.routes],
    }
    con.print(render.exact_table(inst.name, key, outcome))

    if raw.proven_optimal and bks and abs(raw.cost - bks) < 1e-6:
        con.print("[green]The proved optimum equals the published best-known solution.[/green]")
    elif raw.proven_optimal and bks:
        con.print(f"[yellow]The proved optimum differs from the published value "
                  f"({raw.cost:.2f} against {bks:.2f}); check the rounding convention.[/yellow]")

    if bounds:
        from qroute.exact.bounds import bracket
        with con.status("computing lower bounds"):
            upper = float(raw.cost) if raw.cost < float("inf") else (float(bks) if bks else None)
            report_ = bracket(inst, upper_bound=upper)
        con.print()
        con.print(render.bounds_table(report_))
        outcome["bounds"] = report_.as_dict()

    if json_out is not None:
        _write_json(json_out, outcome)


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
@app.command()
def instances(
    family: Annotated[Optional[str], typer.Option("--family", "-f",
                      help="cvrp or vrptw; both when omitted.")] = None,
    max_size: Annotated[Optional[int], typer.Option("--max-size",
                        help="Only instances with at most this many customers.")] = None,
    with_bks: Annotated[bool, typer.Option("--with-bks/--all",
                        help="Only instances that ship a best-known solution.")] = False,
) -> None:
    """List the benchmark instances available on this machine."""
    _anchor_data_root()
    from qroute.problems.loaders import DATA_ROOT, list_instances, load

    available = list_instances()
    wanted = ["cvrp", "vrptw"] if family is None else [family.strip().lower()]
    for f in wanted:
        if f not in available:
            raise _fail(f"unknown family {f!r}", "use cvrp or vrptw")
    if not any(available.values()):
        raise _fail(
            f"no benchmark instances were found under {Path(DATA_ROOT).resolve()}",
            "that directory should contain benchmarks/cvrplib; set QROUTE_DATA to "
            "the data directory of a qroute checkout if it lives elsewhere",
        )

    total = sum(len(available[f]) for f in wanted)
    rows_by_family: dict[str, list[dict]] = {f: [] for f in wanted}
    with _progress() as progress:
        task = progress.add_task("reading instances", total=total)
        for f in wanted:
            for name in available[f]:
                progress.update(task, description=f"reading {name}")
                progress.advance(task)
                try:
                    inst = load(name)
                except Exception as exc:            # a broken file must not hide the rest
                    err.print(f"[yellow]skipping {name}: {exc}[/yellow]")
                    continue
                if max_size is not None and inst.n_customers > max_size:
                    continue
                if with_bks and not inst.meta.get("bks"):
                    continue
                rows_by_family[f].append({
                    "name": name,
                    "n_customers": inst.n_customers,
                    "capacity": float(inst.capacity),
                    "reference_k": inst.meta.get("reference_k") or inst.meta.get("bks_routes"),
                    "bks": inst.meta.get("bks"),
                    "time_windows": inst.has_time_windows,
                })

    for f in wanted:
        rows = rows_by_family[f]
        if not rows:
            con.print(f"[yellow]no {f} instances matched[/yellow]")
            continue
        con.print(render.instances_table(rows, f))
        con.print()


# ---------------------------------------------------------------------------
# osm
# ---------------------------------------------------------------------------
def _load_network(place: str):
    """Load a GraphML road network and wrap it in a :class:`RoadNetwork`."""
    from qroute.graph.network import RoadNetwork
    from qroute.graph.osm import OSM_DIR, list_graphs, load_graph

    path = Path(place)
    if not path.exists():
        known = list_graphs()
        if place in known:
            # Resolved against the graph module's own directory rather than a
            # literal "data/osm", which would only be right when the command was
            # run from inside the checkout.
            path = OSM_DIR / f"{place}.graphml"
        else:
            raise _fail(
                f"road network {place!r} was not found",
                ("available: " + ", ".join(known)) if known else
                f"no road graphs under {OSM_DIR}; run `qroute osm fetch` to build them",
            )
    graph = load_graph(path)
    return RoadNetwork(graph, name=path.stem)


def _network_table(network) -> Any:
    """The size and shape of a road network, as the graph component reports it."""
    summary = network.summary()
    rows = [("nodes", f"{network.n_nodes:,}"), ("edges", f"{network.n_edges:,}")]
    for key, label, fmt in (("total_length_km", "total length", "{:,.1f} km"),
                            ("mean_free_flow_kph", "mean free-flow speed", "{:.1f} km/h"),
                            ("mean_congestion", "mean congestion level", "{:.3f}")):
        if key in summary:
            rows.append((label, fmt.format(summary[key])))
    return render.kv_table(f"road network: {network.name}", rows)


def _congestion_summary(state: dict) -> tuple[float, float]:
    """Length-weighted mean congestion level and the fraction of slowed edges.

    ``congestion_level`` is the proportional increase over free-flow travel
    time, so 0 is free flow and 0.5 is half as long again. The fraction is taken
    from the simulator's own bands rather than recomputed, so the CLI and the
    dashboard cannot disagree about what counts as congested.
    """
    cong = state.get("congestion", {})
    mean = float(cong.get("mean_level_length_weighted", cong.get("mean_level", float("nan"))))
    bands = cong.get("bands", {})
    total = sum(bands.values()) if bands else 0
    slowed = (total - bands.get("free", 0)) / total if total else float("nan")
    return mean, slowed


def _instance_table(inst) -> Any:
    import numpy as np

    return render.kv_table(f"instance: {inst.name}", [
        ("customers", str(inst.n_customers)),
        ("capacity", f"{inst.capacity:g}"),
        ("total demand", f"{inst.demand.sum():g}"),
        ("vehicles (lower bound)", str(inst.min_vehicles)),
        ("mean travel time between stops",
         render.format_duration_hms(float(np.mean(inst.duration)))),
        ("objective weights", str(inst.weights.as_dict())),
    ])


def _instance_json(inst) -> dict:
    """Serialise a road-network instance, including its matrices.

    The matrices are the expensive part to rebuild (an all-pairs Dijkstra over
    tens of thousands of edges), so an instance saved here can be re-solved
    later without touching osmnx at all.
    """
    return {
        "name": inst.name,
        "n_customers": inst.n_customers,
        "capacity": float(inst.capacity),
        "n_vehicles": inst.n_vehicles,
        "demand": inst.demand.tolist(),
        "distance": inst.distance.tolist(),
        "duration": inst.duration.tolist(),
        "congestion": inst.congestion.tolist() if inst.congestion is not None else None,
        "coords": inst.coords.tolist() if inst.coords is not None else None,
        "node_ids": list(inst.node_ids) if inst.node_ids else None,
        "weights": inst.weights.as_dict(),
        "meta": inst.meta,
    }


@osm_app.command("fetch")
def osm_fetch(
    network: str = typer.Option("all", "--network", "-n",
                                help="Which network to rebuild, or 'all'."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir",
                                           help="Where the .graphml files are written; "
                                                "the installation's data/osm by default."),
    force: bool = typer.Option(False, "--force",
                               help="Rebuild even if the file already exists."),
) -> None:
    """Rebuild the bundled road networks from OpenStreetMap.

    The .graphml files are around forty megabytes and are therefore not kept in
    version control. This command regenerates them from the exact recipe
    recorded in ``data/osm/networks.json``, so a fresh clone can run the
    demonstration. It needs an internet connection and takes roughly a minute
    per network.
    """
    _anchor_data_root()

    import json as _json

    from qroute.graph.osm import OSM_DIR

    # The recipe and the graphs live together, and both are found relative to the
    # installation rather than to wherever the shell happens to be.
    out_dir = Path(out_dir) if out_dir is not None else OSM_DIR
    manifest_path = out_dir / "networks.json"
    if not manifest_path.exists():
        # ``_fail`` returns the exception rather than raising it, so a bare call
        # printed the error and carried straight on into json.loads of a file
        # that does not exist.
        raise _fail(f"no manifest at {manifest_path}",
                    "it records the recipe for each network, and lives beside the graphs")
    manifest = _json.loads(manifest_path.read_text())
    available = manifest["networks"]

    wanted = list(available) if network == "all" else [network]
    unknown = [w for w in wanted if w not in available]
    if unknown:
        raise _fail(f"unknown network {unknown[0]!r}",
                    f"available: {', '.join(available)}")

    import osmnx as ox

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in wanted:
        spec = available[name]
        target = out_dir / f"{name}.graphml"
        if target.exists() and not force:
            rows.append((name, "already present", "", ""))
            continue
        started = time.perf_counter()
        graph = ox.graph_from_point(
            tuple(spec["center"]), dist=spec["radius_m"],
            network_type=spec.get("network_type", "drive"),
            simplify=spec.get("simplify", True),
        )
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        ox.save_graphml(graph, target)
        elapsed = time.perf_counter() - started
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        expected_n = spec.get("expected_nodes_raw")
        note = ""
        if expected_n:
            drift = 100.0 * abs(n - expected_n) / expected_n
            note = "as recorded" if drift < 2 else f"{drift:.1f}% from recorded {expected_n}"
        rows.append((name, f"{n} nodes, {m} edges", f"{elapsed:.0f} s", note))

    table = render._table(
        "Road networks",
        ("network", {}),
        ("result", {}),
        ("time", {"justify": "right"}),
        ("note", {"style": "dim"}),
    )
    for row in rows:
        table.add_row(*row)
    out = render.console()
    out.print(table)
    out.print("[dim]Data (c) OpenStreetMap contributors, ODbL.[/dim]")


@osm_app.command("build")
def osm_build(
    place_file: Annotated[str, typer.Option("--place-file", "-g",
                          help="GraphML path, or the name of a bundled network.")],
    customers: Annotated[int, typer.Option("--customers", "-n", help="Number of stops.")] = 40,
    seed: Annotated[int, typer.Option("--seed", help="Seed for stop sampling and demands.")] = 1,
    hour: Annotated[Optional[float], typer.Option("--hour",
                    help="Apply simulated traffic for this hour of the day first.")] = None,
    save: Annotated[Optional[Path], typer.Option("--save", help="Write the instance as JSON.")] = None,
    do_solve: Annotated[bool, typer.Option("--solve/--no-solve",
                        help="Also solve the instance once it is built.")] = False,
    algorithm: Annotated[str, typer.Option("--algorithm", "-a", help="Solver to use.")] = "qpso",
    seconds: Annotated[float, typer.Option("--seconds", "-t", help="Budget when solving.")] = 10.0,
    geojson: Annotated[Optional[Path], typer.Option("--geojson",
                       help="Write the solved routes as GeoJSON polylines.")] = None,
) -> None:
    """Build a routing instance from a road network, and optionally solve it."""
    _anchor_data_root()
    from qroute.graph.builder import build_instance, routes_geojson

    seed = _check_seed(seed)
    seconds = _check_seconds(seconds)
    customers = _check_count(customers, "--customers", minimum=2)
    with con.status(f"loading {place_file}"):
        network = _load_network(place_file)
    con.print(_network_table(network))

    if hour is not None:
        from qroute.traffic.simulator import TrafficSimulator
        with con.status(f"simulating traffic at {hour:g}:00"):
            sim = TrafficSimulator(network, seed=seed)
            sim.set_clock(hour)
            sim.apply_to(network)
        mean_level, slowed = _congestion_summary(sim.state())
        con.print(f"[dim]traffic at {hour:g}:00: mean congestion level {mean_level:.3f}, "
                  f"{slowed * 100:.0f}% of edges slower than free flow[/dim]")

    with con.status(f"selecting {customers} stops and building the travel-time matrix"):
        started = time.perf_counter()
        inst, matrices = build_instance(network, customers, seed=seed, return_matrices=True)
        build_seconds = time.perf_counter() - started
    con.print(_instance_table(inst))
    con.print(f"[dim]matrix built in {render.format_seconds(build_seconds)}[/dim]")

    if save is not None:
        _write_json(save, _instance_json(inst))

    if do_solve:
        _warm(algorithm)
        with interruptible(f"{algorithm} on {inst.name}"):
            result = _solve_once(algorithm, inst, seconds, seed, {})
        con.print()
        con.print(render.solve_table(result, inst))
        con.print(f"[dim]objective is travel time in seconds; total drive time "
                  f"{render.format_duration_hms(result.best.stats.duration)}, distance "
                  f"{result.best.stats.distance / 1000.0:.2f} km[/dim]")
        if result.history:
            line, first, last = render.convergence_line(result.history)
            con.print(f"[bold]convergence[/bold] {line}  {first:,.0f} -> {last:,.0f}")
        if geojson is not None:
            _write_json(geojson, routes_geojson(network, matrices, result.best.routes))
    elif geojson is not None:
        raise _fail("--geojson needs --solve", "there are no routes to write yet")


@osm_app.command("demo")
def osm_demo(
    network_name: Annotated[str, typer.Option("--network", "-g",
                            help="Bundled network name or GraphML path.")] = "bengaluru_koramangala",
    hour: Annotated[float, typer.Option("--hour", help="Hour of the day for the first plan.")] = 9.0,
    customers: Annotated[int, typer.Option("--customers", "-n", help="Number of stops.")] = 40,
    seed: Annotated[int, typer.Option("--seed", help="Seed for stops, demands and traffic.")] = 1,
    seconds: Annotated[float, typer.Option("--seconds", "-t",
                       help="Budget for the first optimisation.")] = 10.0,
    reopt_seconds: Annotated[Optional[float], typer.Option("--reopt-seconds",
                             help="Budget for the re-optimisation; half the first by default.")] = None,
    incident_minutes: Annotated[float, typer.Option("--incident-minutes",
                                help="How long the incident lasts.")] = 45.0,
    incident_edges: Annotated[int, typer.Option("--incident-edges",
                              help="How many of the plan's busiest arcs the incident hits.")] = 40,
    json_out: Annotated[Optional[Path], typer.Option("--json", help="Write the whole story here.")] = None,
) -> None:
    """Plan under morning traffic, break a road, and re-optimise from a warm start.

    This is the end-to-end demonstration. Each stage is measured under the
    conditions that actually hold at that moment: the original plan is re-costed
    on the post-incident matrix before the re-optimised plan is compared with it,
    which is the only comparison that means anything. The improvement reported is
    therefore "what re-optimising saves against carrying on with the old plan",
    not the much larger and much less honest "cost after the incident against
    cost before it".
    """
    _anchor_data_root()

    from collections import Counter

    import numpy as np

    from qroute.graph.builder import build_instance, build_matrices, leg_node_paths
    from qroute.traffic.events import slowdown
    from qroute.traffic.simulator import TrafficSimulator

    seed = _check_seed(seed)
    seconds = _check_seconds(seconds)
    customers = _check_count(customers, "--customers", minimum=2)
    if reopt_seconds is not None:
        _check_seconds(reopt_seconds, "--reopt-seconds")
    if not 0.0 <= hour < 24.0:
        raise _fail(f"--hour must be in [0, 24), got {hour:g}",
                    "the traffic simulator reads it as an hour of the day")
    reopt = reopt_seconds if reopt_seconds is not None else max(seconds / 2.0, 1.0)

    with con.status(f"loading {network_name}"):
        network = _load_network(network_name)
    con.print(_network_table(network))

    # ---- stage 1: the network under morning traffic ------------------------
    sim = TrafficSimulator(network, seed=seed)
    sim.set_clock(hour)
    sim.apply_to(network)
    before_state = sim.state()
    mean_before, slowed_before = _congestion_summary(before_state)
    con.print(render.kv_table(f"traffic at {hour:g}:00", [
        ("mean congestion level", f"{mean_before:.3f}"),
        ("edges slower than free flow", f"{slowed_before * 100:.0f}%"),
        ("network travel time against free flow",
         f"{before_state['travel_time_seconds']['network_ratio']:.2f}x"),
        ("simulated clock", f"minute {before_state['time_minutes']:.0f} of the week"),
    ]))

    with con.status(f"building an instance with {customers} stops"):
        inst, matrices = build_instance(network, customers, seed=seed, return_matrices=True)
    con.print(_instance_table(inst))

    # ---- stage 2: the plan for those conditions ----------------------------
    con.print(f"\n[bold]1. planning[/bold] under {hour:g}:00 traffic, {seconds:g} s budget")
    _warm("qpso")
    with interruptible("the demonstration"):
        plan = _solve_once("qpso", inst, seconds, seed, {})
    con.print(f"   objective {plan.best.cost:,.0f} s, {plan.best.n_routes} routes, "
              f"drive time {render.format_duration_hms(plan.best.stats.duration)}, "
              f"{plan.best.stats.distance / 1000:.2f} km")

    # ---- stage 3: the incident ---------------------------------------------
    # Break a road the plan actually uses: an incident somewhere the vehicles
    # never go would change nothing and would prove nothing.
    paths = leg_node_paths(network, matrices, plan.best.routes)
    src, dst = network.edge_endpoints()
    edge_of = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src, dst))}
    usage: Counter[int] = Counter()
    for path in paths:
        for a, b in zip(path, path[1:]):
            idx = edge_of.get((int(a), int(b)))
            if idx is not None:
                usage[idx] += 1
    if not usage:
        raise _fail("could not map the planned routes back onto network edges",
                    "this indicates a mismatch between the travel-time matrix and the graph")
    # Hit the links the fleet leans on hardest rather than a random street: the
    # busiest arcs of the plan are where an incident actually costs something,
    # and an incident that costs nothing would demonstrate nothing.
    blocked = [idx for idx, _count in usage.most_common(incident_edges)]

    # A severe slowdown rather than a hard closure. The road network stores one
    # finite travel time per edge, so an impassable arc has no representation
    # there; a factor-of-ten slowdown is the strongest disruption the weight
    # model can carry honestly, and it is more than enough to make the affected
    # corridors the wrong way to go.
    start = float(before_state["time_minutes"])
    sim.add_event(slowdown(blocked, start_minute=start, duration_minutes=incident_minutes,
                           speed_multiplier=0.1,
                           description="incident on the corridors the plan uses most"))
    # Links leaving the affected junctions carry the traffic that diverts around
    # the incident, so they slow down too. Without this the re-plan would be a
    # trivial hop onto an untouched parallel street.
    affected_nodes = {int(src[i]) for i in blocked} | {int(dst[i]) for i in blocked}
    blocked_set = set(blocked)
    neighbours = [i for i, u in enumerate(src)
                  if int(u) in affected_nodes and i not in blocked_set][:120]
    if neighbours:
        sim.add_event(slowdown(neighbours, start_minute=start,
                               duration_minutes=incident_minutes, speed_multiplier=0.6,
                               description="diverted traffic on adjacent links"))
    sim.advance(1.0)
    sim.apply_to(network)
    after_state = sim.state()
    mean_after, _ = _congestion_summary(after_state)
    con.print(f"\n[bold]2. incident[/bold] the {len(blocked)} arcs the plan uses most are cut to "
              f"a tenth of their speed and {len(neighbours)} adjacent arcs are slowed, "
              f"for {incident_minutes:g} minutes")
    con.print(f"   mean congestion level {mean_before:.3f} -> {mean_after:.3f}; "
              f"network travel time "
              f"{before_state['travel_time_seconds']['network_ratio']:.2f}x -> "
              f"{after_state['travel_time_seconds']['network_ratio']:.2f}x free flow")

    with con.status("recomputing travel times through the incident"):
        node_indices = np.asarray(inst.meta["stop_node_indices"], dtype=np.int64)
        after = build_matrices(network, node_indices, keep_predecessors=True)
    inst_after = inst.with_matrices(distance=after.distance, duration=after.duration,
                                    congestion=after.congestion)

    old_plan_after = inst_after.make_solution(plan.best.routes)

    # ---- stage 4: re-optimise from a warm start ----------------------------
    con.print(f"\n[bold]3. re-optimising[/bold] from the existing plan, {reopt:g} s budget")
    tour = [c for route in plan.best.routes for c in route]
    keys = np.empty(inst.n_customers, dtype=np.float64)
    for position, customer in enumerate(tour):
        keys[customer - 1] = (position + 0.5) / len(tour)
    warm = np.tile(keys, (5, 1))       # a few particles start from the old plan;
                                       # the rest stay random so the swarm keeps
                                       # the diversity it needs to escape it
    with interruptible("the re-optimisation"):
        replan = _solve_once("qpso", inst_after, reopt, seed + 1, {"initial_keys": warm})
        cold = _solve_once("qpso", inst_after, reopt, seed + 1, {})

    base = plan.best.cost
    stages = [
        {"stage": f"plan at {hour:g}:00", "cost": plan.best.cost,
         "duration": plan.best.stats.duration, "distance": plan.best.stats.distance,
         "n_routes": plan.best.n_routes, "delta": None,
         "note": "conditions before the incident"},
        {"stage": "same plan, after", "cost": old_plan_after.cost,
         "duration": old_plan_after.stats.duration, "distance": old_plan_after.stats.distance,
         "n_routes": old_plan_after.n_routes,
         "delta": 100.0 * (old_plan_after.cost - base) / base,
         "note": "driving the original routes through the incident"},
        {"stage": "re-optimised (warm)", "cost": replan.best.cost,
         "duration": replan.best.stats.duration, "distance": replan.best.stats.distance,
         "n_routes": replan.best.n_routes,
         "delta": 100.0 * (replan.best.cost - old_plan_after.cost) / old_plan_after.cost,
         "note": f"{reopt:g} s from the old plan"},
        {"stage": "re-optimised (cold)", "cost": cold.best.cost,
         "duration": cold.best.stats.duration, "distance": cold.best.stats.distance,
         "n_routes": cold.best.n_routes,
         "delta": 100.0 * (cold.best.cost - old_plan_after.cost) / old_plan_after.cost,
         "note": f"{reopt:g} s from scratch, for comparison"},
    ]
    con.print()
    con.print(render.demo_table(stages))

    saved = old_plan_after.cost - replan.best.cost
    changed = sum(1 for a, b in zip(sorted(map(tuple, plan.best.routes)),
                                    sorted(map(tuple, replan.best.routes))) if a != b)
    con.print(f"\n[bold]result[/bold] re-optimising saves "
              f"{render.format_duration_hms(saved)} of driving "
              f"({100.0 * saved / old_plan_after.cost:.1f}% of the disrupted plan) in "
              f"{reopt:g} s of computation.")
    con.print("[dim]Two different adaptations are at work and the table separates them. "
              "Recomputing the travel-time matrix reroutes every leg around the incident by "
              "itself, which is why the original plan still runs at "
              f"{100.0 * (old_plan_after.cost - base) / base:+.1f}% rather than being stranded; "
              "re-optimising then changes which vehicle serves which stop and in what order, "
              f"and that is the further {100.0 * saved / old_plan_after.cost:.1f}%.[/dim]")
    margin = cold.best.cost - replan.best.cost
    if abs(margin) < 1e-6:
        verdict = "reaches the same cost as"
    elif margin > 0:
        verdict = f"beats a cold restart by {margin:,.0f} s"
    else:
        verdict = f"is beaten by a cold restart by {-margin:,.0f} s"
    con.print(f"[dim]{changed} of {plan.best.n_routes} routes differ from the original plan; "
              f"on the same budget the warm start {verdict}"
              f"{' a cold restart' if abs(margin) < 1e-6 else ''}.[/dim]")

    if json_out is not None:
        _write_json(json_out, {
            "network": network.name, "hour": hour, "customers": inst.n_customers,
            "seed": seed, "seconds": seconds, "reopt_seconds": reopt,
            "traffic_before": before_state, "traffic_after": after_state,
            "incident_edges": blocked, "adjacent_slowed_edges": neighbours,
            "stages": stages,
            "plan": plan.best.to_json(),
            "replan": replan.best.to_json(),
        })


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve(
    host: Annotated[Optional[str], typer.Option("--host",
                    help="Interface to bind. Defaults to QROUTE_HOST, then 127.0.0.1.")] = None,
    port: Annotated[Optional[int], typer.Option("--port",
                    help="Port to bind. Defaults to QROUTE_PORT, then 8000.")] = None,
    reload: Annotated[bool, typer.Option("--reload/--no-reload",
                      help="Restart on source changes (development only).")] = False,
    app_path: Annotated[Optional[str], typer.Option("--app",
                        help="Import string of the ASGI application.")] = None,
) -> None:
    """Start the HTTP API with uvicorn."""
    # The settings object is the one source of truth for how the service is
    # configured, so the command line falls back to it rather than carrying its
    # own defaults. Two entry points that disagree about which port to bind is
    # exactly the sort of thing that wastes an afternoon on a deployment.
    try:
        from qroute.config import settings as _settings

        resolved_settings = _settings()
        host = host if host is not None else resolved_settings.host
        port = port if port is not None else resolved_settings.port
    except Exception:
        # Configuration problems are reported by the application at startup with
        # a far better message than this command could produce, so fall back to
        # the documented defaults and let it speak.
        host = host if host is not None else "127.0.0.1"
        port = port if port is not None else 8000
    if not 1 <= port <= 65535:
        raise _fail(f"--port must be between 1 and 65535, got {port}")
    try:
        import uvicorn
    except ImportError:
        raise _fail("uvicorn is not installed", "pip install 'uvicorn[standard]'")

    from importlib import import_module

    candidates = [app_path] if app_path else list(API_CANDIDATES)
    resolved: Optional[str] = None
    for candidate in candidates:
        module_name, _, attr = candidate.partition(":")
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, attr or "app"):
            resolved = candidate
            break
    if resolved is None:
        raise _fail(
            "no API application was found",
            "tried " + ", ".join(candidates) +
            "; the FastAPI service lives in qroute/api and must exist before `qroute serve` works",
        )

    con.print(f"[bold]qroute API[/bold] {resolved} on http://{host}:{port}")
    con.print(f"[dim]interactive documentation at http://{host}:{port}/docs[/dim]")
    uvicorn.run(resolved, host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the versions of qroute and of everything a result depends on."""
    import platform
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    def safe(name: str) -> str:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return "not installed"

    rows = [("qroute", safe("qroute")), ("python", platform.python_version()),
            ("platform", platform.platform())]
    for name in ("numpy", "scipy", "numba", "networkx", "ortools", "pyvrp",
                 "osmnx", "vrplib", "fastapi", "uvicorn", "typer", "rich",
                 "matplotlib"):
        rows.append((name, safe(name)))
    con.print(render.kv_table("versions", rows))

    from qroute.algorithms.registry import catalogue
    table = render.kv_table("algorithms", [(a["name"], a["description"]) for a in catalogue()]
                            + [(k, v) for k, v in EXTERNAL_SOLVERS.items()])
    table.columns[1].justify = "left"
    con.print(table)


if __name__ == "__main__":       # pragma: no cover - module is run via the script
    app()
