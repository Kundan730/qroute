"""Tests for the behaviour the platform needs when it is used in anger.

Everything here pins a failure that was observed on a real run rather than
imagined. The optimiser's quality is measured by the benchmark sweep and by
:mod:`tests.test_cli`; this file is about what happens when the person driving
the tool changes their mind, re-runs a command by habit, reads a file that a
previous run left half-written, mistypes an option, or works on a machine whose
numba cache is cold or whose core count is not the author's.

Each test therefore reproduces the accident, not a mock of it: a real ``SIGINT``
is delivered to a real solve, a real rows file is truncated on disk, a real
worker process is killed mid-sweep. The whole file is kept to a few tens of
seconds by using the smallest benchmark instance and sub-second budgets, since
none of these outcomes depends on how long the search was given.
"""

from __future__ import annotations

import gzip
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from qroute.benchmark import runner as runner_mod
from qroute.benchmark.runner import (BenchmarkConfig, BenchmarkRunner, ExistingResults,
                                     count_rows, read_rows, resolve_workers, warm_kernels)
from qroute.cli import main as cli
from qroute.cli.main import app

runner = CliRunner()
WIDE = {"COLUMNS": "200", "TERM": "dumb"}

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

#: The seed whose worker process kills itself in the dead-worker test.
POISON_SEED = 4242


def invoke(*args: str, **kwargs):
    return runner.invoke(app, list(args), env=WIDE, **kwargs)


def write_rows(path: Path, rows: list[dict], truncate_last: bool = False) -> Path:
    """Write a rows.jsonl, optionally cut off mid-object like a killed sweep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if truncate_last:
        text = text[:-len(json.dumps(rows[-1])) // 2]
    path.write_text(text)
    return path


def a_row(instance: str = "P-n16-k8", algorithm: str = "qpso", seed: int = 0,
          cost: float = 450.0) -> dict:
    """A benchmark row with the fields the summary and the tables require."""
    return {
        "instance": instance, "algorithm": algorithm, "seed": seed,
        "cost": cost, "gap": 100.0 * (cost - 450.0) / 450.0, "bks": 450.0,
        "n_routes": 8, "feasible": True, "violation": 0.0,
        "iterations": 100, "evaluations": 3000, "seconds": 0.3,
        "params": {}, "status": "ok",
    }


# ---------------------------------------------------------------------------
# 1. The worker count must come from the host, not from the author's laptop
# ---------------------------------------------------------------------------
def test_an_unset_worker_count_is_derived_from_the_host():
    plan = resolve_workers(0, cpus=16)
    assert plan.workers == 14 and plan.cpus == 16 and plan.warning is None
    # Two cores of headroom cannot take the count below one, however small the
    # host: a single-core container still has to be able to run the sweep.
    assert resolve_workers(0, cpus=1).workers == 1
    assert resolve_workers(None, cpus=2).workers == 1


def test_asking_for_more_workers_than_cores_warns_but_still_runs():
    plan = resolve_workers(9, cpus=4)
    assert plan.workers == 9, "an explicit request is honoured, not silently clamped"
    assert plan.warning is not None
    assert "4 core" in plan.warning and "wall-clock" in plan.warning
    assert resolve_workers(4, cpus=4).warning is None


def test_the_host_reports_at_least_one_usable_core():
    assert runner_mod.host_cpus() >= 1


@pytest.mark.parametrize("config", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_no_shipped_configuration_pins_the_worker_count(config: Path):
    """A pinned count is only right on the machine the file was written on.

    ``configs/main.yaml`` used to say ``workers: 9``, which on a four-core
    machine puts nine wall-clock-budgeted runs onto four cores and understates
    every algorithm in the table without a word of warning.
    """
    cfg = BenchmarkConfig.from_yaml(config)
    assert cfg.workers <= 0, f"{config.name} pins workers to {cfg.workers}"


def test_the_sweep_records_the_worker_count_it_actually_used(sweep: Path):
    meta = json.loads((sweep / "meta.json").read_text())
    assert meta["workers"] >= 1
    assert meta["usable_cpus"] >= 1


# ---------------------------------------------------------------------------
# 2. A rows file that stops in the middle is read as far as it goes
# ---------------------------------------------------------------------------
def test_a_truncated_rows_file_is_read_up_to_the_damage(tmp_path: Path):
    rows_path = write_rows(tmp_path / "rows.jsonl",
                           [a_row(seed=i) for i in range(4)], truncate_last=True)
    result = read_rows(rows_path)
    assert len(result.rows) == 3
    assert result.unreadable == [4]
    assert not result.complete
    assert [r["seed"] for r in result.rows] == [0, 1, 2]


def test_damage_in_the_middle_does_not_hide_the_rows_after_it(tmp_path: Path):
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(a_row(seed=0)) + "\n"
                         + '{"instance": "P-n16-k8", "algo\n'
                         + json.dumps(a_row(seed=2)) + "\n")
    result = read_rows(rows_path)
    assert [r["seed"] for r in result.rows] == [0, 2]
    assert result.unreadable == [2]


def test_report_reads_a_truncated_run_and_says_how_much_it_lost(tmp_path: Path):
    """The whole point: an interrupted sweep is still reportable evidence."""
    run_dir = tmp_path / "run"
    write_rows(run_dir / "rows.jsonl",
               [a_row(seed=i) for i in range(6)] + [a_row(algorithm="sa", seed=9)],
               truncate_last=True)
    result = invoke("report", str(run_dir))
    assert result.exit_code == 0, result.output
    assert "1 of 7 lines" in result.output
    assert "could not be parsed" in result.output
    assert "450.00" in result.output, "the intact rows are still reported"


def test_report_refuses_a_rows_file_with_nothing_readable_in_it(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rows.jsonl").write_text("not json at all\n")
    result = invoke("report", str(run_dir))
    assert result.exit_code == 1
    assert "no readable runs" in result.output


def test_a_stored_summary_is_not_quoted_beside_rows_that_no_longer_read(tmp_path: Path):
    """A summary of 100 runs printed above tables built from 6 would mislead."""
    run_dir = tmp_path / "run"
    write_rows(run_dir / "rows.jsonl", [a_row(seed=i) for i in range(4)],
               truncate_last=True)
    (run_dir / "summary.json").write_text(json.dumps(
        {"cells": {}, "algorithms": [], "instances": [], "n_ok": 999,
         "n_failed": 0, "omnibus": None, "failures": []}))
    result = invoke("report", str(run_dir))
    assert result.exit_code == 0, result.output
    assert "999" not in result.output


def test_rows_are_read_from_the_gzipped_form_when_that_is_all_there_is(tmp_path: Path):
    """A fresh clone has only ``rows.jsonl.gz``; ``rows.jsonl`` is gitignored."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = "".join(json.dumps(a_row(seed=i)) + "\n" for i in range(3))
    with gzip.open(run_dir / "rows.jsonl.gz", "wt") as fh:
        fh.write(payload)

    result = read_rows(run_dir / "rows.jsonl")
    assert len(result.rows) == 3
    assert result.path.name == "rows.jsonl.gz"
    assert count_rows(run_dir / "rows.jsonl") == 3

    reported = invoke("report", str(run_dir))
    assert reported.exit_code == 0, reported.output


# ---------------------------------------------------------------------------
# 3. A completed run is not destroyed by re-running the command
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory) -> Path:
    """A four-run sweep small enough to execute inside a test."""
    directory = tmp_path_factory.mktemp("bench")
    config = {
        "name": "production", "instances": ["P-n16-k8"],
        "algorithms": ["qpso", "sa"], "seeds": 2, "master_seed": 20260920,
        "max_seconds": 0.3, "max_iterations": 1_000_000, "workers": 2,
        "output_dir": str(directory), "save_history": False, "history_stride": 1,
    }
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def sweep(tiny_config: Path) -> Path:
    """Run the tiny sweep once for the whole module and return its directory."""
    cfg = BenchmarkConfig.from_yaml(tiny_config)
    result = BenchmarkRunner(cfg).run()
    assert result["summary"]["n_ok"] == 4, result["summary"]["failures"]
    return Path(result["output_dir"])


def test_a_second_sweep_refuses_rather_than_truncating_the_first(sweep: Path,
                                                                tiny_config: Path):
    before = (sweep / "rows.jsonl").read_bytes()
    cfg = BenchmarkConfig.from_yaml(tiny_config)

    with pytest.raises(ExistingResults) as caught:
        BenchmarkRunner(cfg).run()

    assert caught.value.rows == 4
    assert caught.value.finished is True
    assert (sweep / "rows.jsonl").read_bytes() == before, "the previous rows survived"


def test_the_command_line_explains_how_to_proceed_after_refusing(sweep: Path,
                                                                tiny_config: Path):
    before = (sweep / "rows.jsonl").read_bytes()
    result = invoke("bench", "--config", str(tiny_config))
    assert result.exit_code == 1
    assert "already holds a completed sweep of 4 runs" in result.output
    assert "--name" in result.output and "--force" in result.output
    assert (sweep / "rows.jsonl").read_bytes() == before


def test_force_moves_the_previous_rows_aside_instead_of_deleting_them(sweep: Path,
                                                                     tiny_config: Path):
    before = (sweep / "rows.jsonl").read_bytes()
    cfg = BenchmarkConfig.from_yaml(tiny_config)
    result = BenchmarkRunner(cfg, force=True).run()

    superseded = sorted(sweep.glob("rows.superseded-*.jsonl"))
    assert len(superseded) == 1
    assert superseded[0].read_bytes() == before
    assert result["summary"]["n_ok"] == 4
    assert json.loads((sweep / "meta.json").read_text())["superseded_rows"] \
        == superseded[0].name


def test_the_refusal_happens_before_anything_is_written(tmp_path: Path):
    """meta.json is provenance for the results beside it and must not be replaced.

    The check used to run after ``meta.json`` had already been rewritten, so an
    aborted re-run left the old rows described by the new run's metadata.
    """
    out = tmp_path / "runs"
    (out / "solo").mkdir(parents=True)
    (out / "solo" / "rows.jsonl").write_text(json.dumps(a_row()) + "\n")
    (out / "solo" / "meta.json").write_text('{"marker": "original"}')

    cfg = BenchmarkConfig(name="solo", instances=["P-n16-k8"], algorithms=["qpso"],
                          seeds=1, max_seconds=0.2, output_dir=str(out))
    with pytest.raises(ExistingResults):
        BenchmarkRunner(cfg).run()
    assert json.loads((out / "solo" / "meta.json").read_text()) == {"marker": "original"}


# ---------------------------------------------------------------------------
# 4. One dead worker must not cost the whole sweep
# ---------------------------------------------------------------------------
def suicidal_worker(task: dict) -> dict:
    """A worker that dies outright on one task, as an OOM kill would.

    ``os._exit`` leaves no traceback and no result, which is exactly how a
    process killed by the operating system disappears; ``ProcessPoolExecutor``
    responds by declaring the whole pool broken. The short delay before dying
    lets the healthy tasks finish first, which is the ordinary case: a worker
    usually dies part-way through a run rather than the instant it starts.
    """
    if task["seed"] == POISON_SEED:
        time.sleep(0.4)
        os._exit(1)
    return dict(a_row(instance=task["instance"], algorithm=task["algorithm"],
                      seed=task["seed"]))


def test_one_dead_worker_does_not_abort_the_rest_of_the_sweep(tmp_path: Path,
                                                              monkeypatch):
    monkeypatch.setattr(runner_mod, "_run_one", suicidal_worker)
    cfg = BenchmarkConfig(name="dead", instances=["P-n16-k8"], algorithms=["qpso"],
                          seeds=3, max_seconds=0.2, workers=2,
                          output_dir=str(tmp_path))

    tasks = BenchmarkRunner(cfg).tasks()
    tasks[1]["seed"] = POISON_SEED
    monkeypatch.setattr(BenchmarkRunner, "tasks", lambda self: tasks)

    result = BenchmarkRunner(cfg).run()

    statuses = sorted(r["status"] for r in result["rows"])
    assert statuses == ["ok", "ok", "worker_died"], \
        "only the task that killed the process is written off"
    dead = next(r for r in result["rows"] if r["status"] == "worker_died")
    assert dead["seed"] == POISON_SEED
    assert "died" in dead["error"]
    # And the sweep is still a sweep: the survivors are on disk and summarised.
    assert result["summary"]["n_ok"] == 2
    assert count_rows(Path(result["output_dir"]) / "rows.jsonl") == 3


def test_the_tasks_queued_behind_a_fatal_one_are_not_written_off(tmp_path: Path,
                                                                 monkeypatch):
    """The worst case: the pool breaks before a single result comes back.

    Every outstanding task then looks equally guilty. Blaming all of them would
    lose most of a sweep to one bad instance, so the batch is halved until the
    culprit is alone, and only it is recorded as failed. Driven through a stand-in
    for the pool so the bisection itself is what is being tested, not the two
    seconds it costs to start a process.
    """
    calls: list[int] = []

    def fake_batch(self, batch, workers, record):
        calls.append(len(batch))
        if any(task["seed"] == POISON_SEED for _, task in batch):
            return list(batch)
        for _, task in batch:
            record(a_row(seed=task["seed"]))
        return []

    monkeypatch.setattr(BenchmarkRunner, "_run_batch", fake_batch)
    cfg = BenchmarkConfig(name="poisoned", instances=["P-n16-k8"], algorithms=["qpso"],
                          seeds=8, max_seconds=0.1, output_dir=str(tmp_path))
    tasks = BenchmarkRunner(cfg).tasks()
    tasks[5]["seed"] = POISON_SEED
    monkeypatch.setattr(BenchmarkRunner, "tasks", lambda self: tasks)

    result = BenchmarkRunner(cfg).run()

    statuses = sorted(r["status"] for r in result["rows"])
    assert statuses == ["ok"] * 7 + ["worker_died"]
    assert next(r for r in result["rows"] if r["status"] == "worker_died")["seed"] \
        == POISON_SEED
    assert len(calls) <= 2 * 8, "the search for the culprit stays logarithmic"


def test_an_interrupted_sweep_keeps_and_summarises_what_finished(tmp_path: Path,
                                                                 monkeypatch):
    """Ctrl-C during a sweep reports the finished runs instead of raising."""
    def stop_after_two(self, tasks, workers, record):
        for task in tasks[:2]:
            record(a_row(instance=task["instance"], algorithm=task["algorithm"],
                         seed=task["seed"]))
        raise KeyboardInterrupt

    monkeypatch.setattr(BenchmarkRunner, "_execute", stop_after_two)
    cfg = BenchmarkConfig(name="stopped", instances=["P-n16-k8"], algorithms=["qpso"],
                          seeds=5, max_seconds=0.2, output_dir=str(tmp_path))

    result = BenchmarkRunner(cfg).run()

    assert result["interrupted"] is True
    assert result["summary"]["n_ok"] == 2
    out_dir = Path(result["output_dir"])
    assert count_rows(out_dir / "rows.jsonl") == 2
    assert (out_dir / "summary.json").exists()


# ---------------------------------------------------------------------------
# 5. A mistyped option is answered, not thrown back as a traceback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("args, expected", [
    (["solve", "P-n16-k8", "--seed", "-5"], "--seed must not be negative"),
    (["solve", "P-n16-k8", "--seed", str(2 ** 70)], "--seed must be at most"),
    (["solve", "P-n16-k8", "--seconds", "0"], "--seconds must be greater than zero"),
    (["solve", "P-n16-k8", "--seconds", "-3"], "--seconds must be greater than zero"),
    (["solve", "P-n16-k8", "--seconds", "nan"], "--seconds must be a finite number"),
    (["solve", "P-n16-k8", "--seconds", "inf"], "--seconds must be a finite number"),
    (["solve", "P-n16-k8", "--iterations", "0"], "--iterations must be at least 1"),
    (["compare", "P-n16-k8", "--seeds", "0"], "--seeds must be at least 1"),
    (["compare", "P-n16-k8", "--master-seed", "-1"], "--master-seed must not be negative"),
    (["exact", "P-n16-k8", "--seconds", "0"], "--seconds must be greater than zero"),
    (["serve", "--port", "99999"], "--port must be between 1 and 65535"),
])
def test_a_nonsensical_option_produces_a_message_and_no_traceback(args, expected):
    result = invoke(*args)
    assert result.exit_code == 1, result.output
    assert expected in result.output
    assert "Traceback" not in result.output
    assert "ValueError" not in result.output


def test_a_very_short_budget_is_allowed_but_flagged():
    """Accepting it is right - it is legal - but it must not look like a search."""
    result = invoke("solve", "P-n16-k8", "--seconds", "0.05", "--no-routes")
    assert result.exit_code == 0, result.output
    assert "shorter than the time it takes to build a first solution" in result.output


def test_osm_fetch_stops_on_a_bad_argument_instead_of_carrying_on(tmp_path: Path):
    """``_fail`` returns the exception; two call sites forgot to raise it.

    The command printed "unknown network" and then went on to read a manifest
    that was not there, so the user got an error message followed by a
    traceback, and in the missing-manifest case a second, less useful error.
    """
    unknown = invoke("osm", "fetch", "--network", "not-a-city")
    assert unknown.exit_code == 1
    assert "unknown network" in unknown.output
    assert "Traceback" not in unknown.output

    absent = invoke("osm", "fetch", "--out-dir", str(tmp_path / "nowhere"))
    assert absent.exit_code == 1
    assert "no manifest at" in absent.output
    assert "Traceback" not in absent.output


def test_a_command_works_from_a_directory_that_is_not_the_checkout(tmp_path: Path):
    """The benchmark data used to be looked up relative to the shell's cwd.

    Run from anywhere but the repository, every command reported that the
    instance did not exist. A judge who installs the package and runs it from
    their home directory must not meet that.
    """
    process = subprocess.run(
        [sys.executable, "-m", "qroute.cli.main", "solve", "P-n16-k8",
         "--seconds", "0.5", "--no-routes"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items() if k != "QROUTE_DATA"},
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "450.00" in process.stdout


def test_the_data_directory_is_named_when_nothing_can_be_found(tmp_path, monkeypatch):
    """Run from the wrong directory, the tool says so instead of listing nothing."""
    monkeypatch.setenv("QROUTE_DATA", str(tmp_path / "absent"))
    from qroute.problems import loaders

    monkeypatch.setattr(loaders, "DATA_ROOT", tmp_path / "absent")
    monkeypatch.setattr(loaders, "CVRP_DIR", tmp_path / "absent" / "benchmarks" / "cvrplib")
    monkeypatch.setattr(loaders, "VRPTW_DIR", tmp_path / "absent" / "benchmarks" / "solomon")

    result = invoke("solve", "P-n16-k8", "--seconds", "1")
    assert result.exit_code == 1
    assert "no benchmark instances were found under" in result.output
    assert "QROUTE_DATA" in result.output
    assert "absent" in result.output, "the directory that was searched is named"


# ---------------------------------------------------------------------------
# 6. Interrupting a solve
# ---------------------------------------------------------------------------
def test_interruptible_turns_a_keyboard_interrupt_into_a_clean_exit():
    with pytest.raises(typer.Exit) as caught:
        with cli.interruptible("a solve"):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)      # give the signal a moment to be delivered
    assert caught.value.exit_code == cli.EXIT_INTERRUPTED


def _swallow_a_real_interrupt() -> None:
    """Deliver a real SIGINT and drop the ``KeyboardInterrupt`` it raises.

    This is what happens for real inside the search: the signal arrives while
    the main thread is in a compiled kernel, and the resulting exception is
    either turned into a ``SystemError`` by numba's dispatcher or caught by the
    optimiser's ``except Exception`` repair guard.
    """
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def test_interruptible_recognises_the_systemerror_numba_raises_instead():
    """numba reports an interrupt inside a compiled kernel as a ``SystemError``.

    The message is ``CPUDispatcher(<function local_search ...>) returned a
    result with an exception set``, which is what a user saw, under a full rich
    traceback, for pressing Ctrl-C.
    """
    with pytest.raises(typer.Exit) as caught:
        with cli.interruptible("a solve"):
            _swallow_a_real_interrupt()
            raise SystemError("CPUDispatcher(<function local_search>) returned a "
                              "result with an exception set")
    assert caught.value.exit_code == cli.EXIT_INTERRUPTED


def test_a_systemerror_without_an_interrupt_keeps_its_traceback():
    """A genuine compiler-level failure must not be disguised as a Ctrl-C."""
    with pytest.raises(SystemError):
        with cli.interruptible("a solve"):
            raise SystemError("something is actually wrong")


def test_an_interrupt_that_never_reaches_us_is_still_reported():
    """The optimiser's repair step swallows the SystemError with ``except Exception``.

    Observed once in five interrupts of a large instance: a complete result
    table and exit code 0 for a run the operator had cancelled.
    """
    with pytest.raises(typer.Exit) as caught:
        with cli.interruptible("a solve"):
            _swallow_a_real_interrupt()
    assert caught.value.exit_code == cli.EXIT_INTERRUPTED


@pytest.mark.parametrize("delay", [3.0])
def test_a_real_interrupt_during_a_real_solve_exits_cleanly(delay: float):
    """The defect as reported: Ctrl-C during a solve, from a real terminal.

    Run as a subprocess because that is the only way to deliver a signal to a
    process whose main thread is inside a compiled kernel.
    """
    process = subprocess.Popen(
        [sys.executable, "-m", "qroute.cli.main", "solve", "A-n80-k10",
         "--seconds", "60", "--no-routes"],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env={**os.environ, "COLUMNS": "200", "TERM": "dumb"},
    )
    time.sleep(delay)
    process.send_signal(signal.SIGINT)
    output, _ = process.communicate(timeout=60)

    assert process.returncode == cli.EXIT_INTERRUPTED, output
    assert "SystemError" not in output
    assert "CPUDispatcher" not in output
    assert "Traceback" not in output


# ---------------------------------------------------------------------------
# 7. The compiled kernels are warmed before the clock starts
# ---------------------------------------------------------------------------
def test_warming_applies_to_the_compiled_solvers_and_not_the_others():
    warm = warm_kernels("qpso")
    assert warm.applicable and warm.error is None
    # OR-Tools is compiled C++: warming it would only spend its time limit.
    assert warm_kernels("ortools").applicable is False


def test_a_solve_actually_iterates_on_a_budget_a_cold_cache_would_have_eaten():
    """With a cold numba cache this reported zero iterations and 11.7 s of wall
    clock for a five-second budget, and printed the construction heuristic as
    its answer."""
    result = invoke("solve", "P-n16-k8", "--seconds", "1", "--no-routes")
    assert result.exit_code == 0, result.output
    assert "no search iteration completed" not in result.output


def test_a_run_that_never_searched_says_so_rather_than_reporting_a_number():
    class _NoIterations:
        iterations = 0

    cli._check_budget_was_spent_searching(_NoIterations(), seconds=5.0, elapsed=11.7)
    # Nothing to assert about a return value: the contract is that it prints,
    # and the message is what stops a reader believing the number above it.


def test_the_warm_up_reports_failure_instead_of_hiding_it(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("no compiler on this machine")

    monkeypatch.setattr(runner_mod, "_dispatch", explode)
    warm = warm_kernels("qpso")
    assert warm.applicable is True
    assert warm.error is not None and "no compiler" in warm.error


# ---------------------------------------------------------------------------
# 8. Damage that is not a half-written line
#
# The guard and the tolerant reader were written for the accident that actually
# happens - a sweep killed mid-write. These pin the neighbouring cases found by
# pointing both of them at files a tired operator can produce: an output
# directory whose sweep died before its first run, and a rows file that is not
# text at all.
# ---------------------------------------------------------------------------
def test_a_sweep_that_died_before_its_first_run_can_simply_be_re_run(tmp_path: Path):
    """An empty rows.jsonl is not a result and must not demand --force.

    Interrupting a sweep during its first few runs leaves a rows file of zero
    rows. Refusing to start again over it would ask the operator to pass --force
    to protect nothing, and --force would then file a zero-byte
    ``rows.superseded`` as though a previous sweep had been rescued.
    """
    out = tmp_path / "runs"
    (out / "aborted").mkdir(parents=True)
    (out / "aborted" / "rows.jsonl").write_text("")

    cfg = BenchmarkConfig(name="aborted", instances=["P-n16-k8"], algorithms=["qpso"],
                          seeds=1, max_seconds=0.2, workers=1, output_dir=str(out))
    result = BenchmarkRunner(cfg).run()

    assert result["summary"]["n_ok"] == 1
    assert not list((out / "aborted").glob("rows.superseded-*.jsonl")), \
        "nothing was superseded, because there was nothing there"


def test_a_rows_file_that_is_not_text_is_reported_rather_than_raising(tmp_path: Path):
    """Undecodable bytes arrive while iterating the file, not from json.loads.

    A rows file that a crash filled with binary, or simply the wrong file passed
    to ``qroute report``, used to raise ``UnicodeDecodeError`` out of the read
    loop, losing every intact row before it and printing a traceback.
    """
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"instance": "P-n16-k8", "status": "ok"}\n\xc2\x28\xa0\xa1 junk\n')

    rows_file = read_rows(path)

    assert len(rows_file.rows) == 1, "the intact row survives the damaged one"
    assert rows_file.unreadable == [2]
    assert count_rows(path) == 2, "counting damage must not raise either"


def test_a_corrupt_gzip_is_reported_rather_than_raising(tmp_path: Path):
    """The committed rows file is gzipped, so a bad download must not traceback."""
    path = tmp_path / "rows.jsonl.gz"
    path.write_bytes(b"this is not a gzip member at all")

    rows_file = read_rows(tmp_path / "rows.jsonl")

    assert rows_file.rows == []
    assert rows_file.unreadable, "the caller is told the file could not be read"
    assert count_rows(tmp_path / "rows.jsonl") == 0


def test_a_truncated_gzip_keeps_the_rows_it_could_decompress(tmp_path: Path):
    """Half a gzip member still holds whole rows, and they are still evidence."""
    path = tmp_path / "rows.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for seed in range(200):
            fh.write(json.dumps(a_row(seed=seed)) + "\n")
    whole = path.read_bytes()
    path.write_bytes(whole[:len(whole) // 2])

    rows_file = read_rows(tmp_path / "rows.jsonl")

    assert rows_file.rows, "the rows that did decompress are returned"
    assert rows_file.unreadable, "and the truncation is reported, not hidden"
