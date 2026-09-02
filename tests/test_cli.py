"""Tests for the command line interface.

The CLI is the surface a reviewer touches first, so these tests exercise it the
way a person would: through the console script's Typer application, checking
exit codes and what is actually printed, rather than calling the underlying
functions directly. Budgets are kept to a second or two per run because the
point here is that the plumbing works end to end, not that the optimiser is
good; algorithm quality is measured by the benchmark sweep, not by the tests.

The heavier commands (a benchmark sweep, a road-network build) are still run
for real rather than mocked, because the failure mode this file exists to catch
is exactly the one mocking hides: a signature that no longer matches the
library the CLI is a front end for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from qroute.cli import render
from qroute.cli.main import _instance_json, _parse_params, app

runner = CliRunner()
WIDE = {"COLUMNS": "200", "TERM": "dumb"}

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
OSM_DIR = REPO_ROOT / "data" / "osm"


def invoke(*args: str, **kwargs):
    """Run the application with a wide terminal so tables are not truncated."""
    return runner.invoke(app, list(args), env=WIDE, **kwargs)


# ---------------------------------------------------------------------------
# Help and discovery
# ---------------------------------------------------------------------------
def test_help_lists_every_command():
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("solve", "compare", "bench", "report", "exact",
                    "instances", "osm", "serve", "version"):
        assert command in result.output


@pytest.mark.parametrize("command", ["solve", "compare", "bench", "report", "exact",
                                     "instances", "serve", "version", "osm"])
def test_each_command_has_help(command):
    result = invoke(command, "--help")
    assert result.exit_code == 0, result.output


def test_osm_subcommands_have_help():
    for sub in ("build", "demo"):
        result = invoke("osm", sub, "--help")
        assert result.exit_code == 0, result.output


def test_version_reports_the_libraries_results_depend_on():
    result = invoke("version")
    assert result.exit_code == 0
    for package in ("numpy", "ortools", "numba", "qpso"):
        assert package in result.output


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def test_parse_params_infers_types():
    parsed = _parse_params(["swarm_size=60", "beta_end=0.4", "local_search=false",
                            "beta_schedule=linear"])
    assert parsed == {"swarm_size": 60, "beta_end": 0.4, "local_search": False,
                      "beta_schedule": "linear"}
    assert isinstance(parsed["swarm_size"], int)


def test_unknown_instance_exits_non_zero():
    result = invoke("solve", "definitely-not-an-instance", "--seconds", "1")
    assert result.exit_code == 1


def test_unknown_algorithm_exits_non_zero():
    result = invoke("solve", "P-n16-k8", "--algorithm", "quantum-annealer", "--seconds", "1")
    assert result.exit_code == 1


def test_malformed_parameter_exits_non_zero():
    result = invoke("solve", "P-n16-k8", "--seconds", "1", "--params", "swarm_size")
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
def test_instances_lists_small_cvrp_instances():
    result = invoke("instances", "--family", "cvrp", "--max-size", "20")
    assert result.exit_code == 0
    assert "P-n16-k8" in result.output
    assert "A-n80-k10" not in result.output          # filtered out by --max-size


def test_instances_rejects_an_unknown_family():
    result = invoke("instances", "--family", "tsp")
    assert result.exit_code == 1


def test_instances_vrptw_reports_time_windows():
    result = invoke("instances", "--family", "vrptw", "--max-size", "100")
    assert result.exit_code == 0
    assert "C101" in result.output
    assert "yes" in result.output


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def test_solve_reaches_the_optimum_of_a_tiny_instance(tmp_path: Path):
    out = tmp_path / "solution.json"
    result = invoke("solve", "P-n16-k8", "--seconds", "2", "--seed", "0",
                    "--json", str(out))
    assert result.exit_code == 0, result.output
    assert "gap to best known" in result.output

    payload = json.loads(out.read_text())
    assert payload["instance"] == "P-n16-k8"
    assert payload["bks"] == 450.0
    assert payload["feasible"] is True
    # P-n16-k8 has 15 customers; two seconds of any of these searches closes it.
    assert payload["best_cost"] == pytest.approx(450.0)
    assert payload["gap"] == pytest.approx(0.0, abs=1e-9)
    assert sorted(c for route in payload["routes"] for c in route) == list(range(1, 16))


def test_solve_accepts_algorithm_parameters():
    result = invoke("solve", "P-n16-k8", "--seconds", "1", "--algorithm", "qpso",
                    "--params", "swarm_size=12", "--no-routes")
    assert result.exit_code == 0, result.output
    assert "QPSO" in result.output or "qpso" in result.output


def test_solve_runs_the_random_restart_control():
    result = invoke("solve", "P-n16-k8", "--seconds", "1", "--algorithm", "random",
                    "--no-routes")
    assert result.exit_code == 0, result.output


def test_solve_is_deterministic_for_a_fixed_seed(tmp_path: Path):
    costs = []
    for k in range(2):
        out = tmp_path / f"run{k}.json"
        result = invoke("solve", "A-n32-k5", "--seconds", "2", "--seed", "7",
                        "--json", str(out), "--no-routes")
        assert result.exit_code == 0, result.output
        costs.append(json.loads(out.read_text())["best_cost"])
    assert costs[0] == costs[1]


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def test_compare_reports_every_algorithm_and_a_test(tmp_path: Path):
    out = tmp_path / "compare.json"
    result = invoke("compare", "P-n16-k8", "--algorithms", "qpso,sa", "--seeds", "2",
                    "--seconds", "1", "--json", str(out))
    assert result.exit_code == 0, result.output
    assert "qpso" in result.output and "sa" in result.output

    payload = json.loads(out.read_text())
    assert set(payload["runs"]) == {"qpso", "sa"}
    assert len(payload["runs"]["qpso"]) == 2
    assert len(payload["seeds"]) == 2
    assert payload["seeds"][0] != payload["seeds"][1]
    for entry in payload["summary"]:
        assert entry["runs"] == 2
        assert entry["cost_best"] <= entry["cost_mean"] + 1e-9


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------
def test_exact_proves_the_optimum_of_p_n16_k8(tmp_path: Path):
    out = tmp_path / "exact.json"
    result = invoke("exact", "P-n16-k8", "--method", "cpsat", "--seconds", "60",
                    "--no-bounds", "--json", str(out))
    assert result.exit_code == 0, result.output

    payload = json.loads(out.read_text())
    assert payload["proven_optimal"] is True
    assert payload["cost"] == pytest.approx(450.0)
    assert payload["lower_bound"] == pytest.approx(450.0)
    assert payload["gap"] == pytest.approx(0.0, abs=1e-6)


def test_exact_bounds_bracket_the_optimum():
    result = invoke("exact", "P-n16-k8", "--method", "heldkarp")
    assert result.exit_code == 0, result.output
    assert "lower bounds" in result.output
    assert "optimality proved" in result.output


def test_exact_rejects_an_unknown_method():
    result = invoke("exact", "P-n16-k8", "--method", "branch-and-price")
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# bench and report
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def swept(tmp_path_factory) -> Path:
    """Run one very small sweep and reuse it for every report test."""
    out_dir = tmp_path_factory.mktemp("runs")
    config = out_dir / "test.yaml"
    config.write_text(yaml.safe_dump({
        "name": "cli-test",
        "instances": ["P-n16-k8", "A-n32-k5"],
        "algorithms": ["qpso", "random"],
        "seeds": 2,
        "master_seed": 20260920,
        "max_seconds": 1,
        "workers": 2,
        "output_dir": str(out_dir),
        "save_history": True,
    }))
    result = runner.invoke(app, ["bench", "--config", str(config)], env=WIDE)
    assert result.exit_code == 0, result.output
    return out_dir / "cli-test"


def test_bench_writes_the_reproducibility_record(swept: Path):
    meta = json.loads((swept / "meta.json").read_text())
    assert meta["n_tasks"] == 2 * 2 * 2
    assert meta["config"]["master_seed"] == 20260920
    assert meta["environment"]["packages"]["numpy"]
    rows = [json.loads(line) for line in (swept / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert all(r["status"] == "ok" for r in rows), [r for r in rows if r["status"] != "ok"]


def test_bench_rejects_an_unknown_configuration_key(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump({"name": "bad", "instance": ["P-n16-k8"]}))
    result = invoke("bench", "--config", str(config))
    assert result.exit_code == 1


def test_bench_rejects_a_missing_configuration(tmp_path: Path):
    result = invoke("bench", "--config", str(tmp_path / "nope.yaml"))
    assert result.exit_code == 1


def test_report_table(swept: Path):
    result = invoke("report", str(swept))
    assert result.exit_code == 0, result.output
    assert "P-n16-k8" in result.output
    assert "qpso" in result.output and "random" in result.output


def test_report_markdown_is_a_valid_table(swept: Path, tmp_path: Path):
    out = tmp_path / "report.md"
    result = invoke("report", str(swept), "--format", "markdown", "--out", str(out))
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert text.startswith("# Benchmark: cli-test")
    assert "| instance |" in text
    header = next(line for line in text.splitlines() if line.startswith("| instance |"))
    assert header.count("|") == len(("instance", "qpso", "random")) + 1


def test_report_csv_has_one_line_per_run(swept: Path, tmp_path: Path):
    out = tmp_path / "report.csv"
    result = invoke("report", str(swept), "--format", "csv", "--out", str(out))
    assert result.exit_code == 0, result.output
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("instance,algorithm,seed,cost,gap")
    assert len(lines) == 1 + 8


def test_report_plots_are_written(swept: Path):
    result = invoke("report", str(swept), "--plots")
    assert result.exit_code == 0, result.output
    figures = swept / "figures"
    assert (figures / "convergence.png").exists()
    assert (figures / "gap_distribution.png").stat().st_size > 1000


def test_report_rejects_an_unknown_format(swept: Path):
    assert invoke("report", str(swept), "--format", "latex").exit_code == 1


def test_report_rejects_a_directory_without_results(tmp_path: Path):
    assert invoke("report", str(tmp_path)).exit_code == 1


# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["quick", "tier0", "tier1", "tier2", "vrptw"])
def test_shipped_config_is_valid_and_names_instances_we_have(name):
    from qroute.algorithms.registry import ALGORITHMS
    from qroute.benchmark.runner import BenchmarkConfig
    from qroute.problems.loaders import list_instances

    config = BenchmarkConfig.from_yaml(CONFIG_DIR / f"{name}.yaml")
    assert config.name == name
    assert config.instances and config.algorithms and config.seeds >= 1

    available = set(sum(list_instances().values(), []))
    missing = [i for i in config.instances if i not in available]
    assert not missing, f"{name}.yaml names instances that are not on disk: {missing}"

    known = set(ALGORITHMS) | {"ortools", "pyvrp", "cpsat", "random"}
    assert not set(config.algorithms) - known


def test_tier0_instances_are_small_enough_for_an_exact_solver():
    from qroute.benchmark.runner import BenchmarkConfig
    from qroute.problems.loaders import load

    config = BenchmarkConfig.from_yaml(CONFIG_DIR / "tier0.yaml")
    for name in config.instances:
        assert load(name).n_customers <= 35, name


def test_vrptw_config_only_names_time_window_instances():
    from qroute.benchmark.runner import BenchmarkConfig
    from qroute.problems.loaders import load

    config = BenchmarkConfig.from_yaml(CONFIG_DIR / "vrptw.yaml")
    for name in config.instances:
        assert load(name).has_time_windows, name


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
def test_serve_reports_a_missing_application_rather_than_crashing():
    result = invoke("serve", "--app", "qroute.no_such_module:app")
    assert result.exit_code == 1


def test_serve_starts_uvicorn_with_the_resolved_application(monkeypatch):
    calls: list[tuple] = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run",
                        lambda target, **kw: calls.append((target, kw)))
    result = invoke("serve", "--host", "0.0.0.0", "--port", "9999")
    if result.exit_code == 1:
        pytest.skip("the API application is not importable in this checkout")
    assert result.exit_code == 0, result.output
    assert calls and calls[0][1]["port"] == 9999
    assert calls[0][1]["host"] == "0.0.0.0"


# ---------------------------------------------------------------------------
# Road networks
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (OSM_DIR / "delhi_connaught.graphml").exists(),
                    reason="road network extracts are not present")
def test_osm_build_writes_a_reusable_instance(tmp_path: Path):
    out = tmp_path / "instance.json"
    result = invoke("osm", "build", "--place-file", str(OSM_DIR / "delhi_connaught.graphml"),
                    "--customers", "8", "--seed", "1", "--save", str(out))
    assert result.exit_code == 0, result.output

    payload = json.loads(out.read_text())
    assert payload["n_customers"] == 8
    assert len(payload["duration"]) == 9
    assert len(payload["node_ids"]) == 9
    assert payload["meta"]["network"] == "delhi_connaught"
    # Travel times on a strongly connected network are finite and positive off
    # the diagonal; an infinity here would poison every cost comparison.
    for i, row in enumerate(payload["duration"]):
        for j, value in enumerate(row):
            assert math.isfinite(value)
            assert value > 0 or i == j


@pytest.mark.skipif(not (OSM_DIR / "delhi_connaught.graphml").exists(),
                    reason="road network extracts are not present")
def test_osm_build_can_solve_under_traffic():
    result = invoke("osm", "build", "--place-file", "delhi_connaught",
                    "--customers", "10", "--seed", "2", "--hour", "9",
                    "--solve", "--seconds", "2")
    assert result.exit_code == 0, result.output
    assert "traffic at 9:00" in result.output
    assert "objective is travel time in seconds" in result.output


def test_osm_build_rejects_an_unknown_network():
    result = invoke("osm", "build", "--place-file", "atlantis_downtown", "--customers", "5")
    assert result.exit_code == 1


def test_osm_build_refuses_geojson_without_a_solve(tmp_path: Path):
    result = invoke("osm", "build", "--place-file", "delhi_connaught", "--customers", "5",
                    "--no-solve", "--geojson", str(tmp_path / "routes.geojson"))
    assert result.exit_code == 1


@pytest.mark.skipif(not (OSM_DIR / "delhi_connaught.graphml").exists(),
                    reason="road network extracts are not present")
def test_osm_demo_reoptimises_after_an_incident(tmp_path: Path):
    out = tmp_path / "demo.json"
    result = invoke("osm", "demo", "--network", "delhi_connaught", "--hour", "9",
                    "--customers", "15", "--seconds", "3", "--json", str(out))
    assert result.exit_code == 0, result.output

    payload = json.loads(out.read_text())
    stages = {s["stage"]: s for s in payload["stages"]}
    assert len(stages) == 4
    disrupted = next(s for name, s in stages.items() if name == "same plan, after")
    # The incident is placed on the arcs the plan uses most, so carrying on with
    # the original routes must cost more than the plan did before it.
    assert disrupted["cost"] > payload["stages"][0]["cost"]
    warm = next(s for name, s in stages.items() if name == "re-optimised (warm)")
    assert warm["cost"] <= disrupted["cost"] + 1e-6
    assert payload["incident_edges"]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def test_gap_style_bands_are_ordered_from_good_to_bad():
    assert render.gap_style(0.0) == "bold bright_green"
    assert render.gap_style(0.2) == "green"
    assert render.gap_style(1.0) == "yellow"
    assert render.gap_style(3.0) == "dark_orange"
    assert render.gap_style(20.0) == render.GAP_WORST_STYLE
    assert render.gap_style(None) == "dim"
    assert render.gap_style(float("nan")) == "dim"


def test_format_gap_and_numbers_never_print_nan():
    assert str(render.format_gap(None)) == "-"
    assert str(render.format_gap(float("inf"))) == "-"
    assert str(render.format_gap(1.5)) == "+1.50%"
    assert render.format_number(float("nan")) == "-"
    assert render.format_number(1234.5) == "1,234.50"
    assert render.format_seconds(0.25) == "250 ms"
    assert render.format_seconds(3.5) == "3.50 s"
    assert render.format_seconds(300) == "5.0 min"
    assert render.format_duration_hms(3725) == "1h 02m"
    assert render.format_duration_hms(65) == "1m 05s"


def test_sparkline_shape():
    assert render.sparkline([]) == ""
    assert render.sparkline([5.0, 5.0, 5.0]) == render.BLOCKS[0] * 3
    line = render.sparkline([10.0, 8.0, 6.0, 4.0])
    assert line[0] == render.BLOCKS[-1] and line[-1] == render.BLOCKS[0]
    assert len(render.sparkline(list(range(1000)), width=40)) == 40
    # A non-finite entry must be dropped rather than blowing up the scale.
    assert len(render.sparkline([1.0, float("nan"), 2.0])) == 2


def test_convergence_line_accepts_stored_history_dictionaries():
    history = [{"t": 0.1, "i": 1, "c": 10.0}, {"t": 0.2, "i": 2, "c": 9.0}]
    line, first, last = render.convergence_line(history)
    assert len(line) == 2
    assert (first, last) == (10.0, 9.0)


def test_csv_rows_keeps_the_documented_columns():
    text = render.csv_rows([{"instance": "A", "algorithm": "qpso", "cost": 1.0,
                             "unexpected": "ignored"}])
    header, row = text.strip().splitlines()
    assert header.split(",") == list(render.CSV_COLUMNS)
    assert "ignored" not in row


def test_markdown_report_handles_a_run_without_an_omnibus_test():
    summary = {
        "algorithms": ["qpso"], "instances": ["A-n32-k5"],
        "cells": {"A-n32-k5|qpso": {
            "instance": "A-n32-k5", "algorithm": "qpso", "runs": 2,
            "cost": {"median": 784.0}, "gap": {"best": 0.0, "median": 0.0,
                                               "mean": 0.0, "std": 0.0},
            "feasible_runs": 2, "hit_bks": 2,
        }},
        "n_ok": 2, "n_failed": 0, "omnibus": None,
    }
    text = render.markdown_report(summary)
    assert "A-n32-k5" in text
    assert "Statistical comparison" not in text


def test_instance_json_round_trips_through_the_instance_model():
    from qroute.problems.instance import Instance
    from qroute.problems.loaders import load

    original = load("P-n16-k8")
    payload = _instance_json(original)
    rebuilt = Instance(
        name=payload["name"],
        distance=payload["distance"],
        duration=payload["duration"],
        demand=payload["demand"],
        capacity=payload["capacity"],
    )
    assert rebuilt.n_customers == original.n_customers
    assert rebuilt.evaluate([[1, 2, 3]]).distance == original.evaluate([[1, 2, 3]]).distance
