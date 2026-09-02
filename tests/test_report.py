"""Tests for the reporting and plotting layer.

No solver runs here. The point of these tests is the presentation logic, so the
input is a synthetic result set built to contain exactly the awkward cases a
real run produces: an instance with no best-known cost, an algorithm that never
reaches a target, a run that crashed, and a single-run cell whose standard
deviation is undefined. What is checked is that those cases come out as an em
dash or as an explicit "not reached" rather than as ``None``, a crash, or a
silently dropped row.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qroute.benchmark import plots, report
from qroute.benchmark.report import EM_DASH
from qroute.core.rng import make_rng

ALGORITHMS = ["qpso", "qpso-nols", "pso", "ga", "random"]

# Mean quality of each algorithm, as a multiplier on the best known cost. These
# are fixed so that qpso is the best algorithm on every instance, which is what
# the best-cell marking is checked against.
QUALITY = {"qpso": 1.010, "qpso-nols": 1.035, "pso": 1.060, "ga": 1.045, "random": 1.090}

INSTANCES = [
    # name, size, best known cost
    ("T-n20-k3", 20, 500.0),
    ("T-n55-k5", 55, 900.0),
    ("T-n120-k9", 120, 1700.0),
    ("T-n40-k4", 40, None),      # no reference solution: every gap is missing
]


def _history(rng, best_start: float, best_end: float, n: int, seconds: float) -> list[dict]:
    """A plausible monotone convergence trace."""
    out = []
    cost = best_start
    for i in range(1, n + 1):
        frac = i / n
        target = best_start + (best_end - best_start) * frac ** 0.6
        cost = min(cost, float(target))
        out.append({
            "t": round(seconds * frac, 4),
            "i": i,
            "c": cost,
            "m": cost * (1.0 + 0.05 * (1.0 - frac)),
            "d": float(max(0.0, 1.0 - frac) * 10.0 + rng.uniform(0, 0.2)),
        })
    return out


def synthetic_rows(seeds: int = 4) -> list[dict]:
    """A complete, deterministic stand-in for ``rows.jsonl``."""
    rng = make_rng(20260920)
    rows: list[dict] = []
    for name, _size, bks in INSTANCES:
        reference = bks if bks is not None else 1000.0
        for algo in ALGORITHMS:
            for k in range(seeds):
                factor = QUALITY[algo] * (1.0 + rng.uniform(-0.004, 0.004))
                cost = reference * factor
                seconds = 8.0
                n_iters = {"qpso": 120, "qpso-nols": 260, "pso": 300, "ga": 400,
                           "random": 900}[algo]
                row = {
                    "instance": name,
                    "algorithm": algo,
                    "seed": 1000 + k,
                    "cost": float(cost),
                    "gap": (100.0 * (cost - bks) / bks) if bks else None,
                    "bks": bks,
                    "n_routes": 5,
                    "feasible": True,
                    "violation": 0.0,
                    "iterations": n_iters,
                    "evaluations": n_iters * 30,
                    "seconds": seconds,
                    "params": {},
                    "status": "ok",
                    "history": _history(rng, reference * 1.6, cost, 24, seconds),
                }
                rows.append(row)
    # One run that produced a solution but for which the history was not saved,
    # and one that crashed outright.
    rows.append({**rows[0], "seed": 9999, "history": []})
    rows.append({"instance": "T-n55-k5", "algorithm": "ga", "seed": 4242,
                 "status": "error", "error": "RuntimeError: synthetic failure",
                 "seconds": 0.4})
    return rows


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return synthetic_rows()


@pytest.fixture(scope="module")
def tables(rows) -> dict[str, report.Table]:
    return report.all_tables(rows)


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_values_render_as_an_em_dash(value):
    assert report.fmt(value) == EM_DASH
    assert report.fmt_int(value) == EM_DASH
    assert report.fmt_gap(value) == EM_DASH
    assert report.fmt_seconds(value) == EM_DASH


def test_numbers_are_formatted_consistently():
    assert report.fmt(1.23456, 2) == "1.23"
    assert report.fmt_gap(0.0) == "0.00"
    assert report.fmt_seconds(1.0) == "1.000"
    assert report.fmt_int(3.7) == "4"


def test_instance_size_is_read_from_the_name_and_from_an_override():
    assert report.instance_size("T-n120-k9") == 120
    assert report.instance_size("whatever", {"whatever": 77}) == 77
    assert report.instance_size("no-size-here-at-all") is None
    assert report.tier_of(20).startswith("small")
    assert report.tier_of(55).startswith("medium")
    assert report.tier_of(120).startswith("large")
    assert report.tier_of(None) == "unknown size"


# ---------------------------------------------------------------------------
# Every table renders in every format
# ---------------------------------------------------------------------------
EXPECTED_TABLES = {"main_results", "tier_summary", "statistics", "convergence",
                   "ablation", "per_instance_detail"}


def test_all_tables_are_built(tables):
    assert EXPECTED_TABLES <= set(tables)


@pytest.mark.parametrize("stem", sorted(EXPECTED_TABLES))
def test_every_table_renders_in_every_format(tables, stem):
    table = tables[stem]
    md = table.to_markdown()
    csv_text = table.to_csv()
    text = table.to_text()
    rich_table = table.to_rich()

    assert table.rows, f"{stem} produced no rows"
    assert md.startswith("###") or md.startswith("##")
    # Header, separator and one line per row, at minimum.
    assert len(md.splitlines()) > len(table.rows)
    assert len(csv_text.splitlines()) == len(table.rows) + 1
    assert table.columns[0] in text
    assert rich_table.row_count == len(table.rows)


@pytest.mark.parametrize("stem", sorted(EXPECTED_TABLES))
def test_no_table_ever_prints_none(tables, stem):
    table = tables[stem]
    for row in table.rows:
        for cell in row:
            assert cell is not None
            assert str(cell) not in ("None", "nan", "inf", "-inf", "NaN")
    assert "None" not in table.to_markdown()
    assert "None" not in table.to_csv()


def test_csv_is_machine_readable_without_markup(tables):
    import csv
    import io

    text = tables["main_results"].to_csv()
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0][0] == "Instance"
    assert all(len(r) == len(parsed[0]) for r in parsed)
    assert "**" not in text


# ---------------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------------
def test_main_results_marks_the_best_cell_on_every_row(rows):
    table = report.main_results_table(rows)
    algos = table.columns[3:-1]
    assert set(algos) == set(ALGORITHMS)
    marked_rows = {r for r, _ in table.highlight}

    for r, row in enumerate(table.rows):
        if row[0] == "T-n40-k4":            # no best known cost, so no winner
            continue
        assert r in marked_rows, f"row {row[0]} has no marked cell"
        cols = sorted(c for rr, c in table.highlight if rr == r)
        # Every marked cell must show the smallest mean gap on that row, and no
        # unmarked cell may match it.
        means = []
        for c, algo in enumerate(algos, start=3):
            cell = row[c]
            means.append(math.inf if cell == EM_DASH else float(cell.split(" ")[0]))
        best = min(means)
        assert {i for i, m in enumerate(means) if m == best} == {c - 3 for c in cols}
        assert len(cols) == 1, "the fixture has no ties"
        assert row[-1] == algos[cols[0] - 3]
        assert row[-1] == "qpso"            # by construction of the fixture


def test_main_results_bolds_the_best_cell_in_markdown(rows):
    md = report.main_results_table(rows).to_markdown()
    assert "**" in md
    # One bolded cell for each instance that has a best known cost.
    assert md.count("**") == 2 * sum(1 for _n, _s, b in INSTANCES if b is not None)


def test_tied_algorithms_are_all_marked_rather_than_one_chosen_arbitrarily():
    """Two algorithms that print the same gap are tied, and the table must say so
    instead of letting the column order pick a winner."""
    tied = []
    for algo in ("alpha", "omega"):
        for k in range(3):
            tied.append({"instance": "T-n20-k3", "algorithm": algo, "seed": k,
                         "cost": 505.0, "gap": 1.0, "bks": 500.0, "feasible": True,
                         "seconds": 1.0, "iterations": 10, "status": "ok"})
    table = report.main_results_table(tied)
    assert len(table.highlight) == 2
    assert table.rows[0][-1] == "tie (all equal)"
    assert table.to_markdown().count("**") == 4


def test_instance_without_a_best_known_cost_is_all_em_dashes(rows):
    table = report.main_results_table(rows)
    row = next(r for r in table.rows if r[0] == "T-n40-k4")
    assert row[2] == EM_DASH                       # best known column
    assert all(cell == EM_DASH for cell in row[3:])


def test_failed_runs_are_reported_rather_than_hidden(rows):
    table = report.main_results_table(rows)
    assert any("failed" in note for note in table.notes)
    assert len(report.failed_rows(rows)) == 1


# ---------------------------------------------------------------------------
# Tier, statistics, convergence, ablation, detail
# ---------------------------------------------------------------------------
def test_tier_summary_covers_every_tier_present(tables):
    tiers = {row[0] for row in tables["tier_summary"].rows}
    assert any(t.startswith("small") for t in tiers)
    assert any(t.startswith("medium") for t in tiers)
    assert any(t.startswith("large") for t in tiers)
    for row in tables["tier_summary"].rows:
        assert "/" in row[5]                       # runs at best known, as k/n


def test_statistical_table_gives_ranks_a_p_value_and_sentences(tables):
    table = tables["statistics"]
    assert {row[0] for row in table.rows} == set(ALGORITHMS)
    assert any("Friedman" in note and "p =" in note for note in table.notes)
    assert any("Holm" in note for note in table.notes)
    control_rows = [row for row in table.rows if "control" in row[4]]
    assert len(control_rows) == 1
    assert control_rows[0][2] == EM_DASH           # a control has no p against itself
    for row in table.rows:
        assert len(row[4].split()) >= 3, "the finding must read as a sentence"


def test_statistical_table_declines_politely_when_there_is_too_little_data(rows):
    few = [r for r in rows if r["algorithm"] in ("qpso", "pso")]
    table = report.statistical_table(few)
    assert any("Not computed" in note for note in table.notes)
    assert all(row[1] == EM_DASH for row in table.rows)


def test_convergence_table_says_not_reached_instead_of_dropping_runs(rows):
    # A target that no synthetic run can reach: every cell must say so.
    table = report.convergence_table(rows, targets=(0.001,))
    assert table.rows
    assert all(row[2] == "not reached" for row in table.rows)
    assert all(row[4].startswith("0/") for row in table.rows)
    assert any("not reached" in note for note in table.notes)


def test_convergence_table_reports_reached_targets_with_counts(rows):
    table = report.convergence_table(rows, targets=(5.0,))
    reached = [row for row in table.rows if row[2] != "not reached"]
    assert reached, "some algorithm should reach a 5% target in the fixture"
    for row in reached:
        hit, total = row[4].split("/")
        assert 0 < int(hit) <= int(total)


def test_ablation_table_compares_the_arms_against_the_full_method(tables):
    table = tables["ablation"]
    variants = [row[0] for row in table.rows]
    assert variants[0] == "qpso"
    assert "qpso-nols" in variants
    assert "random" in variants
    assert table.rows[0][5] == EM_DASH             # the reference has no delta
    for row in table.rows[1:]:
        assert row[5][0] in "+-"
    # Removing local search must show up as a worse mean gap in this fixture.
    nols = next(row for row in table.rows if row[0] == "qpso-nols")
    assert float(nols[5]) > 0


def test_ablation_returns_none_when_there_is_nothing_to_ablate(rows):
    only_pso = [r for r in rows if r["algorithm"] == "pso"]
    assert report.ablation_table(only_pso) is None


def test_per_instance_detail_has_a_full_distribution(tables):
    table = tables["per_instance_detail"]
    for name in ("Best cost", "Mean cost", "Median cost", "Std dev", "Worst cost",
                 "Feasible runs"):
        assert name in table.columns
    i_best = table.columns.index("Best cost")
    i_worst = table.columns.index("Worst cost")
    for row in table.rows:
        assert float(row[i_best]) <= float(row[i_worst])


def test_single_run_cell_reports_a_zero_standard_deviation(rows):
    one = [r for r in rows if r["algorithm"] == "pso" and r["instance"] == "T-n20-k3"][:1]
    table = report.per_instance_detail_table(one)
    assert table.rows[0][table.columns.index("Std dev")] == "0.00"


# ---------------------------------------------------------------------------
# Whole report
# ---------------------------------------------------------------------------
def test_build_report_writes_markdown_and_a_csv_per_table(rows, tmp_path):
    meta = {"config": {"name": "unit-test", "max_seconds": 8, "seeds": 4,
                       "master_seed": 20260920},
            "environment": {"python": "3.13.0", "platform": "test", "cpu_count": 10,
                            "packages": {"numpy": "2.0.0"}}}
    out = report.build_report(rows, tmp_path, meta=meta)
    md = out["report_md"].read_text()
    assert md.startswith("# Benchmark report")
    assert "1 runs failed" in md or "runs failed" in md
    assert "None" not in md
    for stem in EXPECTED_TABLES:
        path = Path(tmp_path) / f"{stem}.csv"
        assert path.exists() and path.stat().st_size > 0


def test_render_console_does_not_raise(tables):
    from rich.console import Console

    console = Console(file=open("/dev/null", "w"), width=200)
    report.render_console(tables, console=console)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _nonempty(paths) -> None:
    assert paths, "the plot function wrote nothing"
    for p in paths:
        p = Path(p)
        assert p.exists(), f"{p} was not written"
        assert p.stat().st_size > 1000, f"{p} is suspiciously small"
    assert {p.suffix for p in map(Path, paths)} == {".png", ".svg"}


def test_convergence_time_writes_a_figure_per_instance(rows, tmp_path):
    paths = plots.convergence_time(rows, tmp_path)
    _nonempty(paths)
    assert len(paths) == 2 * len(INSTANCES)


def test_convergence_iterations_writes_files(rows, tmp_path):
    _nonempty(plots.convergence_iterations(rows, tmp_path, instance="T-n55-k5"))


def test_time_to_target_writes_files(rows, tmp_path):
    _nonempty(plots.time_to_target(rows, tmp_path, pct=5.0))


def test_time_to_target_still_draws_when_nothing_reaches_the_target(rows, tmp_path):
    _nonempty(plots.time_to_target(rows, tmp_path, pct=0.001))


def test_gap_distribution_writes_files(rows, tmp_path):
    _nonempty(plots.gap_distribution(rows, tmp_path))


def test_scalability_writes_files(rows, tmp_path):
    _nonempty(plots.scalability(rows, tmp_path))


def test_diversity_writes_files(rows, tmp_path):
    _nonempty(plots.diversity(rows, tmp_path, instance="T-n120-k9"))


def test_per_instance_gap_bars_writes_files(rows, tmp_path):
    _nonempty(plots.per_instance_gap_bars(rows, tmp_path))


def test_all_plots_writes_every_figure(rows, tmp_path):
    paths = plots.all_plots(rows, tmp_path)
    _nonempty(paths)
    stems = {Path(p).stem for p in paths}
    assert any(s.startswith("convergence_time_") for s in stems)
    assert any(s.startswith("convergence_iterations_") for s in stems)
    assert any(s.startswith("time_to_target_") for s in stems)
    assert any(s.startswith("diversity_") for s in stems)
    assert {"gap_distribution", "scalability", "per_instance_gap"} <= stems


def test_plots_are_deterministic(rows, tmp_path):
    a = Path(tmp_path) / "a"
    b = Path(tmp_path) / "b"
    first = plots.gap_distribution(rows, a)[0].read_bytes()
    second = plots.gap_distribution(rows, b)[0].read_bytes()
    assert first == second
