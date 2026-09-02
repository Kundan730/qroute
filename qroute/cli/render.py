"""Terminal rendering for the command line interface.

Everything the CLI puts on screen is built here, so the commands themselves
stay a thin layer over the library. Three things are worth explaining.

*Gap colouring.* A gap to the best-known solution is the single number a judge
will look for, so it is coloured by how good it is rather than printed as an
undifferentiated float: zero (the published optimum was matched) is bright
green, a fraction of a percent is green, a couple of percent is yellow, and
anything worse fades through orange to red. The thresholds are stated once in
:data:`GAP_BANDS` and are the same in every table.

*Sparklines.* A convergence curve is the other thing worth seeing immediately,
and opening a PNG breaks the flow of a terminal session. :func:`sparkline`
compresses a run's best-cost history into a line of Unicode block characters
whose height is the cost relative to the run's own range, which is enough to
tell "converged early and sat there" from "still improving when the budget ran
out".

*Plain text fallbacks.* The same tables are also produced as Markdown and CSV
by :func:`markdown_report` and :func:`csv_rows`, because a report that can only
be read in a terminal cannot be pasted into the submission document.

No colour is used to carry information that is not also carried by the number
itself, so piping the output through a file loses nothing.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

# Block characters from lowest to highest, used by :func:`sparkline`.
BLOCKS = "▁▂▃▄▅▆▇█"

#: ``(upper bound on gap in percent, rich style)``, checked in order. A gap at
#: or below the bound takes that style; anything above every bound is red.
GAP_BANDS: tuple[tuple[float, str], ...] = (
    (1e-9, "bold bright_green"),   # matched or beat the best known solution
    (0.5, "green"),
    (1.5, "yellow"),
    (5.0, "dark_orange"),
)
GAP_WORST_STYLE = "bold red"


def console(**kwargs: Any) -> Console:
    """A console configured the way every command wants it.

    ``soft_wrap`` is off so wide tables are truncated rather than reflowed into
    an unreadable stack, and highlighting is off so numbers are not recoloured
    behind the deliberate gap colouring.
    """
    kwargs.setdefault("highlight", False)
    return Console(**kwargs)


# ---------------------------------------------------------------------------
# Scalar formatting
# ---------------------------------------------------------------------------
def gap_style(gap: Optional[float]) -> str:
    """Rich style for a percentage gap; see :data:`GAP_BANDS`."""
    if gap is None or not math.isfinite(gap):
        return "dim"
    for bound, style in GAP_BANDS:
        if gap <= bound:
            return style
    return GAP_WORST_STYLE


def format_gap(gap: Optional[float], digits: int = 2) -> Text:
    """A gap as coloured text, or a dim dash when there is no reference."""
    if gap is None or not math.isfinite(gap):
        return Text("-", style="dim")
    return Text(f"{gap:+.{digits}f}%", style=gap_style(gap))


def format_number(value: Optional[float], digits: int = 2) -> str:
    """Compact fixed-point formatting that never prints ``nan`` or ``inf``."""
    if value is None or not math.isfinite(value):
        return "-"
    if abs(value) >= 1e6:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def format_seconds(seconds: Optional[float]) -> str:
    """Durations in the unit a reader expects at that magnitude."""
    if seconds is None or not math.isfinite(seconds):
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def format_duration_hms(seconds: Optional[float]) -> str:
    """Travel times on a road network, which are naturally hours and minutes."""
    if seconds is None or not math.isfinite(seconds):
        return "-"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def feasibility(flag: bool) -> Text:
    return Text("yes", style="green") if flag else Text("NO", style="bold red")


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------
def sparkline(values: Sequence[float], width: int = 48) -> str:
    """A one-line Unicode plot of ``values``.

    The series is resampled to ``width`` points by taking the minimum of each
    bucket -- the running best is what a convergence curve means -- and scaled
    to its own minimum and maximum, so the line shows the *shape* of the
    descent, not its absolute magnitude. A flat series renders as a flat low
    line rather than as noise from dividing by a zero range.
    """
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return ""
    if len(finite) > width > 0:
        step = len(finite) / width
        buckets = []
        for i in range(width):
            lo = int(i * step)
            hi = max(lo + 1, int((i + 1) * step))
            buckets.append(min(finite[lo:hi]))
        finite = buckets
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 0:
        return BLOCKS[0] * len(finite)
    scale = len(BLOCKS) - 1
    return "".join(BLOCKS[int(round((v - lo) / span * scale))] for v in finite)


def convergence_line(history: Sequence[Mapping[str, Any]] | Sequence[Any],
                     width: int = 48) -> tuple[str, float, float]:
    """Sparkline plus first and last best cost of a history.

    Accepts either :class:`~qroute.algorithms.base.IterationRecord` objects or
    the plain dictionaries stored in ``rows.jsonl`` (whose key is ``"c"``).
    """
    costs: list[float] = []
    for h in history:
        if isinstance(h, Mapping):
            value = h.get("c", h.get("best_cost"))
        else:
            value = getattr(h, "best_cost", None)
        if value is not None:
            costs.append(float(value))
    if not costs:
        return "", float("nan"), float("nan")
    return sparkline(costs, width), costs[0], costs[-1]


# ---------------------------------------------------------------------------
# Single-run tables
# ---------------------------------------------------------------------------
def _table(title: str, *columns: tuple[str, dict], box_title_style: str = "bold") -> Table:
    table = Table(title=title, title_style=box_title_style, title_justify="left",
                  header_style="bold", expand=False)
    for name, opts in columns:
        table.add_column(name, **opts)
    return table


def solve_table(result: Any, instance: Any, elapsed: Optional[float] = None) -> Table:
    """The headline table of ``qroute solve``: what was found and how good it is."""
    bks = instance.meta.get("bks")
    gap = result.gap_to(bks) if bks else None
    seconds = result.seconds if elapsed is None else elapsed
    evals_per_s = result.evaluations / seconds if seconds > 0 else float("nan")

    table = _table(
        f"{result.algorithm} on {instance.name}",
        ("quantity", {"style": "bold", "no_wrap": True}),
        ("value", {"justify": "right"}),
    )
    table.add_row("cost", format_number(result.best.cost))
    if bks:
        table.add_row("best known", format_number(float(bks)))
        table.add_row("gap to best known", format_gap(gap))
    else:
        table.add_row("best known", Text("not available", style="dim"))
    table.add_row("routes", str(result.best.n_routes))
    ref_k = instance.meta.get("reference_k") or instance.meta.get("bks_routes")
    if ref_k:
        table.add_row("routes in reference", str(ref_k))
    table.add_row("feasible", feasibility(result.best.is_feasible))
    if not result.best.is_feasible:
        table.add_row("total violation", format_number(result.best.stats.total_violation, 4))
    table.add_row("distance", format_number(result.best.stats.distance))
    table.add_row("duration", format_number(result.best.stats.duration))
    table.add_row("iterations", f"{result.iterations:,}")
    table.add_row("evaluations", f"{result.evaluations:,}")
    table.add_row("evaluations / second", format_number(evals_per_s, 0))
    table.add_row("wall clock", format_seconds(seconds))
    table.add_row("seed", "-" if result.seed is None else str(result.seed))
    if bks:
        t1 = result.time_to_within(float(bks), 1.0)
        table.add_row("time to within 1%", format_seconds(t1) if t1 is not None else "not reached")
    return table


def routes_table(instance: Any, solution: Any, max_rows: int = 25) -> Table:
    """Per-route load, distance and duration of a solution."""
    table = _table(
        "routes",
        ("#", {"justify": "right", "style": "bold"}),
        ("stops", {"justify": "right"}),
        ("load", {"justify": "right"}),
        ("capacity used", {"justify": "right"}),
        ("distance", {"justify": "right"}),
        ("duration", {"justify": "right"}),
        ("sequence", {"overflow": "ellipsis", "no_wrap": True, "max_width": 60}),
    )
    demand = instance.demand
    dist = instance.distance
    dur = instance.duration
    routes = [r for r in solution.routes if r]
    for k, route in enumerate(routes[:max_rows], start=1):
        load = float(sum(demand[c] for c in route))
        prev = 0
        d = t = 0.0
        for c in route:
            d += float(dist[prev, c])
            t += float(dur[prev, c])
            prev = c
        d += float(dist[prev, 0])
        t += float(dur[prev, 0])
        used = 100.0 * load / instance.capacity if instance.capacity else float("nan")
        style = "red" if used > 100.0 + 1e-9 else ""
        table.add_row(
            str(k), str(len(route)), format_number(load, 0),
            Text(f"{used:.0f}%", style=style),
            format_number(d), format_number(t),
            " ".join(str(c) for c in route),
        )
    if len(routes) > max_rows:
        table.caption = f"{len(routes) - max_rows} further routes not shown"
    return table


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def compare_table(entries: Sequence[Mapping[str, Any]], instance_name: str,
                  bks: Optional[float], control: Optional[str]) -> Table:
    """One row per algorithm: dispersion over seeds and the test against the control.

    ``entries`` come from ``qroute compare`` and hold the per-seed costs and
    gaps. Reporting best, mean and standard deviation together is deliberate: a
    metaheuristic's best over five seeds flatters it, and the spread is what
    tells a reader whether a single run can be trusted.
    """
    reference = f"gap to best known ({format_number(bks)})" if bks else "cost"
    table = _table(
        f"{instance_name}: {len(entries)} algorithms, "
        f"{entries[0]['runs'] if entries else 0} seeds each",
        ("algorithm", {"style": "bold", "no_wrap": True}),
        ("runs", {"justify": "right"}),
        ("best " + ("gap" if bks else "cost"), {"justify": "right"}),
        ("mean " + ("gap" if bks else "cost"), {"justify": "right"}),
        ("std", {"justify": "right"}),
        ("mean cost", {"justify": "right"}),
        ("routes", {"justify": "right"}),
        ("feasible", {"justify": "right"}),
        ("iters/s", {"justify": "right"}),
        (f"vs {control}" if control else "test", {"overflow": "fold"}),
    )
    table.caption = reference
    for e in entries:
        if bks:
            best_cell = format_gap(e["gap_best"])
            mean_cell = format_gap(e["gap_mean"])
            std_cell = format_number(e["gap_std"], 3)
        else:
            best_cell = Text(format_number(e["cost_best"]))
            mean_cell = Text(format_number(e["cost_mean"]))
            std_cell = format_number(e["cost_std"], 3)
        table.add_row(
            e["algorithm"], str(e["runs"]), best_cell, mean_cell, std_cell,
            format_number(e["cost_mean"]),
            format_number(e["routes_mean"], 1),
            f"{e['feasible_runs']}/{e['runs']}",
            format_number(e["iterations_per_second"], 1),
            e.get("test") or Text("control", style="dim"),
        )
    return table


# ---------------------------------------------------------------------------
# Benchmark summary tables
# ---------------------------------------------------------------------------
def summary_table(summary: Mapping[str, Any], metric: str = "gap") -> Table:
    """Instances down the side, algorithms across the top, median gap in the cells.

    The best cell of each row is underlined, which is the fastest way to read
    who won an instance without hunting through decimals.
    """
    algorithms = list(summary.get("algorithms", []))
    instances = list(summary.get("instances", []))
    cells = summary.get("cells", {})
    table = _table(
        f"median {metric} by instance",
        ("instance", {"style": "bold", "no_wrap": True}),
        *[(a, {"justify": "right"}) for a in algorithms],
    )
    for inst in instances:
        values: list[Optional[float]] = []
        for algo in algorithms:
            cell = cells.get(f"{inst}|{algo}")
            block = (cell or {}).get(metric)
            values.append(block.get("median") if block else None)
        finite = [v for v in values if v is not None and math.isfinite(v)]
        best = min(finite) if finite else None
        rendered = []
        for v in values:
            if v is None:
                rendered.append(Text("-", style="dim"))
                continue
            text = format_gap(v) if metric == "gap" else Text(format_number(v))
            if best is not None and abs(v - best) < 1e-12:
                text.stylize("underline")
            rendered.append(text)
        table.add_row(inst, *rendered)
    return table


def cell_table(summary: Mapping[str, Any]) -> Table:
    """Full per (instance, algorithm) detail: dispersion, feasibility, timing."""
    table = _table(
        "per instance and algorithm",
        ("instance", {"style": "bold", "no_wrap": True}),
        ("algorithm", {"no_wrap": True}),
        ("runs", {"justify": "right"}),
        ("best gap", {"justify": "right"}),
        ("median gap", {"justify": "right"}),
        ("mean gap", {"justify": "right"}),
        ("std", {"justify": "right"}),
        ("median cost", {"justify": "right"}),
        ("feasible", {"justify": "right"}),
        ("hit bks", {"justify": "right"}),
        ("median t to 1%", {"justify": "right"}),
    )
    cells = summary.get("cells", {})
    for key in sorted(cells):
        cell = cells[key]
        gap = cell.get("gap") or {}
        table.add_row(
            cell["instance"], cell["algorithm"], str(cell["runs"]),
            format_gap(gap.get("best")), format_gap(gap.get("median")),
            format_gap(gap.get("mean")), format_number(gap.get("std"), 3),
            format_number((cell.get("cost") or {}).get("median")),
            f"{cell['feasible_runs']}/{cell['runs']}",
            str(cell.get("hit_bks", 0)),
            format_seconds(cell.get("median_time_to_1pct")),
        )
    return table


def omnibus_table(omnibus: Mapping[str, Any]) -> Table:
    """Friedman mean ranks and the Holm-corrected comparisons to the control."""
    ranks = omnibus.get("mean_ranks", {})
    table = _table(
        f"Friedman test over {len(omnibus.get('instances_used', []))} instances "
        f"(chi2 = {format_number(omnibus.get('statistic'))}, "
        f"p = {omnibus.get('p_value', float('nan')):.3g})",
        ("algorithm", {"style": "bold", "no_wrap": True}),
        ("mean rank", {"justify": "right"}),
        ("comparison with the control", {"overflow": "fold"}),
    )
    control = omnibus.get("control")
    post = {c["b"]: c for c in omnibus.get("post_hoc", [])}
    for algo, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if algo == control:
            note = Text("control (best mean rank)", style="bold")
        else:
            c = post.get(algo)
            note = Text(c["text"]) if c else Text("-", style="dim")
        table.add_row(algo, format_number(rank, 2), note)
    return table


def failure_table(summary: Mapping[str, Any]) -> Optional[Table]:
    """Runs that raised, so a sweep never looks complete when it was not."""
    failures = summary.get("failures") or []
    if not failures:
        return None
    table = _table(
        f"{summary.get('n_failed', len(failures))} failed runs",
        ("instance", {"style": "bold"}),
        ("algorithm", {}),
        ("error", {"overflow": "fold"}),
    )
    for f in failures:
        table.add_row(f.get("instance", "?"), f.get("algorithm", "?"),
                      Text(str(f.get("error")), style="red"))
    return table


# ---------------------------------------------------------------------------
# Other tables
# ---------------------------------------------------------------------------
def instances_table(rows: Sequence[Mapping[str, Any]], family: str) -> Table:
    """Local benchmark instances with their size and reference solution."""
    table = _table(
        f"{len(rows)} {family} instances",
        ("name", {"style": "bold", "no_wrap": True}),
        ("customers", {"justify": "right"}),
        ("capacity", {"justify": "right"}),
        ("reference k", {"justify": "right"}),
        ("best known", {"justify": "right"}),
        ("time windows", {"justify": "right"}),
    )
    for r in rows:
        table.add_row(
            r["name"], str(r["n_customers"]), format_number(r["capacity"], 0),
            str(r.get("reference_k") or "-"),
            format_number(r.get("bks")) if r.get("bks") else Text("-", style="dim"),
            "yes" if r.get("time_windows") else "no",
        )
    return table


def bounds_table(report: Any) -> Table:
    """Every lower bound computed for an instance, and which one is binding."""
    table = _table(
        f"lower bounds for {report.instance}",
        ("bound", {"style": "bold", "no_wrap": True}),
        ("value", {"justify": "right"}),
    )
    best = report.best
    for name, value in sorted(report.bounds.items(), key=lambda kv: -kv[1]):
        text = Text(format_number(value))
        if math.isfinite(value) and math.isfinite(best) and abs(value - best) < 1e-9:
            text.stylize("bold green")
        table.add_row(name, text)
    table.add_row("vehicles (lower bound)", str(report.vehicles_lb))
    if report.upper_bound is not None:
        table.add_row("upper bound (incumbent)", format_number(report.upper_bound))
        table.add_row("bracket width", format_gap(report.gap_percent))
    return table


def exact_table(name: str, method: str, outcome: Mapping[str, Any]) -> Table:
    """What an exact method proved, stated without overclaiming."""
    table = _table(
        f"{method} on {name}",
        ("quantity", {"style": "bold", "no_wrap": True}),
        ("value", {"justify": "right"}),
    )
    table.add_row("status", outcome.get("status", "?"))
    table.add_row("incumbent cost", format_number(outcome.get("cost")))
    table.add_row("lower bound", format_number(outcome.get("lower_bound")))
    proven = bool(outcome.get("proven_optimal"))
    table.add_row("optimality proved",
                  Text("yes", style="bold green") if proven else Text("no", style="yellow"))
    table.add_row("remaining gap", format_gap(outcome.get("gap")))
    if outcome.get("bks") is not None:
        table.add_row("best known", format_number(outcome["bks"]))
        table.add_row("gap to best known", format_gap(outcome.get("gap_to_bks")))
    table.add_row("routes", str(outcome.get("n_vehicles", "-")))
    table.add_row("wall clock", format_seconds(outcome.get("seconds")))
    return table


def kv_table(title: str, pairs: Iterable[tuple[str, Any]]) -> Table:
    """A generic two-column table, used for network and environment summaries."""
    table = _table(title, ("quantity", {"style": "bold", "no_wrap": True}),
                   ("value", {"justify": "right"}))
    for key, value in pairs:
        table.add_row(str(key), value if isinstance(value, Text) else str(value))
    return table


def demo_table(stages: Sequence[Mapping[str, Any]]) -> Table:
    """The four stages of the road-network demonstration, side by side."""
    table = _table(
        "plan cost through the incident",
        ("stage", {"style": "bold", "no_wrap": True}),
        ("objective (s)", {"justify": "right"}),
        ("drive time", {"justify": "right"}),
        ("distance (km)", {"justify": "right"}),
        ("routes", {"justify": "right"}),
        ("change", {"justify": "right"}),
        ("note", {"overflow": "fold"}),
    )
    for s in stages:
        delta = s.get("delta")
        if delta is None:
            change = Text("-", style="dim")
        else:
            style = "green" if delta < -1e-9 else ("red" if delta > 1e-9 else "dim")
            change = Text(f"{delta:+.1f}%", style=style)
        table.add_row(
            s["stage"], format_number(s["cost"], 0),
            format_duration_hms(s.get("duration")),
            format_number(s.get("distance", 0.0) / 1000.0, 2),
            str(s.get("n_routes", "-")), change, s.get("note", ""),
        )
    return table


# ---------------------------------------------------------------------------
# Text exports
# ---------------------------------------------------------------------------
def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def markdown_report(summary: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> str:
    """The benchmark summary as Markdown, ready to paste into the report."""
    algorithms = list(summary.get("algorithms", []))
    instances = list(summary.get("instances", []))
    cells = summary.get("cells", {})
    out: list[str] = []

    name = (meta or {}).get("config", {}).get("name", "benchmark")
    out.append(f"# Benchmark: {name}")
    out.append("")
    if meta:
        cfg = meta.get("config", {})
        env = meta.get("environment", {})
        out.append(f"* {len(instances)} instances, {len(algorithms)} algorithms, "
                   f"{cfg.get('seeds')} seeds, {cfg.get('max_seconds')} s per run")
        out.append(f"* master seed {cfg.get('master_seed')}, {meta.get('workers')} workers")
        out.append(f"* python {env.get('python')} on {env.get('platform')}, "
                   f"commit {(env.get('git_commit') or 'unknown')[:12]}"
                   f"{' (dirty tree)' if env.get('git_dirty') else ''}")
        out.append("")

    out.append("## Median gap to best known solution (percent)")
    out.append("")
    out.append(_md_row(["instance", *algorithms]))
    out.append(_md_row(["---", *["---:" for _ in algorithms]]))
    for inst in instances:
        row = [inst]
        for algo in algorithms:
            block = (cells.get(f"{inst}|{algo}") or {}).get("gap")
            row.append(f"{block['median']:.2f}" if block else "-")
        out.append(_md_row(row))
    out.append("")

    out.append("## Dispersion over seeds")
    out.append("")
    header = ["instance", "algorithm", "runs", "best gap", "median gap", "mean gap",
              "std", "median cost", "feasible", "hit bks"]
    out.append(_md_row(header))
    out.append(_md_row(["---"] * len(header)))
    for key in sorted(cells):
        c = cells[key]
        gap = c.get("gap") or {}
        out.append(_md_row([
            c["instance"], c["algorithm"], str(c["runs"]),
            f"{gap.get('best', float('nan')):.2f}" if gap else "-",
            f"{gap.get('median', float('nan')):.2f}" if gap else "-",
            f"{gap.get('mean', float('nan')):.2f}" if gap else "-",
            f"{gap.get('std', float('nan')):.3f}" if gap else "-",
            f"{(c.get('cost') or {}).get('median', float('nan')):.2f}",
            f"{c['feasible_runs']}/{c['runs']}", str(c.get("hit_bks", 0)),
        ]))
    out.append("")

    omnibus = summary.get("omnibus")
    if omnibus:
        out.append("## Statistical comparison")
        out.append("")
        out.append(f"Friedman test over {len(omnibus.get('instances_used', []))} instances: "
                   f"chi2 = {omnibus.get('statistic'):.3f}, p = {omnibus.get('p_value'):.3g}. "
                   f"Control algorithm: {omnibus.get('control')}.")
        out.append("")
        out.append(_md_row(["algorithm", "mean rank"]))
        out.append(_md_row(["---", "---:"]))
        for algo, rank in sorted(omnibus.get("mean_ranks", {}).items(), key=lambda kv: kv[1]):
            out.append(_md_row([algo, f"{rank:.2f}"]))
        out.append("")
        out.append("Holm-corrected pairwise comparisons against the control:")
        out.append("")
        for c in omnibus.get("post_hoc", []):
            out.append(f"* {c['text']}")
        out.append("")

    if summary.get("n_failed"):
        out.append(f"## {summary['n_failed']} failed runs")
        out.append("")
        for f in summary.get("failures", []):
            out.append(f"* {f.get('instance')} / {f.get('algorithm')}: {f.get('error')}")
        out.append("")
    return "\n".join(out)


CSV_COLUMNS = ("instance", "algorithm", "seed", "cost", "gap", "bks", "n_routes",
               "feasible", "violation", "iterations", "evaluations", "seconds",
               "time_to_1pct", "iters_to_1pct", "status")


def csv_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Per-run results as CSV, one line per (instance, algorithm, seed)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def write_plots(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
                out_dir: str | Path) -> list[Path]:
    """Write the report figures as PNG files and return their paths.

    Two figures are produced, both of which the submission needs: the mean
    convergence curve per algorithm (best cost against wall-clock time, so
    algorithms with different iteration costs are compared fairly) and the
    distribution of gaps per algorithm over every run.

    Matplotlib is imported here rather than at module level: it costs about a
    second to import and no other command needs it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ok = [r for r in rows if r.get("status") == "ok"]
    algorithms = sorted({r["algorithm"] for r in ok})

    # ---- convergence, one panel per instance that has any history ----------
    with_history = sorted({r["instance"] for r in ok if r.get("history")})
    if with_history:
        n = len(with_history)
        cols = min(3, n)
        panels = (n + cols - 1) // cols
        fig, axes = plt.subplots(panels, cols, figsize=(5.2 * cols, 3.6 * panels),
                                 squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        for k, inst in enumerate(with_history):
            ax = axes[k // cols][k % cols]
            for algo in algorithms:
                runs = [r for r in ok if r["instance"] == inst
                        and r["algorithm"] == algo and r.get("history")]
                if not runs:
                    continue
                # Interpolate every run onto a common time grid before averaging:
                # runs record at their own iteration boundaries, so averaging the
                # raw points would mix different moments in the search.
                horizon = min(max(h["t"] for h in r["history"]) for r in runs)
                grid = np.linspace(0.0, max(horizon, 1e-6), 120)
                curves = []
                for r in runs:
                    t = np.array([h["t"] for h in r["history"]], dtype=float)
                    c = np.array([h["c"] for h in r["history"]], dtype=float)
                    c = np.minimum.accumulate(c)
                    curves.append(np.interp(grid, t, c, left=c[0], right=c[-1]))
                mean = np.mean(curves, axis=0)
                ax.plot(grid, mean, label=algo, linewidth=1.6)
                if len(curves) > 1:
                    ax.fill_between(grid, np.min(curves, axis=0), np.max(curves, axis=0),
                                    alpha=0.12)
            bks = next((r.get("bks") for r in ok if r["instance"] == inst and r.get("bks")), None)
            if bks:
                ax.axhline(float(bks), color="black", linestyle="--", linewidth=1.0,
                           label="best known")
            ax.set_title(inst)
            ax.set_xlabel("wall clock (s)")
            ax.set_ylabel("best cost")
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / "convergence.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # ---- gap distribution --------------------------------------------------
    data = [[r["gap"] for r in ok if r["algorithm"] == a and r.get("gap") is not None]
            for a in algorithms]
    if any(data):
        fig, ax = plt.subplots(figsize=(1.7 * max(len(algorithms), 3) + 2, 4.0))
        keep = [(a, d) for a, d in zip(algorithms, data) if d]
        ax.boxplot([d for _, d in keep], tick_labels=[a for a, _ in keep], showmeans=True)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("gap to best known (%)")
        ax.set_title(f"gap distribution over {summary.get('n_ok', len(ok))} runs")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / "gap_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written
