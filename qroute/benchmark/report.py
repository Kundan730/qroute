"""Tables that turn a saved benchmark run into something a reader can judge.

:mod:`qroute.benchmark.runner` produces one row per ``(instance, algorithm,
seed)`` triple. Those rows are the evidence; this module is the presentation of
that evidence, and it exists because a benchmark that cannot be read is a
benchmark that cannot be checked.

Every table is built once as a :class:`Table` and then rendered three ways: as
GitHub-flavoured markdown for the written report, as CSV for anyone who wants to
recompute a number in a spreadsheet, and as a ``rich`` table for the terminal.
The three renderings come from the same formatted cells, so a figure quoted in
the report is by construction the figure in the CSV.

Two rules are enforced throughout:

* Nothing is invented. A statistic that cannot be computed from the rows -- a
  missing best-known cost, a run that never reached a target, an algorithm that
  crashed -- is printed as an em dash, and the count of runs behind every
  aggregate is printed next to it. Silently dropping a failed run would flatter
  the algorithm that failed.
* Numbers are formatted in one place, so a gap is always two decimals and a
  time is always three, and columns line up when read as plain text.

The entry point for a whole run is :func:`build_report`, which writes
``report.md`` plus one CSV per table into an output directory.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "EM_DASH",
    "Table",
    "first_within",
    "main_results_table",
    "tier_summary_table",
    "statistical_table",
    "convergence_table",
    "ablation_table",
    "per_instance_detail_table",
    "build_report",
    "instance_size",
    "tier_of",
]

EM_DASH = "—"

#: Human-readable descriptions for the ablation arms we expect to see. An arm
#: that is not listed simply gets an em dash rather than a guessed description.
ABLATION_LABELS: dict[str, str] = {
    "qpso": "full method: quantum-behaved swarm with local search",
    "qpso-nols": "local search removed",
    "qpso_nols": "local search removed",
    "qpso-nols-norestart": "local search and restarts removed",
    "qpso-norestart": "diversity restarts removed",
    "qpso-uniform": "unweighted mean best position",
    "random": "control: random restart with the same local search",
    "restart": "control: random restart with the same local search",
}

#: Instance-size tiers. CVRPLIB and Solomon instances are conventionally
#: discussed in these bands, and the boundaries are stated so a reader can see
#: exactly which instances landed where.
TIERS: tuple[tuple[str, int, float], ...] = (
    ("small (< 50 nodes)", 0, 50),
    ("medium (50-99 nodes)", 50, 100),
    ("large (>= 100 nodes)", 100, math.inf),
)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _missing(value: Any) -> bool:
    """True when a value cannot honestly be printed as a number."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    """Format a number to a fixed number of decimals, or an em dash if missing."""
    if _missing(value):
        return EM_DASH
    return f"{float(value):.{digits}f}{suffix}"


def fmt_int(value: Any) -> str:
    if _missing(value):
        return EM_DASH
    return f"{int(round(float(value)))}"


def fmt_gap(value: Any) -> str:
    """A percentage gap above the best known cost, always two decimals."""
    return fmt(value, 2)


def fmt_seconds(value: Any) -> str:
    return fmt(value, 3)


def _fraction(reached: int, total: int) -> str:
    return f"{reached}/{total}"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
@dataclass
class Table:
    """A rendered table: cells are already strings, so every output agrees.

    ``highlight`` holds ``(row, column)`` indices that should be emphasised --
    used for the best cell in a row of the main results table. Emphasis is
    markup, so it appears in markdown and in the terminal but not in the CSV,
    which stays machine-readable; the main table therefore also carries an
    explicit column naming the winner.
    """

    title: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    align: list[str] = field(default_factory=list)      # "l" or "r" per column
    notes: list[str] = field(default_factory=list)
    highlight: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.align:
            self.align = ["l"] + ["r"] * (len(self.columns) - 1)
        if len(self.align) != len(self.columns):
            raise ValueError("align must have one entry per column")
        for i, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(f"row {i} has {len(row)} cells, expected {len(self.columns)}")

    # ---------------------------------------------------------------- render
    def to_markdown(self, heading_level: int = 3) -> str:
        head = "#" * max(1, heading_level)
        out = [f"{head} {self.title}", ""]
        out.append("| " + " | ".join(self.columns) + " |")
        out.append("| " + " | ".join("---:" if a == "r" else ":---" for a in self.align) + " |")
        for r, row in enumerate(self.rows):
            cells = []
            for c, cell in enumerate(row):
                text = str(cell).replace("|", "\\|")
                if (r, c) in self.highlight and text != EM_DASH:
                    text = f"**{text}**"
                cells.append(text)
            out.append("| " + " | ".join(cells) + " |")
        if self.notes:
            out.append("")
            out.extend(f"{note}" for note in self.notes)
        out.append("")
        return "\n".join(out)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buf.getvalue()

    def to_text(self) -> str:
        """Fixed-width plain text, for a log file or a terminal without rich."""
        widths = [len(c) for c in self.columns]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        def line(cells: Sequence[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                text = str(cell)
                parts.append(text.rjust(widths[i]) if self.align[i] == "r"
                             else text.ljust(widths[i]))
            return "  ".join(parts).rstrip()

        out = [self.title, "=" * len(self.title), line(self.columns),
               "  ".join("-" * w for w in widths)]
        out.extend(line(row) for row in self.rows)
        if self.notes:
            out.append("")
            out.extend(self.notes)
        return "\n".join(out)

    def to_rich(self):
        """Build a ``rich.table.Table``. Imported lazily so report.py stays importable
        in environments without rich installed."""
        from rich import box
        from rich.table import Table as RichTable

        table = RichTable(title=self.title, box=box.SIMPLE_HEAVY, header_style="bold")
        for name, a in zip(self.columns, self.align):
            table.add_column(name, justify="right" if a == "r" else "left", overflow="fold")
        for r, row in enumerate(self.rows):
            cells = []
            for c, cell in enumerate(row):
                text = str(cell)
                if (r, c) in self.highlight and text != EM_DASH:
                    text = f"[bold green]{text}[/bold green]"
                cells.append(text)
            table.add_row(*cells)
        if self.notes:
            table.caption = "\n".join(self.notes)
        return table

    def write(self, out_dir: str | Path, stem: str) -> dict[str, Path]:
        """Write the markdown and CSV renderings side by side."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        md = d / f"{stem}.md"
        csv_path = d / f"{stem}.csv"
        md.write_text(self.to_markdown())
        csv_path.write_text(self.to_csv())
        return {"markdown": md, "csv": csv_path}


# ---------------------------------------------------------------------------
# Row handling
# ---------------------------------------------------------------------------
def ok_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Rows from runs that completed. Failures are counted, never averaged in."""
    return [dict(r) for r in rows if r.get("status", "ok") == "ok"]


def failed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [dict(r) for r in rows if r.get("status", "ok") != "ok"]


def algorithms_in(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(r["algorithm"]) for r in rows})


def instances_in(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Instance names ordered by size, then alphabetically, so tables read
    small-to-large the way the scalability discussion does."""
    names = {str(r["instance"]) for r in rows}
    return sorted(names, key=lambda n: (instance_size(n) or 10 ** 9, n))


def group(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(str(r[k]) for k in keys), []).append(dict(r))
    return out


_SIZE_RE = re.compile(r"[-_]?n(\d+)", re.IGNORECASE)
_size_cache: dict[str, int | None] = {}


def instance_size(name: str, sizes: Mapping[str, int] | None = None) -> int | None:
    """Number of nodes in an instance.

    Resolved from an explicit mapping first, then from the CVRPLIB naming
    convention (``A-n32-k5`` has 32 nodes), and only then by loading the
    instance from disk. Returns ``None`` when the size genuinely is not known,
    which the tables print as an em dash rather than guessing.
    """
    if sizes and name in sizes:
        return int(sizes[name])
    if name in _size_cache:
        return _size_cache[name]
    size: int | None = None
    m = _SIZE_RE.search(name)
    if m:
        size = int(m.group(1))
    else:
        try:                                     # last resort: read the file
            from qroute.problems.loaders import load
            size = int(load(name).size)
        except Exception:
            size = None
    _size_cache[name] = size
    return size


def tier_of(size: int | None) -> str:
    if size is None:
        return "unknown size"
    for label, lo, hi in TIERS:
        if lo <= size < hi:
            return label
    return "unknown size"


def _bks(rows: Sequence[Mapping[str, Any]]) -> float | None:
    for r in rows:
        v = r.get("bks")
        if v is not None and math.isfinite(float(v)) and float(v) > 0:
            return float(v)
    return None


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            out.append(v)
    return out


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


# ---------------------------------------------------------------------------
# 1. Main results table
# ---------------------------------------------------------------------------
def main_results_table(rows: Sequence[Mapping[str, Any]],
                       summary: Mapping[str, Any] | None = None,
                       sizes: Mapping[str, int] | None = None,
                       title: str = "Main results: gap to best known, mean (best) over seeds",
                       ) -> Table:
    """One row per instance, one column per algorithm, cells ``mean (best)`` gap.

    ``summary`` is accepted for symmetry with the rest of the reporting API and
    is used only for the algorithm ordering when present; every number is
    recomputed from ``rows`` so the table cannot drift from the raw evidence.
    """
    good = ok_rows(rows)
    algos = list(summary.get("algorithms")) if summary and summary.get("algorithms") \
        else algorithms_in(good or rows)
    insts = instances_in(good or rows)
    by = group(good, "instance", "algorithm")

    columns = ["Instance", "Size", "Best known"] + list(algos) + ["Best on row"]
    align = ["l", "r", "r"] + ["r"] * len(algos) + ["l"]
    table = Table(title=title, columns=columns, align=align)

    for i, inst in enumerate(insts):
        size = instance_size(inst, sizes)
        bks = _bks([r for r in good if r["instance"] == inst] or
                   [r for r in rows if str(r.get("instance")) == inst])
        cells = [inst, fmt_int(size), fmt_int(bks)]
        means: list[float | None] = []
        for algo in algos:
            rs = by.get((inst, algo), [])
            gaps = _values(rs, "gap")
            if not gaps:
                cells.append(EM_DASH)
                means.append(None)
                continue
            cells.append(f"{fmt_gap(float(np.mean(gaps)))} ({fmt_gap(float(np.min(gaps)))})")
            means.append(float(np.mean(gaps)))
        # Compare on the value actually printed. Two algorithms that both show
        # 0.00 are tied as far as this table can tell, and marking one of them
        # as the winner would be an artefact of the sort order.
        finite = [(round(m, 2), a) for m, a in zip(means, algos) if m is not None]
        if finite:
            best_mean = min(m for m, _ in finite)
            winners = [a for m, a in finite if m == best_mean]
            cells.append(", ".join(winners) if len(winners) < len(algos) else "tie (all equal)")
            for a in winners:
                table.highlight.add((i, 3 + algos.index(a)))
        else:
            cells.append(EM_DASH)
        table.rows.append(cells)

    table.notes.append(
        "Cells are the mean percentage gap above the best known cost over all seeds, "
        "with the single best seed in brackets. Lower is better; the best mean on each "
        "row is emphasised and named in the last column, and every algorithm tied with "
        "it at two decimal places is marked too. An em dash means no run of that "
        "algorithm produced a comparable result."
    )
    bad = failed_rows(rows)
    if bad:
        table.notes.append(f"{len(bad)} of {len(rows)} runs failed and are excluded from the means.")
    return table


# ---------------------------------------------------------------------------
# 2. Summary by size tier
# ---------------------------------------------------------------------------
def tier_summary_table(rows: Sequence[Mapping[str, Any]],
                       sizes: Mapping[str, int] | None = None,
                       title: str = "Summary by instance-size tier",
                       ) -> Table:
    """Mean gap, runs that matched the best known cost, and mean run time per tier."""
    good = ok_rows(rows)
    for r in good:
        r["_tier"] = tier_of(instance_size(str(r["instance"]), sizes))
    order = [label for label, _, _ in TIERS] + ["unknown size"]
    algos = algorithms_in(good)

    table = Table(
        title=title,
        columns=["Tier", "Algorithm", "Instances", "Runs", "Mean gap %",
                 "Runs at best known", "Mean seconds"],
        align=["l", "l", "r", "r", "r", "r", "r"],
    )
    for tier in order:
        tier_rows = [r for r in good if r["_tier"] == tier]
        if not tier_rows:
            continue
        n_inst = len({r["instance"] for r in tier_rows})
        for algo in algos:
            rs = [r for r in tier_rows if r["algorithm"] == algo]
            if not rs:
                continue
            gaps = _values(rs, "gap")
            hits = sum(1 for g in gaps if g <= 1e-9)
            table.rows.append([
                tier, algo, fmt_int(n_inst), fmt_int(len(rs)),
                fmt_gap(_mean(gaps)),
                _fraction(hits, len(rs)) if gaps else EM_DASH,
                fmt_seconds(_mean(_values(rs, "seconds"))),
            ])
    table.notes.append(
        "Tier boundaries are on node count: " +
        ", ".join(f"{label}" for label, _, _ in TIERS) + ". "
        "\"Runs at best known\" counts runs whose final cost equalled the best known cost."
    )
    return table


# ---------------------------------------------------------------------------
# 3. Statistical table
# ---------------------------------------------------------------------------
def statistical_table(rows: Sequence[Mapping[str, Any]],
                      control: str | None = None,
                      alpha: float = 0.05,
                      title: str = "Statistical comparison: Friedman ranks and Holm-corrected pairwise tests",
                      ) -> Table:
    """Friedman mean ranks with the omnibus p-value, then Holm-corrected
    comparisons against the control algorithm, each written out as a sentence.

    The omnibus test needs a complete matrix, so only instances on which every
    algorithm produced a gap take part; which instances those were is stated in
    the notes rather than left implicit.
    """
    from qroute.benchmark.stats import friedman

    good = ok_rows(rows)
    algos = algorithms_in(good)
    insts = instances_in(good)
    by = group(good, "instance", "algorithm")

    per_algo: dict[str, list[float]] = {a: [] for a in algos}
    used: list[str] = []
    for inst in insts:
        medians = {}
        for a in algos:
            gaps = _values(by.get((inst, a), []), "gap")
            if not gaps:
                medians = {}
                break
            medians[a] = float(np.median(gaps))
        if medians:
            used.append(inst)
            for a in algos:
                per_algo[a].append(medians[a])

    table = Table(
        title=title,
        columns=["Algorithm", "Mean rank", "Holm-corrected p vs control", "Effect size", "Finding"],
        align=["l", "r", "r", "r", "l"],
    )

    if len(algos) < 3 or len(used) < 3:
        table.notes.append(
            f"Not computed: the Friedman test needs at least three algorithms and three "
            f"instances measured on all of them; this run has {len(algos)} algorithm(s) "
            f"and {len(used)} such instance(s)."
        )
        for a in algos:
            table.rows.append([a, EM_DASH, EM_DASH, EM_DASH, "not enough data for a test"])
        return table

    fr = friedman(per_algo, control=control, alpha=alpha)
    ctrl = fr.control
    comparisons = {c.b if c.a == ctrl else c.a: c for c in fr.post_hoc}
    for name, rank in fr.ranking():
        if name == ctrl:
            table.rows.append([name, fmt(rank, 2), EM_DASH, EM_DASH,
                               "control algorithm (lowest mean rank)" if control is None
                               else "control algorithm"])
            continue
        c = comparisons.get(name)
        if c is None:
            table.rows.append([name, fmt(rank, 2), EM_DASH, EM_DASH, EM_DASH])
            continue
        p = c.p_adjusted if c.p_adjusted is not None else c.p_value
        if not math.isfinite(p) or p > alpha:
            sentence = (f"no significant difference from {ctrl} at alpha = {alpha:g} "
                        f"(n = {c.n} instances)")
        else:
            better = c.winner or (c.a if c.median_a < c.median_b else c.b)
            worse = c.b if better == c.a else c.a
            sentence = (f"{better} beats {worse} significantly "
                        f"(n = {c.n} instances, effect size {abs(c.effect_size):.2f})")
        table.rows.append([name, fmt(rank, 2), f"{p:.3g}", fmt(abs(c.effect_size), 2), sentence])

    verdict = ("the algorithms are not all equivalent" if fr.p_value <= alpha
               else "no overall difference between the algorithms was detected")
    table.notes.append(
        f"Friedman omnibus test over {fr.n_instances} instances and {len(algos)} algorithms: "
        f"chi-square = {fr.statistic:.3f}, p = {fr.p_value:.3g}, so at alpha = {alpha:g} {verdict}. "
        f"Ranks are over the per-instance median gap, 1 = best."
    )
    table.notes.append(
        "Pairwise tests are paired Wilcoxon signed-rank tests against the control "
        f"({ctrl}), with Holm's step-down correction over the {len(fr.post_hoc)} comparisons. "
        f"Instances used: {', '.join(used)}."
    )
    return table


# ---------------------------------------------------------------------------
# 4. Convergence table
# ---------------------------------------------------------------------------
def first_within(row: Mapping[str, Any], pct: float) -> tuple[int | None, float | None]:
    """Iteration and elapsed time at which a run first came within ``pct``% of
    the best known cost, or ``(None, None)`` if it never did.

    Computed from the recorded history when the run saved one, which is what
    makes an arbitrary target percentage possible; otherwise the runner's own
    precomputed 1% and 2% fields are used, and if neither is available the run
    counts as "not reached" rather than being dropped.
    """
    bks = row.get("bks")
    history = row.get("history") or []
    if bks and history:
        threshold = float(bks) * (1.0 + pct / 100.0)
        for h in history:
            if float(h["c"]) <= threshold:
                return int(h["i"]), float(h["t"])
        return None, None
    key_t = f"time_to_{int(pct)}pct" if float(pct).is_integer() else None
    key_i = f"iters_to_{int(pct)}pct" if float(pct).is_integer() else None
    t = row.get(key_t) if key_t else None
    i = row.get(key_i) if key_i else None
    return (int(i) if i is not None else None, float(t) if t is not None else None)


def convergence_table(rows: Sequence[Mapping[str, Any]],
                      targets: Sequence[float] = (1.0, 2.0),
                      title: str = "Convergence: median effort to reach a target quality",
                      ) -> Table:
    """Median iterations and seconds to come within each target percentage of the
    best known cost.

    Medians are taken over the runs that actually reached the target, and the
    number that did is printed beside them. A cell reading "not reached" means
    no run of that algorithm on that instance ever got there within the budget,
    which is a result, not a gap in the data.
    """
    good = ok_rows(rows)
    algos = algorithms_in(good)
    insts = instances_in(good)
    by = group(good, "instance", "algorithm")

    columns = ["Instance", "Algorithm"]
    align = ["l", "l"]
    for pct in targets:
        columns += [f"Iterations to {pct:g}%", f"Seconds to {pct:g}%", f"Runs reaching {pct:g}%"]
        align += ["r", "r", "r"]
    table = Table(title=title, columns=columns, align=align)

    any_history = any(r.get("history") for r in good)
    for inst in insts:
        for algo in algos:
            rs = by.get((inst, algo), [])
            if not rs:
                continue
            cells = [inst, algo]
            for pct in targets:
                hits = [first_within(r, pct) for r in rs]
                iters = [i for i, _ in hits if i is not None]
                secs = [t for _, t in hits if t is not None]
                reached = sum(1 for i, t in hits if i is not None or t is not None)
                if reached == 0:
                    cells += ["not reached", "not reached", _fraction(0, len(rs))]
                else:
                    cells += [fmt_int(_median(iters)) if iters else EM_DASH,
                              fmt_seconds(_median(secs)) if secs else EM_DASH,
                              _fraction(reached, len(rs))]
            table.rows.append(cells)

    table.notes.append(
        "Medians are over the runs that reached the target; the last column of each block "
        "gives how many of the seeds did. \"not reached\" means no seed reached that target "
        "within the time budget."
    )
    if not any_history:
        table.notes.append(
            "No convergence history was saved with this run, so these figures come from the "
            "runner's precomputed 1% and 2% fields only."
        )
    return table


# ---------------------------------------------------------------------------
# 5. Ablation table
# ---------------------------------------------------------------------------
def _default_variants(algos: Sequence[str], base: str = "qpso") -> list[str]:
    """Arms of an ablation: the base method, anything derived from it by a
    suffix, and the random-restart control if it was run."""
    family = [a for a in algos if a == base or a.startswith(base + "-") or a.startswith(base + "_")]
    control = [a for a in algos if a in ("random", "restart")]
    ordered = ([base] if base in family else []) + sorted(a for a in family if a != base)
    return ordered + control


def ablation_table(rows: Sequence[Mapping[str, Any]],
                   variants: Sequence[str] | None = None,
                   base: str = "qpso",
                   labels: Mapping[str, str] | None = None,
                   title: str = "Ablation: contribution of each component",
                   ) -> Table | None:
    """What each component of the proposed method is worth.

    Returns ``None`` when the run does not contain at least two comparable arms,
    because an ablation table with one row would say nothing. Instances are
    restricted to those solved by every arm, so the deltas compare like with
    like.
    """
    good = ok_rows(rows)
    algos = algorithms_in(good)
    arms = list(variants) if variants else _default_variants(algos, base)
    arms = [a for a in arms if a in algos]
    if len(arms) < 2:
        return None

    by = group(good, "instance", "algorithm")
    common = [i for i in instances_in(good)
              if all(_values(by.get((i, a), []), "gap") for a in arms)]

    label_map = dict(ABLATION_LABELS)
    if labels:
        label_map.update(labels)

    reference = arms[0]
    ref_gap = _mean([g for i in common for g in _values(by[(i, reference)], "gap")])

    table = Table(
        title=title,
        columns=["Variant", "What changed", "Instances", "Runs", "Mean gap %",
                 "Change vs full method", "Runs at best known", "Mean seconds"],
        align=["l", "l", "r", "r", "r", "r", "r", "r"],
    )
    for arm in arms:
        rs = [r for i in common for r in by.get((i, arm), [])]
        gaps = _values(rs, "gap")
        mean_gap = _mean(gaps)
        if mean_gap is None or ref_gap is None:
            delta = EM_DASH
        elif arm == reference:
            delta = EM_DASH
        else:
            delta = f"{mean_gap - ref_gap:+.2f}"
        hits = sum(1 for g in gaps if g <= 1e-9)
        table.rows.append([
            arm, label_map.get(arm, EM_DASH), fmt_int(len(common)), fmt_int(len(rs)),
            fmt_gap(mean_gap), delta,
            _fraction(hits, len(rs)) if gaps else EM_DASH,
            fmt_seconds(_mean(_values(rs, "seconds"))),
        ])
    table.notes.append(
        f"All arms are compared on the {len(common)} instance(s) that every arm solved, "
        f"with the same seeds and the same time budget. \"Change vs full method\" is the "
        f"difference in mean gap against {reference}; a positive number means removing that "
        f"component made the result worse."
    )
    return table


# ---------------------------------------------------------------------------
# 6. Per-instance detail
# ---------------------------------------------------------------------------
def per_instance_detail_table(rows: Sequence[Mapping[str, Any]],
                              instance: str | None = None,
                              sizes: Mapping[str, int] | None = None,
                              title: str | None = None,
                              ) -> Table:
    """Per ``(instance, algorithm)``: the whole distribution over seeds.

    A mean alone hides whether an algorithm is reliably mediocre or wildly
    variable, which for a stochastic method is the more interesting question.
    """
    good = ok_rows(rows)
    if instance is not None:
        good = [r for r in good if str(r["instance"]) == instance]
    insts = instances_in(good)
    algos = algorithms_in(good)
    by = group(good, "instance", "algorithm")

    table = Table(
        title=title or ("Per-instance detail" + (f": {instance}" if instance else "")),
        columns=["Instance", "Size", "Best known", "Algorithm", "Runs", "Feasible runs",
                 "Best cost", "Mean cost", "Median cost", "Std dev", "Worst cost",
                 "Best gap %", "Mean gap %"],
        align=["l", "r", "r", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
    )
    for inst in insts:
        size = instance_size(inst, sizes)
        bks = _bks([r for r in good if r["instance"] == inst])
        for algo in algos:
            rs = by.get((inst, algo), [])
            if not rs:
                continue
            costs = _values(rs, "cost")
            gaps = _values(rs, "gap")
            std = float(np.std(costs, ddof=1)) if len(costs) > 1 else (0.0 if costs else None)
            table.rows.append([
                inst, fmt_int(size), fmt_int(bks), algo, fmt_int(len(rs)),
                fmt_int(sum(1 for r in rs if r.get("feasible"))),
                fmt(min(costs), 2) if costs else EM_DASH,
                fmt(_mean(costs), 2), fmt(_median(costs), 2), fmt(std, 2),
                fmt(max(costs), 2) if costs else EM_DASH,
                fmt_gap(min(gaps)) if gaps else EM_DASH,
                fmt_gap(_mean(gaps)),
            ])
    table.notes.append(
        "Costs are the final cost of each run, re-scored by the reference evaluator. "
        "The standard deviation is the sample standard deviation over seeds; it is 0.00 "
        "for a single run and an em dash when no run produced a cost."
    )
    return table


# ---------------------------------------------------------------------------
# Whole report
# ---------------------------------------------------------------------------
def all_tables(rows: Sequence[Mapping[str, Any]],
               summary: Mapping[str, Any] | None = None,
               sizes: Mapping[str, int] | None = None,
               control: str | None = None,
               ablation_base: str = "qpso",
               ) -> dict[str, Table]:
    """Every table this module knows how to build, keyed by file stem.

    The ablation entry is simply absent when the run contains no comparable
    arms, rather than present and empty.
    """
    tables: dict[str, Table] = {
        "main_results": main_results_table(rows, summary, sizes),
        "tier_summary": tier_summary_table(rows, sizes),
        "statistics": statistical_table(rows, control=control),
        "convergence": convergence_table(rows),
        "per_instance_detail": per_instance_detail_table(rows, sizes=sizes),
    }
    abl = ablation_table(rows, base=ablation_base)
    if abl is not None:
        tables["ablation"] = abl
    return tables


def _preamble(rows: Sequence[Mapping[str, Any]],
              meta: Mapping[str, Any] | None) -> list[str]:
    good = ok_rows(rows)
    bad = failed_rows(rows)
    lines = ["# Benchmark report", ""]
    if meta:
        cfg = meta.get("config") or {}
        env = meta.get("environment") or {}
        lines += [
            f"Run name: `{cfg.get('name', EM_DASH)}`  ",
            f"Budget: {cfg.get('max_seconds', EM_DASH)} s wall clock per run, "
            f"{cfg.get('seeds', EM_DASH)} seeds from master seed {cfg.get('master_seed', EM_DASH)}  ",
            f"Environment: Python {env.get('python', EM_DASH)} on {env.get('platform', EM_DASH)}, "
            f"{env.get('cpu_count', EM_DASH)} CPUs, numpy "
            f"{(env.get('packages') or {}).get('numpy', EM_DASH)}  ",
        ]
        commit = env.get("git_commit")
        if commit:
            lines.append(f"Git commit: `{commit[:12]}`"
                         + (" (working tree dirty)" if env.get("git_dirty") else "") + "  ")
        lines.append("")
    lines += [
        f"{len(good)} completed runs over {len(instances_in(good))} instances and "
        f"{len(algorithms_in(good))} algorithms"
        + (f"; {len(bad)} runs failed and are excluded from every aggregate." if bad
           else "; no runs failed."),
        "",
        "All gaps are percentages above the best known cost, so lower is better. "
        "A value that could not be computed is printed as an em dash.",
        "",
    ]
    if bad:
        lines.append("Failures:")
        lines.append("")
        for r in bad[:20]:
            lines.append(f"* `{r.get('instance')}` / `{r.get('algorithm')}` "
                         f"seed {r.get('seed')}: {r.get('error')}")
        lines.append("")
    return lines


def build_report(rows: Sequence[Mapping[str, Any]],
                 out_dir: str | Path,
                 summary: Mapping[str, Any] | None = None,
                 meta: Mapping[str, Any] | None = None,
                 sizes: Mapping[str, int] | None = None,
                 control: str | None = None,
                 ablation_base: str = "qpso",
                 ) -> dict[str, Any]:
    """Build every table, write ``report.md`` and one CSV per table.

    Returns the tables plus the paths written, so a caller can go on to render
    the same objects to a terminal without recomputing anything.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    tables = all_tables(rows, summary=summary, sizes=sizes, control=control,
                        ablation_base=ablation_base)

    written: dict[str, dict[str, Path]] = {}
    for stem, table in tables.items():
        written[stem] = {"csv": d / f"{stem}.csv"}
        (d / f"{stem}.csv").write_text(table.to_csv())

    order = ["main_results", "tier_summary", "statistics", "convergence",
             "ablation", "per_instance_detail"]
    parts = _preamble(rows, meta)
    for stem in order:
        if stem in tables:
            parts.append(tables[stem].to_markdown(heading_level=2))
    report_md = d / "report.md"
    report_md.write_text("\n".join(parts))
    return {"tables": tables, "report_md": report_md, "csv": written, "output_dir": d}


def render_console(tables: Mapping[str, Table], console=None) -> None:
    """Print the tables to a terminal using rich."""
    from rich.console import Console

    console = console or Console()
    for table in tables.values():
        console.print(table.to_rich())
        console.print()
