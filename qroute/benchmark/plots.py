"""Figures for the convergence and benchmarking deliverable.

Each function here takes the rows produced by
:class:`qroute.benchmark.runner.BenchmarkRunner` and an output directory, and
writes one figure as both PNG at 150 dpi (for slides and the written report) and
SVG (so it can be scaled or edited without resampling).

The figures are meant to answer questions rather than to decorate a document,
and each one exists because a different question is being asked:

* Cost against wall-clock time says which algorithm you should actually run,
  because the budget in the field is seconds, not iterations.
* Cost against iteration count says whether an advantage comes from a better
  search or merely from cheaper iterations. The two plots regularly disagree,
  and reporting only the flattering one would be dishonest.
* A time-to-target plot reports the whole distribution over seeds instead of a
  single average, which for a stochastic method is the only fair summary.
* A gap distribution shows spread, so a method that is occasionally excellent
  and often poor cannot hide behind its mean.
* Scalability against instance size shows where a method stops working.
* Diversity beside best cost makes premature convergence visible: a swarm whose
  diversity collapses while its cost is still falling has stopped searching.

Conventions: the palette is the Okabe-Ito colour-blind-safe set, every axis is
labelled with its unit, the band around a median line is the inter-quartile
range across seeds, and the best known cost is drawn as a horizontal reference
so a reader can see the remaining gap directly. Runs are never dropped
silently; when a curve stops early it is because those runs stopped early.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# These figures are written to files, never shown, and the benchmark runs
# headless in worker processes, so the non-interactive backend is selected
# before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from qroute.benchmark.report import (     # noqa: E402
    algorithms_in,
    instance_size,
    instances_in,
    ok_rows,
)
from qroute.benchmark.stats import time_to_target_curve   # noqa: E402

__all__ = [
    "PALETTE",
    "convergence_time",
    "convergence_iterations",
    "time_to_target",
    "gap_distribution",
    "scalability",
    "diversity",
    "per_instance_gap_bars",
    "all_plots",
]

#: Okabe-Ito qualitative palette: eight hues that remain distinguishable under
#: deuteranopia, protanopia and tritanopia, and in greyscale print.
PALETTE: tuple[str, ...] = (
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#CC79A7",   # reddish purple
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
    "#000000",   # black
)

_LINESTYLES = ("-", "--", "-.", ":")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "figure.constrained_layout.use": True,
}


def _style(algorithms: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Fixed colour, dash pattern and marker per algorithm.

    Keyed off the sorted algorithm list so the same algorithm keeps the same
    colour across every figure in a report, and dash patterns vary as well as
    colour so the figures survive being printed in black and white.
    """
    out = {}
    for i, name in enumerate(algorithms):
        out[name] = {
            "color": PALETTE[i % len(PALETTE)],
            "linestyle": _LINESTYLES[(i // len(PALETTE)) % len(_LINESTYLES)],
            "marker": _MARKERS[i % len(_MARKERS)],
        }
    return out


def _save(fig, out_dir: str | Path, stem: str) -> list[Path]:
    """Write a figure as PNG (150 dpi) and SVG, and close it."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = d / f"{stem}.{ext}"
        fig.savefig(p, format=ext, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _history(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Elapsed seconds, iteration index, best cost and diversity for one run."""
    hist = row.get("history") or []
    t = np.array([float(h["t"]) for h in hist], dtype=float)
    i = np.array([float(h["i"]) for h in hist], dtype=float)
    c = np.array([float(h["c"]) for h in hist], dtype=float)
    d = np.array([float(h.get("d", np.nan)) for h in hist], dtype=float)
    return t, i, c, d


def _step_band(curves: Sequence[tuple[np.ndarray, np.ndarray]],
               grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and inter-quartile range of several step curves on a common grid.

    Each curve is a monotone record of the incumbent, so the value at a grid
    point is the last recorded value at or before it. Grid points before a run
    had recorded anything are left as NaN and excluded from that column's
    statistics, which is why the band can be based on fewer runs early on; the
    count of contributing runs is returned so a caller can say so.
    """
    stacked = []
    for x, y in curves:
        if x.size == 0:
            continue
        idx = np.searchsorted(x, grid, side="right") - 1
        vals = np.where(idx >= 0, y[np.clip(idx, 0, y.size - 1)], np.nan)
        stacked.append(vals)
    if not stacked:
        empty = np.full(grid.shape, np.nan)
        return empty, empty, empty, np.zeros(grid.shape, dtype=int)
    arr = np.vstack(stacked)
    counts = np.sum(np.isfinite(arr), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
    return med, q1, q3, counts


def _bks_of(rows: Sequence[Mapping[str, Any]]) -> float | None:
    for r in rows:
        v = r.get("bks")
        if v is not None and math.isfinite(float(v)) and float(v) > 0:
            return float(v)
    return None


def _target_instances(rows: Sequence[Mapping[str, Any]], instance: str | None) -> list[str]:
    return [instance] if instance is not None else instances_in(rows)


# ---------------------------------------------------------------------------
# 1 and 2. Convergence
# ---------------------------------------------------------------------------
def _convergence(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                 instance: str | None, axis: str) -> list[Path]:
    good = ok_rows(rows)
    algos = algorithms_in(good)
    styles = _style(algos)
    paths: list[Path] = []

    for inst in _target_instances(good, instance):
        inst_rows = [r for r in good if str(r["instance"]) == inst and r.get("history")]
        if not inst_rows:
            continue
        bks = _bks_of(inst_rows)

        curves_by_algo: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for r in inst_rows:
            t, it, c, _ = _history(r)
            x = t if axis == "time" else it
            if x.size:
                curves_by_algo.setdefault(str(r["algorithm"]), []).append((x, c))
        if not curves_by_algo:
            continue

        all_x = np.concatenate([x for cs in curves_by_algo.values() for x, _ in cs])
        if axis == "time":
            lo = float(np.min(all_x[all_x > 0])) if np.any(all_x > 0) else 1e-3
            hi = float(np.max(all_x))
            grid = np.geomspace(max(lo, 1e-4), max(hi, lo * 10), 240)
        else:
            hi = float(np.max(all_x))
            grid = np.linspace(1.0, max(hi, 2.0), 240)

        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            for algo in algos:
                cs = curves_by_algo.get(algo)
                if not cs:
                    continue
                med, q1, q3, counts = _step_band(cs, grid)
                # Early on, only the fastest seeds have reported anything, and a
                # median over one seed is not a median. Curves therefore start
                # where at least half the seeds have data.
                enough = counts >= max(1, (len(cs) + 1) // 2)
                med, q1, q3 = (np.where(enough, v, np.nan) for v in (med, q1, q3))
                st = styles[algo]
                ax.plot(grid, med, label=f"{algo} (n = {len(cs)})",
                        color=st["color"], linestyle=st["linestyle"])
                ax.fill_between(grid, q1, q3, color=st["color"], alpha=0.15, linewidth=0)
            if bks:
                ax.axhline(bks, color="0.35", linewidth=1.0, linestyle=(0, (4, 3)))
                # Blended transform: x in axes fraction, y in data units, so the
                # label sits just inside the left spine whatever the x scale is.
                ax.text(0.012, bks, f"best known {bks:g}", fontsize=8, color="0.35",
                        va="bottom", ha="left", transform=ax.get_yaxis_transform())
            if axis == "time":
                ax.set_xscale("log")
                ax.set_xlabel("wall-clock time (s, log scale)")
            else:
                ax.set_xlabel("iteration")
            ax.set_ylabel("best cost found (distance units)")
            caveat = ("a curve starts once at least half its seeds have recorded an "
                      "incumbent" if axis == "time" else
                      "a run that has stopped holds its final cost, so the right of "
                      "this plot mixes runs that stopped at different iterations")
            ax.set_title(f"Convergence on {inst}: median over seeds, band is the "
                         f"inter-quartile range\n({caveat})")
            ax.legend(loc="upper right", ncols=2)
            stem = f"convergence_{'time' if axis == 'time' else 'iterations'}_{_safe(inst)}"
            paths += _save(fig, out_dir, stem)
    return paths


def convergence_time(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                     instance: str | None = None) -> list[Path]:
    """Median best cost against wall-clock time, one figure per instance.

    The time axis is logarithmic because the interesting differences between
    metaheuristics happen in the first fraction of a second and would otherwise
    be compressed into the left-hand edge of the plot.
    """
    return _convergence(rows, out_dir, instance, axis="time")


def convergence_iterations(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                           instance: str | None = None) -> list[Path]:
    """Median best cost against iteration count, one figure per instance.

    Read together with :func:`convergence_time` this separates a better search
    from a cheaper one: an algorithm that wins per iteration but loses per
    second is paying too much for each step.
    """
    return _convergence(rows, out_dir, instance, axis="iterations")


# ---------------------------------------------------------------------------
# 3. Time to target
# ---------------------------------------------------------------------------
def time_to_target(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                   pct: float = 1.0, instance: str | None = None) -> list[Path]:
    """Empirical probability of having reached within ``pct``% of the best known.

    Runs that never reached the target are kept in the sample as infinite times,
    following Aiex, Resende and Ribeiro, so a curve that ends at 0.6 means forty
    percent of runs failed rather than that the data ran out.
    """
    from qroute.benchmark.report import first_within

    good = ok_rows(rows)
    if instance is not None:
        good = [r for r in good if str(r["instance"]) == instance]
    algos = algorithms_in(good)
    styles = _style(algos)
    if not good:
        return []

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        plotted = False
        for algo in algos:
            rs = [r for r in good if str(r["algorithm"]) == algo]
            times = []
            for r in rs:
                _, t = first_within(r, pct)
                times.append(float(t) if t is not None else math.inf)
            if not times:
                continue
            t_sorted, probs = time_to_target_curve(times)
            finite = np.isfinite(t_sorted)
            st = styles[algo]
            reached = int(finite.sum())
            label = f"{algo} ({reached}/{len(times)} reached)"
            if reached == 0:
                # Nothing to draw, but the algorithm must still appear in the
                # legend, otherwise its total failure is invisible.
                ax.plot([], [], color=st["color"], linestyle=st["linestyle"], label=label)
                continue
            ax.step(t_sorted[finite], probs[finite], where="post",
                    color=st["color"], linestyle=st["linestyle"], label=label)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, f"no run reached within {pct:g}% of the best known cost",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        else:
            ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("wall-clock time to reach the target (s, log scale)")
        ax.set_ylabel("empirical probability")
        scope = instance if instance else "all instances"
        ax.set_title(f"Time to reach within {pct:g}% of the best known cost ({scope})")
        # The curves rise to the right, so the upper left is always clear.
        ax.legend(loc="upper left")
        stem = f"time_to_target_{pct:g}pct" + (f"_{_safe(instance)}" if instance else "")
        return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# 4. Gap distribution
# ---------------------------------------------------------------------------
def gap_distribution(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                     seed: int = 0) -> list[Path]:
    """Box plot of the final gap of every run, one box per algorithm.

    Individual runs are drawn as jittered points on top of the box, because with
    only a handful of seeds per instance a box plot alone can suggest more data
    than there is. The jitter is drawn from a seeded generator so the figure is
    reproducible.
    """
    from qroute.core.rng import make_rng

    good = ok_rows(rows)
    algos = algorithms_in(good)
    styles = _style(algos)
    rng = make_rng(seed)

    data, labels, colours = [], [], []
    for algo in algos:
        gaps = [float(r["gap"]) for r in good
                if str(r["algorithm"]) == algo and r.get("gap") is not None
                and math.isfinite(float(r["gap"]))]
        if not gaps:
            continue
        data.append(np.asarray(gaps))
        labels.append(f"{algo}\n(n = {len(gaps)})")
        colours.append(styles[algo]["color"])
    if not data:
        return []

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(1.4 * len(data) + 2.6, 4.2))
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.55,
                        medianprops={"color": "black", "linewidth": 1.4},
                        patch_artist=True)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.30)
            patch.set_edgecolor(colour)
        for i, (values, colour) in enumerate(zip(data, colours), start=1):
            x = i + rng.uniform(-0.14, 0.14, size=values.size)
            ax.plot(x, values, linestyle="none", marker="o", markersize=3,
                    color=colour, alpha=0.75, markeredgewidth=0)
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_ylabel("gap above best known cost (%)")
        ax.set_title("Distribution of final gaps over all instances and seeds "
                     "(lower is better; 0 is the best known cost)")
        return _save(fig, out_dir, "gap_distribution")


# ---------------------------------------------------------------------------
# 5. Scalability
# ---------------------------------------------------------------------------
def scalability(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                sizes: Mapping[str, int] | None = None) -> list[Path]:
    """Mean gap and mean run time against instance size.

    Two stacked panels rather than one twin-axis plot: a twin axis invites the
    reader to compare two quantities whose scales have nothing to do with each
    other.
    """
    good = ok_rows(rows)
    algos = algorithms_in(good)
    styles = _style(algos)

    points: dict[str, list[tuple[int, float, float]]] = {}
    for algo in algos:
        for inst in instances_in(good):
            size = instance_size(inst, sizes)
            if size is None:
                continue
            rs = [r for r in good if str(r["instance"]) == inst and str(r["algorithm"]) == algo]
            gaps = [float(r["gap"]) for r in rs
                    if r.get("gap") is not None and math.isfinite(float(r["gap"]))]
            secs = [float(r["seconds"]) for r in rs if r.get("seconds") is not None]
            if not gaps or not secs:
                continue
            points.setdefault(algo, []).append((size, float(np.mean(gaps)), float(np.mean(secs))))
    if not points:
        return []

    with plt.rc_context(_RC):
        fig, (ax_gap, ax_time) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
        for algo in algos:
            pts = sorted(points.get(algo, []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            st = styles[algo]
            ax_gap.plot(xs, [p[1] for p in pts], label=algo, color=st["color"],
                        linestyle=st["linestyle"], marker=st["marker"], markersize=4)
            ax_time.plot(xs, [p[2] for p in pts], label=algo, color=st["color"],
                         linestyle=st["linestyle"], marker=st["marker"], markersize=4)
        ax_gap.set_ylabel("mean gap above best known (%)")
        ax_gap.set_title("Scalability: solution quality and run time against instance size")
        ax_time.set_ylabel("mean run time (s)")
        ax_time.set_xlabel("instance size (nodes, log scale)")
        ax_time.set_xscale("log")

        # Label the ticks with the instance sizes themselves; the default log
        # locator writes them as powers of ten, which is unreadable for a node
        # count.
        all_sizes = sorted({p[0] for pts in points.values() for p in pts})
        ax_time.set_xticks(all_sizes, minor=False)
        ax_time.set_xticks([], minor=True)
        ax_time.set_xticklabels([str(s) for s in all_sizes])

        # Under an equal wall-clock budget every algorithm runs for the same
        # time by construction, so say so rather than letting a reader read
        # meaning into a flat panel.
        all_secs = [p[2] for pts in points.values() for p in pts]
        spread = (max(all_secs) - min(all_secs)) / max(1e-9, float(np.mean(all_secs)))
        if spread < 0.05:
            ax_time.text(0.99, 0.04,
                         "run time is fixed by the benchmark budget, so this panel "
                         "shows only overhead",
                         transform=ax_time.transAxes, ha="right", va="bottom",
                         fontsize=8, color="0.35")
        ax_gap.legend(loc="upper left", ncols=2)
        return _save(fig, out_dir, "scalability")


# ---------------------------------------------------------------------------
# 6. Diversity
# ---------------------------------------------------------------------------
def diversity(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
              instance: str | None = None) -> list[Path]:
    """Population diversity beside best cost, both against iteration.

    Premature convergence is exactly the pattern of a diversity curve that
    reaches zero while the cost curve is still descending: the population has
    collapsed onto one point and any further improvement can only come from the
    local search, not from the swarm.
    """
    good = ok_rows(rows)
    algos = algorithms_in(good)
    styles = _style(algos)
    paths: list[Path] = []

    for inst in _target_instances(good, instance):
        inst_rows = [r for r in good if str(r["instance"]) == inst and r.get("history")]
        if not inst_rows:
            continue
        div_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        cost_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        max_it = 0.0
        for r in inst_rows:
            _, it, c, d = _history(r)
            if it.size == 0:
                continue
            max_it = max(max_it, float(it.max()))
            algo = str(r["algorithm"])
            cost_curves.setdefault(algo, []).append((it, c))
            if np.any(np.isfinite(d)):
                div_curves.setdefault(algo, []).append((it, d))
        if not cost_curves:
            continue
        grid = np.linspace(1.0, max(max_it, 2.0), 240)
        bks = _bks_of(inst_rows)

        with plt.rc_context(_RC):
            fig, (ax_d, ax_c) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
            for algo in algos:
                st = styles[algo]
                for axis, source in ((ax_d, div_curves), (ax_c, cost_curves)):
                    if algo not in source:
                        continue
                    curves = source[algo]
                    med, q1, q3, counts = _step_band(curves, grid)
                    enough = counts >= max(1, (len(curves) + 1) // 2)
                    med, q1, q3 = (np.where(enough, v, np.nan) for v in (med, q1, q3))
                    axis.plot(grid, med, color=st["color"], linestyle=st["linestyle"],
                              label=algo)
                    axis.fill_between(grid, q1, q3, color=st["color"], alpha=0.15,
                                      linewidth=0)
            if not div_curves:
                ax_d.text(0.5, 0.5, "no diversity was recorded for these runs",
                          ha="center", va="center", transform=ax_d.transAxes, fontsize=9)
            if bks:
                ax_c.axhline(bks, color="0.35", linewidth=1.0, linestyle=(0, (4, 3)))
            ax_d.set_ylabel("population diversity\n(mean pairwise distance)")
            ax_d.set_title(f"Diversity and best cost on {inst}: median over seeds, "
                           f"band is the inter-quartile range")
            ax_c.set_ylabel("best cost found (distance units)")
            ax_c.set_xlabel("iteration")
            # "best" rather than a fixed corner: the diversity curves of the
            # population methods sit at the top of their panel on some instances
            # and at the bottom on others, so a fixed legend would cover data.
            ax_d.legend(loc="best", ncols=2)
            paths += _save(fig, out_dir, f"diversity_{_safe(inst)}")
    return paths


# ---------------------------------------------------------------------------
# 7. Per-instance bars
# ---------------------------------------------------------------------------
def per_instance_gap_bars(rows: Sequence[Mapping[str, Any]],
                          out_dir: str | Path) -> list[Path]:
    """Grouped bars of mean gap per instance, with one standard deviation over
    seeds as the error bar. Instances are ordered by size."""
    good = ok_rows(rows)
    algos = algorithms_in(good)
    insts = instances_in(good)
    styles = _style(algos)
    if not insts or not algos:
        return []

    width = 0.8 / len(algos)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(insts) + 2.0), 4.2))
        base = np.arange(len(insts), dtype=float)
        for k, algo in enumerate(algos):
            means, errs = [], []
            for inst in insts:
                gaps = [float(r["gap"]) for r in good
                        if str(r["instance"]) == inst and str(r["algorithm"]) == algo
                        and r.get("gap") is not None and math.isfinite(float(r["gap"]))]
                means.append(float(np.mean(gaps)) if gaps else np.nan)
                errs.append(float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0)
            ax.bar(base + (k - (len(algos) - 1) / 2) * width, means, width * 0.92,
                   yerr=errs, capsize=2.5, label=algo, color=styles[algo]["color"],
                   alpha=0.85, error_kw={"linewidth": 0.9, "ecolor": "0.25"})
        ax.set_xticks(base)
        ax.set_xticklabels(insts, rotation=0 if len(insts) <= 8 else 45,
                           ha="center" if len(insts) <= 8 else "right")
        ax.set_ylabel("mean gap above best known (%)")
        ax.set_xlabel("instance (ordered by size)")
        ax.set_title("Mean gap per instance, error bars are one standard deviation over seeds")
        ax.legend(loc="upper left", ncols=min(len(algos), 4))
        return _save(fig, out_dir, "per_instance_gap")


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------
def all_plots(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
              targets: Iterable[float] = (1.0, 2.0),
              sizes: Mapping[str, int] | None = None) -> list[Path]:
    """Write every figure for a run and return the paths in the order written."""
    paths: list[Path] = []
    paths += convergence_time(rows, out_dir)
    paths += convergence_iterations(rows, out_dir)
    for pct in targets:
        paths += time_to_target(rows, out_dir, pct=pct)
    paths += gap_distribution(rows, out_dir)
    paths += scalability(rows, out_dir, sizes=sizes)
    paths += diversity(rows, out_dir)
    paths += per_instance_gap_bars(rows, out_dir)
    return paths
