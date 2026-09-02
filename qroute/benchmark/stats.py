"""Statistical comparison of optimisation algorithms.

Comparing metaheuristics honestly needs more than a table of best-found costs.
A metaheuristic is a random variable: one lucky seed proves nothing. The tests
here follow the protocol recommended by Derrac, Garcia, Molina and Herrera
(2011) for comparing evolutionary algorithms, which is the standard reference in
this literature.

The procedure used by the benchmark reports is:

1. Run every algorithm on every instance with the same set of seeds and the same
   wall-clock budget, so runs are *paired* by seed.
2. Compare a pair of algorithms across instances with the **Wilcoxon signed-rank
   test**, which is non-parametric and therefore does not assume that gaps are
   normally distributed. They are not: gaps are bounded below and skewed.
3. Compare *all* algorithms at once with the **Friedman test** on the per-instance
   ranks, followed by **Holm's step-down correction** on the pairwise
   comparisons against the control algorithm. Correction is necessary: with six
   algorithms there are fifteen pairwise comparisons, and at a five percent
   significance level roughly one of them is expected to look significant by
   chance alone.

A p-value is not a claim of practical importance, so the reports also give the
effect size, and a difference is only described as meaningful when both the test
and the size of the difference support it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


@dataclass
class PairwiseResult:
    """Outcome of comparing two algorithms across paired observations."""

    a: str
    b: str
    n: int
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    p_adjusted: float | None = None
    effect_size: float = 0.0        # matched-pairs rank-biserial correlation
    winner: str | None = None

    def describe(self, alpha: float = 0.05) -> str:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        if not np.isfinite(p) or p > alpha:
            return f"{self.a} vs {self.b}: no significant difference (p = {p:.3g}, n = {self.n})"
        better = self.winner or (self.a if self.median_a < self.median_b else self.b)
        return (f"{better} is better (p = {p:.3g}, n = {self.n}, "
                f"effect size {abs(self.effect_size):.2f})")


@dataclass
class FriedmanResult:
    """Outcome of the omnibus test across all algorithms."""

    algorithms: list[str]
    mean_ranks: dict[str, float]
    statistic: float
    p_value: float
    n_instances: int
    post_hoc: list[PairwiseResult] = field(default_factory=list)
    control: str | None = None

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.mean_ranks.items(), key=lambda kv: kv[1])


def wilcoxon(a: Sequence[float], b: Sequence[float], names: tuple[str, str] = ("A", "B")) -> PairwiseResult:
    """Paired Wilcoxon signed-rank test on two equal-length samples.

    ``a`` and ``b`` must be paired: element *k* of each is the same instance and
    the same seed. Lower is better, as these are costs or gaps.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("wilcoxon requires paired samples of equal length")
    diff = x - y
    nonzero = np.count_nonzero(np.abs(diff) > 1e-12)
    if nonzero == 0:
        # Identical samples: the test is undefined, and reporting p = 1 is the
        # honest answer rather than letting scipy raise.
        return PairwiseResult(names[0], names[1], len(x), float(np.median(x)),
                              float(np.median(y)), 0.0, 1.0, effect_size=0.0)
    try:
        stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return PairwiseResult(names[0], names[1], len(x), float(np.median(x)),
                              float(np.median(y)), float("nan"), 1.0)
    # Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-)
    d = diff[np.abs(diff) > 1e-12]
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = w_pos + w_neg
    effect = float((w_pos - w_neg) / total) if total > 0 else 0.0
    winner = names[1] if effect > 0 else names[0]   # positive means a > b, so b wins
    return PairwiseResult(names[0], names[1], len(x), float(np.median(x)), float(np.median(y)),
                          float(stat), float(p), effect_size=effect, winner=winner)


def holm_correction(results: Sequence[PairwiseResult], alpha: float = 0.05) -> list[PairwiseResult]:
    """Apply Holm's step-down correction to a family of comparisons.

    Holm is uniformly more powerful than Bonferroni and makes no independence
    assumption, which matters here because comparisons that share a control
    algorithm are correlated by construction.
    """
    ordered = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    out = list(results)
    running_max = 0.0
    for rank, idx in enumerate(ordered):
        adj = min(1.0, (m - rank) * results[idx].p_value)
        running_max = max(running_max, adj)   # enforce monotonicity
        r = results[idx]
        out[idx] = PairwiseResult(r.a, r.b, r.n, r.median_a, r.median_b, r.statistic,
                                  r.p_value, running_max, r.effect_size, r.winner)
    return out


def friedman(per_instance: Mapping[str, Sequence[float]], control: str | None = None,
             alpha: float = 0.05) -> FriedmanResult:
    """Friedman test over algorithms, plus Holm-corrected comparisons to a control.

    ``per_instance`` maps an algorithm name to its score on each instance, in a
    consistent instance order (typically the median gap over seeds). Lower is
    better.
    """
    names = list(per_instance)
    if len(names) < 3:
        raise ValueError("the Friedman test needs at least three algorithms; use wilcoxon for two")
    matrix = np.array([np.asarray(per_instance[n], dtype=float) for n in names])
    if matrix.ndim != 2:
        raise ValueError("every algorithm must be scored on the same instances")
    n_instances = matrix.shape[1]

    # Rank algorithms within each instance, 1 = best. Ties share the average rank.
    ranks = np.apply_along_axis(stats.rankdata, 0, matrix)
    mean_ranks = {n: float(ranks[i].mean()) for i, n in enumerate(names)}

    stat, p = stats.friedmanchisquare(*[matrix[i] for i in range(len(names))])

    ctrl = control or min(mean_ranks, key=mean_ranks.get)
    comps = [wilcoxon(per_instance[ctrl], per_instance[n], (ctrl, n))
             for n in names if n != ctrl]
    comps = holm_correction(comps, alpha)
    return FriedmanResult(names, mean_ranks, float(stat), float(p), n_instances, comps, ctrl)


def summarise(values: Iterable[float]) -> dict[str, float]:
    """Best, mean, median, standard deviation and inter-quartile range."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "best": float(v.min()),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "worst": float(v.max()),
    }


def time_to_target_curve(times: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Empirical cumulative distribution of times to reach a target.

    Returns sorted times and their plotting positions ``(i - 0.5) / n``, the
    convention used for time-to-target plots by Aiex, Resende and Ribeiro. A
    run that never reached the target should be passed as infinity so that the
    curve honestly stops below one rather than silently dropping the failure.
    """
    t = np.sort(np.asarray(times, dtype=float))
    n = t.size
    if n == 0:
        return np.array([]), np.array([])
    probs = (np.arange(1, n + 1) - 0.5) / n
    return t, probs
