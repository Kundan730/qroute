"""Integer scaling of cost matrices for integer-programming solvers.

CP-SAT, OR-Tools routing and PyVRP all require integral arc costs, while the
project's :class:`~qroute.problems.instance.Instance` carries ``float64``
matrices. This module centralises the conversion so that every solver in the
platform scales the same way and their objective values remain comparable.

Why scaling matters for correctness
-----------------------------------
If a solver silently truncates ``12.7`` to ``12`` it will optimise a *different*
problem and its "optimal" answer can be cheaper than the true optimum. That is
exactly the failure mode that makes a benchmark meaningless. The rule used here
is therefore:

* find the smallest power of ten that makes every entry integral to within
  ``tol``; if one exists the transformation is exact and the solver's objective
  is the true objective times that factor;
* if no power of ten up to ``max_scale`` works, fall back to ``max_scale`` and
  *round*. Rounding is reported through :attr:`Scaling.exact` so a caller can
  refuse to claim optimality on a scaled model.

The two benchmark families both land in the exact case: CVRPLIB ``EUC_2D``
distances are already integers (scale 1) and Solomon distances are truncated to
one decimal (scale 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Powers of ten tried in order. Ten thousand is already beyond any benchmark we
# use and keeps the largest CP-SAT coefficient well inside int64.
_CANDIDATES = (1, 10, 100, 1000, 10000)


@dataclass(frozen=True)
class Scaling:
    """The multiplier used to turn float costs into integers.

    Attributes
    ----------
    factor:
        Multiply a float cost by this to obtain the integer cost.
    exact:
        ``True`` when every input entry was integral after scaling, i.e. the
        integer model is equivalent to the float model rather than an
        approximation of it.
    max_error:
        Largest absolute rounding error introduced on a single arc, in the
        instance's own units. Zero when ``exact``.
    """

    factor: int
    exact: bool
    max_error: float

    def to_int(self, values: np.ndarray) -> np.ndarray:
        """Scale and round ``values`` to a C-contiguous ``int64`` array."""
        return np.ascontiguousarray(np.rint(np.asarray(values, dtype=np.float64) * self.factor).astype(np.int64))

    def to_float(self, value: float | int) -> float:
        """Inverse transform, for reporting a scaled objective in native units."""
        return float(value) / self.factor


def integer_scaling(*arrays: np.ndarray, max_scale: int = 10000, tol: float = 1e-6) -> Scaling:
    """Smallest power-of-ten multiplier making every array integral.

    All arrays are considered together so that, for example, a distance matrix
    and a duration matrix end up on the same scale and can be mixed inside one
    model.
    """
    stacked = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if a is not None]
    if not stacked:
        return Scaling(1, True, 0.0)
    flat = np.concatenate(stacked)
    if flat.size == 0:
        return Scaling(1, True, 0.0)

    for factor in _CANDIDATES:
        if factor > max_scale:
            break
        scaled = flat * factor
        err = np.abs(scaled - np.rint(scaled)).max()
        if err <= tol:
            return Scaling(factor, True, 0.0)

    factor = min(max_scale, _CANDIDATES[-1])
    scaled = flat * factor
    err = float(np.abs(scaled - np.rint(scaled)).max()) / factor
    return Scaling(factor, False, err)


def integer_demands(demand: np.ndarray, capacity: float) -> tuple[np.ndarray, int, int]:
    """Integral demands and capacity.

    Demands are integral in every benchmark family we support, so a
    non-integral demand is treated as an error rather than silently rounded:
    rounding a demand up makes the problem harder and rounding it down makes a
    reported solution potentially infeasible.
    """
    d = np.asarray(demand, dtype=np.float64)
    if np.abs(d - np.rint(d)).max() > 1e-9:
        raise ValueError(
            "non-integral demands are not supported by the integer-programming "
            "models; scale the instance's demands and capacity yourself"
        )
    cap = int(np.floor(capacity + 1e-9))
    if abs(capacity - cap) > 1e-9:
        # Flooring a capacity only ever removes feasible solutions, so refuse.
        raise ValueError("non-integral vehicle capacity is not supported")
    return np.rint(d).astype(np.int64), cap, int(np.rint(d).sum())
