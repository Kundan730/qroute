"""Seeded random number generation.

Every stochastic component in the platform draws from a :class:`numpy.random.Generator`
created here, so that a single integer seed reproduces an entire experiment.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a NumPy generator for ``seed``.

    Passing an existing generator returns it unchanged, which lets components
    share a stream when that is desirable.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_seeds(seed: int, n: int) -> list[int]:
    """Derive ``n`` independent, reproducible child seeds from ``seed``.

    Used by the benchmark runner so that run *k* of an algorithm is identical
    no matter which order the runs execute in, or whether they run in parallel.
    """
    ss = np.random.SeedSequence(seed)
    return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(n)]
