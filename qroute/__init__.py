"""qroute - Quantum-inspired intelligent traffic route optimisation.

SIH 2026 Problem Statement 26137 (Egreen Quanta).

The package is organised as:

* :mod:`qroute.problems`   - problem instances (CVRP / VRPTW / time-dependent VRP)
* :mod:`qroute.graph`      - road network modelling and travel-time matrices
* :mod:`qroute.traffic`    - congestion models and the dynamic weight-update engine
* :mod:`qroute.algorithms` - quantum-inspired optimisers (QPSO, QIEA) and classical baselines
* :mod:`qroute.exact`      - exact methods (MILP / dynamic programming) used as ground truth
* :mod:`qroute.benchmark`  - reproducible benchmarking, statistics and reporting
* :mod:`qroute.api`        - FastAPI service backing the web platform
* :mod:`qroute.cli`        - command line entry points
"""

__version__ = "0.1.0"


def _configure_numba_cache() -> None:
    """Point Numba's compilation cache somewhere guaranteed to be writable.

    Numba caches compiled kernels next to the source that defines them, inside
    the package's own ``__pycache__``. That is fine on a developer's machine and
    wrong almost everywhere else: in a container the package is installed into
    site-packages owned by root and the service runs as an unprivileged user, so
    the directory is not writable, the cache is silently never written, and
    every process recompiles from scratch.

    The cost is not marginal. Measured on this codebase, a solve with a warm
    cache takes 0.49 seconds and the same solve with a cold one takes 6.34
    seconds, and the penalty is paid again by every worker process the pool
    spawns. Against a ten-second budget that is most of the run spent in LLVM
    rather than in the search, and the answer reported would be the construction
    heuristic rather than anything the optimiser found.

    The variable has to be set before Numba is imported, which is why this runs
    at package import and not from the settings module: by the time anything
    calls :func:`qroute.config.settings` the compiler is already configured.
    An explicit ``NUMBA_CACHE_DIR`` is always respected.
    """
    import os
    import tempfile
    from pathlib import Path

    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    # Prefer a real cache location, fall back to the temporary directory. Both
    # are per-user, so two accounts on one host cannot collide.
    base = os.environ.get("XDG_CACHE_HOME")
    candidates = [
        Path(base) / "qroute" if base else Path.home() / ".cache" / "qroute",
        Path(tempfile.gettempdir()) / f"qroute-numba-{os.getuid() if hasattr(os, 'getuid') else 'cache'}",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".writable"
            probe.touch()
            probe.unlink()
        except OSError:
            continue
        os.environ["NUMBA_CACHE_DIR"] = str(candidate)
        return
    # Nothing writable was found. Leave the variable unset so Numba keeps its
    # own behaviour and reports its own warning, rather than failing here for a
    # problem that only makes the program slower.


_configure_numba_cache()
