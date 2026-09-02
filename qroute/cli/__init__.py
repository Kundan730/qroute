"""Command line interface for the qroute platform.

The console script declared in ``pyproject.toml`` points at
:data:`qroute.cli.main.app`, so this package is what a judge or a reviewer
touches first. Everything the platform can do -- solving one instance,
comparing algorithms with a statistical test, running a reproducible benchmark
sweep, proving optimality on small instances, building an instance from a real
road network and running the live-traffic demonstration -- is reachable from
here without writing any Python.

The heavy imports (numba-compiled kernels, OR-Tools, osmnx, uvicorn) are
deliberately kept out of module import time and are pulled in inside the
command that needs them, so ``qroute --help`` stays instant.
"""

from __future__ import annotations

__all__ = ["main", "render"]
