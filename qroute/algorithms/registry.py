"""Name-to-class registry for the solvers.

The CLI, the benchmark runner and the HTTP API all need to turn a string such as
``"qpso"`` into a configured optimiser, and all three need to list the available
names without paying for the import of every algorithm module. Numba-compiled
kernels are imported transitively by all of them, and importing a module whose
kernels are not yet cached costs seconds, so ``ALGORITHMS`` maps names to
*import paths* and the class object is only loaded when it is actually asked
for. :func:`names` therefore stays a dictionary lookup.

Classes are resolved once and memoised, so repeated ``build`` calls in a
benchmark sweep do not re-import.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from qroute.algorithms.base import Optimizer, StopCriteria
    from qroute.problems.instance import Instance

#: Registered solver name -> ``"module:attribute"``. Names are lowercase and are
#: the strings accepted on the command line and in API requests.
ALGORITHMS: dict[str, str] = {
    "qpso": "qroute.algorithms.qpso:QPSO",
    "pso": "qroute.algorithms.pso:PSO",
    "ga": "qroute.algorithms.ga:GeneticAlgorithm",
    "sa": "qroute.algorithms.sa:SimulatedAnnealing",
    "aco": "qroute.algorithms.aco:AntColony",
}

#: Human-readable one-liners, used by ``--help`` output and the API's
#: capability endpoint. Kept here so listing algorithms needs no imports.
DESCRIPTIONS: dict[str, str] = {
    "qpso": "Quantum-behaved particle swarm (the proposed method)",
    "pso": "Classical particle swarm with constriction coefficients",
    "ga": "Steady-state genetic algorithm with order crossover",
    "sa": "Simulated annealing over route-level neighbourhoods",
    "aco": "Ant Colony System with MAX-MIN trail bounds",
}

_CACHE: dict[str, type] = {}


def names() -> list[str]:
    """Registered algorithm names, in a stable order."""
    return list(ALGORITHMS)


def catalogue() -> list[dict[str, str]]:
    """``[{"name": ..., "description": ...}]`` for the API and the CLI."""
    return [{"name": n, "description": DESCRIPTIONS.get(n, "")} for n in ALGORITHMS]


def get(name: str) -> type:
    """Resolve a name to its optimiser class, importing it on first use."""
    key = name.strip().lower()
    if key in _CACHE:
        return _CACHE[key]
    try:
        path = ALGORITHMS[key]
    except KeyError:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {', '.join(ALGORITHMS)}"
        ) from None
    module_name, _, attr = path.partition(":")
    cls = getattr(import_module(module_name), attr)
    _CACHE[key] = cls
    return cls


def build(name: str, instance: "Instance", stop: "Optional[StopCriteria]" = None,
          seed: Optional[int] = None, callback: Any = None, **params) -> "Optimizer":
    """Construct a configured optimiser.

    Every solver takes the same first four arguments, which is what makes a
    benchmark sweep a one-line loop over :data:`ALGORITHMS`. Algorithm-specific
    settings are passed through as keyword arguments; each class accepts and
    ignores unknown keywords via ``**kw`` so a shared configuration dictionary
    can be handed to all of them without filtering.
    """
    return get(name)(instance, stop=stop, seed=seed, callback=callback, **params)
