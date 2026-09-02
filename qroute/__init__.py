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
