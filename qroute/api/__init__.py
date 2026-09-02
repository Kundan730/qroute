"""The HTTP service that backs the qroute web platform.

The package is deliberately thin: every endpoint is a translation layer over
functionality that already exists in :mod:`qroute.problems`,
:mod:`qroute.graph`, :mod:`qroute.traffic`, :mod:`qroute.algorithms` and
:mod:`qroute.benchmark`. No optimisation, no traffic modelling and no
statistics are implemented here, so the numbers the browser shows are exactly
the numbers the library produces on the command line.

Modules
-------
:mod:`qroute.api.app`
    The application factory, CORS policy, static-file mounting and the
    endpoints that do not need shared mutable state.
:mod:`qroute.api.schemas`
    Pydantic request and response models. They are the executable form of the
    wire contract that ``frontend/src/api/types.ts`` declares on the other side.
:mod:`qroute.api.state`
    Process-wide state: lazily loaded road networks with their traffic
    simulators, the generated-instance store, and the run registry.
:mod:`qroute.api.runs`
    Runs solvers in separate processes and streams their convergence to the
    browser over Server-Sent Events.
:mod:`qroute.api.networks`
    Road-network, traffic-simulator and shortest-path endpoints.

Run it with ``uvicorn qroute.api.app:app`` or ``python -m qroute.api.app``.
"""

from __future__ import annotations

from qroute.api.app import create_app

__all__ = ["create_app"]
