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

:mod:`qroute.config`
    Not part of this package, but read by all of it: the typed settings object
    that decides where the data is, what CORS permits and how long a run may
    take.

Run it with ``uvicorn qroute.api.app:app`` or ``python -m qroute.api.app``.

Import order
------------
Importing anything from this package resolves the settings and publishes the
resolved absolute data paths into the environment *first*. That has to happen
before :mod:`qroute.problems.loaders` or :mod:`qroute.graph.osm` are imported,
because both read ``QROUTE_DATA`` at import time and default it to the relative
string ``"data"`` - which is why, before this, starting the server from anywhere
but the repository root produced an API that served an empty catalogue instead
of an error. Neither module is imported at the top of any module in this
package, so doing it here is sufficient; see
:meth:`qroute.config.Settings.export_to_environment` for why the bridge exists
at all and what should replace it.
"""

from __future__ import annotations

from qroute.config import settings

settings().export_to_environment()

from qroute.api.app import create_app  # noqa: E402  (must follow the path bridge above)

__all__ = ["create_app", "settings"]
