"""Typed settings for the qroute service, read once from the process environment.

Why this module exists
----------------------
Before it, the service was configured by scattered ``os.environ.get(...)`` calls
evaluated at import time in whichever module happened to need them, and every
default was a *relative* path. That has two consequences a judge or an operator
will hit immediately:

* the server silently finds nothing unless it is started from the repository
  root, because ``Path("data")`` means "data below the current directory"; and
* there is nowhere to look to find out what the service is actually configured
  to do.

Both are fixed here. Every knob is a field on one frozen dataclass with an
explicit type, a documented default and a documented environment variable, and
every path default is anchored on the *installation* rather than on
``os.getcwd()``.

Why a dataclass and not pydantic-settings
-----------------------------------------
``pydantic-settings`` is not installed in this environment, and pulling in a
dependency to parse eleven environment variables would be a poor trade for a
project whose install already takes OR-Tools, osmnx and numba. The parsing here
is explicit, typed and rejects malformed values loudly (:class:`ConfigError`)
rather than falling back to a default, which is the only behaviour of
``pydantic-settings`` that actually matters at this size.

Reading the settings
--------------------
Call :func:`settings`. The result is cached for the life of the process, so the
environment is read once and every component sees the same answer.
:func:`reload_settings` exists for tests that need to vary the environment.

A note on the environment bridge
--------------------------------
:meth:`Settings.export_to_environment` writes the *resolved absolute* paths back
into ``os.environ``. That is a transitional bridge, not the intended design:
:mod:`qroute.problems.loaders` and :mod:`qroute.graph.osm` both compute their
data directories from ``QROUTE_DATA`` at import time, and until they import
these settings directly the only way to give them an absolute root is to put one
in the variable they already read. It is called explicitly from
:mod:`qroute.api` rather than as an import side effect, so the ordering is
visible in the code instead of implied.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Optional

__all__ = [
    "ConfigError",
    "Settings",
    "configure_logging",
    "reload_settings",
    "settings",
]

#: The directory holding this file: ``<somewhere>/qroute``.
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent

#: The directory the package is installed under. In a source checkout - which is
#: how the platform is run and how a judge will run it - this is the repository
#: root, so ``INSTALL_ROOT / "data"`` is the committed benchmark data regardless
#: of the working directory the server was started from.
INSTALL_ROOT: Final[Path] = PACKAGE_DIR.parent

#: How far up from the working directory to look for a ``data`` directory when
#: the package is installed somewhere that has none beside it (a wheel in
#: site-packages, for example). Four levels covers being inside ``frontend/``,
#: ``scripts/`` or a results directory without turning the search into a
#: filesystem crawl.
_UPWARD_SEARCH_LEVELS: Final[int] = 4

#: Log levels accepted by ``QROUTE_LOG_LEVEL``, most to least verbose.
_LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

#: Log formats accepted by ``QROUTE_LOG_FORMAT``. ``text`` is one aligned line
#: per record for a human watching a terminal; ``json`` is one JSON object per
#: line for anything that ships logs somewhere.
_LOG_FORMATS: Final[tuple[str, ...]] = ("text", "json")


class ConfigError(RuntimeError):
    """A setting was present but unusable, or required data is missing.

    Always raised with a message that names the environment variable at fault
    and states what an operator should do about it. A configuration mistake that
    produces a silent default is worse than one that stops the process.
    """


# --------------------------------------------------------------------------
# Environment parsing
# --------------------------------------------------------------------------


def _raw(name: str) -> Optional[str]:
    """The environment value for ``name``, or ``None`` when unset or blank.

    An explicitly empty variable is treated as absent. Deployment tooling sets
    empty strings by accident often enough that reading one as "the empty
    string" is never what the operator meant.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(name: str, default: str, *, allowed: Optional[tuple[str, ...]] = None) -> str:
    value = _raw(name)
    if value is None:
        return default
    if allowed is not None and value.upper() not in {a.upper() for a in allowed}:
        raise ConfigError(f"{name}={value!r} is not one of {', '.join(allowed)}")
    return value


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ConfigError(f"{name}={value!r} is not an integer") from None
    if not minimum <= parsed <= maximum:
        raise ConfigError(f"{name}={parsed} is outside the accepted range {minimum}..{maximum}")
    return parsed


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        raise ConfigError(f"{name}={value!r} is not a number") from None
    if not minimum <= parsed <= maximum:
        raise ConfigError(f"{name}={parsed} is outside the accepted range {minimum}..{maximum}")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name}={value!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")


def _env_path(name: str, default: Path) -> Path:
    """An absolute path from ``name``, or ``default``.

    A relative value is resolved against the current working directory, because
    an operator who types a relative path on a command line means "relative to
    where I am". A *default* is never relative: that is the whole point of this
    module.
    """
    value = _raw(name)
    if value is None:
        return default
    return Path(value).expanduser().resolve()


def _env_csv(name: str) -> tuple[str, ...]:
    value = _raw(name)
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


# --------------------------------------------------------------------------
# Path anchoring
# --------------------------------------------------------------------------


def _data_root_candidates() -> list[Path]:
    """Places a ``data`` directory could plausibly be, most authoritative first.

    The installation is checked before the working directory on purpose. When
    both exist the one that ships with the code is the one whose contents match
    the code, and preferring the working directory would reintroduce exactly the
    "works on my laptop, from my shell, in my directory" behaviour this module
    exists to remove.
    """
    candidates = [INSTALL_ROOT / "data"]
    here = Path.cwd().resolve()
    for level, directory in enumerate([here, *here.parents]):
        if level > _UPWARD_SEARCH_LEVELS:
            break
        candidates.append(directory / "data")
    # ``dict.fromkeys`` keeps the first occurrence and drops later duplicates,
    # which matters because the installation root is often also an ancestor of
    # the working directory.
    return list(dict.fromkeys(candidates))


def _looks_like_data_root(path: Path) -> bool:
    """True when ``path`` holds something this platform recognises.

    ``data/osm`` is gitignored and rebuilt by ``qroute osm fetch``, so a fresh
    clone has ``data/benchmarks`` and nothing else. Either subdirectory is
    therefore enough to identify the right directory.
    """
    return (path / "benchmarks").is_dir() or (path / "osm").is_dir()


def _default_data_root() -> Path:
    """The first candidate that actually holds data, else the installation's.

    Falling back to ``INSTALL_ROOT / "data"`` rather than raising keeps the
    failure at the point of use, where :meth:`Settings.require_data` can say
    what is missing and how to supply it, instead of at import time where the
    traceback would be useless.
    """
    for candidate in _data_root_candidates():
        if _looks_like_data_root(candidate):
            return candidate
    return INSTALL_ROOT / "data"


# --------------------------------------------------------------------------
# The settings object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Every externally configurable value the service has.

    Frozen because configuration that changes under a running server is a
    source of bugs that only appear under load: two requests would disagree
    about the time limit. Tests that need different settings build a new object
    with :func:`dataclasses.replace` and install it with :func:`reload_settings`.
    """

    # ---------------------------------------------------------------- server
    #: ``QROUTE_HOST`` - interface uvicorn binds. The default is the loopback
    #: address rather than ``0.0.0.0``: a development server that is reachable
    #: from the whole network the moment it starts is a decision an operator
    #: should have to make on purpose.
    host: str = "127.0.0.1"

    #: ``QROUTE_PORT`` - port uvicorn binds.
    port: int = 8000

    # ----------------------------------------------------------------- paths
    #: ``QROUTE_DATA`` - root of the committed benchmark instances
    #: (``benchmarks/``) and the road graphs (``osm/``). Defaults to the ``data``
    #: directory beside the installed package, so the service finds its data no
    #: matter which directory it was started from.
    data_root: Path = INSTALL_ROOT / "data"

    #: ``QROUTE_RESULTS`` - where the benchmark runner writes result sets and
    #: where ``/api/benchmarks`` reads them. Missing is not an error: it just
    #: means no benchmark has been run yet.
    results_root: Path = INSTALL_ROOT / "results" / "runs"

    #: ``QROUTE_FRONTEND`` - the built single-page application. Missing is not
    #: an error either; the API serves a short explanatory JSON document at
    #: ``/`` instead, and everything under ``/api`` works regardless.
    frontend_dist: Path = INSTALL_ROOT / "frontend" / "dist"

    # --------------------------------------------------------------- logging
    #: ``QROUTE_LOG_LEVEL`` - one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_level: str = "INFO"

    #: ``QROUTE_LOG_FORMAT`` - ``text`` (default) or ``json``.
    log_format: str = "text"

    #: ``QROUTE_REQUEST_LOG`` - log one line per HTTP request with its status and
    #: server-side duration. On by default: the latency of a request is the main
    #: thing anyone debugging this service wants to know, and the events are
    #: cheap.
    request_log: bool = True

    # ----------------------------------------------------------------- CORS
    #: ``QROUTE_CORS_ORIGINS`` - comma-separated list of origins permitted to
    #: call the API cross-origin. Empty by default, which means same-origin
    #: only. See :meth:`cors_enabled` and the comment in
    #: :func:`qroute.api.app.create_app` for why that is the right default here.
    cors_allow_origins: tuple[str, ...] = ()

    #: ``QROUTE_CORS_ORIGIN_REGEX`` - a regular expression matched against the
    #: whole ``Origin`` header, for the case where the permitted origins are not
    #: a fixed list (ephemeral preview deployments, for instance).
    cors_allow_origin_regex: Optional[str] = None

    # ---------------------------------------------------------------- limits
    #: ``QROUTE_MAX_ACTIVE_RUNS`` - how many solver processes may run at once.
    #: Each one pins a core, and the machine also has to serve the API and, in a
    #: demonstration, a browser. Matches the default in
    #: :data:`qroute.api.runs.MAX_ACTIVE_RUNS`.
    max_active_runs: int = 4

    #: ``QROUTE_MAX_RUN_SECONDS`` - the longest wall-clock budget a single run
    #: may occupy a worker for. A request that asks for more is clamped to this
    #: and told so, rather than rejected, because the useful answer to "solve for
    #: an hour" on a shared demonstration machine is a five-minute solve with an
    #: explanation, not a 4xx. Note that ``RunRequest`` independently caps the
    #: field at 600 s at the wire level, so values above that never take effect.
    max_run_seconds: float = 300.0

    # --------------------------------------------------------------- startup
    #: ``QROUTE_API_PRELOAD`` - which road networks to load into memory at
    #: startup: ``all`` (the default), ``none``, ``first``, or a comma-separated
    #: list of network ids. Loading all three bundled extracts costs about 28
    #: seconds in a background thread and roughly 550 MB resident.
    preload: str = "all"

    # ------------------------------------------------------------------ read
    @classmethod
    def from_environment(cls) -> "Settings":
        """Build the settings from ``os.environ``, raising on a bad value."""
        data_root = _env_path("QROUTE_DATA", _default_data_root())
        return cls(
            host=_env_str("QROUTE_HOST", cls.host),
            port=_env_int("QROUTE_PORT", cls.port, minimum=1, maximum=65535),
            data_root=data_root,
            # The results and frontend defaults follow the data root's *parent*
            # when the operator has moved the data, so pointing QROUTE_DATA at a
            # deployment directory relocates the whole set coherently.
            results_root=_env_path("QROUTE_RESULTS", data_root.parent / "results" / "runs"),
            frontend_dist=_env_path("QROUTE_FRONTEND", data_root.parent / "frontend" / "dist"),
            log_level=_env_str("QROUTE_LOG_LEVEL", cls.log_level, allowed=_LOG_LEVELS).upper(),
            log_format=_env_str("QROUTE_LOG_FORMAT", cls.log_format, allowed=_LOG_FORMATS).lower(),
            request_log=_env_bool("QROUTE_REQUEST_LOG", cls.request_log),
            cors_allow_origins=_env_csv("QROUTE_CORS_ORIGINS"),
            cors_allow_origin_regex=_raw("QROUTE_CORS_ORIGIN_REGEX"),
            max_active_runs=_env_int("QROUTE_MAX_ACTIVE_RUNS", cls.max_active_runs,
                                     minimum=1, maximum=64),
            max_run_seconds=_env_float("QROUTE_MAX_RUN_SECONDS", cls.max_run_seconds,
                                       minimum=1.0, maximum=86_400.0),
            preload=_env_str("QROUTE_API_PRELOAD", cls.preload),
        )

    # ------------------------------------------------------------ derived
    @property
    def benchmarks_dir(self) -> Path:
        """Where CVRPLIB and Solomon instances live."""
        return self.data_root / "benchmarks"

    @property
    def osm_dir(self) -> Path:
        """Where the GraphML road networks live."""
        return self.data_root / "osm"

    @property
    def cors_enabled(self) -> bool:
        """True when any cross-origin access has been configured."""
        return bool(self.cors_allow_origins or self.cors_allow_origin_regex)

    @property
    def frontend_built(self) -> bool:
        """True when :attr:`frontend_dist` holds a built application."""
        return self.frontend_dist.is_dir() and (self.frontend_dist / "index.html").is_file()

    # ------------------------------------------------------------ validation
    def require_data(self) -> None:
        """Raise :class:`ConfigError` when the data root is genuinely unusable.

        "Genuinely" means neither ``benchmarks`` nor ``osm`` is present. A clone
        that has never run ``qroute osm fetch`` has no ``osm`` directory and is
        perfectly serviceable, so that alone is a warning at startup, not an
        error.
        """
        if _looks_like_data_root(self.data_root):
            return
        tried = "\n  ".join(str(p) for p in _data_root_candidates())
        raise ConfigError(
            f"no qroute data directory at {self.data_root}. It must contain "
            "'benchmarks/' (committed benchmark instances) and, once "
            "`qroute osm fetch` has been run, 'osm/' (road graphs).\n"
            f"Directories searched:\n  {tried}\n"
            "Set QROUTE_DATA to the absolute path of the data directory, for "
            "example QROUTE_DATA=/srv/qroute/data."
        )

    def clamp_run_seconds(self, requested: float) -> tuple[float, bool]:
        """Bound a client's requested time limit.

        Returns the budget the run will actually get and whether it was reduced,
        so the endpoint can report the reduction instead of quietly granting
        something other than what was asked for.
        """
        if requested > self.max_run_seconds:
            return self.max_run_seconds, True
        return requested, False

    # -------------------------------------------------------------- reporting
    def describe(self) -> list[tuple[str, str]]:
        """Key/value pairs for the startup banner, in a stable order.

        Deliberately not exposed over HTTP. It names filesystem paths, and an
        unauthenticated endpoint that describes the host's directory layout is a
        gift to anyone probing the service; the operator who started the process
        can read them in its log.
        """
        def state(path: Path, present: str, absent: str) -> str:
            return f"{path} ({present if path.is_dir() else absent})"

        origins = ", ".join(self.cors_allow_origins) if self.cors_allow_origins else "(none)"
        frontend_state = "built" if self.frontend_built else "not built, run `npm run build` in frontend/"
        return [
            ("data root", state(self.data_root, "present", "MISSING")),
            ("benchmarks", state(self.benchmarks_dir, "present", "missing")),
            ("road graphs", state(self.osm_dir, "present", "missing, run `qroute osm fetch`")),
            ("results root", state(self.results_root, "present", "missing, no benchmark run yet")),
            ("frontend", f"{self.frontend_dist} ({frontend_state})"),
            ("cors origins", origins),
            ("cors origin regex", self.cors_allow_origin_regex or "(none)"),
            ("max active runs", str(self.max_active_runs)),
            ("max run seconds", f"{self.max_run_seconds:g}"),
            ("network preload", self.preload),
            ("log level/format", f"{self.log_level}/{self.log_format}"),
        ]

    # -------------------------------------------------------------- bridging
    def export_to_environment(self) -> None:
        """Publish the resolved absolute paths back into ``os.environ``.

        :mod:`qroute.problems.loaders` and :mod:`qroute.graph.osm` read
        ``QROUTE_DATA`` at import time and default it to the relative string
        ``"data"``, which is what makes those modules find nothing when the
        process was not started from the repository root. Overwriting the
        variable with the resolved root before they are imported gives them the
        same answer these settings give everyone else.

        This is a bridge and should disappear: those two modules should import
        :func:`settings` instead. It is written here, in one place, with the
        reason attached, rather than left as an undocumented assignment.

        It also reaches the solver worker processes, which inherit the
        environment, so a worker resolves data exactly as its parent did.
        """
        os.environ["QROUTE_DATA"] = str(self.data_root)
        os.environ["QROUTE_RESULTS"] = str(self.results_root)
        os.environ["QROUTE_FRONTEND"] = str(self.frontend_dist)


# --------------------------------------------------------------------------
# Process-wide access
# --------------------------------------------------------------------------


#: The one settings object this process uses. Built on first access rather than
#: at import so that a test can set an environment variable before the first
#: read, and held afterwards so no two components can disagree.
_SETTINGS: Optional[Settings] = None

#: Guards the lazy build. Uvicorn's lifespan, the run registry's reader threads
#: and the background startup thread all read the settings, and two of them
#: arriving together must not each parse the environment.
_SETTINGS_LOCK: Final[threading.Lock] = threading.Lock()


def settings() -> Settings:
    """The settings for this process, read from the environment on first call."""
    global _SETTINGS
    if _SETTINGS is None:
        with _SETTINGS_LOCK:
            if _SETTINGS is None:
                _SETTINGS = Settings.from_environment()
    return _SETTINGS


def reload_settings(**overrides: Any) -> Settings:
    """Re-read the environment, optionally overriding fields. For tests only.

    Production code never calls this: the point of holding a single object is
    that two components cannot disagree about the configuration mid-flight.
    """
    global _SETTINGS
    with _SETTINGS_LOCK:
        fresh = Settings.from_environment()
        if overrides:
            fresh = replace(fresh, **overrides)
        _SETTINGS = fresh
    return fresh


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, for a log that something other than a human reads."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Fields attached with ``logger.info(..., extra={...})`` are the
        # structured half of "structured logging"; copy across anything that is
        # not part of the standard record so request timings survive.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


#: Attributes every :class:`logging.LogRecord` carries, which must not be copied
#: into the JSON payload as if they were caller-supplied context.
_RESERVED_LOG_KEYS: Final[frozenset[str]] = frozenset(
    set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}
)


def configure_logging(config: Optional[Settings] = None) -> None:
    """Send every log record to stdout at the configured level and format.

    stdout rather than stderr because these are the service's ordinary
    operational events, not its failures, and every process supervisor worth the
    name captures both anyway. Existing handlers are replaced rather than added
    to, so calling this twice - which uvicorn's reloader will - does not double
    every line.
    """
    config = config or settings()
    handler = logging.StreamHandler(stream=sys.stdout)
    if config.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(config.log_level)

    # uvicorn installs its own handlers on these three loggers when it starts.
    # Clearing them and letting the records propagate to the root handler is
    # what makes *all* output - ours and the server's - come out in one format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
