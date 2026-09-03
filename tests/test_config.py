"""Tests for :mod:`qroute.config`.

These are deliberately cheap: no server, no data loading, no solver. What they
protect is the property that everything else in the platform assumes and that
nothing else tests - that the service resolves its data, its limits and its CORS
policy from an explicit, typed configuration rather than from whichever
directory it happened to be started in.

The working-directory tests are the important ones. Before this module existed
every data path was ``Path("data")``, which silently resolved to nothing when
the server was started from anywhere but the repository root, and the failure
looked like an empty catalogue rather than an error.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from qroute.config import (
    INSTALL_ROOT,
    ConfigError,
    Settings,
    configure_logging,
    reload_settings,
    settings,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Run with every qroute variable unset, and restore the cached settings.

    The settings object is process-wide and cached on purpose, so a test that
    changes it has to put it back or the tests that follow inherit the change.
    """
    for name in (
        "QROUTE_HOST", "QROUTE_PORT", "QROUTE_DATA", "QROUTE_RESULTS",
        "QROUTE_FRONTEND", "QROUTE_LOG_LEVEL", "QROUTE_LOG_FORMAT",
        "QROUTE_REQUEST_LOG", "QROUTE_CORS_ORIGINS", "QROUTE_CORS_ORIGIN_REGEX",
        "QROUTE_MAX_ACTIVE_RUNS", "QROUTE_MAX_RUN_SECONDS", "QROUTE_API_PRELOAD",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    # The environment has to go back before the settings are re-read, and
    # monkeypatch's own teardown runs *after* this fixture's, so undo it here.
    monkeypatch.undo()
    reload_settings()


# --------------------------------------------------------------------------
# Path anchoring
# --------------------------------------------------------------------------


def test_data_root_is_absolute_and_independent_of_the_working_directory(
    clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point: chdir somewhere unrelated and still find the data."""
    from_repo = Settings.from_environment()
    monkeypatch.chdir(tmp_path)
    from_elsewhere = Settings.from_environment()

    assert from_repo.data_root.is_absolute()
    assert from_repo.data_root == from_elsewhere.data_root
    assert from_repo.results_root == from_elsewhere.results_root
    assert from_repo.frontend_dist == from_elsewhere.frontend_dist


def test_data_root_defaults_beside_the_installed_package(clean_env) -> None:
    config = Settings.from_environment()
    assert config.data_root == INSTALL_ROOT / "data"
    assert config.benchmarks_dir == config.data_root / "benchmarks"
    assert config.osm_dir == config.data_root / "osm"


def test_explicit_data_root_relocates_results_and_frontend_together(
    clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pointing QROUTE_DATA at a deployment moves the whole set coherently."""
    monkeypatch.setenv("QROUTE_DATA", str(tmp_path / "srv" / "data"))
    config = Settings.from_environment()
    assert config.data_root == tmp_path / "srv" / "data"
    assert config.results_root == tmp_path / "srv" / "results" / "runs"
    assert config.frontend_dist == tmp_path / "srv" / "frontend" / "dist"


def test_relative_paths_on_the_command_line_are_resolved(
    clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An operator typing a relative path means "relative to where I am"."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QROUTE_RESULTS", "out/runs")
    assert Settings.from_environment().results_root == (tmp_path / "out" / "runs").resolve()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_missing_data_is_an_actionable_error(clean_env, tmp_path: Path) -> None:
    config = replace(Settings.from_environment(), data_root=tmp_path / "absent")
    with pytest.raises(ConfigError) as excinfo:
        config.require_data()
    message = str(excinfo.value)
    assert "QROUTE_DATA" in message
    assert "benchmarks/" in message


def test_the_environment_bridge_does_not_change_the_missing_data_message(
    clean_env, tmp_path: Path
) -> None:
    """An operator who never set QROUTE_DATA must not be told QROUTE_DATA is wrong.

    ``qroute.api`` calls :meth:`Settings.export_to_environment` at import, which
    writes the resolved root into ``QROUTE_DATA``. Deciding which of the two
    messages to show by reading that variable afterwards therefore always chose
    the "you pointed it at the wrong place" one, and hid the list of directories
    that were actually searched - which is the whole diagnosis when nothing was
    found. The provenance is recorded when the settings are built instead.
    """
    config = replace(Settings.from_environment(), data_root=tmp_path / "absent")
    config.export_to_environment()
    assert os.environ["QROUTE_DATA"] == str(tmp_path / "absent")
    with pytest.raises(ConfigError) as excinfo:
        config.require_data()
    message = str(excinfo.value)
    assert "Searched:" in message, message
    assert "points at" not in message, message


def test_present_data_passes_validation(clean_env) -> None:
    """The repository's own data directory is a valid root."""
    Settings.from_environment().require_data()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("QROUTE_PORT", "not-a-number"),
        ("QROUTE_PORT", "99999"),
        ("QROUTE_LOG_LEVEL", "chatty"),
        ("QROUTE_LOG_FORMAT", "xml"),
        ("QROUTE_REQUEST_LOG", "maybe"),
        ("QROUTE_MAX_ACTIVE_RUNS", "0"),
        ("QROUTE_MAX_RUN_SECONDS", "eventually"),
        # Starlette compiles this pattern lazily, on the first request that
        # carries an Origin header, so without a check here an unbalanced
        # bracket becomes a re.PatternError traceback during a demonstration
        # rather than a refusal to start.
        ("QROUTE_CORS_ORIGIN_REGEX", "[unclosed"),
    ],
)
def test_a_malformed_setting_stops_the_process(
    clean_env, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Never fall back to a default: the operator meant something by that value."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_environment()
    assert name in str(excinfo.value)


def test_a_blank_setting_is_treated_as_unset(
    clean_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QROUTE_HOST", "   ")
    assert Settings.from_environment().host == Settings.host


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def test_cors_is_same_origin_by_default(clean_env) -> None:
    config = Settings.from_environment()
    assert config.cors_allow_origins == ()
    assert config.cors_allow_origin_regex is None
    assert config.cors_enabled is False


def test_cors_widens_only_when_configured(
    clean_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QROUTE_CORS_ORIGINS", "http://localhost:5173, https://demo.example")
    config = Settings.from_environment()
    assert config.cors_allow_origins == ("http://localhost:5173", "https://demo.example")
    assert config.cors_enabled is True


def test_run_budget_is_clamped_and_the_reduction_is_visible(clean_env) -> None:
    config = replace(Settings.from_environment(), max_run_seconds=30.0)
    assert config.clamp_run_seconds(10.0) == (10.0, False)
    assert config.clamp_run_seconds(30.0) == (30.0, False)
    assert config.clamp_run_seconds(600.0) == (30.0, True)


# --------------------------------------------------------------------------
# Process-wide access and logging
# --------------------------------------------------------------------------


def test_settings_are_read_once(clean_env) -> None:
    assert settings() is settings()


def test_reload_settings_applies_overrides(clean_env) -> None:
    assert reload_settings(max_run_seconds=7.0).max_run_seconds == 7.0
    assert settings().max_run_seconds == 7.0


def test_configure_logging_installs_exactly_one_stdout_handler(clean_env) -> None:
    """Called twice - as uvicorn's reloader does - it must not double every line."""
    import json
    import logging
    import sys

    root = logging.getLogger()
    # Root-logger state is global and pytest keeps handlers of its own on it, so
    # it is snapshotted and put back rather than reset to a guessed default.
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        config = replace(Settings.from_environment(), log_level="DEBUG", log_format="json")
        configure_logging(config)
        configure_logging(config)

        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert handler.stream is sys.stdout
        assert root.level == logging.DEBUG

        record = logging.LogRecord("t", logging.INFO, "f", 1, "hello %s", ("world",), None)
        record.request_id = "abc123"
        payload = json.loads(handler.formatter.format(record))
        assert payload["message"] == "hello world"
        assert payload["request_id"] == "abc123"
        # Standard record attributes must not leak in as if they were context.
        assert "msecs" not in payload
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
