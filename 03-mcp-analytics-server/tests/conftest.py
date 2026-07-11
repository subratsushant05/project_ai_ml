"""Shared fixtures: every test runs against a temporary seeded database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mcp_analytics import config
from mcp_analytics.db import Database
from mcp_analytics.seed import seed_database


@pytest.fixture(scope="session")
def seeded_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seed one database for the whole test session (it is never mutated)."""
    path = tmp_path_factory.mktemp("data") / "ecommerce.db"
    seed_database(path)
    return path


@pytest.fixture(autouse=True)
def analytics_env(
    seeded_db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Point the server at the temp database and a temp chart directory."""
    monkeypatch.setenv(config.ENV_DB_PATH, str(seeded_db_path))
    monkeypatch.setenv(config.ENV_CHART_DIR, str(tmp_path / "charts"))
    monkeypatch.setenv(config.ENV_MAX_ROWS, "200")
    monkeypatch.setenv(config.ENV_TIMEOUT, "5.0")
    yield


@pytest.fixture()
def db() -> Database:
    """Database accessor bound to the environment set by ``analytics_env``."""
    return Database()
