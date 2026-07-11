"""Runtime configuration for the analytics server.

All settings can be overridden through environment variables so the server
behaves identically when launched by Claude Desktop, Docker, or pytest.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

ENV_DB_PATH = "MCP_ANALYTICS_DB"
ENV_CHART_DIR = "MCP_ANALYTICS_CHART_DIR"
ENV_MAX_ROWS = "MCP_ANALYTICS_MAX_ROWS"
ENV_TIMEOUT = "MCP_ANALYTICS_TIMEOUT_SECONDS"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    """Validated server settings.

    Attributes:
        db_path: Location of the SQLite database file.
        chart_dir: Directory where rendered PNG charts are written.
        max_rows: Hard cap on rows returned by any query tool.
        query_timeout_seconds: Wall-clock budget for a single SQL statement.
    """

    db_path: Path = Field(default=_PACKAGE_ROOT / "data" / "ecommerce.db")
    chart_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "mcp_analytics_charts"
    )
    max_rows: int = Field(default=200, ge=1, le=10_000)
    query_timeout_seconds: float = Field(default=5.0, gt=0.0, le=120.0)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults.

        Returns:
            A validated :class:`Settings` instance.
        """
        overrides: dict[str, object] = {}
        if db_path := os.environ.get(ENV_DB_PATH):
            overrides["db_path"] = Path(db_path)
        if chart_dir := os.environ.get(ENV_CHART_DIR):
            overrides["chart_dir"] = Path(chart_dir)
        if max_rows := os.environ.get(ENV_MAX_ROWS):
            overrides["max_rows"] = int(max_rows)
        if timeout := os.environ.get(ENV_TIMEOUT):
            overrides["query_timeout_seconds"] = float(timeout)
        return cls(**overrides)


def get_settings() -> Settings:
    """Read settings fresh from the environment.

    Intentionally uncached so tests can repoint ``MCP_ANALYTICS_DB`` at a
    temporary database between cases.

    Returns:
        Current :class:`Settings`.
    """
    return Settings.from_env()
