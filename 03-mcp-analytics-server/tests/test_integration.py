"""Integration test: run the real demo client against a real stdio server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_demo_end_to_end_over_stdio(seeded_db_path: Path, tmp_path: Path) -> None:
    """Spawn the demo (which itself spawns the server) and check its output.

    This exercises the full MCP stack: stdio transport, initialize handshake,
    tool listing, resource read, and every tool call -- exactly what Claude
    Desktop does.
    """
    env = os.environ.copy()
    env["MCP_ANALYTICS_DB"] = str(seeded_db_path)
    env["MCP_ANALYTICS_CHART_DIR"] = str(tmp_path / "charts")
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_analytics.demo"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"demo failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    out = proc.stdout
    for marker in (
        "Available tools",
        "list_tables()",
        "schema://database",
        "revenue",
        "Only read-only SELECT statements",  # the DROP TABLE call must be rejected
        "Chart saved:",
        "Demo completed successfully.",
    ):
        assert marker in out, f"missing marker {marker!r} in demo output"
