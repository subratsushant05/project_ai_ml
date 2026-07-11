"""Read-only SQLite access layer with layered safety controls.

Every user-supplied statement passes through three independent defenses:

1. :func:`mcp_analytics.guards.validate_select` -- static statement validation.
2. A SQLite *authorizer callback* that denies every action except reads.
3. A connection opened with ``file:...?mode=ro`` so the OS-level handle
   physically cannot write.

A progress-handler based wall-clock timeout and a row cap bound resource use.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from pydantic import BaseModel

from .config import Settings, get_settings
from .guards import validate_select
from .seed import seed_database

logger = logging.getLogger(__name__)

_PROGRESS_HANDLER_OPCODES = 5_000

_READ_ACTIONS: frozenset[int] = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)


class QueryTimeoutError(RuntimeError):
    """Raised when a statement exceeds the configured wall-clock budget."""


class QueryResult(BaseModel):
    """Result of a guarded SELECT.

    Attributes:
        columns: Column names in result order.
        rows: Row tuples (as lists after validation).
        truncated: True when the row cap cut off additional rows.
    """

    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False


def _authorizer(action: int, *_args: Any) -> int:
    """SQLite authorizer callback: permit read actions only."""
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


class Database:
    """Thin wrapper over a read-only SQLite database.

    Args:
        settings: Resolved server settings; defaults to environment-derived.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ensure_exists(self) -> None:
        """Seed the bundled sample database if the file is missing."""
        if not self.settings.db_path.exists():
            logger.info("Database missing; seeding sample data at %s", self.settings.db_path)
            seed_database(self.settings.db_path)

    def connect(self, trusted: bool = False) -> sqlite3.Connection:
        """Open a read-only connection.

        Args:
            trusted: When False (user SQL), the deny-by-default authorizer is
                installed. Internal schema introspection uses ``trusted=True``
                (still ``mode=ro``) because PRAGMA calls would otherwise be
                denied by our own authorizer.

        Returns:
            An open ``sqlite3.Connection``.
        """
        self.ensure_exists()
        uri = f"file:{self.settings.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        if not trusted:
            if hasattr(conn, "setconfig"):  # Python >= 3.12: harden further
                conn.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
            conn.set_authorizer(_authorizer)
        return conn

    def run_select(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Validate and execute a single SELECT statement.

        Args:
            sql: Raw SQL text from the client.
            max_rows: Optional row cap override (never above the configured cap).

        Returns:
            A :class:`QueryResult` with columns, rows, and a truncation flag.

        Raises:
            mcp_analytics.guards.QueryValidationError: If validation fails.
            QueryTimeoutError: If execution exceeds the time budget.
            sqlite3.Error: For SQL errors surfaced by SQLite itself.
        """
        statement = validate_select(sql)
        cap = min(max_rows or self.settings.max_rows, self.settings.max_rows)
        deadline = time.monotonic() + self.settings.query_timeout_seconds

        conn = self.connect(trusted=False)
        try:
            conn.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0,
                _PROGRESS_HANDLER_OPCODES,
            )
            try:
                cursor = conn.execute(statement)
                columns = [d[0] for d in cursor.description or []]
                fetched = cursor.fetchmany(cap + 1)
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc).lower():
                    raise QueryTimeoutError(
                        f"Query exceeded the {self.settings.query_timeout_seconds:.1f}s "
                        "time budget and was aborted."
                    ) from exc
                raise
        finally:
            conn.close()

        truncated = len(fetched) > cap
        rows = [list(row) for row in fetched[:cap]]
        logger.debug("Query returned %d row(s) (truncated=%s)", len(rows), truncated)
        return QueryResult(columns=columns, rows=rows, truncated=truncated)

    def table_names(self) -> list[str]:
        """Return user table names in the database, alphabetically."""
        conn = self.connect(trusted=True)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def require_table(self, table: str) -> str:
        """Validate a table name against the actual schema.

        Args:
            table: Candidate table name.

        Returns:
            The canonical table name.

        Raises:
            ValueError: If the table does not exist (prevents identifier
                injection in tools that interpolate table names).
        """
        names = self.table_names()
        for name in names:
            if name.lower() == table.strip().lower():
                return name
        raise ValueError(f"Unknown table {table!r}. Available: {', '.join(names)}")

    def table_columns(self, table: str) -> list[dict[str, Any]]:
        """Describe columns of a table via PRAGMA table_info.

        Args:
            table: Table name (validated against the schema first).

        Returns:
            One dict per column: name, type, notnull, default, pk.
        """
        canonical = self.require_table(table)
        conn = self.connect(trusted=True)
        try:
            rows = conn.execute(f'PRAGMA table_info("{canonical}")').fetchall()
        finally:
            conn.close()
        return [
            {"name": r[1], "type": r[2], "notnull": bool(r[3]), "default": r[4], "pk": bool(r[5])}
            for r in rows
        ]

    def row_count(self, table: str) -> int:
        """Return COUNT(*) for a validated table name."""
        canonical = self.require_table(table)
        conn = self.connect(trusted=True)
        try:
            return int(conn.execute(f'SELECT COUNT(*) FROM "{canonical}"').fetchone()[0])
        finally:
            conn.close()

    def schema_ddl(self) -> str:
        """Return the CREATE statements for all user tables and indexes."""
        conn = self.connect(trusted=True)
        try:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rootpage"
            ).fetchall()
        finally:
            conn.close()
        return ";\n\n".join(r[0] for r in rows) + ";"


def get_database() -> Database:
    """Construct a :class:`Database` from current environment settings."""
    return Database(get_settings())
