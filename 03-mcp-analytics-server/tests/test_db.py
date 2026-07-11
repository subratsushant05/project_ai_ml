"""Tests for the read-only database layer: caps, timeout, authorizer."""

from __future__ import annotations

import sqlite3

import pytest

from mcp_analytics.config import ENV_MAX_ROWS, ENV_TIMEOUT
from mcp_analytics.db import Database, QueryTimeoutError
from mcp_analytics.guards import QueryValidationError


def test_run_select_basic(db: Database) -> None:
    result = db.run_select("SELECT customer_id, name FROM customers ORDER BY customer_id")
    assert result.columns == ["customer_id", "name"]
    assert result.rows[0][0] == 1
    assert not result.truncated


def test_row_cap_enforced(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_ROWS, "10")
    capped = Database()
    result = capped.run_select("SELECT * FROM order_items")
    assert len(result.rows) == 10
    assert result.truncated


def test_row_cap_cannot_be_raised_above_config(db: Database) -> None:
    result = db.run_select("SELECT * FROM order_items", max_rows=999_999)
    assert len(result.rows) <= db.settings.max_rows


def test_mutation_rejected_by_validator(db: Database) -> None:
    with pytest.raises(QueryValidationError):
        db.run_select("DELETE FROM orders")


def test_authorizer_blocks_writes_even_without_validator(db: Database) -> None:
    """Defense in depth: raw connection still refuses non-read actions."""
    conn = db.connect(trusted=False)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DROP TABLE customers")
    finally:
        conn.close()


def test_readonly_mode_blocks_writes_on_trusted_connection(db: Database) -> None:
    """Even the trusted (no-authorizer) connection is opened mode=ro."""
    conn = db.connect(trusted=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            conn.execute("INSERT INTO customers VALUES (9999,'x','q@x.com','USA','2024-01-01')")
    finally:
        conn.close()


def test_query_timeout_aborts_runaway_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TIMEOUT, "0.2")
    db = Database()
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT COUNT(*) FROM c"
    )
    with pytest.raises(QueryTimeoutError):
        db.run_select(runaway)


def test_unknown_table_rejected(db: Database) -> None:
    with pytest.raises(ValueError, match="Unknown table"):
        db.require_table("no_such_table; DROP TABLE customers")


def test_schema_ddl_lists_all_tables(db: Database) -> None:
    ddl = db.schema_ddl()
    for table in ("customers", "products", "orders", "order_items"):
        assert f"CREATE TABLE {table}" in ddl
