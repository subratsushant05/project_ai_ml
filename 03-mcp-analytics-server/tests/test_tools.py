"""Unit tests calling the MCP tool functions directly (no transport)."""

from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

import pytest

from mcp_analytics import server
from mcp_analytics.guards import QueryValidationError


def test_list_tables_reports_all_tables_and_counts(seeded_db_path: Path) -> None:
    out = server.list_tables()
    for table in ("customers", "orders", "order_items", "products"):
        assert table in out
    conn = sqlite3.connect(seeded_db_path)
    try:
        n_customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    finally:
        conn.close()
    assert str(n_customers) in out


def test_describe_table_shows_columns_and_samples() -> None:
    out = server.describe_table("orders")
    for column in ("order_id", "customer_id", "order_date", "status", "total_amount"):
        assert column in out
    assert "Sample rows" in out


def test_describe_table_unknown_table_raises() -> None:
    with pytest.raises(ValueError, match="Unknown table"):
        server.describe_table("users")


def test_run_query_returns_markdown_table() -> None:
    out = server.run_query("SELECT category, COUNT(*) AS n FROM products GROUP BY category")
    assert out.startswith("| category | n |")
    assert "| --- |" in out
    assert "row(s)" in out


def test_run_query_blocks_mutations() -> None:
    with pytest.raises(QueryValidationError):
        server.run_query("UPDATE products SET price = 0")


def test_table_stats_matches_manual_computation(seeded_db_path: Path) -> None:
    conn = sqlite3.connect(seeded_db_path)
    try:
        prices = [r[0] for r in conn.execute("SELECT price FROM products")]
    finally:
        conn.close()
    out = server.table_stats("products", "price")
    assert f"mean: {statistics.fmean(prices):.4f}" in out
    assert f"std (population): {statistics.pstdev(prices):.4f}" in out
    assert f"non-null: {len(prices)}" in out
    assert "nulls: 0" in out


def test_table_stats_top_values_for_categorical_column() -> None:
    out = server.table_stats("orders", "status")
    assert "Top values" in out
    assert "delivered" in out
    assert "mean" not in out  # non-numeric column: no mean/std


def test_table_stats_unknown_column_raises() -> None:
    with pytest.raises(ValueError, match="Unknown column"):
        server.table_stats("products", "nope")


def test_plot_data_writes_png(tmp_path: Path) -> None:
    out = server.plot_data(
        sql="SELECT category, AVG(price) AS avg_price FROM products GROUP BY category",
        chart_type="bar",
        x="category",
        y="avg_price",
    )
    assert "Chart saved:" in out
    path = Path(out.split("Chart saved: ", 1)[1].split(" (", 1)[0])
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_plot_data_rejects_bad_chart_type() -> None:
    with pytest.raises(ValueError, match="chart_type"):
        server.plot_data(
            sql="SELECT category, COUNT(*) AS n FROM products GROUP BY category",
            chart_type="pie",
            x="category",
            y="n",
        )


def test_plot_data_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="must both appear"):
        server.plot_data(
            sql="SELECT category FROM products",
            chart_type="bar",
            x="category",
            y="revenue",
        )


def test_schema_resource_returns_ddl() -> None:
    ddl = server.database_schema()
    assert "CREATE TABLE customers" in ddl
    assert "CREATE TABLE order_items" in ddl
