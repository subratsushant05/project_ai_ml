"""Tests for the deterministic seed script."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mcp_analytics.seed import N_CUSTOMERS, N_ORDERS, N_PRODUCTS, seed_database


def _dump(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return "\n".join(conn.iterdump())
    finally:
        conn.close()


def test_seed_is_deterministic(tmp_path: Path) -> None:
    """Two independent seeds must produce byte-identical logical content."""
    a = seed_database(tmp_path / "a.db")
    b = seed_database(tmp_path / "b.db")
    assert _dump(a) == _dump(b)


def test_seed_row_counts(seeded_db_path: Path) -> None:
    conn = sqlite3.connect(seeded_db_path)
    try:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("customers", "products", "orders", "order_items")
        }
    finally:
        conn.close()
    assert counts["customers"] == N_CUSTOMERS
    assert counts["products"] == N_PRODUCTS
    assert counts["orders"] == N_ORDERS
    assert counts["order_items"] >= N_ORDERS  # every order has >= 1 item
    assert 400 <= sum(counts.values()) <= 700  # "~500 rows" overall


def test_referential_integrity(seeded_db_path: Path) -> None:
    """No orphaned foreign keys and order totals match their line items."""
    conn = sqlite3.connect(seeded_db_path)
    try:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM order_items oi "
            "LEFT JOIN orders o USING (order_id) "
            "LEFT JOIN products p USING (product_id) "
            "WHERE o.order_id IS NULL OR p.product_id IS NULL"
        ).fetchone()[0]
        mismatched = conn.execute(
            "SELECT COUNT(*) FROM orders o WHERE ABS(o.total_amount - ("
            "SELECT SUM(quantity * unit_price) FROM order_items "
            "WHERE order_id = o.order_id)) > 0.01"
        ).fetchone()[0]
    finally:
        conn.close()
    assert orphans == 0
    assert mismatched == 0


def test_seed_refuses_overwrite_without_force(tmp_path: Path) -> None:
    path = seed_database(tmp_path / "x.db")
    with pytest.raises(FileExistsError):
        seed_database(path)
    assert seed_database(path, force=True) == path
