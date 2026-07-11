"""Column-level descriptive statistics computed inside SQLite."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .db import Database

logger = logging.getLogger(__name__)

_NUMERIC_TYPES = {"integer", "real"}


class ColumnStats(BaseModel):
    """Descriptive statistics for one column.

    Attributes:
        table: Table name.
        column: Column name.
        total_rows: Total row count of the table.
        non_null: Count of non-NULL values.
        nulls: Count of NULL values.
        distinct: Count of distinct non-NULL values.
        mean: Mean (numeric columns only).
        std: Population standard deviation (numeric columns only).
        minimum: Minimum value.
        maximum: Maximum value.
        top_values: Up to five most frequent values as (value, count) pairs.
    """

    table: str
    column: str
    total_rows: int
    non_null: int
    nulls: int
    distinct: int
    mean: float | None = None
    std: float | None = None
    minimum: Any = None
    maximum: Any = None
    top_values: list[tuple[Any, int]] = []


def _resolve_column(db: Database, table: str, column: str) -> tuple[str, str]:
    """Validate table and column names against the live schema.

    Args:
        db: Database accessor.
        table: Candidate table name.
        column: Candidate column name.

    Returns:
        Tuple of canonical ``(table, column)`` names.

    Raises:
        ValueError: If the table or column does not exist.
    """
    canonical_table = db.require_table(table)
    columns = {c["name"].lower(): c["name"] for c in db.table_columns(canonical_table)}
    key = column.strip().lower()
    if key not in columns:
        raise ValueError(
            f"Unknown column {column!r} on table {canonical_table!r}. "
            f"Available: {', '.join(columns.values())}"
        )
    return canonical_table, columns[key]


def compute_column_stats(db: Database, table: str, column: str) -> ColumnStats:
    """Compute count/null/mean/std/min/max/top-value statistics for a column.

    Identifiers are validated against the schema before being interpolated,
    so this function is not injectable despite building SQL strings.

    Args:
        db: Database accessor.
        table: Table name.
        column: Column name.

    Returns:
        A populated :class:`ColumnStats`.
    """
    t, c = _resolve_column(db, table, column)
    conn = db.connect(trusted=True)
    try:
        total, non_null, distinct, minimum, maximum = conn.execute(
            f'SELECT COUNT(*), COUNT("{c}"), COUNT(DISTINCT "{c}"), '
            f'MIN("{c}"), MAX("{c}") FROM "{t}"'
        ).fetchone()

        numeric_kind = conn.execute(
            f'SELECT typeof("{c}") FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT 1'
        ).fetchone()
        is_numeric = bool(numeric_kind) and numeric_kind[0] in _NUMERIC_TYPES

        mean = std = None
        if is_numeric and non_null:
            mean_sq, mean = conn.execute(
                f'SELECT AVG("{c}" * "{c}" * 1.0), AVG("{c}" * 1.0) '
                f'FROM "{t}" WHERE "{c}" IS NOT NULL'
            ).fetchone()
            variance = max(mean_sq - mean * mean, 0.0)
            std = variance**0.5

        top = conn.execute(
            f'SELECT "{c}", COUNT(*) AS n FROM "{t}" WHERE "{c}" IS NOT NULL '
            f'GROUP BY "{c}" ORDER BY n DESC, "{c}" LIMIT 5'
        ).fetchall()
    finally:
        conn.close()

    return ColumnStats(
        table=t,
        column=c,
        total_rows=total,
        non_null=non_null,
        nulls=total - non_null,
        distinct=distinct,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        top_values=[(value, count) for value, count in top],
    )
