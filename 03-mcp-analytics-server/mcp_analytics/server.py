"""MCP analytics server: safe SQL analytics tools over a bundled SQLite DB.

Run over stdio (the transport Claude Desktop and Cursor use):

    python -m mcp_analytics.server
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .db import get_database
from .plotting import render_chart
from .render import format_cell, to_markdown_table
from .stats import compute_column_stats

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "analytics",
    instructions=(
        "Read-only SQL analytics over a sample e-commerce SQLite database "
        "(customers, orders, order_items, products). Use list_tables and "
        "describe_table to explore, run_query for SELECT statements, "
        "table_stats for column summaries, and plot_data for charts."
    ),
)


@mcp.tool()
def list_tables() -> str:
    """List every table in the database with its column and row counts.

    Returns:
        A markdown table of table name, column count, and row count.
    """
    db = get_database()
    rows = [
        [name, len(db.table_columns(name)), db.row_count(name)]
        for name in db.table_names()
    ]
    return to_markdown_table(["table", "columns", "rows"], rows)


@mcp.tool()
def describe_table(table: str) -> str:
    """Show a table's columns, types, and constraints plus 5 sample rows.

    Args:
        table: Name of the table to describe (e.g. "orders").

    Returns:
        Markdown sections for the column definitions and sample rows.
    """
    db = get_database()
    canonical = db.require_table(table)
    columns = [
        [c["name"], c["type"], "yes" if c["notnull"] else "no", "yes" if c["pk"] else "no"]
        for c in db.table_columns(canonical)
    ]
    schema_md = to_markdown_table(["column", "type", "not null", "primary key"], columns)
    sample = db.run_select(f'SELECT * FROM "{canonical}" LIMIT 5')
    sample_md = to_markdown_table(sample.columns, sample.rows)
    return f"### `{canonical}` columns\n{schema_md}\n\n### Sample rows\n{sample_md}"


@mcp.tool()
def run_query(sql: str) -> str:
    """Execute a single read-only SELECT statement and return a markdown table.

    Mutating or administrative SQL (INSERT/UPDATE/DELETE/DROP/PRAGMA/ATTACH,
    multi-statement input, ...) is rejected; results are capped and the query
    runs under a wall-clock timeout on a read-only connection.

    Args:
        sql: One SELECT (or WITH ... SELECT) statement.

    Returns:
        The result set as a markdown table, with a truncation note if the
        row cap was hit.
    """
    db = get_database()
    result = db.run_select(sql)
    table = to_markdown_table(result.columns, result.rows)
    note = (
        f"\n\n*Truncated to the first {len(result.rows)} rows "
        f"(cap: {db.settings.max_rows}).*"
        if result.truncated
        else f"\n\n*{len(result.rows)} row(s).*"
    )
    return table + note


@mcp.tool()
def table_stats(table: str, column: str) -> str:
    """Summary statistics for one column: counts, nulls, mean/std, top values.

    Args:
        table: Table name (e.g. "products").
        column: Column name within that table (e.g. "price").

    Returns:
        A markdown summary. Mean and standard deviation are included only
        for numeric columns.
    """
    stats = compute_column_stats(get_database(), table, column)
    lines = [
        f"### Statistics for `{stats.table}.{stats.column}`",
        f"- rows: {stats.total_rows}  |  non-null: {stats.non_null}  "
        f"|  nulls: {stats.nulls}  |  distinct: {stats.distinct}",
        f"- min: {format_cell(stats.minimum)}  |  max: {format_cell(stats.maximum)}",
    ]
    if stats.mean is not None and stats.std is not None:
        lines.append(f"- mean: {stats.mean:.4f}  |  std (population): {stats.std:.4f}")
    top = to_markdown_table(
        ["value", "count"], [[value, count] for value, count in stats.top_values]
    )
    lines.append(f"\n**Top values**\n{top}")
    return "\n".join(lines)


@mcp.tool()
def plot_data(sql: str, chart_type: str, x: str, y: str) -> str:
    """Run a SELECT and render the result as a PNG chart on disk.

    Args:
        sql: Read-only SELECT producing the data (same guard as run_query).
        chart_type: One of "bar", "line", or "scatter".
        x: Result column to use for the x axis.
        y: Result column to use for the y axis (numeric).

    Returns:
        A message containing the absolute path of the rendered PNG.
    """
    db = get_database()
    result = db.run_select(sql)
    if x not in result.columns or y not in result.columns:
        raise ValueError(
            f"Columns {x!r} and {y!r} must both appear in the result set "
            f"(got: {', '.join(result.columns)})."
        )
    xi, yi = result.columns.index(x), result.columns.index(y)
    path = render_chart(
        x_values=[row[xi] for row in result.rows],
        y_values=[row[yi] for row in result.rows],
        chart_type=chart_type,
        x_label=x,
        y_label=y,
        out_dir=get_settings().chart_dir,
    )
    return f"Chart saved: {path} ({chart_type}, {len(result.rows)} points)"


@mcp.resource("schema://database")
def database_schema() -> str:
    """Full DDL of the sample database, exposed as an MCP resource.

    Returns:
        The CREATE TABLE / CREATE INDEX statements as SQL text.
    """
    return get_database().schema_ddl()


def main() -> None:
    """Seed the database if needed and serve MCP over stdio."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdout is reserved for the MCP protocol
    )
    db = get_database()
    db.ensure_exists()
    logger.info(
        "Starting mcp_analytics (db=%s, max_rows=%d, timeout=%.1fs, charts=%s)",
        db.settings.db_path,
        db.settings.max_rows,
        db.settings.query_timeout_seconds,
        db.settings.chart_dir,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

__all__ = [
    "mcp",
    "main",
    "list_tables",
    "describe_table",
    "run_query",
    "table_stats",
    "plot_data",
    "database_schema",
]
