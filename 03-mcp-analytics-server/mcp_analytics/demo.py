"""End-to-end demo: spawn the server over stdio and exercise every tool.

Usage:
    python -m mcp_analytics.demo

This uses the official ``mcp`` client library exactly the way Claude Desktop
does (stdio transport), which proves the server works with real MCP clients.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

_RULE = "-" * 72


def _text_of(result: Any) -> str:
    """Extract concatenated text content from a tool call result."""
    parts = [c.text for c in result.content if getattr(c, "text", None)]
    return "\n".join(parts) if parts else "(no text content)"


def _show(title: str, body: str, max_lines: int = 14) -> None:
    """Print a titled, truncated block of tool output."""
    print(f"\n{_RULE}\n>> {title}\n{_RULE}")
    lines = body.splitlines()
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")


async def run_demo() -> int:
    """Connect to the server over stdio and call each tool once.

    Returns:
        Process exit code (0 on success).
    """
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_analytics.server"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        tool_lines = [
            f"- {t.name}: {(t.description or '').strip().splitlines()[0]}"
            for t in tools.tools
        ]
        _show("Available tools", "\n".join(tool_lines))

        resources = await session.list_resources()
        _show(
            "Available resources",
            "\n".join(f"- {r.uri}" for r in resources.resources),
        )

        schema = await session.read_resource(AnyUrl("schema://database"))
        _show("Resource: schema://database", schema.contents[0].text)  # type: ignore[union-attr]

        result = await session.call_tool("list_tables", {})
        _show("list_tables()", _text_of(result))

        result = await session.call_tool("describe_table", {"table": "orders"})
        _show('describe_table("orders")', _text_of(result))

        result = await session.call_tool(
            "run_query",
            {
                "sql": (
                    "SELECT c.country, COUNT(o.order_id) AS orders, "
                    "ROUND(SUM(o.total_amount), 2) AS revenue "
                    "FROM orders o JOIN customers c USING (customer_id) "
                    "GROUP BY c.country ORDER BY revenue DESC"
                )
            },
        )
        _show("run_query(revenue by country)", _text_of(result))

        result = await session.call_tool(
            "run_query", {"sql": "DROP TABLE customers"}
        )
        _show("run_query(DROP TABLE ...) -- expected to be BLOCKED", _text_of(result))

        result = await session.call_tool(
            "table_stats", {"table": "products", "column": "price"}
        )
        _show('table_stats("products", "price")', _text_of(result))

        result = await session.call_tool(
            "plot_data",
            {
                "sql": (
                    "SELECT category, ROUND(AVG(price), 2) AS avg_price "
                    "FROM products GROUP BY category ORDER BY avg_price DESC"
                ),
                "chart_type": "bar",
                "x": "category",
                "y": "avg_price",
            },
        )
        _show("plot_data(avg price by category)", _text_of(result))

    print(f"\n{_RULE}\nDemo completed successfully.\n{_RULE}")
    return 0


def main() -> int:
    """Synchronous wrapper for the async demo."""
    return asyncio.run(run_demo())


if __name__ == "__main__":
    raise SystemExit(main())
