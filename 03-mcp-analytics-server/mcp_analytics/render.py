"""Helpers for rendering query results as GitHub-flavored markdown."""

from __future__ import annotations

from typing import Any

__all__ = ["format_cell", "to_markdown_table"]


def format_cell(value: Any) -> str:
    """Format a single SQL value for display in a markdown cell.

    Args:
        value: Any SQLite-returned scalar.

    Returns:
        A pipe-safe string; floats keep at most 4 decimal places.
    """
    if value is None:
        return "NULL"
    text = f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Render columns and rows as a markdown table.

    Args:
        columns: Header names.
        rows: Row values.

    Returns:
        Markdown table text, or a note when the result set is empty.
    """
    if not columns:
        return "*(no columns returned)*"
    header = "| " + " | ".join(format_cell(c) for c in columns) + " |"
    divider = "|" + "|".join(" --- " for _ in columns) + "|"
    body = ["| " + " | ".join(format_cell(v) for v in row) + " |" for row in rows]
    if not body:
        return header + "\n" + divider + "\n\n*(0 rows)*"
    return "\n".join([header, divider, *body])
