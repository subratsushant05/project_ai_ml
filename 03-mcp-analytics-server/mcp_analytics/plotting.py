"""Chart rendering for query results using matplotlib's Agg backend."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

CHART_TYPES = ("bar", "line", "scatter")


def render_chart(
    x_values: list[Any],
    y_values: list[Any],
    chart_type: str,
    x_label: str,
    y_label: str,
    out_dir: Path,
    title: str | None = None,
) -> Path:
    """Render a chart to a PNG file.

    Args:
        x_values: Values for the x axis (categorical or numeric).
        y_values: Numeric values for the y axis.
        chart_type: One of ``bar``, ``line``, or ``scatter``.
        x_label: Label for the x axis.
        y_label: Label for the y axis.
        out_dir: Directory to write the PNG into (created if missing).
        title: Optional chart title; defaults to "y_label by x_label".

    Returns:
        Path of the written PNG file.

    Raises:
        ValueError: If ``chart_type`` is unsupported or the series is empty.
    """
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported chart_type {chart_type!r}; use one of {CHART_TYPES}.")
    if not x_values or not y_values:
        raise ValueError("Query returned no rows; nothing to plot.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chart_{chart_type}_{uuid.uuid4().hex[:10]}.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    try:
        if chart_type == "bar":
            positions = range(len(x_values))
            ax.bar(positions, y_values, color="#4C72B0")
            ax.set_xticks(list(positions))
            ax.set_xticklabels([str(v) for v in x_values], rotation=30, ha="right")
        elif chart_type == "line":
            ax.plot(x_values, y_values, marker="o", color="#4C72B0")
        else:  # scatter
            ax.scatter(x_values, y_values, color="#4C72B0", alpha=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title or f"{y_label} by {x_label}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
    finally:
        plt.close(fig)

    logger.info("Rendered %s chart with %d point(s) -> %s", chart_type, len(x_values), out_path)
    return out_path
