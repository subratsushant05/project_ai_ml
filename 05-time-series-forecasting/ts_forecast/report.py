"""Reporting: matplotlib figures, metrics.json and a plotly HTML report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

logger = logging.getLogger(__name__)

_HISTORY_TAIL_DAYS = 180


def plot_forecast(
    history: pd.Series,
    intervals: pd.DataFrame,
    path: Path,
    title: str,
    anomalies: pd.DataFrame | None = None,
) -> None:
    """Save a forecast plot with conformal interval band.

    Args:
        history: Full historical series (only the tail is drawn).
        intervals: DataFrame with ``forecast``, ``lower``, ``upper`` columns.
        path: Output PNG path.
        title: Plot title.
        anomalies: Optional output of :func:`ts_forecast.anomaly.flag_anomalies`
            used to mark flagged history points.
    """
    tail = history.iloc[-_HISTORY_TAIL_DAYS:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tail.index, tail.values, color="#30507a", lw=1.2, label="history")
    ax.plot(
        intervals.index, intervals["forecast"], color="#c0392b", lw=1.6, label="forecast"
    )
    ax.fill_between(
        intervals.index,
        intervals["lower"],
        intervals["upper"],
        color="#c0392b",
        alpha=0.18,
        label="conformal interval",
    )
    if anomalies is not None:
        flagged = anomalies.loc[anomalies["is_anomaly"]].loc[tail.index[0] :]
        if not flagged.empty:
            ax.scatter(
                flagged.index, flagged["value"], color="#e67e22", s=28,
                zorder=5, label="anomaly",
            )
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_backtest_errors(metrics: pd.DataFrame, path: Path, title: str) -> None:
    """Save a per-fold MASE comparison plot across models.

    Args:
        metrics: Backtest metrics with ``fold``, ``model``, ``mase`` columns.
        path: Output PNG path.
        title: Plot title.
    """
    pivot = metrics.pivot(index="fold", columns="model", values="mase")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(1.0, color="grey", lw=1, ls="--", label="naive baseline (MASE=1)")
    ax.set_xlabel("backtest fold")
    ax.set_ylabel("MASE")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", path)


def write_metrics_json(payload: dict[str, Any], path: Path) -> None:
    """Serialize the metrics payload to JSON.

    Args:
        payload: JSON-serializable results dictionary.
        path: Output path for ``metrics.json``.
    """
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s", path)


def build_html_report(
    history: pd.Series,
    intervals: pd.DataFrame,
    metrics: pd.DataFrame,
    winner: str,
    coverage: float,
    path: Path,
) -> None:
    """Write a self-contained interactive HTML report via plotly.

    Args:
        history: Historical series.
        intervals: Forecast with lower/upper bounds.
        metrics: Per-fold backtest metrics.
        winner: Name of the selected model.
        coverage: Held-out empirical interval coverage in [0, 1].
        path: Output HTML path.
    """
    tail = history.iloc[-_HISTORY_TAIL_DAYS:]
    fig = go.Figure()
    fig.add_scatter(x=tail.index, y=tail.values, name="history", line={"color": "#30507a"})
    fig.add_scatter(
        x=intervals.index, y=intervals["upper"], name="upper", line={"width": 0},
        showlegend=False,
    )
    fig.add_scatter(
        x=intervals.index, y=intervals["lower"], name="interval", fill="tonexty",
        line={"width": 0}, fillcolor="rgba(192,57,43,0.18)",
    )
    fig.add_scatter(
        x=intervals.index, y=intervals["forecast"], name=f"forecast ({winner})",
        line={"color": "#c0392b"},
    )
    fig.update_layout(
        title=(
            f"{history.name}: {len(intervals)}-day forecast — winner: {winner}, "
            f"held-out interval coverage {coverage:.0%}"
        ),
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    table = (
        metrics.groupby("model")[["mae", "rmse", "smape", "mase"]]
        .mean()
        .sort_values("mase")
        .round(3)
    )
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    table_html = (
        "<div style='max-width:760px;margin:1em auto;font-family:sans-serif'>"
        "<h3>Backtest metrics (mean across folds)</h3>"
        + table.to_html(border=0)
        + "</div></body>"
    )
    path.write_text(html.replace("</body>", table_html))
    logger.info("Wrote %s", path)
