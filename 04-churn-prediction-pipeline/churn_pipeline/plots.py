"""Matplotlib plots written as training artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from churn_pipeline.threshold import ThresholdResult

logger = logging.getLogger(__name__)


def plot_profit_curve(result: ThresholdResult, path: Path) -> None:
    """Plot expected campaign profit versus decision threshold.

    Args:
        result: Output of the threshold sweep.
        path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.thresholds, result.profits, color="#1f77b4", lw=2)
    ax.axvline(
        result.best_threshold,
        color="#d62728",
        ls="--",
        label=f"profit-optimal t={result.best_threshold:.2f} (${result.best_profit:,.0f})",
    )
    ax.axvline(0.5, color="grey", ls=":", label="default t=0.50")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Decision threshold (churn probability)")
    ax.set_ylabel("Expected campaign profit (USD)")
    ax.set_title("Retention campaign profit vs. decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_shap_importance(importance: pd.DataFrame, path: Path) -> None:
    """Plot global SHAP feature importance as a horizontal bar chart.

    Args:
        importance: Frame with ``feature`` and ``mean_abs_shap`` columns.
        path: Destination PNG path.
    """
    data = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance (TreeSHAP)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path
) -> None:
    """Plot holdout ROC curves for one or more models.

    Args:
        curves: Mapping of label to (y_true, y_prob, auc).
        path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (y_true, y_prob, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Holdout ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)
