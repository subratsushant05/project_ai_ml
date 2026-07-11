"""Model evaluation: stratified cross-validation and holdout metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

METRIC_NAMES = ["roc_auc", "pr_auc", "f1", "brier"]


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the standard metric set from probabilities.

    Args:
        y_true: Binary labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Cut-off used for the F1 score.

    Returns:
        Mapping of metric name to value (ROC-AUC, PR-AUC, F1, Brier).
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a pipeline with stratified k-fold cross-validation.

    Args:
        pipeline: Unfitted model pipeline.
        X: Feature frame.
        y: Binary target.
        n_folds: Number of stratified folds.
        seed: Shuffle seed for fold assignment.

    Returns:
        Mean and standard deviation of each metric across folds, keyed as
        ``<metric>`` (mean) and ``<metric>_std``.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}

    for train_idx, val_idx in skf.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y[train_idx])
        y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        for name, value in compute_metrics(y[val_idx], y_prob).items():
            fold_scores[name].append(value)

    result: dict[str, float] = {}
    for name, values in fold_scores.items():
        result[name] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
    return result


def compare_models(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Cross-validate several pipelines and rank them by ROC-AUC.

    Args:
        pipelines: Mapping of model name to unfitted pipeline.
        X: Feature frame.
        y: Binary target.
        n_folds: Number of stratified folds.
        seed: Shuffle seed for fold assignment.

    Returns:
        DataFrame indexed by model name, sorted by mean ROC-AUC descending.
    """
    rows: dict[str, dict[str, Any]] = {}
    for name, pipeline in pipelines.items():
        logger.info("Cross-validating %s (%d folds)...", name, n_folds)
        rows[name] = cross_validate_model(pipeline, X, y, n_folds=n_folds, seed=seed)
        logger.info(
            "%s | ROC-AUC %.4f ± %.4f | PR-AUC %.4f | F1 %.4f | Brier %.4f",
            name,
            rows[name]["roc_auc"],
            rows[name]["roc_auc_std"],
            rows[name]["pr_auc"],
            rows[name]["f1"],
            rows[name]["brier"],
        )
    return pd.DataFrame(rows).T.sort_values("roc_auc", ascending=False)
