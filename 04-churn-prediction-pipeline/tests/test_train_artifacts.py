"""Tests for the end-to-end training entry point's artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def test_training_writes_all_artifacts(trained_artifacts: Path) -> None:
    """A training run produces model, metadata, metrics, and plots."""
    expected = [
        "model.joblib",
        "metadata.json",
        "metrics.json",
        "profit_curve.png",
        "shap_importance.png",
        "roc_curve.png",
        "shap_importance.csv",
    ]
    for name in expected:
        assert (trained_artifacts / name).exists(), f"missing artifact: {name}"


def test_metrics_json_contents(trained_artifacts: Path) -> None:
    """metrics.json contains CV comparison, holdout metrics, and threshold."""
    metrics = json.loads((trained_artifacts / "metrics.json").read_text())
    assert set(metrics["cv_comparison"]) == {
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }
    holdout = metrics["holdout"]
    for key in ("roc_auc", "pr_auc", "f1", "brier"):
        assert key in holdout
    assert 0.5 < holdout["roc_auc"] <= 1.0
    business = metrics["business_threshold"]
    assert 0.0 < business["threshold"] < 1.0
    assert business["offer_cost"] > 0
