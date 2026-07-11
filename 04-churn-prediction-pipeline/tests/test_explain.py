"""Tests for SHAP-based explanations."""

from __future__ import annotations

import numpy as np
import pytest

from churn_pipeline.explain import compute_shap_values, explain_customer, global_importance
from churn_pipeline.features import get_feature_names


def test_shap_values_sum_to_prediction(fitted_pipeline, split) -> None:
    """TreeSHAP contributions plus base value equal the raw model output."""
    _, X_test, _, _ = split
    sample = X_test.iloc[:20]
    values, base, names = compute_shap_values(fitted_pipeline, sample)
    assert values.shape == (20, len(names))

    raw = fitted_pipeline.named_steps["model"].booster_.predict(
        np.asarray(
            fitted_pipeline.named_steps["preprocess"].transform(sample), dtype=float
        ),
        raw_score=True,
    )
    np.testing.assert_allclose(values.sum(axis=1) + base, raw, atol=1e-6)


def test_global_importance_ranks_known_drivers(fitted_pipeline, split) -> None:
    """Importance frame is sorted and surfaces the designed churn drivers."""
    _, X_test, _, _ = split
    importance = global_importance(fitted_pipeline, X_test.iloc[:150], top_n=10)
    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    top_features = " ".join(importance["feature"].head(6))
    assert "tenure" in top_features or "contract" in top_features


def test_explain_customer_text_output(fitted_pipeline, split) -> None:
    """Per-customer explanation exposes probability and top contributors."""
    _, X_test, _, _ = split
    explanation = explain_customer(fitted_pipeline, X_test.iloc[[3]], top_k=3)
    assert 0.0 <= explanation.churn_probability <= 1.0
    assert 1 <= len(explanation.top_positive) <= 3
    assert 1 <= len(explanation.top_negative) <= 3
    assert all(v > 0 for _, v in explanation.top_positive)
    assert all(v < 0 for _, v in explanation.top_negative)

    text = explanation.to_text()
    assert "Churn probability" in text
    assert "increasing churn risk" in text
    known_names = set(get_feature_names(fitted_pipeline.named_steps["preprocess"]))
    assert explanation.top_positive[0][0] in known_names


def test_explain_customer_rejects_multiple_rows(fitted_pipeline, split) -> None:
    """Passing more than one row raises a clear error."""
    _, X_test, _, _ = split
    with pytest.raises(ValueError, match="exactly one row"):
        explain_customer(fitted_pipeline, X_test.iloc[:2])
