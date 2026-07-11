"""Explainability: SHAP values for global and per-customer explanations.

For LightGBM models, SHAP values are computed with the booster's built-in
TreeSHAP implementation (``predict(..., pred_contrib=True)``), which is exact
and adds no heavy dependencies. If the fitted estimator is not a LightGBM
model, the ``shap`` package is used as a fallback when installed.

All contributions are in log-odds space: positive values push the customer
toward churn, negative values away from it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from churn_pipeline.features import get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class CustomerExplanation:
    """Per-customer churn explanation.

    Attributes:
        churn_probability: Predicted probability of churn.
        base_value: Model expected value in log-odds space.
        top_positive: Largest churn-increasing (feature, shap) pairs.
        top_negative: Largest churn-decreasing (feature, shap) pairs.
    """

    churn_probability: float
    base_value: float
    top_positive: list[tuple[str, float]]
    top_negative: list[tuple[str, float]]

    def to_text(self) -> str:
        """Render the explanation as a short human-readable report."""
        lines = [f"Churn probability: {self.churn_probability:.1%}"]
        lines.append("Top factors increasing churn risk:")
        lines += [f"  + {name} (shap={value:+.3f})" for name, value in self.top_positive]
        lines.append("Top factors decreasing churn risk:")
        lines += [f"  - {name} (shap={value:+.3f})" for name, value in self.top_negative]
        return "\n".join(lines)


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[np.ndarray, float, list[str]]:
    """Compute SHAP values for each row of ``X``.

    Args:
        pipeline: Fitted pipeline with ``preprocess`` and ``model`` steps.
        X: Raw feature frame.

    Returns:
        Tuple of (shap value matrix [n_rows, n_features], base value,
        transformed feature names).

    Raises:
        TypeError: If the model is not LightGBM and ``shap`` is unavailable.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    X_t = np.asarray(preprocessor.transform(X), dtype=float)
    names = get_feature_names(preprocessor)

    if isinstance(model, LGBMClassifier):
        contrib = model.booster_.predict(X_t, pred_contrib=True)
        return contrib[:, :-1], float(contrib[0, -1]), names

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only without shap
        raise TypeError(
            f"Native TreeSHAP supports LightGBM only (got {type(model).__name__}); "
            "install the 'shap' package for other models."
        ) from exc
    explainer = shap.Explainer(model, X_t)
    explanation = explainer(X_t)
    values = explanation.values
    if values.ndim == 3:  # multiclass-shaped output: take positive class
        values = values[:, :, 1]
    base = np.ravel(np.asarray(explanation.base_values, dtype=float))
    return values, float(base[0]), names


def global_importance(pipeline: Pipeline, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value over a sample.

    Args:
        pipeline: Fitted model pipeline.
        X: Raw feature frame (a few hundred rows is plenty).
        top_n: Number of features to keep.

    Returns:
        DataFrame with ``feature`` and ``mean_abs_shap`` columns, descending.
    """
    values, _, names = compute_shap_values(pipeline, X)
    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    logger.info("Top SHAP features: %s", ", ".join(importance["feature"].head(5)))
    return importance


def explain_customer(
    pipeline: Pipeline, customer: pd.DataFrame, top_k: int = 3
) -> CustomerExplanation:
    """Explain a single customer's churn prediction.

    Args:
        pipeline: Fitted model pipeline.
        customer: Single-row raw feature frame.
        top_k: Number of positive and negative contributors to report.

    Returns:
        A :class:`CustomerExplanation` with the top drivers either way.

    Raises:
        ValueError: If ``customer`` does not contain exactly one row.
    """
    if len(customer) != 1:
        raise ValueError(f"Expected exactly one row, got {len(customer)}.")
    values, base_value, names = compute_shap_values(pipeline, customer)
    row = values[0]
    probability = float(pipeline.predict_proba(customer)[0, 1])

    order = np.argsort(row)
    positive = [(names[i], float(row[i])) for i in order[::-1] if row[i] > 0][:top_k]
    negative = [(names[i], float(row[i])) for i in order if row[i] < 0][:top_k]
    return CustomerExplanation(
        churn_probability=probability,
        base_value=base_value,
        top_positive=positive,
        top_negative=negative,
    )
