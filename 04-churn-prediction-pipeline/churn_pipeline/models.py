"""Model factory: candidate classifiers wrapped in the shared preprocessor.

Class imbalance is handled at the estimator level via class weights
(``class_weight="balanced"`` / ``is_unbalance``); decision thresholds are
tuned separately (see :mod:`churn_pipeline.threshold`).
"""

from __future__ import annotations

import logging
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_pipeline.features import build_preprocessor

logger = logging.getLogger(__name__)

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]


def build_estimator(name: str, seed: int = 42, **params: Any) -> Any:
    """Create a bare classifier by name.

    Args:
        name: One of ``MODEL_NAMES``.
        seed: Random seed passed to the estimator.
        **params: Extra hyperparameters overriding the defaults.

    Returns:
        An unfitted sklearn-compatible classifier.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    if name == "logistic_regression":
        defaults: dict[str, Any] = {
            "max_iter": 2000,
            "class_weight": "balanced",
            "C": 1.0,
            "random_state": seed,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)
    if name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": seed,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if name == "lightgbm":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(params)
        return LGBMClassifier(**defaults)
    raise ValueError(f"Unknown model '{name}'. Expected one of {MODEL_NAMES}.")


def build_model_pipeline(name: str, seed: int = 42, **params: Any) -> Pipeline:
    """Create a full pipeline: preprocessing plus a classifier.

    Args:
        name: One of ``MODEL_NAMES``.
        seed: Random seed for the estimator.
        **params: Hyperparameters forwarded to :func:`build_estimator`.

    Returns:
        Unfitted sklearn ``Pipeline`` with steps ``preprocess`` and ``model``.
    """
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", build_estimator(name, seed=seed, **params)),
        ]
    )
    logger.debug("Built pipeline for %s with params %s", name, params or "defaults")
    return pipeline
