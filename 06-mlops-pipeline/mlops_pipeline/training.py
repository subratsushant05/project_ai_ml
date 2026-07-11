"""Training stage: model comparison with MLflow experiment tracking.

Trains a small stable of candidate models, evaluates each on a shared
holdout split, and logs params, metrics, the fitted pipeline artifact and
the data-schema hash to MLflow. The best candidate (by holdout AUC) is
returned for the promotion stage.
"""

from __future__ import annotations

import logging

import mlflow
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_pipeline.config import Settings
from mlops_pipeline.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class CandidateResult(BaseModel):
    """The winning candidate of a training round."""

    run_id: str
    model_name: str
    metrics: dict[str, float]


def evaluate_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Score a fitted classifier on a holdout set.

    Args:
        model: Fitted sklearn pipeline exposing ``predict_proba``.
        x: Holdout features.
        y: Holdout binary labels.

    Returns:
        Dict with ``auc``, ``accuracy`` and ``brier`` (lower is better).
    """
    proba = model.predict_proba(x)[:, 1]
    return {
        "auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, proba >= 0.5)),
        "brier": float(brier_score_loss(y, proba)),
    }


def _preprocessor() -> ColumnTransformer:
    """Shared feature encoding: scale numerics, one-hot categoricals."""
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_candidates(seed: int) -> dict[str, Pipeline]:
    """Construct the candidate models compared in each training round.

    Args:
        seed: Random seed forwarded to stochastic estimators.

    Returns:
        Mapping of model name to unfitted sklearn pipeline.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("features", _preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _flat_params(pipeline: Pipeline) -> dict[str, str]:
    """Extract the classifier's hyperparameters for MLflow logging."""
    clf = pipeline.named_steps["clf"]
    return {k: str(v) for k, v in clf.get_params().items() if v is not None}


def train_and_log(
    df: pd.DataFrame,
    settings: Settings,
    schema_hash: str,
) -> tuple[CandidateResult, pd.DataFrame, pd.Series]:
    """Train all candidates, log them to MLflow, and pick the best.

    Args:
        df: Validated training frame (features + target).
        settings: Pipeline configuration (tracking URI, seed, split size).
        schema_hash: Hash of the data schema, logged for lineage.

    Returns:
        Tuple of (best candidate, holdout features, holdout labels). The
        holdout is returned so the caller can score the incumbent
        production model on identical data before deciding on promotion.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=df[TARGET_COLUMN],
    )

    best: CandidateResult | None = None
    for name, pipeline in build_candidates(settings.random_seed).items():
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(x_train, y_train)
            metrics = evaluate_model(pipeline, x_test, y_test)

            mlflow.log_params(_flat_params(pipeline))
            mlflow.log_param("model_name", name)
            mlflow.log_param("n_train_rows", len(x_train))
            mlflow.log_metrics(metrics)
            mlflow.set_tag("data_schema_hash", schema_hash)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                "Trained %s: auc=%.4f accuracy=%.4f brier=%.4f (run %s)",
                name,
                metrics["auc"],
                metrics["accuracy"],
                metrics["brier"],
                run.info.run_id,
            )
            if best is None or metrics["auc"] > best.metrics["auc"]:
                best = CandidateResult(
                    run_id=run.info.run_id, model_name=name, metrics=metrics
                )

    assert best is not None  # noqa: S101 - candidates dict is never empty
    logger.info("Best candidate: %s (auc=%.4f)", best.model_name, best.metrics["auc"])
    return best, x_test, y_test
