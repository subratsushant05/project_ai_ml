"""End-to-end training entry point.

Usage::

    python -m churn_pipeline.train

Steps: generate synthetic data, compare candidate models with stratified CV,
tune LightGBM with Optuna, fit the final model, calibrate the business
decision threshold on the holdout set, compute SHAP explanations, and save
all artifacts (model, metadata, metrics, plots) to the artifacts directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_pipeline.config import Settings, setup_logging
from churn_pipeline.data import TARGET, generate_churn_data
from churn_pipeline.evaluate import compare_models, compute_metrics
from churn_pipeline.explain import explain_customer, global_importance
from churn_pipeline.models import MODEL_NAMES, build_model_pipeline
from churn_pipeline.persistence import save_model
from churn_pipeline.plots import plot_profit_curve, plot_roc_curves, plot_shap_importance
from churn_pipeline.threshold import optimize_threshold
from churn_pipeline.tuning import tune_lightgbm

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split a generated frame into stratified train/holdout parts.

    Args:
        df: Full dataset including the target column.
        test_size: Holdout fraction.
        seed: Split seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test); features exclude
        ``customer_id`` and the target.
    """
    X = df.drop(columns=[TARGET, "customer_id"])
    y = df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def run_training(settings: Settings | None = None) -> dict[str, Any]:
    """Run the full training workflow and write artifacts.

    Args:
        settings: Optional settings override (used by tests and the demo).

    Returns:
        The metrics dictionary that was written to ``metrics.json``.
    """
    settings = settings or Settings()
    setup_logging(settings.log_level)
    artifacts = Path(settings.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/6: generating %d synthetic customers", settings.n_rows)
    df = generate_churn_data(n_rows=settings.n_rows, seed=settings.seed)
    X_train, X_test, y_train, y_test = split_data(df, settings.test_size, settings.seed)

    logger.info("Step 2/6: comparing %s with %d-fold CV", MODEL_NAMES, settings.cv_folds)
    candidates = {name: build_model_pipeline(name, seed=settings.seed) for name in MODEL_NAMES}
    cv_results = compare_models(
        candidates, X_train, y_train, n_folds=settings.cv_folds, seed=settings.seed
    )

    logger.info("Step 3/6: tuning LightGBM with Optuna (%d trials)", settings.n_trials)
    best_params, best_cv_auc = tune_lightgbm(
        X_train,
        y_train,
        n_trials=settings.n_trials,
        cv_folds=settings.tuning_cv_folds,
        seed=settings.seed,
    )

    logger.info("Step 4/6: fitting final model and scoring holdout")
    final = build_model_pipeline("lightgbm", seed=settings.seed, **best_params)
    final.fit(X_train, y_train)
    y_prob = final.predict_proba(X_test)[:, 1]
    holdout_metrics = compute_metrics(y_test, y_prob)
    logger.info("Holdout: %s", {k: round(v, 4) for k, v in holdout_metrics.items()})

    logger.info("Step 5/6: calibrating business decision threshold")
    threshold_result = optimize_threshold(
        y_test,
        y_prob,
        offer_cost=settings.retention_offer_cost,
        churn_loss=settings.churn_loss,
        offer_save_rate=settings.offer_save_rate,
    )
    holdout_metrics["f1_at_business_threshold"] = compute_metrics(
        y_test, y_prob, threshold=threshold_result.best_threshold
    )["f1"]

    logger.info("Step 6/6: explanations, plots, and artifacts")
    sample = X_test.iloc[: settings.shap_sample_size]
    importance = global_importance(final, sample)
    example_explanation = explain_customer(final, X_test.iloc[[0]])

    plot_profit_curve(threshold_result, artifacts / "profit_curve.png")
    plot_shap_importance(importance, artifacts / "shap_importance.png")
    plot_roc_curves({"lightgbm (tuned)": (y_test, y_prob, holdout_metrics["roc_auc"])},
                    artifacts / "roc_curve.png")

    metrics: dict[str, Any] = {
        "cv_comparison": {name: dict(row) for name, row in cv_results.round(4).iterrows()},
        "tuned_lightgbm_cv_roc_auc": round(best_cv_auc, 4),
        "best_params": best_params,
        "holdout": {k: round(v, 4) for k, v in holdout_metrics.items()},
        "business_threshold": {
            "threshold": round(threshold_result.best_threshold, 3),
            "expected_profit_usd": round(threshold_result.best_profit, 2),
            "f1_optimal_threshold": round(threshold_result.f1_threshold, 3),
            "offer_cost": settings.retention_offer_cost,
            "churn_loss": settings.churn_loss,
            "offer_save_rate": settings.offer_save_rate,
        },
        "n_rows": settings.n_rows,
        "holdout_size": int(len(y_test)),
        "churn_rate": round(float(df[TARGET].mean()), 4),
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_model(
        final,
        artifacts,
        metrics=metrics["holdout"],
        threshold=threshold_result.best_threshold,
        extra={"best_params": best_params, "cv_roc_auc": round(best_cv_auc, 4)},
    )

    importance.to_csv(artifacts / "shap_importance.csv", index=False)
    logger.info("Example explanation:\n%s", example_explanation.to_text())
    logger.info("Done. Artifacts in %s", artifacts.resolve())
    return metrics


def main() -> None:
    """CLI entry point."""
    run_training()


if __name__ == "__main__":
    main()
