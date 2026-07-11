"""Hyperparameter tuning for the LightGBM pipeline with Optuna.

The search space is intentionally small so the whole study finishes in well
under a minute on the default dataset; the point is the pattern (pruned,
seeded, CV-scored study), not exhaustive search.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churn_pipeline.models import build_model_pipeline

logger = logging.getLogger(__name__)


def tune_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 15,
    cv_folds: int = 3,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """Run a small Optuna study over LightGBM hyperparameters.

    The objective is mean ROC-AUC over a stratified k-fold split of the
    training data (the holdout set is never touched here).

    Args:
        X: Training feature frame.
        y: Binary training target.
        n_trials: Number of Optuna trials.
        cv_folds: Folds used inside the objective.
        seed: Seed for the sampler and CV shuffling.

    Returns:
        Tuple of (best hyperparameters, best CV ROC-AUC).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            pipeline = build_model_pipeline("lightgbm", seed=seed, **params)
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            y_prob = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], y_prob))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="lightgbm-churn",
    )
    logger.info("Starting Optuna study: %d trials, %d-fold CV objective", n_trials, cv_folds)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Best trial ROC-AUC %.4f with params %s", study.best_value, study.best_params)
    return dict(study.best_params), float(study.best_value)
