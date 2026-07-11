"""Central configuration for the churn pipeline, backed by pydantic-settings.

All knobs can be overridden via environment variables with the ``CHURN_``
prefix, e.g. ``CHURN_N_ROWS=1000 python -m churn_pipeline.train``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime configuration for data generation, training, and serving.

    Attributes:
        seed: Global random seed for reproducibility.
        n_rows: Number of synthetic customers to generate.
        test_size: Fraction of data held out for final evaluation.
        cv_folds: Number of stratified CV folds for model comparison.
        n_trials: Number of Optuna trials for hyperparameter tuning.
        tuning_cv_folds: CV folds used inside the Optuna objective.
        retention_offer_cost: Cost of sending one retention offer (USD).
        churn_loss: Revenue lost when a customer churns (USD).
        offer_save_rate: Probability a retention offer prevents a churn.
        artifacts_dir: Directory where model artifacts and plots are written.
        shap_sample_size: Rows sampled for SHAP global importance.
        log_level: Logging level for the pipeline.
    """

    model_config = SettingsConfigDict(env_prefix="CHURN_", extra="ignore")

    seed: int = 42
    n_rows: int = Field(default=5000, ge=50)
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    cv_folds: int = Field(default=5, ge=2)
    n_trials: int = Field(default=15, ge=1)
    tuning_cv_folds: int = Field(default=3, ge=2)

    retention_offer_cost: float = Field(default=50.0, gt=0.0)
    churn_loss: float = Field(default=600.0, gt=0.0)
    offer_save_rate: float = Field(default=0.4, gt=0.0, le=1.0)

    artifacts_dir: Path = Path("artifacts")
    shap_sample_size: int = Field(default=300, ge=10)
    log_level: str = "INFO"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once with a consistent format.

    Args:
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=False,
    )
