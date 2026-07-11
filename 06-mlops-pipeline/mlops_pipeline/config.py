"""Central configuration for the MLOps pipeline.

All knobs (paths, thresholds, model settings) live in a single
pydantic-settings object so every stage reads the same values and
anything can be overridden via ``MLOPS_``-prefixed environment variables.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables.

    Example:
        ``MLOPS_PROMOTION_MARGIN=0.02 python -m mlops_pipeline.run_pipeline``
    """

    model_config = SettingsConfigDict(env_prefix="MLOPS_", env_file=".env")

    # --- paths ------------------------------------------------------------
    base_dir: Path = Field(default=Path("."), description="Project working directory.")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Reports and audit logs.")
    tracking_db: Path = Field(default=Path("mlruns.db"), description="MLflow SQLite backend.")
    artifact_root: Path = Field(default=Path("mlruns"), description="MLflow artifact root.")

    # --- experiment / registry --------------------------------------------
    experiment_name: str = "loan-default"
    registered_model_name: str = "loan-default-classifier"
    production_alias: str = "production"

    # --- data --------------------------------------------------------------
    n_samples: int = 4000
    random_seed: int = 42
    test_size: float = 0.25

    # --- promotion policy ---------------------------------------------------
    promotion_margin: float = Field(
        default=0.005,
        description="Challenger AUC must beat production AUC by at least this much.",
    )
    min_auc: float = Field(default=0.70, description="Absolute AUC floor for any promotion.")
    max_brier: float = Field(default=0.20, description="Calibration guardrail (Brier score).")

    # --- drift thresholds ----------------------------------------------------
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_alpha: float = 0.05
    retrain_min_alerts: int = Field(
        default=2, description="Feature alerts needed before should_retrain() fires."
    )

    # --- serving --------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - container-friendly default
    api_port: int = 8000

    @property
    def tracking_uri(self) -> str:
        """MLflow tracking URI (SQLite so registry aliases are supported)."""
        return f"sqlite:///{self.base_dir / self.tracking_db}"

    @property
    def audit_log_path(self) -> Path:
        """Path of the JSON-lines registry audit log."""
        return self.base_dir / self.artifacts_dir / "registry_audit.jsonl"

    def ensure_dirs(self) -> None:
        """Create working directories if they do not exist yet."""
        (self.base_dir / self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.artifact_root).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` from the environment."""
    return Settings()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a consistent format."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
