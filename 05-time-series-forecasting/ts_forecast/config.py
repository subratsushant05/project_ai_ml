"""Pipeline configuration via pydantic-settings.

All knobs for the forecasting pipeline live here so the CLI, tests and demo
share one validated source of truth. Values can be overridden with
environment variables prefixed ``TSF_`` (e.g. ``TSF_HORIZON=14``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    """Configuration for the end-to-end forecasting pipeline.

    Attributes:
        horizon: Number of days to forecast ahead.
        n_folds: Number of rolling-origin backtest folds.
        min_train_days: Minimum length of the first training window.
        season_length: Dominant seasonal period in days (7 = weekly).
        alpha: Miscoverage rate for conformal intervals (0.1 -> 90% PI).
        anomaly_z_threshold: |z|-score above which a point is flagged.
        seed: Seed for the synthetic data generator.
        output_dir: Directory for plots, metrics.json and the HTML report.
    """

    model_config = SettingsConfigDict(env_prefix="TSF_", frozen=True)

    horizon: int = Field(default=28, ge=1, le=365)
    n_folds: int = Field(default=5, ge=2, le=20)
    min_train_days: int = Field(default=365, ge=60)
    season_length: int = Field(default=7, ge=1)
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    anomaly_z_threshold: float = Field(default=3.0, gt=0.0)
    seed: int = Field(default=42, ge=0)
    output_dir: Path = Path("docs")

    @field_validator("output_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        """Expand ``~`` in the output directory path."""
        return value.expanduser()
