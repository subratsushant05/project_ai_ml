"""End-to-end tests for the orchestrated pipeline."""

from pathlib import Path

import pytest

from mlops_pipeline.config import Settings
from mlops_pipeline.data import generate_loan_data
from mlops_pipeline.registry import ModelRegistry
from mlops_pipeline.run_pipeline import (
    PipelineResult,
    ValidationFailedError,
    run_pipeline,
)


def test_pipeline_end_to_end(prod_env: tuple[Settings, PipelineResult]) -> None:
    """A fresh run validates, trains, and promotes a first model."""
    settings, result = prod_env

    assert result.validation_passed
    assert result.promoted
    assert result.promoted_version == 1
    assert result.candidate.metrics["auc"] > 0.5

    # Stage artifacts landed on disk.
    artifacts = settings.base_dir / settings.artifacts_dir
    assert (artifacts / "validation_report.json").exists()
    assert settings.audit_log_path.exists()

    # The registry alias resolves to the promoted version.
    registry = ModelRegistry(settings)
    version = registry.production_version()
    assert version is not None
    assert int(version.version) == result.promoted_version


def test_pipeline_halts_on_invalid_data(tmp_path: Path) -> None:
    """Corrupt input stops the pipeline before training."""
    settings = Settings(base_dir=tmp_path, n_samples=100)
    df = generate_loan_data(100, seed=3)
    df.loc[0, "credit_score"] = 99_999

    with pytest.raises(ValidationFailedError, match="credit_score"):
        run_pipeline(settings, df=df)

    # The validation report is still written for debugging.
    report_path = tmp_path / settings.artifacts_dir / "validation_report.json"
    assert report_path.exists()
    # ...and no model store was created.
    assert not (tmp_path / settings.tracking_db).exists()
