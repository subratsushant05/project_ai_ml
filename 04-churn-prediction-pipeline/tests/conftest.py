"""Shared fixtures: small dataset and a fast trained model bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_pipeline.config import Settings
from churn_pipeline.data import TARGET, generate_churn_data
from churn_pipeline.models import build_model_pipeline
from churn_pipeline.train import split_data


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """A small deterministic synthetic dataset."""
    return generate_churn_data(n_rows=1200, seed=123)


@pytest.fixture(scope="session")
def split(small_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split of the small dataset."""
    return split_data(small_df, test_size=0.25, seed=123)


@pytest.fixture(scope="session")
def fitted_pipeline(split):
    """A small LightGBM pipeline fitted on the training split."""
    X_train, _, y_train, _ = split
    pipeline = build_model_pipeline("lightgbm", seed=123, n_estimators=80)
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Artifacts from a fast end-to-end training run (used by API tests)."""
    from churn_pipeline.train import run_training

    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    settings = Settings(
        n_rows=800,
        n_trials=2,
        cv_folds=2,
        tuning_cv_folds=2,
        shap_sample_size=50,
        artifacts_dir=artifacts_dir,
    )
    run_training(settings)
    return artifacts_dir


@pytest.fixture(scope="session")
def api_client(trained_artifacts: Path, monkeypatch_session):
    """FastAPI TestClient wired to the small trained model."""
    from fastapi.testclient import TestClient

    from churn_pipeline import api

    monkeypatch_session.setenv("CHURN_ARTIFACTS_DIR", str(trained_artifacts))
    with TestClient(api.app) as client:
        yield client


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch (pytest's default is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def target_column() -> str:
    """Name of the target column."""
    return TARGET
