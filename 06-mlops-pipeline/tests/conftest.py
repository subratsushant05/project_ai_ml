"""Shared fixtures: session-scoped MLflow environments on tiny data."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlops_pipeline.config import Settings
from mlops_pipeline.run_pipeline import PipelineResult, run_pipeline


def make_settings(base_dir: Path, **overrides: object) -> Settings:
    """Build test settings rooted in a temp directory with tiny data."""
    defaults: dict[str, object] = {
        "base_dir": base_dir,
        "n_samples": 600,
        "random_seed": 7,
        "experiment_name": "test-exp",
        "registered_model_name": "test-model",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def prod_env(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Settings, PipelineResult]]:
    """Run the full pipeline once into a temp MLflow store.

    Yields:
        The settings used and the pipeline result (a promoted model).
    """
    tmp = tmp_path_factory.mktemp("mlops-prod")
    settings = make_settings(tmp)
    cwd = os.getcwd()
    os.chdir(tmp)  # keep MLflow's default artifact root inside the temp dir
    try:
        result = run_pipeline(settings)
    finally:
        os.chdir(cwd)
    yield settings, result


@pytest.fixture(scope="session")
def registry_env(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Settings, list[str]]:
    """Provide a temp MLflow store containing two tiny logged models.

    Returns:
        The settings and the two run ids (in logging order).
    """
    tmp = tmp_path_factory.mktemp("mlops-registry")
    settings = make_settings(tmp, experiment_name="registry-exp")
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.create_experiment(
        settings.experiment_name, artifact_location=str(tmp / "mlruns")
    )
    mlflow.set_experiment(settings.experiment_name)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int)
    run_ids: list[str] = []
    for c in (0.5, 1.0):
        with mlflow.start_run() as run:
            model = LogisticRegression(C=c).fit(x, y)
            mlflow.log_metric("auc", 0.9)
            mlflow.sklearn.log_model(model, artifact_path="model")
            run_ids.append(run.info.run_id)
    return settings, run_ids
