"""Integration tests for the MLflow-backed registry (promote/rollback/audit).

The tests in this module share ``registry_env`` and run top-to-bottom as a
promotion lifecycle: promote v1, promote v2, roll back to v1.
"""

from pathlib import Path

import pytest

from mlops_pipeline.config import Settings
from mlops_pipeline.registry import ModelRegistry
from mlops_pipeline.training import CandidateResult


def _candidate(run_id: str, auc: float) -> CandidateResult:
    return CandidateResult(
        run_id=run_id,
        model_name="logistic_regression",
        metrics={"auc": auc, "accuracy": 0.8, "brier": 0.15},
    )


def test_promote_sets_alias_and_writes_audit(
    registry_env: tuple[Settings, list[str]],
) -> None:
    """Two promotions move the alias and append audit entries."""
    settings, run_ids = registry_env
    registry = ModelRegistry(settings)

    v1 = registry.promote(_candidate(run_ids[0], 0.80), reason="first promotion")
    assert v1 == 1
    assert registry.production_version() is not None

    v2 = registry.promote(_candidate(run_ids[1], 0.85), reason="beats v1")
    assert v2 == 2
    version = registry.production_version()
    assert version is not None and int(version.version) == 2

    events = [e["event"] for e in registry.audit_entries()]
    assert events == ["promote", "promote"]
    assert registry.audit_entries()[-1]["previous_version"] == 1


def test_rollback_restores_previous_version(
    registry_env: tuple[Settings, list[str]],
) -> None:
    """Rollback re-points the alias at the previously promoted version."""
    settings, _ = registry_env
    registry = ModelRegistry(settings)

    target = registry.rollback(reason="v2 misbehaving in prod")
    assert target == 1
    version = registry.production_version()
    assert version is not None and int(version.version) == 1

    last = registry.audit_entries()[-1]
    assert last["event"] == "rollback"
    assert last["version"] == 1
    assert last["previous_version"] == 2


def test_load_production_returns_working_model(
    registry_env: tuple[Settings, list[str]],
) -> None:
    """The production alias resolves to a loadable, predicting model."""
    settings, _ = registry_env
    registry = ModelRegistry(settings)
    model = registry.load_production()
    assert model.predict([[1.0, 0.0, 0.0]]) is not None
    info = registry.production_info()
    assert info is not None
    assert info["version"] == 1


def test_rollback_without_history_raises(tmp_path: Path) -> None:
    """Rolling back an empty registry is a hard error."""
    settings = Settings(base_dir=tmp_path, registered_model_name="ghost-model")
    registry = ModelRegistry(settings)
    with pytest.raises(RuntimeError, match="No previous production version"):
        registry.rollback(reason="nothing to roll back")
