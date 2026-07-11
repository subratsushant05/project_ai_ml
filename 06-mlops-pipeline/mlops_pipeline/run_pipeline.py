"""Pipeline orchestrator: validate -> train -> evaluate -> promote.

Run with ``python -m mlops_pipeline.run_pipeline``. Each stage is logged
with timing, the run is idempotent (re-running never demotes a good
production model), and validation failure halts the pipeline with a
non-zero exit code.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from mlops_pipeline.config import Settings, get_settings, setup_logging
from mlops_pipeline.data import generate_loan_data
from mlops_pipeline.registry import (
    ModelRegistry,
    PromotionPolicy,
    promotion_decision,
)
from mlops_pipeline.training import CandidateResult, evaluate_model, train_and_log
from mlops_pipeline.validation import loan_schema

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ValidationFailedError(RuntimeError):
    """Raised when the data validation gate fails."""


class PipelineResult(BaseModel):
    """Summary of one orchestrated pipeline run."""

    validation_passed: bool
    candidate: CandidateResult
    incumbent_metrics: dict[str, float] | None
    promoted: bool
    promoted_version: int | None
    reasons: list[str]


def _stage(name: str, fn: Callable[[], T]) -> T:
    """Run one pipeline stage with start/finish logging and timing."""
    logger.info("stage=%s status=started", name)
    start = time.perf_counter()
    result = fn()
    logger.info("stage=%s status=finished elapsed=%.2fs", name, time.perf_counter() - start)
    return result


def run_pipeline(
    settings: Settings | None = None,
    df: pd.DataFrame | None = None,
) -> PipelineResult:
    """Execute the full train-and-promote pipeline.

    Args:
        settings: Pipeline configuration; loaded from the environment when
            omitted.
        df: Optional pre-built training frame (used by the drift simulator
            to retrain on recent traffic); generated deterministically when
            omitted.

    Returns:
        A :class:`PipelineResult` describing what happened.

    Raises:
        ValidationFailedError: When the data fails schema validation.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    frame = _stage(
        "ingest",
        lambda: df
        if df is not None
        else generate_loan_data(settings.n_samples, settings.random_seed),
    )

    schema = loan_schema()
    report = _stage("validate", lambda: schema.validate_frame(frame))
    report.save(settings.base_dir / settings.artifacts_dir / "validation_report.json")
    if not report.passed:
        failures = "; ".join(
            f"{c.column}/{c.check}: {c.detail}" for c in report.failures()
        )
        raise ValidationFailedError(f"Data validation failed: {failures}")

    candidate, x_holdout, y_holdout = _stage(
        "train", lambda: train_and_log(frame, settings, schema.schema_hash())
    )

    registry = ModelRegistry(settings)

    def _evaluate_incumbent() -> dict[str, float] | None:
        if registry.production_version() is None:
            logger.info("No production model yet - candidate races an empty field")
            return None
        incumbent = registry.load_production()
        return evaluate_model(incumbent, x_holdout, y_holdout)

    incumbent_metrics = _stage("evaluate_incumbent", _evaluate_incumbent)

    policy = PromotionPolicy(
        margin=settings.promotion_margin,
        min_auc=settings.min_auc,
        max_brier=settings.max_brier,
    )
    decision = promotion_decision(candidate.metrics, incumbent_metrics, policy)
    reason = "; ".join(decision.reasons)

    promoted_version: int | None = None
    if decision.promote:
        promoted_version = _stage(
            "promote", lambda: registry.promote(candidate, reason)
        )
    else:
        _stage("reject", lambda: registry.record_rejection(candidate, reason))

    return PipelineResult(
        validation_passed=True,
        candidate=candidate,
        incumbent_metrics=incumbent_metrics,
        promoted=decision.promote,
        promoted_version=promoted_version,
        reasons=decision.reasons,
    )


def main() -> int:
    """CLI entry point."""
    setup_logging()
    try:
        result = run_pipeline()
    except ValidationFailedError as exc:
        logger.error("%s", exc)
        return 1

    print("\n=== Pipeline summary ===")
    print(f"best candidate : {result.candidate.model_name}")
    print(
        "holdout metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in result.candidate.metrics.items())
    )
    if result.incumbent_metrics:
        print(f"incumbent auc  : {result.incumbent_metrics['auc']:.4f}")
    verdict = (
        f"PROMOTED as version {result.promoted_version}"
        if result.promoted
        else "REJECTED (production model kept)"
    )
    print(f"decision       : {verdict}")
    print(f"reasons        : {'; '.join(result.reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
