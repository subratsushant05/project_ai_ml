"""Model registry with gated promotion, rollback, and an audit trail.

Wraps the MLflow model registry: a candidate becomes ``production`` (via
a registry alias) only when it beats the incumbent by a configurable AUC
margin *and* passes absolute quality guardrails. Every promotion and
rollback is appended to a JSON-lines audit log.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from mlops_pipeline.config import Settings
from mlops_pipeline.training import CandidateResult

logger = logging.getLogger(__name__)


class PromotionPolicy(BaseModel):
    """Rules a candidate must satisfy to replace the production model."""

    margin: float = 0.005
    min_auc: float = 0.70
    max_brier: float = 0.20


class PromotionDecision(BaseModel):
    """Outcome of applying a :class:`PromotionPolicy` to a candidate."""

    promote: bool
    reasons: list[str]


def promotion_decision(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    policy: PromotionPolicy,
) -> PromotionDecision:
    """Decide whether a candidate model should be promoted.

    Args:
        candidate: Candidate metrics (``auc``, ``brier``) on the holdout.
        incumbent: Production model metrics on the *same* holdout, or
            ``None`` when no production model exists yet.
        policy: Margin and guardrail configuration.

    Returns:
        A :class:`PromotionDecision` with human-readable reasons.
    """
    reasons: list[str] = []
    ok = True

    if candidate["auc"] < policy.min_auc:
        ok = False
        reasons.append(
            f"candidate auc {candidate['auc']:.4f} below floor {policy.min_auc:.2f}"
        )
    if candidate.get("brier", 0.0) > policy.max_brier:
        ok = False
        reasons.append(
            f"candidate brier {candidate['brier']:.4f} above cap {policy.max_brier:.2f}"
        )

    if incumbent is None:
        reasons.append("no production model - first promotion" if ok else "")
        return PromotionDecision(promote=ok, reasons=[r for r in reasons if r])

    required = incumbent["auc"] + policy.margin
    if candidate["auc"] >= required:
        reasons.append(
            f"candidate auc {candidate['auc']:.4f} beats production "
            f"{incumbent['auc']:.4f} + margin {policy.margin:.3f}"
        )
    else:
        ok = False
        reasons.append(
            f"candidate auc {candidate['auc']:.4f} < required {required:.4f} "
            f"(production {incumbent['auc']:.4f} + margin {policy.margin:.3f})"
        )
    return PromotionDecision(promote=ok, reasons=reasons)


class ModelRegistry:
    """Alias-based production registry over MLflow with an audit log."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the client against the configured tracking store."""
        self.settings = settings
        mlflow.set_tracking_uri(settings.tracking_uri)
        self.client = MlflowClient(tracking_uri=settings.tracking_uri)
        self.model_name = settings.registered_model_name
        self.alias = settings.production_alias

    # -- queries -----------------------------------------------------------

    def production_version(self) -> ModelVersion | None:
        """Return the model version behind the production alias, if any."""
        try:
            return self.client.get_model_version_by_alias(self.model_name, self.alias)
        except MlflowException:
            return None

    def production_metrics(self) -> dict[str, float] | None:
        """Return training-time metrics of the current production model."""
        version = self.production_version()
        if version is None or not version.run_id:
            return None
        return dict(self.client.get_run(version.run_id).data.metrics)

    def production_info(self) -> dict[str, Any] | None:
        """Summarise the production model for the serving /model/info route."""
        version = self.production_version()
        if version is None:
            return None
        run = self.client.get_run(version.run_id)
        return {
            "name": self.model_name,
            "alias": self.alias,
            "version": int(version.version),
            "run_id": version.run_id,
            "model_type": run.data.params.get("model_name", "unknown"),
            "metrics": dict(run.data.metrics),
            "data_schema_hash": run.data.tags.get("data_schema_hash", ""),
        }

    def load_production(self) -> Any:
        """Load the production model artifact (raises when none exists)."""
        return mlflow.sklearn.load_model(f"models:/{self.model_name}@{self.alias}")

    # -- transitions ----------------------------------------------------------

    def promote(self, candidate: CandidateResult, reason: str) -> int:
        """Register a candidate run and point the production alias at it.

        Args:
            candidate: Winning training run to promote.
            reason: Human-readable justification recorded in the audit log.

        Returns:
            The newly promoted registry version number.
        """
        with contextlib.suppress(MlflowException):  # already exists
            self.client.create_registered_model(self.model_name)

        previous = self.production_version()
        version = self.client.create_model_version(
            name=self.model_name,
            source=f"runs:/{candidate.run_id}/model",
            run_id=candidate.run_id,
        )
        self.client.set_registered_model_alias(
            self.model_name, self.alias, version.version
        )
        self._audit(
            event="promote",
            version=int(version.version),
            previous_version=int(previous.version) if previous else None,
            run_id=candidate.run_id,
            model_name=candidate.model_name,
            metrics=candidate.metrics,
            reason=reason,
        )
        logger.info("Promoted version %s to '%s' (%s)", version.version, self.alias, reason)
        return int(version.version)

    def rollback(self, reason: str) -> int:
        """Point the production alias back at the previously promoted version.

        Args:
            reason: Justification recorded in the audit log.

        Returns:
            The version number production was rolled back to.

        Raises:
            RuntimeError: If there is no earlier version to roll back to.
        """
        current = self.production_version()
        history = [
            entry["version"]
            for entry in self.audit_entries()
            if entry["event"] in {"promote", "rollback"}
        ]
        current_version = int(current.version) if current else None
        target = next(
            (v for v in reversed(history) if v != current_version), None
        )
        if target is None:
            msg = "No previous production version available for rollback"
            raise RuntimeError(msg)

        self.client.set_registered_model_alias(self.model_name, self.alias, target)
        self._audit(
            event="rollback",
            version=target,
            previous_version=current_version,
            reason=reason,
        )
        logger.info("Rolled back '%s' to version %s (%s)", self.alias, target, reason)
        return target

    def record_rejection(self, candidate: CandidateResult, reason: str) -> None:
        """Record a candidate that failed the promotion gate.

        Args:
            candidate: The rejected training run.
            reason: Why the promotion policy rejected it.
        """
        self._audit(
            event="reject",
            run_id=candidate.run_id,
            model_name=candidate.model_name,
            metrics=candidate.metrics,
            reason=reason,
        )
        logger.info("Rejected candidate %s (%s)", candidate.run_id, reason)

    # -- audit log ---------------------------------------------------------------

    def audit_entries(self) -> list[dict[str, Any]]:
        """Read all audit-log entries (oldest first)."""
        path = self.settings.audit_log_path
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def _audit(self, event: str, **payload: Any) -> None:
        """Append one JSON line to the audit log."""
        path = self.settings.audit_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "model": self.model_name,
            **payload,
        }
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
