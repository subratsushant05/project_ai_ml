"""Model persistence: joblib artifact plus a git-friendly metadata sidecar."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from churn_pipeline import __version__

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass
class ModelBundle:
    """A fitted pipeline together with its metadata.

    Attributes:
        pipeline: Fitted sklearn pipeline (preprocessing + classifier).
        metadata: Metrics, threshold, and provenance information.
    """

    pipeline: Pipeline
    metadata: dict[str, Any]


def save_model(
    pipeline: Pipeline,
    artifacts_dir: Path,
    metrics: dict[str, Any],
    threshold: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted pipeline and a JSON metadata sidecar.

    Args:
        pipeline: Fitted model pipeline.
        artifacts_dir: Output directory (created if missing).
        metrics: Evaluation metrics to embed in the metadata.
        threshold: Business decision threshold to apply at serving time.
        extra: Additional metadata entries (e.g. best hyperparameters).

    Returns:
        Path to the saved model file.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)

    model_step = pipeline.named_steps["model"]
    metadata: dict[str, Any] = {
        "package_version": __version__,
        "model_class": type(model_step).__name__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "decision_threshold": threshold,
        "metrics": metrics,
    }
    metadata.update(extra or {})
    metadata_path = artifacts_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    logger.info("Saved model to %s and metadata to %s", model_path, metadata_path)
    return model_path


def load_model(artifacts_dir: Path) -> ModelBundle:
    """Load a persisted pipeline and its metadata.

    Args:
        artifacts_dir: Directory containing ``model.joblib`` and
            ``metadata.json``.

    Returns:
        A :class:`ModelBundle`.

    Raises:
        FileNotFoundError: If either artifact file is missing.
    """
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / MODEL_FILENAME
    metadata_path = artifacts_dir / METADATA_FILENAME
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in '{artifacts_dir}'. "
            "Run 'python -m churn_pipeline.train' first."
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    logger.info("Loaded %s trained at %s", metadata.get("model_class"), metadata.get("trained_at"))
    return ModelBundle(pipeline=pipeline, metadata=metadata)
