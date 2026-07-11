"""FastAPI service exposing the trained churn model.

Run locally with::

    uvicorn churn_pipeline.api:app --host 0.0.0.0 --port 8000

The model is loaded once at startup from ``Settings.artifacts_dir``
(override with ``CHURN_ARTIFACTS_DIR``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_pipeline import __version__
from churn_pipeline.config import Settings, setup_logging
from churn_pipeline.explain import explain_customer
from churn_pipeline.persistence import ModelBundle, load_model
from churn_pipeline.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    Contribution,
    Customer,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

_state: dict[str, ModelBundle] = {}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the model bundle once at startup."""
    settings = Settings()
    setup_logging(settings.log_level)
    try:
        _state["bundle"] = load_model(settings.artifacts_dir)
        logger.info("Model loaded from %s", settings.artifacts_dir)
    except FileNotFoundError:
        logger.warning("No model artifacts in %s; /predict will return 503", settings.artifacts_dir)
    yield
    _state.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Serve churn probabilities with business-calibrated decisions.",
    version=__version__,
    lifespan=lifespan,
)


def _get_bundle() -> ModelBundle:
    """Return the loaded model bundle or raise 503."""
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    return bundle


def _predict_one(bundle: ModelBundle, customer: Customer, explain: bool) -> PredictionResponse:
    """Score a single customer and optionally attach SHAP contributors."""
    threshold = float(bundle.metadata["decision_threshold"])
    frame = customer.to_frame()
    probability = float(bundle.pipeline.predict_proba(frame)[0, 1])
    top_positive = top_negative = None
    if explain:
        explanation = explain_customer(bundle.pipeline, frame)
        top_positive = [Contribution(feature=f, shap_value=v) for f, v in explanation.top_positive]
        top_negative = [Contribution(feature=f, shap_value=v) for f, v in explanation.top_negative]
    return PredictionResponse(
        churn_probability=probability,
        churn_predicted=probability >= threshold,
        decision_threshold=threshold,
        top_positive=top_positive,
        top_negative=top_negative,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe reporting whether a model is loaded."""
    return HealthResponse(status="ok", model_loaded="bundle" in _state)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Metadata of the currently served model."""
    metadata = _get_bundle().metadata
    return ModelInfoResponse(
        model_class=metadata["model_class"],
        package_version=metadata["package_version"],
        trained_at=metadata["trained_at"],
        decision_threshold=metadata["decision_threshold"],
        metrics=metadata["metrics"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer, explain: bool = True) -> PredictionResponse:
    """Score one customer; include top SHAP drivers unless ``explain=false``."""
    return _predict_one(_get_bundle(), customer, explain=explain)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to 1000 customers in one call (no per-row explanations)."""
    bundle = _get_bundle()
    threshold = float(bundle.metadata["decision_threshold"])
    frame = pd.DataFrame([c.model_dump() for c in request.customers])
    probabilities = bundle.pipeline.predict_proba(frame)[:, 1]
    predictions = [
        PredictionResponse(
            churn_probability=float(p),
            churn_predicted=bool(p >= threshold),
            decision_threshold=threshold,
        )
        for p in probabilities
    ]
    return BatchPredictionResponse(predictions=predictions)
