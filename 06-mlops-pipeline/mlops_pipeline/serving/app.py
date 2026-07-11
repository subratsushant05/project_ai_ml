"""FastAPI app serving the production model with Prometheus monitoring.

The app loads whatever model currently holds the ``production`` registry
alias at startup, validates every request with pydantic, and exposes
request-count, latency, and prediction-distribution metrics on /metrics.

Run locally with::

    uvicorn mlops_pipeline.serving.app:app --port 8000
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from mlops_pipeline.config import Settings, get_settings
from mlops_pipeline.data import FEATURE_COLUMNS
from mlops_pipeline.registry import ModelRegistry

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """A single loan application to score."""

    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    loan_amount: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)
    debt_to_income: float = Field(ge=0.0, le=1.0)
    num_prior_defaults: int = Field(ge=0, le=5)
    employment_status: Literal["employed", "self_employed", "unemployed", "retired"]
    loan_term_months: Literal[12, 24, 36, 48, 60]


class PredictionResponse(BaseModel):
    """Model output for one application."""

    default_probability: float
    prediction: Literal["default", "no_default"]
    model_version: int


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the serving app with its own Prometheus registry.

    Args:
        settings: Pipeline configuration; loaded from the environment when
            omitted. Tests pass a settings object pointed at a temp store.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    settings = settings or get_settings()
    metrics_registry = CollectorRegistry()
    request_count = Counter(
        "http_requests_total",
        "HTTP requests by endpoint and status code.",
        ["method", "endpoint", "status"],
        registry=metrics_registry,
    )
    request_latency = Histogram(
        "http_request_latency_seconds",
        "Request latency by endpoint.",
        ["endpoint"],
        registry=metrics_registry,
    )
    prediction_probability = Histogram(
        "prediction_probability",
        "Distribution of predicted default probabilities.",
        buckets=[i / 10 for i in range(11)],
        registry=metrics_registry,
    )
    predictions_total = Counter(
        "predictions_total",
        "Predictions served, by predicted class.",
        ["predicted_class"],
        registry=metrics_registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = ModelRegistry(settings)
        info = registry.production_info()
        if info is None:
            msg = (
                "No production model found - run "
                "`python -m mlops_pipeline.run_pipeline` first"
            )
            raise RuntimeError(msg)
        app.state.model = registry.load_production()
        app.state.model_info = info
        logger.info(
            "Loaded %s v%s for serving", info["model_type"], info["version"]
        )
        yield

    app = FastAPI(
        title="Loan Default Scoring API",
        description="Serves the current production model from the MLflow registry.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        endpoint = request.url.path
        request_latency.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        request_count.labels(
            method=request.method, endpoint=endpoint, status=response.status_code
        ).inc()
        return response

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Liveness/readiness probe."""
        loaded = getattr(request.app.state, "model", None) is not None
        return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}

    @app.get("/model/info")
    async def model_info(request: Request) -> dict[str, Any]:
        """Metadata about the model currently behind the production alias."""
        info = getattr(request.app.state, "model_info", None)
        if info is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return dict(info)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
        """Score one loan application with the production model."""
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        frame = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
        probability = float(model.predict_proba(frame)[0, 1])
        label = "default" if probability >= 0.5 else "no_default"

        prediction_probability.observe(probability)
        predictions_total.labels(predicted_class=label).inc()
        return PredictionResponse(
            default_probability=round(probability, 6),
            prediction=label,
            model_version=request.app.state.model_info["version"],
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus scrape endpoint (text exposition format)."""
        return Response(generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
