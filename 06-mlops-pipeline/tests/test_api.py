"""Tests for the FastAPI serving layer via TestClient."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.config import Settings
from mlops_pipeline.run_pipeline import PipelineResult
from mlops_pipeline.serving.app import create_app

VALID_PAYLOAD = {
    "age": 34,
    "income": 55_000.0,
    "loan_amount": 12_000.0,
    "credit_score": 700,
    "debt_to_income": 0.25,
    "num_prior_defaults": 0,
    "employment_status": "employed",
    "loan_term_months": 36,
}


@pytest.fixture(scope="module")
def client(prod_env: tuple[Settings, PipelineResult]) -> Iterator[TestClient]:
    """Serve the model promoted by the session-scoped pipeline run."""
    settings, _ = prod_env
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    """The health probe reports a loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_request(client: TestClient) -> None:
    """A valid application returns a probability and a class."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in {"default", "no_default"}
    assert body["model_version"] >= 1


def test_predict_rejects_invalid_payload(client: TestClient) -> None:
    """Pydantic validation returns 422 for out-of-domain values."""
    bad = dict(VALID_PAYLOAD, credit_score=9000)
    assert client.post("/predict", json=bad).status_code == 422
    bad = dict(VALID_PAYLOAD, employment_status="astronaut")
    assert client.post("/predict", json=bad).status_code == 422
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "income"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_metrics_endpoint_exposes_prometheus_series(client: TestClient) -> None:
    """/metrics exposes request counts, latency, and prediction histogram."""
    client.post("/predict", json=VALID_PAYLOAD)
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert 'http_requests_total{endpoint="/predict"' in text
    assert "http_request_latency_seconds_bucket" in text
    assert "prediction_probability_bucket" in text
    assert "predictions_total" in text


def test_model_info(client: TestClient, prod_env: tuple[Settings, PipelineResult]) -> None:
    """/model/info reflects the promoted registry version."""
    settings, result = prod_env
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == settings.registered_model_name
    assert body["version"] == result.promoted_version
    assert body["model_type"] == result.candidate.model_name
    assert "auc" in body["metrics"]
    assert body["data_schema_hash"]
