"""API tests using FastAPI's TestClient and a small trained model."""

from __future__ import annotations

VALID_CUSTOMER = {
    "tenure_months": 2,
    "monthly_charges": 88.0,
    "total_charges": 176.0,
    "num_support_calls": 5,
    "avg_download_gb": 40.0,
    "late_payments_12m": 3,
    "contract_type": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "electronic_check",
    "tech_support": "no",
    "paperless_billing": "yes",
}

LOYAL_CUSTOMER = {
    **VALID_CUSTOMER,
    "tenure_months": 70,
    "num_support_calls": 0,
    "late_payments_12m": 0,
    "contract_type": "two-year",
    "internet_service": "dsl",
    "payment_method": "credit_card",
    "tech_support": "yes",
    "monthly_charges": 45.0,
    "total_charges": 3200.0,
}


def test_health(api_client) -> None:
    """Health endpoint reports a loaded model."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(api_client) -> None:
    """Model info exposes metadata and metrics of the served model."""
    response = api_client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_class"] == "LGBMClassifier"
    assert 0.0 < body["decision_threshold"] < 1.0
    assert "roc_auc" in body["metrics"]


def test_predict_single_with_explanation(api_client) -> None:
    """Single prediction returns probability, decision, and SHAP drivers."""
    response = api_client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_predicted"], bool)
    assert body["top_positive"], "expected at least one positive contributor"
    assert {"feature", "shap_value"} == set(body["top_positive"][0])


def test_predict_risk_ordering(api_client) -> None:
    """A risky profile must score higher than a loyal one."""
    risky = api_client.post("/predict", json=VALID_CUSTOMER, params={"explain": False}).json()
    loyal = api_client.post("/predict", json=LOYAL_CUSTOMER, params={"explain": False}).json()
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["top_positive"] is None  # explain=false skips SHAP


def test_predict_missing_optional_fields(api_client) -> None:
    """Optional fields may be omitted; imputers fill them downstream."""
    optional = ("total_charges", "tech_support")
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k not in optional}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_validation_errors(api_client) -> None:
    """Bad enum values and negative numbers are rejected with 422."""
    bad_contract = {**VALID_CUSTOMER, "contract_type": "weekly"}
    assert api_client.post("/predict", json=bad_contract).status_code == 422
    negative_tenure = {**VALID_CUSTOMER, "tenure_months": -4}
    assert api_client.post("/predict", json=negative_tenure).status_code == 422
    extra_field = {**VALID_CUSTOMER, "definitely_not_a_feature": 1}
    assert api_client.post("/predict", json=extra_field).status_code == 422


def test_predict_batch(api_client) -> None:
    """Batch endpoint scores all rows and keeps input order."""
    payload = {"customers": [VALID_CUSTOMER, LOYAL_CUSTOMER]}
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["churn_probability"] > predictions[1]["churn_probability"]


def test_predict_batch_empty_rejected(api_client) -> None:
    """An empty batch violates the schema."""
    assert api_client.post("/predict/batch", json={"customers": []}).status_code == 422
