"""Pydantic schemas for the serving API."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Customer(BaseModel):
    """Raw input features for a single customer.

    Optional fields (``total_charges``, ``tech_support``) may be omitted;
    the pipeline's imputers handle missing values.
    """

    model_config = ConfigDict(extra="forbid")

    tenure_months: float = Field(ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(gt=0, le=500, description="Current monthly bill (USD)")
    total_charges: float | None = Field(default=None, ge=0, description="Lifetime spend (USD)")
    num_support_calls: float = Field(ge=0, le=50, description="Support calls, last 12 months")
    avg_download_gb: float = Field(ge=0, description="Average monthly download volume (GB)")
    late_payments_12m: float = Field(ge=0, le=12, description="Late payments, last 12 months")
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
    tech_support: Literal["yes", "no"] | None = None
    paperless_billing: Literal["yes", "no"]

    def to_frame(self) -> pd.DataFrame:
        """Convert the customer to a single-row DataFrame for the pipeline."""
        return pd.DataFrame([self.model_dump()])


class BatchPredictionRequest(BaseModel):
    """Batch of customers to score."""

    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Contribution(BaseModel):
    """One feature's SHAP contribution (log-odds space)."""

    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    """Churn prediction for a single customer."""

    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_predicted: bool
    decision_threshold: float
    top_positive: list[Contribution] | None = None
    top_negative: list[Contribution] | None = None


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch of customers, in input order."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Service liveness and model readiness."""

    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Metadata of the currently served model."""

    model_class: str
    package_version: str
    trained_at: str
    decision_threshold: float
    metrics: dict[str, object]
