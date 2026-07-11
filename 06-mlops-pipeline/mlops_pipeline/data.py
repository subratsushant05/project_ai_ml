"""Deterministic synthetic loan-default dataset.

The generator produces a realistic-looking tabular dataset whose label
depends on the features through a fixed logistic relationship, so models
can learn a real signal and distribution shifts measurably change both
the features and the achievable performance.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

NUMERIC_FEATURES: list[str] = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "num_prior_defaults",
]
CATEGORICAL_FEATURES: list[str] = ["employment_status", "loan_term_months"]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "default"

EMPLOYMENT_STATUSES: list[str] = ["employed", "self_employed", "unemployed", "retired"]
LOAN_TERMS: list[int] = [12, 24, 36, 48, 60]

# (scale, offset) applied to numeric features to simulate covariate shift.
ShiftSpec = Mapping[str, tuple[float, float]]


def _default_probability(
    df: pd.DataFrame, rng: np.random.Generator, macro_shock: float = 0.0
) -> np.ndarray:
    """Compute per-row default probability from the generated features.

    Args:
        df: Frame containing all feature columns.
        rng: Source of latent noise, so labels are not perfectly separable.
        macro_shock: Concept-drift knob mimicking a credit-cycle downturn:
            debt-to-income becomes more predictive of default while credit
            score decouples (score inflation), so a model trained before
            the shock ranks applicants noticeably worse.

    Returns:
        Array of probabilities in ``(0, 1)``.
    """
    logit = (
        -3.0
        + (4.8 + macro_shock) * df["debt_to_income"].to_numpy()
        - (0.012 / (1.0 + macro_shock)) * (df["credit_score"].to_numpy() - 680.0)
        - 1.1 * np.log(np.maximum(df["income"].to_numpy(), 1.0) / 52_000.0)
        + 0.85 * np.minimum(df["num_prior_defaults"].to_numpy(), 3)
        + 0.7 * (df["employment_status"] == "unemployed").to_numpy()
        + 0.22 * (df["loan_amount"].to_numpy() / 25_000.0)
        + rng.normal(0.0, 0.25, size=len(df))
    )
    return 1.0 / (1.0 + np.exp(-logit))


def generate_loan_data(
    n_samples: int = 4000,
    seed: int = 42,
    shift: ShiftSpec | None = None,
    macro_shock: float = 0.0,
) -> pd.DataFrame:
    """Generate a labelled synthetic loan-application dataset.

    Args:
        n_samples: Number of rows to generate.
        seed: RNG seed; identical inputs always yield identical frames.
        shift: Optional ``{feature: (scale, offset)}`` map applied to numeric
            features *before* labels are drawn, so a covariate shift also
            shifts the outcome distribution (as it would in production).
        macro_shock: Concept-drift knob forwarded to the label model; a
            positive value makes debt-to-income more predictive of default.

    Returns:
        DataFrame with all feature columns plus the binary ``default`` target.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(21, 70, size=n_samples),
            "income": rng.lognormal(mean=10.85, sigma=0.45, size=n_samples).round(2),
            "loan_amount": rng.lognormal(mean=9.6, sigma=0.65, size=n_samples).round(2),
            "credit_score": np.clip(
                rng.normal(682, 72, size=n_samples), 300, 850
            ).astype(np.int64),
            "debt_to_income": rng.beta(2.0, 5.0, size=n_samples).round(4),
            "num_prior_defaults": np.minimum(rng.poisson(0.35, size=n_samples), 5),
            "employment_status": rng.choice(
                EMPLOYMENT_STATUSES, size=n_samples, p=[0.65, 0.15, 0.10, 0.10]
            ),
            "loan_term_months": rng.choice(
                LOAN_TERMS, size=n_samples, p=[0.10, 0.20, 0.30, 0.20, 0.20]
            ),
        }
    )

    if shift:
        for feature, (scale, offset) in shift.items():
            if feature not in NUMERIC_FEATURES:
                msg = f"Can only shift numeric features, got '{feature}'"
                raise ValueError(msg)
            df[feature] = df[feature] * scale + offset
        df["credit_score"] = np.clip(df["credit_score"], 300, 850).astype(np.int64)
        df["debt_to_income"] = np.clip(df["debt_to_income"], 0.0, 1.0)
        df["income"] = np.maximum(df["income"], 0.0)

    probabilities = _default_probability(df, rng, macro_shock=macro_shock)
    df[TARGET_COLUMN] = (rng.uniform(size=n_samples) < probabilities).astype(np.int64)
    return df
