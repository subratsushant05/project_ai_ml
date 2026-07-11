"""Synthetic telco-style churn dataset generator.

The data is fully synthetic and generated from a hand-designed, nonlinear
churn mechanism so that models can learn genuine signal. The generating
process is documented in the README; no real customer data is involved.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "churned"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "avg_download_gb",
    "late_payments_12m",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "tech_support",
    "paperless_billing",
]

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
INTERNET_SERVICES = ["dsl", "fiber", "none"]
PAYMENT_METHODS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
YES_NO = ["yes", "no"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_churn_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic telco churn dataset.

    The churn probability is a nonlinear function of the features:

    * churn hazard decays exponentially with tenure,
    * month-to-month contracts churn far more, amplified at low tenure,
    * fiber customers are price-sensitive (charges interact with service),
    * support calls hurt quadratically once above two calls,
    * a "promo cliff" spikes churn around months 10-16 (discount expiry),
    * premium prices with low usage ("value mismatch") add risk,
    * tech support and autopay-style payment methods are protective.

    Roughly 4% of ``total_charges`` and 3% of ``tech_support`` values are
    masked to missing to exercise imputation.

    Args:
        n_rows: Number of customers to generate.
        seed: Random seed; identical seeds yield identical frames.

    Returns:
        DataFrame with ``NUMERIC_FEATURES``, ``CATEGORICAL_FEATURES``, a
        ``customer_id`` column, and the binary ``churned`` target.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n_rows).astype(float)
    contract = rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET_SERVICES, size=n_rows, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENT_METHODS, size=n_rows, p=[0.35, 0.15, 0.25, 0.25])
    tech_support = rng.choice(YES_NO, size=n_rows, p=[0.35, 0.65])
    paperless = rng.choice(YES_NO, size=n_rows, p=[0.60, 0.40])

    base_price = np.where(internet == "fiber", 75.0, np.where(internet == "dsl", 50.0, 22.0))
    monthly = base_price + rng.normal(0.0, 12.0, size=n_rows) + 8.0 * (tech_support == "yes")
    monthly = np.clip(monthly, 15.0, 140.0).round(2)
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(0.9, 1.1, size=n_rows)).round(2)

    support_calls = rng.poisson(lam=np.where(internet == "fiber", 1.6, 1.0)).astype(float)
    download = np.where(
        internet == "none", 0.0, rng.gamma(shape=2.0, scale=25.0, size=n_rows)
    ).round(1)
    late_payments = rng.binomial(6, 0.12, size=n_rows).astype(float)

    # Nonlinear churn mechanism (log-odds).
    logit = (
        -3.6
        + 1.9 * (contract == "month-to-month")
        + 0.5 * (contract == "one-year")
        + 2.8 * np.exp(-tenure / 14.0)
        + 0.045 * (monthly - 65.0) * (internet == "fiber")
        + 0.015 * (monthly - 65.0)
        + 0.30 * np.square(np.maximum(support_calls - 2.0, 0.0))
        + 0.50 * late_payments
        + 0.60 * (payment == "electronic_check")
        - 1.0 * (tech_support == "yes")
        - 1.1 * (internet == "none")
        + 1.2 * (contract == "month-to-month") * np.exp(-tenure / 10.0)
        # Promo cliff: churn spikes when a first-year discount expires.
        + 1.4 * ((tenure >= 10) & (tenure <= 16))
        # Value mismatch: paying a premium while barely using the service.
        + 0.9 * ((monthly > 80.0) & (download < 20.0))
    )
    churned = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n_rows)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": support_calls,
            "avg_download_gb": download,
            "late_payments_12m": late_payments,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "tech_support": tech_support,
            "paperless_billing": paperless,
            TARGET: churned,
        }
    )

    # Inject missing values to make the imputation stage meaningful.
    total_mask = rng.random(n_rows) < 0.04
    support_mask = rng.random(n_rows) < 0.03
    df.loc[total_mask, "total_charges"] = np.nan
    df.loc[support_mask, "tech_support"] = np.nan

    logger.info(
        "Generated %d rows | churn rate %.1f%% | %d missing total_charges, %d missing tech_support",
        n_rows,
        100.0 * df[TARGET].mean(),
        int(total_mask.sum()),
        int(support_mask.sum()),
    )
    return df
