"""Tests for the synthetic data generator."""

from __future__ import annotations

import pandas as pd

from churn_pipeline.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    generate_churn_data,
)


def test_generator_is_deterministic() -> None:
    """Identical seeds must produce identical frames."""
    a = generate_churn_data(n_rows=300, seed=7)
    b = generate_churn_data(n_rows=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generator_seed_changes_data() -> None:
    """Different seeds must produce different data."""
    a = generate_churn_data(n_rows=300, seed=7)
    b = generate_churn_data(n_rows=300, seed=8)
    assert not a.drop(columns=["customer_id"]).equals(b.drop(columns=["customer_id"]))


def test_generator_schema(small_df: pd.DataFrame) -> None:
    """Frame contains all documented columns with expected dtypes."""
    expected = {"customer_id", TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    assert set(small_df.columns) == expected
    assert len(small_df) == 1200
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(small_df[col]), col
    assert set(small_df[TARGET].unique()) <= {0, 1}


def test_generator_has_missing_values(small_df: pd.DataFrame) -> None:
    """Missingness is injected into the documented columns only."""
    assert small_df["total_charges"].isna().sum() > 0
    assert small_df["tech_support"].isna().sum() > 0
    assert small_df["monthly_charges"].isna().sum() == 0


def test_churn_rate_is_plausible(small_df: pd.DataFrame) -> None:
    """Churn rate should be imbalanced but learnable (10%-45%)."""
    rate = small_df[TARGET].mean()
    assert 0.10 < rate < 0.45
