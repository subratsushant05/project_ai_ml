"""Tests for the synthetic data generator."""

import pandas as pd

from mlops_pipeline.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_loan_data


def test_generator_is_deterministic() -> None:
    """Identical seeds must produce identical frames."""
    a = generate_loan_data(200, seed=123)
    b = generate_loan_data(200, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_label_rate() -> None:
    """Output has all feature columns and a plausible default rate."""
    df = generate_loan_data(1000, seed=5)
    assert list(df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert len(df) == 1000
    assert 0.10 < df[TARGET_COLUMN].mean() < 0.50


def test_shift_moves_feature_distribution() -> None:
    """A (scale, offset) shift changes the shifted column only."""
    base = generate_loan_data(500, seed=9)
    shifted = generate_loan_data(500, seed=9, shift={"income": (0.5, 0.0)})
    assert shifted["income"].mean() < base["income"].mean() * 0.6
    pd.testing.assert_series_equal(base["age"], shifted["age"])
