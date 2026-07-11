"""Tests for the schema validation gate."""

import numpy as np

from mlops_pipeline.data import generate_loan_data
from mlops_pipeline.validation import loan_schema


def test_valid_frame_passes() -> None:
    """The generator's output satisfies the canonical schema."""
    report = loan_schema().validate_frame(generate_loan_data(300, seed=1))
    assert report.passed
    assert report.failures() == []
    assert report.n_rows == 300


def test_out_of_range_value_fails() -> None:
    """A credit score above 850 must fail the range check."""
    df = generate_loan_data(100, seed=2)
    df.loc[0, "credit_score"] = 9001
    report = loan_schema().validate_frame(df)
    assert not report.passed
    assert any(
        c.column == "credit_score" and c.check == "range" for c in report.failures()
    )


def test_unknown_category_fails() -> None:
    """A value outside the employment-status domain must fail."""
    df = generate_loan_data(100, seed=3)
    df.loc[0, "employment_status"] = "astronaut"
    report = loan_schema().validate_frame(df)
    assert any(c.check == "category_domain" for c in report.failures())


def test_null_threshold_enforced() -> None:
    """Nulls beyond the configured fraction must fail; zero nulls pass."""
    df = generate_loan_data(100, seed=4)
    df["income"] = df["income"].astype(float)
    df.loc[df.index[:10], "income"] = np.nan
    report = loan_schema().validate_frame(df)
    assert any(
        c.column == "income" and c.check == "null_fraction"
        for c in report.failures()
    )


def test_missing_column_fails() -> None:
    """Dropping a required column must fail the presence check."""
    df = generate_loan_data(100, seed=5).drop(columns=["debt_to_income"])
    report = loan_schema().validate_frame(df)
    assert any(
        c.column == "debt_to_income" and c.check == "presence"
        for c in report.failures()
    )


def test_schema_hash_is_stable_and_sensitive() -> None:
    """The hash is deterministic and changes when the schema changes."""
    a, b = loan_schema(), loan_schema()
    assert a.schema_hash() == b.schema_hash()
    b.columns[0].max_value = 123
    assert a.schema_hash() != b.schema_hash()
