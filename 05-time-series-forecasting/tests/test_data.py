"""Tests for the synthetic generator and CSV loader."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_forecast.data import DEFAULT_SPECS, generate_datasets, generate_series, load_csv


def test_generator_is_deterministic() -> None:
    """Identical seeds must produce byte-identical series."""
    a = generate_datasets(seed=42, periods=200)
    b = generate_datasets(seed=42, periods=200)
    assert set(a) == set(b)
    for name in a:
        pd.testing.assert_series_equal(a[name], b[name])


def test_generator_seed_changes_output() -> None:
    """Different seeds must produce different noise realizations."""
    a = generate_series(DEFAULT_SPECS[0], seed=1, periods=200)
    b = generate_series(DEFAULT_SPECS[0], seed=2, periods=200)
    assert not a.equals(b)


def test_generator_shape_and_properties() -> None:
    """Series have the right length, daily frequency and no NaNs."""
    series = generate_datasets(seed=0, periods=365)
    assert len(series) == 3
    for y in series.values():
        assert len(y) == 365
        assert pd.infer_freq(y.index) == "D"
        assert y.notna().all()
        assert (y >= 0).all()


def test_load_csv_roundtrip(tmp_path) -> None:
    """A well-formed CSV loads into an equivalent daily series."""
    y = generate_series(DEFAULT_SPECS[0], seed=3, periods=120)
    path = tmp_path / "sales.csv"
    pd.DataFrame({"date": y.index, "value": y.values}).to_csv(path, index=False)
    loaded = load_csv(path)
    assert loaded.name == "sales"
    pd.testing.assert_index_equal(loaded.index, y.index)
    assert (loaded.to_numpy() == pytest.approx(y.to_numpy()))


def test_load_csv_rejects_bad_columns(tmp_path) -> None:
    """Missing required columns raises a clear error."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"day": ["2023-01-01"], "amount": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="date"):
        load_csv(path)
