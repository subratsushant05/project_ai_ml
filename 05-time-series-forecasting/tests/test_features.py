"""Tests for feature engineering, focused on leakage prevention."""

from __future__ import annotations

import numpy as np

from tests.helpers import small_series
from ts_forecast.features import make_features, make_training_table


def test_lag_features_match_shifted_values() -> None:
    """lag_k at time t must equal y at t-k exactly."""
    y = small_series(100)
    feats = make_features(y, lags=(1, 7), roll_windows=(7,))
    for k in (1, 7):
        expected = y.shift(k)
        assert feats[f"lag_{k}"].dropna().equals(expected.dropna())


def test_features_do_not_leak_future() -> None:
    """Changing values at or after time t must not change features at t."""
    y = small_series(120)
    t = 80
    feats_before = make_features(y).iloc[:t].copy()

    corrupted = y.copy()
    corrupted.iloc[t:] = 1e9  # radically change the "future"
    feats_after = make_features(corrupted).iloc[:t]

    assert np.allclose(
        feats_before.to_numpy(dtype=float),
        feats_after.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_rolling_mean_excludes_current_value() -> None:
    """roll_mean_7 at t is the mean of y[t-7:t], never including y[t]."""
    y = small_series(60)
    feats = make_features(y, lags=(1,), roll_windows=(7,))
    t = 30
    expected = y.iloc[t - 7 : t].mean()
    assert feats["roll_mean_7"].iloc[t] == np.float64(expected)


def test_training_table_has_no_nans_and_aligns() -> None:
    """Warm-up rows are dropped and X/y stay index-aligned."""
    y = small_series(90)
    x_train, y_train = make_training_table(y)
    assert x_train.notna().all().all()
    assert x_train.index.equals(y_train.index)
    # Longest warm-up: lag_28 / 28-day rolling window over shifted values.
    assert len(x_train) == 90 - 28
