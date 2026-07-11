"""Metric math verified against hand-computed toy cases."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_forecast.metrics import mae, mase, rmse, smape


def test_mae_and_rmse_hand_computed() -> None:
    """MAE and RMSE match manual arithmetic on a tiny case."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([2.0, 2.0, 5.0])
    # abs errors: 1, 0, 2 -> MAE = 1.0 ; squared: 1, 0, 4 -> RMSE = sqrt(5/3)
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert rmse(y_true, y_pred) == pytest.approx((5.0 / 3.0) ** 0.5)


def test_smape_hand_computed() -> None:
    """sMAPE matches the 200*|t-p|/(|t|+|p|) definition."""
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([110.0, 180.0])
    # terms: 200*10/210 = 9.5238..., 200*20/380 = 10.5263...; mean = 10.0250...
    expected = (200 * 10 / 210 + 200 * 20 / 380) / 2
    assert smape(y_true, y_pred) == pytest.approx(expected)


def test_mase_hand_computed() -> None:
    """MASE matches a fully hand-worked seasonal example.

    Train = [10, 12, 14, 16], season = 2:
    naive errors |14-10|, |16-12| -> scale = 4.
    Forecast errors |20-18|, |22-24| -> MAE = 2. MASE = 2/4 = 0.5.
    """
    y_train = pd.Series([10.0, 12.0, 14.0, 16.0])
    y_true = pd.Series([20.0, 22.0])
    y_pred = pd.Series([18.0, 24.0])
    assert mase(y_true, y_pred, y_train, season_length=2) == pytest.approx(0.5)


def test_mase_rejects_constant_training_series() -> None:
    """Zero seasonal-naive error makes the scale undefined."""
    y_train = pd.Series([5.0] * 10)
    with pytest.raises(ValueError, match="zero"):
        mase(pd.Series([5.0]), pd.Series([5.0]), y_train, season_length=1)


def test_metrics_reject_length_mismatch() -> None:
    """Misaligned inputs raise instead of silently truncating."""
    with pytest.raises(ValueError, match="mismatch"):
        mae(pd.Series([1.0, 2.0]), pd.Series([1.0]))
