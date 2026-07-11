"""Every forecaster produces correct-length, finite, date-aligned output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import small_series
from ts_forecast.models import (
    ExponentialSmoothingForecaster,
    Forecaster,
    LightGBMForecaster,
    SARIMAForecaster,
    SeasonalNaive,
)

HORIZON = 14


@pytest.fixture(scope="module")
def y() -> pd.Series:
    """A small series shared by all model tests (module-scoped for speed)."""
    return small_series(240)


def _check_forecast(model: Forecaster, y: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Fit, predict and run the shared output contract assertions."""
    forecast = model.fit(y).predict(horizon)
    assert len(forecast) == horizon
    assert np.isfinite(forecast.to_numpy(dtype=float)).all()
    assert forecast.index[0] == y.index[-1] + pd.Timedelta(days=1)
    assert (forecast.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    return forecast


def test_seasonal_naive_output_and_tiling(y: pd.Series) -> None:
    """Baseline output is valid and exactly repeats the last weekly cycle."""
    forecast = _check_forecast(SeasonalNaive(season_length=7), y)
    expected = np.tile(y.to_numpy()[-7:], 2)[:HORIZON]
    assert forecast.to_numpy() == pytest.approx(expected)


def test_exponential_smoothing_output(y: pd.Series) -> None:
    """Holt-Winters output is valid and in a sane range."""
    forecast = _check_forecast(ExponentialSmoothingForecaster(season_length=7), y)
    assert forecast.mean() == pytest.approx(y.iloc[-28:].mean(), rel=0.5)


def test_sarima_output(y: pd.Series) -> None:
    """SARIMA output is valid and in a sane range."""
    model = SARIMAForecaster(season_length=7, maxiter=50)
    forecast = _check_forecast(model, y)
    assert forecast.mean() == pytest.approx(y.iloc[-28:].mean(), rel=0.5)


def test_lightgbm_output(y: pd.Series) -> None:
    """Recursive LightGBM output is valid and in a sane range."""
    forecast = _check_forecast(LightGBMForecaster(n_estimators=60), y)
    assert forecast.mean() == pytest.approx(y.iloc[-28:].mean(), rel=0.5)


def test_predict_before_fit_raises() -> None:
    """Calling predict on an unfitted model raises a clear error."""
    with pytest.raises(RuntimeError, match="not fitted"):
        SeasonalNaive().predict(7)
