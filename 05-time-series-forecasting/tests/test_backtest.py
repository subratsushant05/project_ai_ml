"""Backtest fold construction, leakage prevention and model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.helpers import small_series
from ts_forecast.backtest import make_folds, rolling_origin_backtest, select_best
from ts_forecast.models.base import Forecaster


class SpyForecaster(Forecaster):
    """Records exactly what data it is trained on; predicts a constant."""

    def __init__(self, log: list[pd.Series], constant: float = 0.0) -> None:
        self.log = log
        self.constant = constant
        self._y: pd.Series | None = None

    @property
    def name(self) -> str:
        return "spy"

    def fit(self, y: pd.Series) -> SpyForecaster:
        self.log.append(y)
        self._y = y
        return self

    def predict(self, horizon: int) -> pd.Series:
        values = np.full(horizon, self.constant)
        return pd.Series(values, index=self._future_index(horizon), name=self.name)


class OracleForecaster(SpyForecaster):
    """Cheap 'good' model: repeats the last training value."""

    def predict(self, horizon: int) -> pd.Series:
        y = self._check_fitted()
        values = np.full(horizon, float(y.iloc[-1]))
        return pd.Series(values, index=self._future_index(horizon), name=self.name)


def test_fold_boundaries_are_contiguous_and_expanding() -> None:
    """Test windows tile the tail of the series; training always precedes them."""
    folds = make_folds(n_obs=400, horizon=30, n_folds=4, min_train=200)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_end == fold.test_start  # train strictly before test
        assert fold.test_end - fold.test_start == 30
    for prev, cur in zip(folds, folds[1:], strict=False):
        assert cur.test_start == prev.test_end  # non-overlapping, contiguous
        assert cur.train_end > prev.train_end  # expanding window
    assert folds[-1].test_end == 400  # last fold ends at the final observation


def test_backtest_never_shows_models_test_data() -> None:
    """Every fold's training series must end before its test window starts."""
    y = small_series(300)
    log: list[pd.Series] = []
    result = rolling_origin_backtest(
        y,
        {"spy": lambda: SpyForecaster(log)},
        horizon=14,
        n_folds=3,
        min_train=200,
    )
    assert len(log) == 3
    for fold, trained_on in zip(result.folds, log, strict=True):
        test_index = y.index[fold.test_start : fold.test_end]
        assert trained_on.index.max() < test_index.min()
        assert len(trained_on) == fold.train_end


def test_backtest_metrics_shape() -> None:
    """One metrics row per (fold, model) with all four metric columns."""
    y = small_series(300)
    result = rolling_origin_backtest(
        y,
        {"a": lambda: SpyForecaster([], 100.0), "b": lambda: OracleForecaster([])},
        horizon=14,
        n_folds=3,
        min_train=200,
    )
    assert len(result.metrics) == 3 * 2
    assert {"mae", "rmse", "smape", "mase"} <= set(result.metrics.columns)
    assert result.metrics[["mae", "rmse", "smape", "mase"]].notna().all().all()


def test_model_selection_picks_lower_mase_model() -> None:
    """A rigged contest: the oracle must beat a wildly-off constant model."""
    y = small_series(300)
    result = rolling_origin_backtest(
        y,
        {
            "bad": lambda: SpyForecaster([], constant=-500.0),
            "good": lambda: OracleForecaster([]),
        },
        horizon=14,
        n_folds=3,
        min_train=200,
    )
    assert select_best(result, metric="mase") == "good"
    means = result.mean_metric("mase")
    assert means["good"] < means["bad"]
