"""Holt-Winters exponential smoothing wrapper around statsmodels."""

from __future__ import annotations

import logging
import warnings

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ts_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class ExponentialSmoothingForecaster(Forecaster):
    """Additive Holt-Winters with damped trend and weekly seasonality."""

    def __init__(self, season_length: int = 7, damped_trend: bool = True) -> None:
        """Initialize the model.

        Args:
            season_length: Seasonal period in days.
            damped_trend: Whether to damp the linear trend (safer long-range).
        """
        self.season_length = season_length
        self.damped_trend = damped_trend
        self._y: pd.Series | None = None
        self._fitted = None

    @property
    def name(self) -> str:
        """Short model name."""
        return "exp_smoothing"

    def fit(self, y: pd.Series) -> ExponentialSmoothingForecaster:
        """Fit Holt-Winters by maximum likelihood.

        Args:
            y: Daily training series (at least two seasonal cycles).

        Returns:
            The fitted forecaster.
        """
        if len(y) < 2 * self.season_length:
            raise ValueError("Need at least two full seasonal cycles to fit ETS")
        self._y = y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y.to_numpy(),
                trend="add",
                damped_trend=self.damped_trend,
                seasonal="add",
                seasonal_periods=self.season_length,
                initialization_method="estimated",
            )
            self._fitted = model.fit(optimized=True)
        logger.debug("ETS fitted on %d observations", len(y))
        return self

    def predict(self, horizon: int) -> pd.Series:
        """Forecast the next ``horizon`` days.

        Args:
            horizon: Number of days to forecast.

        Returns:
            Forecast series of length ``horizon``.
        """
        self._check_fitted()
        assert self._fitted is not None
        values = self._fitted.forecast(horizon)
        return pd.Series(values, index=self._future_index(horizon), name=self.name)
