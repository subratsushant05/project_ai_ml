"""SARIMA wrapper around statsmodels SARIMAX with a small fixed order."""

from __future__ import annotations

import logging
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ts_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class SARIMAForecaster(Forecaster):
    """SARIMA(1,1,1)(1,0,1,s) — a compact, robust default for daily data.

    A fixed small order keeps backtests fast and avoids overfitting the
    order-selection step to a single series; the weekly seasonal AR/MA terms
    capture the dominant day-of-week structure after first differencing.
    """

    def __init__(
        self,
        season_length: int = 7,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int] | None = None,
        maxiter: int = 100,
    ) -> None:
        """Initialize the model.

        Args:
            season_length: Seasonal period in days.
            order: Non-seasonal (p, d, q) order.
            seasonal_order: Seasonal (P, D, Q); defaults to (1, 0, 1).
            maxiter: Optimizer iteration cap (lower = faster, slightly rougher).
        """
        self.season_length = season_length
        self.order = order
        self.seasonal_order = (
            *(seasonal_order or (1, 0, 1)),
            season_length,
        )
        self.maxiter = maxiter
        self._y: pd.Series | None = None
        self._fitted = None

    @property
    def name(self) -> str:
        """Short model name."""
        return "sarima"

    def fit(self, y: pd.Series) -> SARIMAForecaster:
        """Fit SARIMAX by maximum likelihood.

        Args:
            y: Daily training series.

        Returns:
            The fitted forecaster.
        """
        if len(y) < 3 * self.season_length:
            raise ValueError("Need at least three seasonal cycles to fit SARIMA")
        self._y = y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y.to_numpy(),
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fitted = model.fit(disp=False, maxiter=self.maxiter)
        logger.debug("SARIMA fitted on %d observations", len(y))
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
        values = self._fitted.forecast(steps=horizon)
        return pd.Series(values, index=self._future_index(horizon), name=self.name)
