"""Common interface for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    """Abstract base class every model implements.

    The contract is deliberately small: ``fit`` on a daily series, then
    ``predict`` a fixed horizon. Models must be stateless before ``fit`` so a
    fresh instance per backtest fold guarantees no information carry-over.
    """

    _y: pd.Series | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable model name."""

    @abstractmethod
    def fit(self, y: pd.Series) -> Forecaster:
        """Fit the model on a daily series.

        Args:
            y: Series with a daily ``DatetimeIndex`` and float values.

        Returns:
            The fitted forecaster (for chaining).
        """

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Forecast ``horizon`` days past the end of the training series.

        Args:
            horizon: Number of daily steps to forecast.

        Returns:
            Series of length ``horizon`` indexed by future dates.
        """

    def _check_fitted(self) -> pd.Series:
        """Return the training series or raise if ``fit`` was never called."""
        if self._y is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit() first")
        return self._y

    def _future_index(self, horizon: int) -> pd.DatetimeIndex:
        """Build the daily index for the next ``horizon`` days after training."""
        y = self._check_fitted()
        start = y.index[-1] + pd.Timedelta(days=1)
        return pd.date_range(start=start, periods=horizon, freq="D")
