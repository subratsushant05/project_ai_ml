"""Seasonal naive baseline: repeat the last observed seasonal cycle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_forecast.models.base import Forecaster


class SeasonalNaive(Forecaster):
    """Forecast ``y[t] = y[t - season_length]``, tiling the last cycle.

    This is the standard baseline any real model must beat, and it is also
    the denominator model in the MASE metric.
    """

    def __init__(self, season_length: int = 7) -> None:
        """Initialize the baseline.

        Args:
            season_length: Seasonal period in days (7 = weekly).
        """
        self.season_length = season_length
        self._y: pd.Series | None = None

    @property
    def name(self) -> str:
        """Short model name."""
        return "seasonal_naive"

    def fit(self, y: pd.Series) -> SeasonalNaive:
        """Store the training series (no parameters to learn).

        Args:
            y: Daily training series.

        Returns:
            The fitted forecaster.
        """
        if len(y) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations, got {len(y)}"
            )
        self._y = y
        return self

    def predict(self, horizon: int) -> pd.Series:
        """Tile the last full seasonal cycle over the horizon.

        Args:
            horizon: Number of days to forecast.

        Returns:
            Forecast series of length ``horizon``.
        """
        y = self._check_fitted()
        last_cycle = y.to_numpy()[-self.season_length :]
        reps = int(np.ceil(horizon / self.season_length))
        values = np.tile(last_cycle, reps)[:horizon]
        return pd.Series(values, index=self._future_index(horizon), name=self.name)
