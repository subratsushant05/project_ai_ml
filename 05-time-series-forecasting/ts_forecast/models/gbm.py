"""LightGBM forecaster with engineered features and recursive multi-step."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ts_forecast.features import DEFAULT_LAGS, DEFAULT_ROLL_WINDOWS, make_features
from ts_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class LightGBMForecaster(Forecaster):
    """Gradient-boosted trees on lag/rolling/calendar features.

    Multi-step forecasts are produced recursively: each predicted day is
    appended to the history so the next day's lag features can be computed.
    """

    def __init__(
        self,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 0,
    ) -> None:
        """Initialize the model.

        Args:
            lags: Lag orders used as features.
            roll_windows: Rolling mean/std window sizes.
            n_estimators: Number of boosting rounds.
            learning_rate: Boosting learning rate.
            num_leaves: Maximum leaves per tree.
            random_state: Seed for LightGBM (deterministic single-thread trees).
        """
        self.lags = lags
        self.roll_windows = roll_windows
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "verbosity": -1,
        }
        self._y: pd.Series | None = None
        self._model: LGBMRegressor | None = None
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        """Short model name."""
        return "lightgbm"

    def fit(self, y: pd.Series) -> LightGBMForecaster:
        """Build the feature table and train the regressor.

        Args:
            y: Daily training series.

        Returns:
            The fitted forecaster.
        """
        min_len = max(max(self.lags), max(self.roll_windows) + 1) + 30
        if len(y) < min_len:
            raise ValueError(f"Need at least {min_len} observations, got {len(y)}")
        self._y = y
        features = make_features(y, lags=self.lags, roll_windows=self.roll_windows)
        mask = features.notna().all(axis=1)
        x_train, y_train = features.loc[mask], y.loc[mask]
        self._columns = list(x_train.columns)
        self._model = LGBMRegressor(**self._params)
        self._model.fit(x_train, y_train)
        logger.debug("LightGBM fitted on %d rows, %d features", *x_train.shape)
        return self

    def predict(self, horizon: int) -> pd.Series:
        """Forecast recursively, one day at a time.

        Args:
            horizon: Number of days to forecast.

        Returns:
            Forecast series of length ``horizon``.
        """
        y = self._check_fitted()
        assert self._model is not None
        future_index = self._future_index(horizon)
        # Keep only the history the features can actually see, for speed.
        window = max(max(self.lags), max(self.roll_windows) + 1) + 1
        history = y.iloc[-window:].copy()

        preds: list[float] = []
        for ts in future_index:
            extended = pd.concat([history, pd.Series([np.nan], index=[ts])])
            row = make_features(
                extended, lags=self.lags, roll_windows=self.roll_windows
            ).iloc[[-1]][self._columns]
            yhat = float(self._model.predict(row)[0])
            preds.append(yhat)
            history.loc[ts] = yhat
            history = history.iloc[-window:]
        return pd.Series(preds, index=future_index, name=self.name)
