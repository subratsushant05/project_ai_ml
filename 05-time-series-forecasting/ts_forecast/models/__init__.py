"""Forecasting models behind a common :class:`Forecaster` interface."""

from __future__ import annotations

from collections.abc import Callable

from ts_forecast.models.base import Forecaster
from ts_forecast.models.ets import ExponentialSmoothingForecaster
from ts_forecast.models.gbm import LightGBMForecaster
from ts_forecast.models.naive import SeasonalNaive
from ts_forecast.models.sarima import SARIMAForecaster

ForecasterFactory = Callable[[], Forecaster]


def default_model_factories(season_length: int = 7) -> dict[str, ForecasterFactory]:
    """Return factories for the standard model zoo.

    Factories (rather than instances) are used so every backtest fold gets a
    fresh, unfitted model.

    Args:
        season_length: Dominant seasonal period in days.

    Returns:
        Mapping of model name to zero-argument factory.
    """
    return {
        "seasonal_naive": lambda: SeasonalNaive(season_length=season_length),
        "exp_smoothing": lambda: ExponentialSmoothingForecaster(
            season_length=season_length
        ),
        "sarima": lambda: SARIMAForecaster(season_length=season_length),
        "lightgbm": lambda: LightGBMForecaster(),
    }


__all__ = [
    "ExponentialSmoothingForecaster",
    "Forecaster",
    "ForecasterFactory",
    "LightGBMForecaster",
    "SARIMAForecaster",
    "SeasonalNaive",
    "default_model_factories",
]
