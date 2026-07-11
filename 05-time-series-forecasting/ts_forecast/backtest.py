"""Rolling-origin (expanding window) backtesting and model selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ts_forecast.metrics import compute_all
from ts_forecast.models import ForecasterFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One train/test split of a rolling-origin backtest.

    Attributes:
        fold_id: Zero-based fold number (0 = earliest origin).
        train_end: Index of the last training observation (exclusive slice end).
        test_start: Index of the first test observation.
        test_end: Index one past the last test observation.
    """

    fold_id: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes:
        metrics: One row per (fold, model) with mae/rmse/smape/mase columns.
        predictions: model name -> forecast series concatenated across folds.
        actuals: Actual values over all test windows (concatenated).
        residuals: model name -> (actual - forecast) across all folds.
        folds: The fold boundary definitions used.
    """

    metrics: pd.DataFrame
    predictions: dict[str, pd.Series]
    actuals: pd.Series
    residuals: dict[str, pd.Series] = field(default_factory=dict)
    folds: list[Fold] = field(default_factory=list)

    def mean_metric(self, metric: str = "mase") -> pd.Series:
        """Average a metric across folds, per model.

        Args:
            metric: Column name in :attr:`metrics`.

        Returns:
            Series indexed by model name, sorted ascending.
        """
        return self.metrics.groupby("model")[metric].mean().sort_values()


def make_folds(n_obs: int, horizon: int, n_folds: int, min_train: int) -> list[Fold]:
    """Compute expanding-window fold boundaries.

    The last fold's test window ends at the final observation; earlier folds
    step back by ``horizon`` each time. Test windows never overlap and every
    training window ends strictly before its test window begins.

    Args:
        n_obs: Total number of observations.
        horizon: Test window length per fold.
        n_folds: Number of folds requested.
        min_train: Minimum size of the first (smallest) training window.

    Returns:
        Folds ordered from earliest origin to latest.

    Raises:
        ValueError: If the series is too short for the requested layout.
    """
    needed = min_train + n_folds * horizon
    if n_obs < needed:
        raise ValueError(
            f"Series has {n_obs} obs but {needed} are needed "
            f"(min_train={min_train}, n_folds={n_folds}, horizon={horizon})"
        )
    folds = []
    for i in range(n_folds):
        test_end = n_obs - (n_folds - 1 - i) * horizon
        test_start = test_end - horizon
        folds.append(Fold(i, train_end=test_start, test_start=test_start, test_end=test_end))
    return folds


def rolling_origin_backtest(
    y: pd.Series,
    factories: dict[str, ForecasterFactory],
    horizon: int,
    n_folds: int,
    min_train: int,
    season_length: int = 7,
) -> BacktestResult:
    """Run an expanding-window backtest for several models.

    Each fold trains a *fresh* model instance on ``y[:train_end]`` and
    evaluates on the following ``horizon`` days, so no future information can
    leak into training.

    Args:
        y: Full daily series.
        factories: Model name -> factory producing an unfitted forecaster.
        horizon: Forecast length per fold.
        n_folds: Number of folds.
        min_train: Minimum first training window length.
        season_length: Season used for MASE scaling.

    Returns:
        A :class:`BacktestResult` with per-fold metrics and pooled residuals.
    """
    folds = make_folds(len(y), horizon, n_folds, min_train)
    rows: list[dict[str, object]] = []
    preds: dict[str, list[pd.Series]] = {name: [] for name in factories}
    actual_chunks: list[pd.Series] = []

    for fold in folds:
        y_train = y.iloc[: fold.train_end]
        y_test = y.iloc[fold.test_start : fold.test_end]
        actual_chunks.append(y_test)
        for model_name, factory in factories.items():
            model = factory().fit(y_train)
            forecast = model.predict(horizon)
            forecast.index = y_test.index  # align defensively
            preds[model_name].append(forecast)
            metric_values = compute_all(y_test, forecast, y_train, season_length)
            rows.append({"fold": fold.fold_id, "model": model_name, **metric_values})
            logger.info(
                "fold=%d model=%s mase=%.3f", fold.fold_id, model_name, metric_values["mase"]
            )

    actuals = pd.concat(actual_chunks)
    predictions = {name: pd.concat(chunks) for name, chunks in preds.items()}
    residuals = {name: actuals - fc for name, fc in predictions.items()}
    return BacktestResult(
        metrics=pd.DataFrame(rows),
        predictions=predictions,
        actuals=actuals,
        residuals=residuals,
        folds=folds,
    )


def select_best(result: BacktestResult, metric: str = "mase") -> str:
    """Pick the model with the lowest mean metric across folds.

    Args:
        result: Backtest output.
        metric: Metric column to rank by (default MASE).

    Returns:
        Name of the winning model.
    """
    ranking = result.mean_metric(metric)
    winner = str(ranking.index[0])
    logger.info("Model selection by %s: %s (%.3f)", metric, winner, ranking.iloc[0])
    return winner
