"""Forecast accuracy metrics: MAE, RMSE, sMAPE and MASE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_arrays(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes and convert to float arrays."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    return y_true.to_numpy(dtype=float), y_pred.to_numpy(dtype=float)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error.

    Args:
        y_true: Observed values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        MAE in the units of the series.
    """
    t, p = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error.

    Args:
        y_true: Observed values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        RMSE in the units of the series.
    """
    t, p = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric mean absolute percentage error, in percent.

    Uses the standard definition ``200 * |t - p| / (|t| + |p|)`` with terms
    where both values are zero counted as zero error.

    Args:
        y_true: Observed values.
        y_pred: Forecast values, aligned with ``y_true``.

    Returns:
        sMAPE in percent (0 to 200).
    """
    t, p = _to_arrays(y_true, y_pred)
    denom = np.abs(t) + np.abs(p)
    ratio = np.where(denom == 0.0, 0.0, 2.0 * np.abs(t - p) / np.where(denom == 0, 1, denom))
    return float(100.0 * np.mean(ratio))


def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    season_length: int = 1,
) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    The forecast MAE is scaled by the in-sample MAE of a seasonal-naive
    forecast on the training data, making the metric unit-free and comparable
    across series. Values below 1 beat the naive baseline.

    Args:
        y_true: Observed test values.
        y_pred: Forecast values, aligned with ``y_true``.
        y_train: Training series used to compute the naive scale.
        season_length: Season for the naive scale (1 = plain naive).

    Returns:
        MASE (dimensionless).

    Raises:
        ValueError: If the training series is too short or has zero
            seasonal-naive error (scale undefined).
    """
    if len(y_train) <= season_length:
        raise ValueError("Training series shorter than one season; MASE undefined")
    train = y_train.to_numpy(dtype=float)
    scale = float(np.mean(np.abs(train[season_length:] - train[:-season_length])))
    if scale == 0.0:
        raise ValueError("Seasonal-naive training error is zero; MASE undefined")
    return mae(y_true, y_pred) / scale


def compute_all(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    season_length: int = 7,
) -> dict[str, float]:
    """Compute the full metric set for one forecast.

    Args:
        y_true: Observed test values.
        y_pred: Forecast values.
        y_train: Training series (for MASE scaling).
        season_length: Season used by the MASE denominator.

    Returns:
        Dict with keys ``mae``, ``rmse``, ``smape``, ``mase``.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, season_length=season_length),
    }
