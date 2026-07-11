"""Feature engineering for tree-based forecasting.

All lag and rolling features are built strictly from past values (rolling
windows are applied to ``y.shift(1)``), so a row at time ``t`` never sees
``y[t]`` or anything later. This is verified by tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 28)
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (7, 28)


def make_features(
    y: pd.Series,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> pd.DataFrame:
    """Build a leakage-free feature matrix for a daily series.

    Args:
        y: Daily series with a ``DatetimeIndex``.
        lags: Lag orders; feature ``lag_k`` at time t equals ``y[t - k]``.
        roll_windows: Window sizes for rolling mean/std over past values only
            (the window ends at ``t - 1``, never including ``y[t]``).

    Returns:
        DataFrame aligned with ``y`` containing lag, rolling and calendar
        features. Early rows contain NaNs where history is insufficient.
    """
    feats = pd.DataFrame(index=y.index)
    for k in lags:
        feats[f"lag_{k}"] = y.shift(k)

    past = y.shift(1)  # exclude the current value from every rolling window
    for w in roll_windows:
        feats[f"roll_mean_{w}"] = past.rolling(w).mean()
        feats[f"roll_std_{w}"] = past.rolling(w).std()

    feats = feats.join(calendar_features(y.index))
    return feats


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build deterministic calendar features for a datetime index.

    Args:
        index: Daily datetime index.

    Returns:
        DataFrame with day-of-week, month and smooth annual-cycle encodings.
    """
    doy = index.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "dayofweek": index.dayofweek.to_numpy(dtype=np.int16),
            "month": index.month.to_numpy(dtype=np.int16),
            "is_weekend": (index.dayofweek >= 5).astype(np.int16),
            "is_month_start": index.is_month_start.astype(np.int16),
            "is_month_end": index.is_month_end.astype(np.int16),
            "doy_sin": np.sin(2.0 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2.0 * np.pi * doy / 365.25),
        },
        index=index,
    )


def make_training_table(
    y: pd.Series,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build an (X, y) training table with warm-up NaN rows dropped.

    Args:
        y: Daily target series.
        lags: Lag orders passed to :func:`make_features`.
        roll_windows: Rolling windows passed to :func:`make_features`.

    Returns:
        Tuple of feature matrix and aligned target with no missing values.
    """
    features = make_features(y, lags=lags, roll_windows=roll_windows)
    mask = features.notna().all(axis=1)
    return features.loc[mask], y.loc[mask]
