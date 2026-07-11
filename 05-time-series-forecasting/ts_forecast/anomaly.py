"""Residual-based anomaly flagging on historical data.

A point is anomalous when its seasonal-difference residual sits far outside
the local residual distribution. Using the seasonal difference ``y[t] -
y[t-s]`` removes trend and day-of-week structure without fitting a model, and
a rolling robust z-score adapts to slowly changing noise levels.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def flag_anomalies(
    y: pd.Series,
    season_length: int = 7,
    window: int = 56,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag historical anomalies via rolling robust z-scores.

    Args:
        y: Daily series.
        season_length: Seasonal period used for differencing.
        window: Rolling window (days) for the robust location/scale estimate.
        z_threshold: |z| above which a point is flagged.

    Returns:
        DataFrame indexed like ``y`` with columns ``value``, ``zscore`` and
        ``is_anomaly``. Warm-up rows have ``zscore = NaN`` and are not flagged.
    """
    resid = y - y.shift(season_length)
    med = resid.rolling(window, min_periods=window // 2).median()
    # 1.4826 * MAD is a consistent robust estimate of the standard deviation.
    mad = (resid - med).abs().rolling(window, min_periods=window // 2).median()
    scale = 1.4826 * mad
    z = (resid - med) / scale.replace(0.0, np.nan)

    flags = z.abs() > z_threshold
    flags &= z.notna()
    n_flagged = int(flags.sum())
    logger.info("Flagged %d/%d points as anomalies (|z| > %.1f)", n_flagged, len(y), z_threshold)
    return pd.DataFrame({"value": y, "zscore": z, "is_anomaly": flags})
