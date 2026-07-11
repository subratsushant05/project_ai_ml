"""Split-conformal prediction intervals from backtest residuals.

The classic split-conformal recipe: take absolute residuals from a
calibration set the model never trained on, use their finite-sample-corrected
``(1 - alpha)`` quantile as a symmetric interval half-width. Under
exchangeability the interval covers with probability at least ``1 - alpha``.
Time series are not perfectly exchangeable, so we also *measure* coverage on
held-out folds and report it honestly instead of assuming the guarantee.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformalInterval:
    """A symmetric conformal interval around point forecasts.

    Attributes:
        half_width: Interval half-width in series units.
        alpha: Nominal miscoverage rate (0.1 -> 90% target coverage).
        n_calibration: Number of calibration residuals used.
    """

    half_width: float
    alpha: float
    n_calibration: int

    def apply(self, forecast: pd.Series) -> pd.DataFrame:
        """Attach lower/upper bounds to a point forecast.

        Args:
            forecast: Point forecast series.

        Returns:
            DataFrame with columns ``forecast``, ``lower``, ``upper``.
        """
        return pd.DataFrame(
            {
                "forecast": forecast,
                "lower": forecast - self.half_width,
                "upper": forecast + self.half_width,
            },
            index=forecast.index,
        )


def conformal_half_width(residuals: pd.Series, alpha: float) -> ConformalInterval:
    """Compute the split-conformal half-width from calibration residuals.

    Uses the finite-sample quantile level ``ceil((n + 1)(1 - alpha)) / n``
    applied to absolute residuals, which restores the coverage guarantee lost
    by plugging in the empirical quantile directly.

    Args:
        residuals: Calibration residuals (actual - forecast).
        alpha: Nominal miscoverage rate in (0, 1).

    Returns:
        A :class:`ConformalInterval`.

    Raises:
        ValueError: If there are too few residuals for the requested alpha.
    """
    abs_res = np.abs(residuals.dropna().to_numpy(dtype=float))
    n = len(abs_res)
    if n == 0 or (n + 1) * (1 - alpha) > n:
        raise ValueError(
            f"Need at least {math.ceil(1 / alpha)} calibration residuals for "
            f"alpha={alpha}, got {n}"
        )
    level = math.ceil((n + 1) * (1 - alpha)) / n
    half_width = float(np.quantile(abs_res, level, method="higher"))
    logger.debug("Conformal half-width %.4f from %d residuals", half_width, n)
    return ConformalInterval(half_width=half_width, alpha=alpha, n_calibration=n)


def empirical_coverage(
    actuals: pd.Series, forecasts: pd.Series, half_width: float
) -> float:
    """Fraction of actuals falling inside ``forecast +/- half_width``.

    Args:
        actuals: Observed values.
        forecasts: Point forecasts aligned with ``actuals``.
        half_width: Interval half-width.

    Returns:
        Coverage in [0, 1].
    """
    errors = np.abs(actuals.to_numpy(dtype=float) - forecasts.to_numpy(dtype=float))
    return float(np.mean(errors <= half_width))


def calibrate_and_evaluate(
    residuals_by_fold: list[pd.Series],
    predictions_by_fold: list[pd.Series],
    actuals_by_fold: list[pd.Series],
    alpha: float,
) -> tuple[ConformalInterval, float]:
    """Calibrate on all folds except the last; measure coverage on the last.

    This mimics deployment: the interval width is chosen using only residuals
    available *before* the final evaluation window, so the reported coverage
    is an honest out-of-sample estimate.

    Args:
        residuals_by_fold: Per-fold residual series (chronological order).
        predictions_by_fold: Per-fold point forecasts (same order).
        actuals_by_fold: Per-fold actuals (same order).
        alpha: Nominal miscoverage rate.

    Returns:
        Tuple of the calibrated interval and its empirical coverage on the
        held-out final fold.

    Raises:
        ValueError: If fewer than two folds are provided.
    """
    if len(residuals_by_fold) < 2:
        raise ValueError("Need at least two folds: calibration + evaluation")
    calibration = pd.concat(residuals_by_fold[:-1])
    interval = conformal_half_width(calibration, alpha)
    coverage = empirical_coverage(
        actuals_by_fold[-1], predictions_by_fold[-1], interval.half_width
    )
    logger.info(
        "Conformal interval: +/-%.3f (target %.0f%%, held-out coverage %.1f%%)",
        interval.half_width,
        100 * (1 - alpha),
        100 * coverage,
    )
    return interval, coverage
