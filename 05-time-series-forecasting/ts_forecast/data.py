"""Synthetic data generation and CSV loading.

The generator produces realistic daily series combining trend, weekly and
yearly seasonality, holiday effects, noise, and occasional level shifts.
Everything is driven by a single seed so runs are fully reproducible.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_START = "2022-01-01"
_DEFAULT_PERIODS = 1095  # ~3 years of daily data

# (month, day) -> additive effect multiplier applied to holiday_scale.
_HOLIDAYS: dict[tuple[int, int], float] = {
    (1, 1): -0.8,  # New Year's Day: closed / low activity
    (7, 4): 0.6,  # Independence Day spike
    (11, 25): 1.2,  # Late-November shopping bump
    (12, 24): 1.5,  # Christmas Eve peak
    (12, 25): -1.0,  # Christmas Day trough
    (12, 31): 0.9,  # New Year's Eve
}


@dataclass(frozen=True)
class SeriesSpec:
    """Recipe for one synthetic series.

    Attributes:
        name: Series identifier used in reports and file names.
        base: Baseline level of the series.
        trend_per_day: Linear trend slope (units per day).
        weekly_amp: Amplitude of the weekly seasonal pattern.
        yearly_amp: Amplitude of the yearly seasonal pattern.
        noise_scale: Standard deviation of Gaussian noise.
        holiday_scale: Multiplier for holiday effects.
        n_level_shifts: Number of random permanent level shifts.
        weekly_pattern: Additive weights for Mon..Sun, scaled by weekly_amp.
    """

    name: str
    base: float
    trend_per_day: float
    weekly_amp: float
    yearly_amp: float
    noise_scale: float
    holiday_scale: float
    n_level_shifts: int
    weekly_pattern: tuple[float, ...] = field(
        default=(0.1, 0.0, -0.1, 0.0, 0.4, 1.0, 0.6)
    )


DEFAULT_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec("retail_sales", 200.0, 0.05, 30.0, 40.0, 8.0, 25.0, 2),
    SeriesSpec("web_traffic", 5000.0, 1.2, 600.0, 500.0, 150.0, 400.0, 1),
    SeriesSpec("energy_demand", 900.0, -0.02, 60.0, 180.0, 20.0, 30.0, 2),
)


def _holiday_effect(index: pd.DatetimeIndex, scale: float) -> np.ndarray:
    """Return the additive holiday effect for each date in ``index``."""
    effect = np.zeros(len(index))
    for i, ts in enumerate(index):
        mult = _HOLIDAYS.get((ts.month, ts.day))
        if mult is not None:
            effect[i] = mult * scale
    return effect


def generate_series(
    spec: SeriesSpec,
    seed: int,
    start: str = _DEFAULT_START,
    periods: int = _DEFAULT_PERIODS,
) -> pd.Series:
    """Generate one synthetic daily series from a spec.

    Args:
        spec: Recipe describing the series components.
        seed: Base seed; combined with the spec name for independence.
        start: First date (ISO string).
        periods: Number of daily observations.

    Returns:
        A float series with a daily ``DatetimeIndex`` named ``spec.name``.
    """
    # Derive a per-series seed so each series is independent yet deterministic.
    rng = np.random.default_rng([seed, zlib.crc32(spec.name.encode())])
    index = pd.date_range(start=start, periods=periods, freq="D")
    t = np.arange(periods, dtype=float)

    trend = spec.base + spec.trend_per_day * t
    weekly = spec.weekly_amp * np.array(
        [spec.weekly_pattern[d] for d in index.dayofweek]
    )
    yearly = spec.yearly_amp * np.sin(2.0 * np.pi * (index.dayofyear - 30) / 365.25)
    holidays = _holiday_effect(index, spec.holiday_scale)
    noise = rng.normal(0.0, spec.noise_scale, periods)

    shifts = np.zeros(periods)
    if spec.n_level_shifts > 0:
        shift_points = rng.integers(periods // 4, periods - 60, spec.n_level_shifts)
        for point in shift_points:
            shifts[point:] += rng.normal(0.0, spec.base * 0.05)

    values = trend + weekly + yearly + holidays + noise + shifts
    values = np.maximum(values, 0.0)
    logger.debug("Generated series %s with %d points", spec.name, periods)
    return pd.Series(values, index=index, name=spec.name)


def generate_datasets(
    seed: int = 42,
    start: str = _DEFAULT_START,
    periods: int = _DEFAULT_PERIODS,
) -> dict[str, pd.Series]:
    """Generate the default bundle of synthetic series.

    Args:
        seed: Master seed controlling all randomness.
        start: First date for every series.
        periods: Number of daily observations per series.

    Returns:
        Mapping of series name to daily series.
    """
    return {
        spec.name: generate_series(spec, seed=seed, start=start, periods=periods)
        for spec in DEFAULT_SPECS
    }


def load_csv(path: str | Path) -> pd.Series:
    """Load a user CSV with ``date`` and ``value`` columns into a daily series.

    Missing days are filled by linear interpolation so downstream models see a
    regular daily grid.

    Args:
        path: Path to a CSV file with columns ``date`` and ``value``.

    Returns:
        Daily float series named after the file stem.

    Raises:
        ValueError: If required columns are missing, dates are duplicated,
            or fewer than 60 observations are present.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    frame.columns = [c.strip().lower() for c in frame.columns]
    missing = {"date", "value"} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV must contain columns 'date' and 'value'; missing {missing}")

    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError("CSV contains duplicate dates")

    series = (
        frame.set_index("date")["value"].astype(float).sort_index().asfreq("D")
    )
    n_gaps = int(series.isna().sum())
    if n_gaps:
        logger.warning("Filling %d missing days by interpolation", n_gaps)
        series = series.interpolate(method="linear")
    if len(series) < 60:
        raise ValueError(f"Need at least 60 daily observations, got {len(series)}")
    series.name = path.stem
    series.index.name = None
    return series
