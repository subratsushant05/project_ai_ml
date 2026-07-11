"""Shared test fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def small_series(n: int = 240, seed: int = 0, name: str = "test") -> pd.Series:
    """Build a small deterministic daily series with weekly seasonality.

    Args:
        n: Number of daily observations.
        seed: RNG seed.
        name: Series name.

    Returns:
        Daily series with trend + weekly pattern + noise.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    weekly = 10.0 * np.sin(2.0 * np.pi * index.dayofweek / 7.0)
    values = 100.0 + 0.05 * t + weekly + rng.normal(0, 2.0, n)
    return pd.Series(values, index=index, name=name)
