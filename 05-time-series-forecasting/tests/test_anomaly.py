"""Anomaly flagging on history with injected spikes."""

from __future__ import annotations

from tests.helpers import small_series
from ts_forecast.anomaly import flag_anomalies


def test_injected_spike_is_flagged() -> None:
    """A large one-day spike must be caught; the output aligns with input."""
    y = small_series(300)
    spike_pos = 200
    y.iloc[spike_pos] += 60.0  # ~30 sigma given noise scale of 2

    result = flag_anomalies(y, season_length=7, z_threshold=3.0)
    assert result.index.equals(y.index)
    assert bool(result["is_anomaly"].iloc[spike_pos])


def test_clean_series_flags_almost_nothing() -> None:
    """A well-behaved series should produce very few false alarms."""
    y = small_series(300, seed=5)
    result = flag_anomalies(y, season_length=7, z_threshold=3.5)
    assert result["is_anomaly"].mean() < 0.02
