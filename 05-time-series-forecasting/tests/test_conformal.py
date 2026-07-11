"""Conformal interval math and empirical coverage on synthetic noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_forecast.conformal import (
    calibrate_and_evaluate,
    conformal_half_width,
    empirical_coverage,
)


def test_conformal_quantile_finite_sample_correction() -> None:
    """Half-width equals the ceil((n+1)(1-alpha))/n-th order statistic.

    With residuals 1..10 and alpha=0.2: level = ceil(11*0.8)/10 = 0.9, and the
    'higher' quantile of {1..10} at 0.9 is 10 (index ceil is conservative).
    """
    residuals = pd.Series(np.arange(1.0, 11.0))
    interval = conformal_half_width(residuals, alpha=0.2)
    assert interval.half_width == pytest.approx(10.0)
    assert interval.n_calibration == 10


def test_conformal_rejects_tiny_calibration_set() -> None:
    """Too few residuals for the requested alpha raises."""
    with pytest.raises(ValueError, match="calibration"):
        conformal_half_width(pd.Series([1.0, 2.0]), alpha=0.1)


def test_conformal_coverage_on_iid_noise() -> None:
    """On exchangeable Gaussian noise, coverage lands near the nominal level."""
    rng = np.random.default_rng(7)
    alpha = 0.1
    n_cal, n_test = 500, 2000
    cal_residuals = pd.Series(rng.normal(0, 3.0, n_cal))
    interval = conformal_half_width(cal_residuals, alpha=alpha)

    truth = pd.Series(rng.normal(0, 3.0, n_test))
    forecasts = pd.Series(np.zeros(n_test))
    coverage = empirical_coverage(truth, forecasts, interval.half_width)
    assert coverage == pytest.approx(1 - alpha, abs=0.03)


def test_calibrate_and_evaluate_holds_out_last_fold() -> None:
    """Coverage is measured on the final fold only, using earlier folds' width."""
    rng = np.random.default_rng(11)
    folds = 4
    residuals = [pd.Series(rng.normal(0, 1.0, 200)) for _ in range(folds)]
    predictions = [pd.Series(np.zeros(200)) for _ in range(folds)]
    actuals = [predictions[i] + residuals[i] for i in range(folds)]
    interval, coverage = calibrate_and_evaluate(residuals, predictions, actuals, alpha=0.1)
    # Width from the first 3 folds must reproduce the reported coverage.
    expected = empirical_coverage(actuals[-1], predictions[-1], interval.half_width)
    assert coverage == pytest.approx(expected)
    assert 0.8 <= coverage <= 1.0


def test_interval_apply_produces_bounds() -> None:
    """apply() yields lower <= forecast <= upper everywhere."""
    interval = conformal_half_width(pd.Series(np.arange(1.0, 51.0)), alpha=0.1)
    forecast = pd.Series([10.0, 20.0, 30.0])
    frame = interval.apply(forecast)
    assert (frame["lower"] < frame["forecast"]).all()
    assert (frame["upper"] > frame["forecast"]).all()
    widths = (frame["upper"] - frame["lower"]).to_numpy()
    assert widths == pytest.approx(2 * interval.half_width)
