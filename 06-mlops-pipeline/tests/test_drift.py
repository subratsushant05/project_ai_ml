"""Tests for the drift-detection math against hand-computed values."""

import math

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.data import generate_loan_data
from mlops_pipeline.drift import (
    DriftReport,
    FeatureDrift,
    Severity,
    categorical_psi,
    classify_severity,
    detect_drift,
    ks_critical_value,
    ks_statistic,
    population_stability_index,
)


def test_psi_identical_samples_is_near_zero() -> None:
    """PSI of a distribution against itself is ~0."""
    sample = np.random.default_rng(0).normal(size=2000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_psi_detects_mean_shift() -> None:
    """A one-sigma mean shift produces a PSI well above the alert level."""
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, size=5000)
    cur = rng.normal(1.0, 1.0, size=5000)
    assert population_stability_index(ref, cur) > 0.25


def test_categorical_psi_hand_computed() -> None:
    """PSI for 50/50 vs 80/20 equals 0.3*ln(1.6) + (-0.3)*ln(0.4)."""
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 80 + ["b"] * 20)
    expected = 0.3 * math.log(0.8 / 0.5) + (-0.3) * math.log(0.2 / 0.5)
    assert categorical_psi(ref, cur) == pytest.approx(expected, rel=1e-9)


def test_ks_statistic_hand_computed() -> None:
    """For [1,2,3,4] vs [3,4,5,6] the max ECDF gap is exactly 0.5."""
    assert ks_statistic(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])) == 0.5
    same = np.arange(10.0)
    assert ks_statistic(same, same) == 0.0


def test_ks_critical_value_hand_computed() -> None:
    """c(0.05)=sqrt(-0.5*ln(0.025)); n=m=100 gives ~0.19207."""
    expected = math.sqrt(-0.5 * math.log(0.025)) * math.sqrt(200 / 10_000)
    assert ks_critical_value(100, 100, alpha=0.05) == pytest.approx(expected)
    assert ks_critical_value(100, 100, alpha=0.05) == pytest.approx(0.19207, abs=1e-4)


def test_severity_thresholds() -> None:
    """PSI drives severity; a KS rejection escalates NONE to WARN."""
    assert classify_severity(0.05, 0.10, 0.25) is Severity.NONE
    assert classify_severity(0.15, 0.10, 0.25) is Severity.WARN
    assert classify_severity(0.30, 0.10, 0.25) is Severity.ALERT
    assert classify_severity(0.05, 0.10, 0.25, ks_drifted=True) is Severity.WARN


def test_should_retrain_requires_min_alerts() -> None:
    """Retraining fires only at or above the configured alert count."""

    def report(n_alerts: int) -> DriftReport:
        features = [
            FeatureDrift(feature=f"f{i}", psi=0.9, severity=Severity.ALERT)
            for i in range(n_alerts)
        ] + [FeatureDrift(feature="calm", psi=0.0, severity=Severity.NONE)]
        return DriftReport(
            n_reference=100, n_current=100, features=features, retrain_min_alerts=2
        )

    assert not report(0).should_retrain()
    assert not report(1).should_retrain()
    assert report(2).should_retrain()
    assert report(3).should_retrain()


def test_detect_drift_end_to_end() -> None:
    """Stable windows raise no alerts; a strong shift is flagged."""
    reference = generate_loan_data(1500, seed=10)
    stable = generate_loan_data(800, seed=11)
    shifted = generate_loan_data(
        800, seed=12, shift={"income": (0.5, 0.0), "credit_score": (1.0, -80.0)}
    )

    calm = detect_drift(reference, stable)
    assert calm.alerts == []
    assert not calm.should_retrain()

    noisy = detect_drift(reference, shifted)
    flagged = {f.feature for f in noisy.alerts}
    assert {"income", "credit_score"} <= flagged
    assert noisy.should_retrain()
