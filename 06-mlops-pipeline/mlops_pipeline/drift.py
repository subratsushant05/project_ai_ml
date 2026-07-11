"""Feature drift detection: PSI and two-sample KS tests.

Both statistics are implemented from scratch with plain numpy so the math
is transparent and unit-testable against hand-computed values.
"""

from __future__ import annotations

import logging
import math
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mlops_pipeline.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

_EPSILON = 1e-4


class Severity(StrEnum):
    """Drift severity level for a single feature."""

    NONE = "none"
    WARN = "warn"
    ALERT = "alert"


class FeatureDrift(BaseModel):
    """Drift measurements for one feature."""

    feature: str
    psi: float
    ks_statistic: float | None = None
    ks_critical: float | None = None
    severity: Severity


class DriftReport(BaseModel):
    """Aggregated drift results for a scoring window."""

    n_reference: int
    n_current: int
    features: list[FeatureDrift]
    retrain_min_alerts: int

    @property
    def alerts(self) -> list[FeatureDrift]:
        """Features whose drift severity is ALERT."""
        return [f for f in self.features if f.severity is Severity.ALERT]

    def should_retrain(self) -> bool:
        """Decide whether drift is severe enough to trigger retraining."""
        return len(self.alerts) >= self.retrain_min_alerts

    def save(self, path: Path) -> None:
        """Write the report as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Compute PSI between two numeric samples.

    Bin edges are taken from ``reference`` quantiles so each reference bin
    holds roughly equal mass; proportions are floored at a small epsilon to
    keep the logarithm finite for empty bins.

    Args:
        reference: Baseline sample (e.g. training data).
        current: Incoming sample to compare.
        bins: Number of quantile bins.

    Returns:
        The PSI value (0 = identical distributions).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_prop = np.maximum(ref_counts / max(len(reference), 1), _EPSILON)
    cur_prop = np.maximum(cur_counts / max(len(current), 1), _EPSILON)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Compute PSI over category frequencies.

    Args:
        reference: Baseline categorical sample.
        current: Incoming categorical sample.

    Returns:
        The PSI value over the union of observed categories.
    """
    categories = sorted(set(reference.unique()) | set(current.unique()), key=str)
    ref_prop = np.array(
        [max((reference == c).mean(), _EPSILON) for c in categories]
    )
    cur_prop = np.array([max((current == c).mean(), _EPSILON) for c in categories])
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF distance)."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    pooled = np.concatenate([reference, current])
    cdf_ref = np.searchsorted(reference, pooled, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, pooled, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def ks_critical_value(n_reference: int, n_current: int, alpha: float = 0.05) -> float:
    """Asymptotic KS rejection threshold ``c(alpha) * sqrt((n+m)/(n*m))``."""
    c_alpha = math.sqrt(-0.5 * math.log(alpha / 2.0))
    return c_alpha * math.sqrt((n_reference + n_current) / (n_reference * n_current))


def classify_severity(
    psi: float,
    psi_warn: float,
    psi_alert: float,
    ks_drifted: bool | None = None,
) -> Severity:
    """Map PSI (and optionally a KS rejection) to a severity level.

    PSI drives the level; a KS rejection escalates NONE to WARN so subtle
    shape changes that PSI's binning smooths over are still surfaced.
    """
    if psi >= psi_alert:
        return Severity.ALERT
    if psi >= psi_warn:
        return Severity.WARN
    if ks_drifted:
        return Severity.WARN
    return Severity.NONE


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    ks_alpha: float = 0.05,
    retrain_min_alerts: int = 2,
) -> DriftReport:
    """Run PSI + KS drift checks for every model feature.

    Args:
        reference: Baseline window (typically the training data).
        current: Incoming window of production traffic.
        psi_warn: PSI at or above this is a warning.
        psi_alert: PSI at or above this is an alert.
        ks_alpha: Significance level for the KS test on numeric features.
        retrain_min_alerts: Alerts needed for ``should_retrain()`` to fire.

    Returns:
        A :class:`DriftReport` with per-feature results.
    """
    results: list[FeatureDrift] = []
    for feature in NUMERIC_FEATURES:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur)
        stat = ks_statistic(ref, cur)
        critical = ks_critical_value(len(ref), len(cur), ks_alpha)
        severity = classify_severity(psi, psi_warn, psi_alert, ks_drifted=stat > critical)
        results.append(
            FeatureDrift(
                feature=feature,
                psi=round(psi, 6),
                ks_statistic=round(stat, 6),
                ks_critical=round(critical, 6),
                severity=severity,
            )
        )
    for feature in CATEGORICAL_FEATURES:
        psi = categorical_psi(reference[feature], current[feature])
        severity = classify_severity(psi, psi_warn, psi_alert)
        results.append(
            FeatureDrift(feature=feature, psi=round(psi, 6), severity=severity)
        )

    report = DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        features=results,
        retrain_min_alerts=retrain_min_alerts,
    )
    logger.info(
        "Drift check: %d/%d features on alert (retrain=%s)",
        len(report.alerts),
        len(results),
        report.should_retrain(),
    )
    return report
