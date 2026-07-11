"""Retraining-trigger simulation: drift in, retrain out.

Run with ``python -m mlops_pipeline.simulate_drift``. The script scores
two incoming windows against the training reference: a stable window
(no action) and a shifted window that trips the drift detector, fires
``should_retrain()``, and kicks off a retrain whose winner is promoted
only if it beats the incumbent on identical holdout data.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from mlops_pipeline.config import Settings, get_settings, setup_logging
from mlops_pipeline.data import generate_loan_data
from mlops_pipeline.drift import DriftReport, detect_drift
from mlops_pipeline.registry import ModelRegistry
from mlops_pipeline.run_pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Simulated credit-cycle downturn. Covariate shift: incomes sag, credit
# scores slip, debt-to-income creeps up. Concept shift (MACRO_SHOCK):
# leverage becomes more predictive while credit score decouples.
DRIFT_SHIFT: dict[str, tuple[float, float]] = {
    "income": (0.82, 0.0),
    "credit_score": (1.0, -50.0),
    "debt_to_income": (1.0, 0.06),
}
MACRO_SHOCK = 2.0
RETRAIN_BATCH_ROWS = 3000


def _run_drift_check(
    name: str,
    reference: pd.DataFrame,
    window: pd.DataFrame,
    settings: Settings,
) -> DriftReport:
    """Detect drift for one window, print a table, and save the report."""
    report = detect_drift(
        reference,
        window,
        psi_warn=settings.psi_warn,
        psi_alert=settings.psi_alert,
        ks_alpha=settings.ks_alpha,
        retrain_min_alerts=settings.retrain_min_alerts,
    )
    report.save(settings.base_dir / settings.artifacts_dir / f"drift_report_{name}.json")

    print(f"\n--- Drift check: {name} window ({report.n_current} rows) ---")
    print(f"{'feature':<20} {'PSI':>8} {'KS stat':>9} {'KS crit':>9}  severity")
    for feat in report.features:
        ks_stat = f"{feat.ks_statistic:.4f}" if feat.ks_statistic is not None else "-"
        ks_crit = f"{feat.ks_critical:.4f}" if feat.ks_critical is not None else "-"
        print(
            f"{feat.feature:<20} {feat.psi:>8.4f} {ks_stat:>9} {ks_crit:>9}  "
            f"{feat.severity.value.upper()}"
        )
    print(
        f"alerts: {len(report.alerts)}/{len(report.features)} features "
        f"(retrain threshold: {report.retrain_min_alerts})"
    )
    print(f"should_retrain() -> {report.should_retrain()}")
    return report


def main() -> int:
    """CLI entry point for the drift-to-retrain walkthrough."""
    setup_logging(logging.WARNING)
    settings = get_settings()
    settings.ensure_dirs()

    registry = ModelRegistry(settings)
    if registry.production_version() is None:
        print("No production model found - running the training pipeline first...")
        run_pipeline(settings)

    info = registry.production_info()
    assert info is not None  # noqa: S101 - guaranteed by the block above
    print(
        f"production model: {info['model_type']} v{info['version']} "
        f"(training auc={info['metrics'].get('auc', float('nan')):.4f})"
    )

    reference = generate_loan_data(settings.n_samples, settings.random_seed)

    stable_window = generate_loan_data(1000, settings.random_seed + 100)
    _run_drift_check("stable", reference, stable_window, settings)

    shifted_window = generate_loan_data(
        1000, settings.random_seed + 200, shift=DRIFT_SHIFT, macro_shock=MACRO_SHOCK
    )
    shifted_report = _run_drift_check("shifted", reference, shifted_window, settings)

    if not shifted_report.should_retrain():
        print("\nNo retraining triggered - done.")
        return 0

    print("\n--- Retraining triggered: fitting on recent labelled traffic ---")
    retrain_batch = generate_loan_data(
        RETRAIN_BATCH_ROWS,
        settings.random_seed + 300,
        shift=DRIFT_SHIFT,
        macro_shock=MACRO_SHOCK,
    )
    combined = pd.concat([shifted_window, retrain_batch], ignore_index=True)
    result = run_pipeline(settings, df=combined)

    print(f"retrained best : {result.candidate.model_name}")
    print(f"candidate auc  : {result.candidate.metrics['auc']:.4f} (new holdout)")
    if result.incumbent_metrics is not None:
        print(f"incumbent auc  : {result.incumbent_metrics['auc']:.4f} (same holdout)")
    if result.promoted:
        print(f"outcome        : PROMOTED as version {result.promoted_version}")
    else:
        print("outcome        : REJECTED - incumbent stays in production")
    print(f"reasons        : {'; '.join(result.reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
