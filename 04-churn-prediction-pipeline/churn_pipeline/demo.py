"""Quick end-to-end demo on a small dataset.

Usage::

    python -m churn_pipeline.demo

Trains a reduced pipeline (1500 rows, 5 Optuna trials), prints holdout
metrics, the business threshold, and a per-customer explanation, then scores
one example customer the same way the API would.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from churn_pipeline.config import Settings, setup_logging
from churn_pipeline.data import TARGET, generate_churn_data
from churn_pipeline.explain import explain_customer
from churn_pipeline.persistence import load_model
from churn_pipeline.schemas import Customer
from churn_pipeline.train import run_training

logger = logging.getLogger(__name__)

EXAMPLE_CUSTOMER = Customer(
    tenure_months=3,
    monthly_charges=95.5,
    total_charges=290.0,
    num_support_calls=4,
    avg_download_gb=60.2,
    late_payments_12m=2,
    contract_type="month-to-month",
    internet_service="fiber",
    payment_method="electronic_check",
    tech_support="no",
    paperless_billing="yes",
)


def main() -> None:
    """Run a fast train-predict-explain round trip."""
    setup_logging("INFO")
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            n_rows=1500,
            n_trials=5,
            cv_folds=3,
            artifacts_dir=Path(tmp),
        )
        logger.info("Training demo model (%d rows, %d trials)...", settings.n_rows,
                    settings.n_trials)
        metrics = run_training(settings)

        print("\n=== Holdout metrics ===")
        print(json.dumps(metrics["holdout"], indent=2))
        print("\n=== Business threshold ===")
        print(json.dumps(metrics["business_threshold"], indent=2))

        bundle = load_model(Path(tmp))
        frame = EXAMPLE_CUSTOMER.to_frame()
        probability = float(bundle.pipeline.predict_proba(frame)[0, 1])
        threshold = float(bundle.metadata["decision_threshold"])
        print("\n=== Example customer ===")
        print(f"churn probability: {probability:.1%} "
              f"(threshold {threshold:.2f} -> "
              f"{'SEND retention offer' if probability >= threshold else 'no action'})")
        print(explain_customer(bundle.pipeline, frame).to_text())

        holdout_df = generate_churn_data(n_rows=200, seed=7)
        rate = holdout_df[TARGET].mean()
        print(f"\nSanity check: churn rate on a fresh sample = {rate:.1%}")


if __name__ == "__main__":
    main()
