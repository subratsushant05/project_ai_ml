"""Demo entry point: run the full pipeline on synthetic data.

Usage:
    python -m ts_forecast.demo
"""

from __future__ import annotations

import logging

from ts_forecast.config import PipelineConfig
from ts_forecast.data import generate_datasets
from ts_forecast.pipeline import run_and_report


def main() -> None:
    """Generate synthetic series, run the pipeline, and print a summary."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    config = PipelineConfig()
    series = generate_datasets(seed=config.seed)
    results = run_and_report(series, config)

    print("\n=== Backtest summary (mean across folds) ===")
    for name, result in results.items():
        table = result.backtest.mean_metric("mase").round(3)
        print(f"\n{name} — winner: {result.winner}, "
              f"held-out {1 - config.alpha:.0%} interval coverage: {result.coverage:.1%}")
        for model, value in table.items():
            marker = " <-- selected" if model == result.winner else ""
            print(f"  {model:<16} MASE={value:.3f}{marker}")
    print(f"\nArtifacts written to {config.output_dir.resolve()}/")


if __name__ == "__main__":
    main()
