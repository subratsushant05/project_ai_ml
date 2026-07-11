"""CLI for forecasting a user-provided CSV.

Usage:
    python -m ts_forecast.run --csv data.csv [--horizon 28] [--folds 5]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ts_forecast.config import PipelineConfig
from ts_forecast.data import load_csv
from ts_forecast.pipeline import run_and_report


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CSV runner."""
    parser = argparse.ArgumentParser(
        prog="python -m ts_forecast.run",
        description="Forecast a daily time series from a CSV with columns date,value.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to input CSV")
    parser.add_argument("--horizon", type=int, default=28, help="Forecast days ahead")
    parser.add_argument("--folds", type=int, default=5, help="Backtest folds")
    parser.add_argument(
        "--min-train-days", type=int, default=365, help="Minimum first training window"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Interval miscoverage (0.1 = 90%% PI)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Artifact directory"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse args, load the CSV and run the full pipeline.

    Args:
        argv: Optional argument list (for testing); defaults to ``sys.argv``.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        horizon=args.horizon,
        n_folds=args.folds,
        min_train_days=args.min_train_days,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
    y = load_csv(args.csv)
    results = run_and_report({str(y.name): y}, config)
    result = results[str(y.name)]
    print(f"\nWinner: {result.winner}  (held-out coverage {result.coverage:.1%})")
    print(f"Artifacts written to {config.output_dir.resolve()}/")


if __name__ == "__main__":
    main()
