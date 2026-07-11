"""End-to-end pipeline: backtest, select, calibrate intervals, report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ts_forecast.anomaly import flag_anomalies
from ts_forecast.backtest import BacktestResult, rolling_origin_backtest, select_best
from ts_forecast.config import PipelineConfig
from ts_forecast.conformal import ConformalInterval, calibrate_and_evaluate
from ts_forecast.models import default_model_factories
from ts_forecast.report import (
    build_html_report,
    plot_backtest_errors,
    plot_forecast,
    write_metrics_json,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline produces for one series.

    Attributes:
        series_name: Name of the input series.
        backtest: Full backtest result.
        winner: Model selected by mean MASE.
        forecast: Final forecast with ``forecast``/``lower``/``upper`` columns.
        interval: Calibrated conformal interval.
        coverage: Empirical coverage on the held-out final fold.
        anomalies: Historical anomaly flags.
    """

    series_name: str
    backtest: BacktestResult
    winner: str
    forecast: pd.DataFrame
    interval: ConformalInterval
    coverage: float
    anomalies: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        """Build a JSON-serializable summary of this run."""
        table = (
            self.backtest.metrics.groupby("model")[["mae", "rmse", "smape", "mase"]]
            .mean()
            .sort_values("mase")
            .round(4)
        )
        return {
            "series": self.series_name,
            "winner": self.winner,
            "backtest_mean_metrics": table.to_dict(orient="index"),
            "conformal": {
                "alpha": self.interval.alpha,
                "half_width": round(self.interval.half_width, 4),
                "n_calibration": self.interval.n_calibration,
                "held_out_coverage": round(self.coverage, 4),
            },
            "n_anomalies_flagged": int(self.anomalies["is_anomaly"].sum()),
        }


def _split_by_fold(result: BacktestResult, model: str) -> tuple[
    list[pd.Series], list[pd.Series], list[pd.Series]
]:
    """Slice pooled backtest outputs back into per-fold chunks."""
    residuals, predictions, actuals = [], [], []
    horizon = result.folds[0].test_end - result.folds[0].test_start
    for i, _fold in enumerate(result.folds):
        sl = slice(i * horizon, (i + 1) * horizon)
        residuals.append(result.residuals[model].iloc[sl])
        predictions.append(result.predictions[model].iloc[sl])
        actuals.append(result.actuals.iloc[sl])
    return residuals, predictions, actuals


def run_pipeline(y: pd.Series, config: PipelineConfig) -> PipelineResult:
    """Run backtest, model selection, conformal calibration and anomaly scan.

    Args:
        y: Daily series to forecast.
        config: Pipeline configuration.

    Returns:
        A :class:`PipelineResult`.
    """
    name = str(y.name or "series")
    logger.info("=== %s: %d observations ===", name, len(y))
    factories = default_model_factories(season_length=config.season_length)
    backtest = rolling_origin_backtest(
        y,
        factories,
        horizon=config.horizon,
        n_folds=config.n_folds,
        min_train=config.min_train_days,
        season_length=config.season_length,
    )
    winner = select_best(backtest, metric="mase")

    residuals, predictions, actuals = _split_by_fold(backtest, winner)
    interval, coverage = calibrate_and_evaluate(
        residuals, predictions, actuals, alpha=config.alpha
    )

    final_model = factories[winner]().fit(y)
    point_forecast = final_model.predict(config.horizon)
    forecast = interval.apply(point_forecast)

    anomalies = flag_anomalies(
        y,
        season_length=config.season_length,
        z_threshold=config.anomaly_z_threshold,
    )
    return PipelineResult(
        series_name=name,
        backtest=backtest,
        winner=winner,
        forecast=forecast,
        interval=interval,
        coverage=coverage,
        anomalies=anomalies,
    )


def write_reports(result: PipelineResult, y: pd.Series, output_dir: Path) -> None:
    """Write all artifacts (plots, HTML report) for one pipeline run.

    Args:
        result: Pipeline output for the series.
        y: The historical series.
        output_dir: Directory to write into (created if needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name = result.series_name
    plot_forecast(
        y,
        result.forecast,
        output_dir / f"{name}_forecast.png",
        title=(
            f"{name}: {len(result.forecast)}-day forecast ({result.winner}) "
            f"with {1 - result.interval.alpha:.0%} conformal interval"
        ),
        anomalies=result.anomalies,
    )
    plot_backtest_errors(
        result.backtest.metrics,
        output_dir / f"{name}_backtest_mase.png",
        title=f"{name}: MASE by backtest fold",
    )
    build_html_report(
        y,
        result.forecast,
        result.backtest.metrics,
        result.winner,
        result.coverage,
        output_dir / f"{name}_report.html",
    )


def run_and_report(
    series: dict[str, pd.Series], config: PipelineConfig
) -> dict[str, PipelineResult]:
    """Run the pipeline for several series and write a combined metrics.json.

    Args:
        series: Mapping of name to daily series.
        config: Pipeline configuration.

    Returns:
        Mapping of series name to pipeline result.
    """
    results: dict[str, PipelineResult] = {}
    for name, y in series.items():
        result = run_pipeline(y, config)
        write_reports(result, y, config.output_dir)
        results[name] = result
    payload = {name: res.summary() for name, res in results.items()}
    write_metrics_json(payload, config.output_dir / "metrics.json")
    return results
