"""End-to-end pipeline smoke test on a tiny configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.helpers import small_series
from ts_forecast.backtest import rolling_origin_backtest, select_best
from ts_forecast.config import PipelineConfig
from ts_forecast.conformal import calibrate_and_evaluate
from ts_forecast.models import SeasonalNaive
from ts_forecast.pipeline import PipelineResult, run_and_report, run_pipeline


def _tiny_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        horizon=7,
        n_folds=3,
        min_train_days=150,
        season_length=7,
        alpha=0.2,
        output_dir=tmp_path / "out",
    )


def test_pipeline_with_fast_models(tmp_path: Path) -> None:
    """A fast two-model pipeline yields a coherent, calibrated result."""
    y = small_series(260, name="tiny")
    factories = {
        "seasonal_naive": lambda: SeasonalNaive(7),
        "seasonal_naive_14": lambda: SeasonalNaive(14),
    }
    backtest = rolling_origin_backtest(
        y, factories, horizon=7, n_folds=3, min_train=150, season_length=7
    )
    winner = select_best(backtest)
    assert winner in factories

    residuals = [backtest.residuals[winner].iloc[i * 7 : (i + 1) * 7] for i in range(3)]
    predictions = [backtest.predictions[winner].iloc[i * 7 : (i + 1) * 7] for i in range(3)]
    actuals = [backtest.actuals.iloc[i * 7 : (i + 1) * 7] for i in range(3)]
    interval, coverage = calibrate_and_evaluate(residuals, predictions, actuals, alpha=0.2)
    assert interval.half_width > 0
    assert 0.0 <= coverage <= 1.0


def test_run_and_report_writes_artifacts(tmp_path: Path) -> None:
    """The full pipeline writes plots, HTML and metrics.json for a series."""
    config = _tiny_config(tmp_path)
    y = small_series(260, name="tiny")
    results = run_and_report({"tiny": y}, config)

    result = results["tiny"]
    assert isinstance(result, PipelineResult)
    assert len(result.forecast) == config.horizon
    assert np.isfinite(result.forecast.to_numpy(dtype=float)).all()
    assert (result.forecast["lower"] <= result.forecast["forecast"]).all()

    out = config.output_dir
    assert (out / "tiny_forecast.png").exists()
    assert (out / "tiny_backtest_mase.png").exists()
    assert (out / "tiny_report.html").exists()
    payload = json.loads((out / "metrics.json").read_text())
    assert payload["tiny"]["winner"] == result.winner
    assert "mase" in next(iter(payload["tiny"]["backtest_mean_metrics"].values()))


def test_run_pipeline_summary_is_serializable(tmp_path: Path) -> None:
    """summary() returns a JSON-serializable dict with expected keys."""
    config = _tiny_config(tmp_path)
    y = small_series(260, name="tiny")
    result = run_pipeline(y, config)
    summary = result.summary()
    json.dumps(summary)  # must not raise
    assert {"series", "winner", "backtest_mean_metrics", "conformal"} <= set(summary)
