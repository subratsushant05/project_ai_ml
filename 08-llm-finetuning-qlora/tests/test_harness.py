"""Tests for the evaluation harness on the bundled sample predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_tune.evaluation.harness import evaluate_predictions, load_predictions, render_report

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sample_predictions.jsonl"


def test_bundled_predictions_load() -> None:
    """The bundled sample file parses and has every required field."""
    rows = load_predictions(SAMPLE)
    assert len(rows) == 10
    assert {row["category"] for row in rows} == {
        "password_reset",
        "vpn",
        "hardware",
        "software_install",
        "access_request",
    }


def test_finetuned_beats_base_on_bundled_sample() -> None:
    """On the sample file the fine-tuned outputs outscore base on all metrics."""
    report = evaluate_predictions(load_predictions(SAMPLE))
    base, tuned = report.scores["base"], report.scores["finetuned"]
    assert tuned.rouge_l_f1 > base.rouge_l_f1
    assert tuned.keyword_coverage > base.keyword_coverage
    assert tuned.exact_match >= base.exact_match
    assert 0 <= base.rouge_l_f1 <= 1 and 0 <= tuned.rouge_l_f1 <= 1


def test_report_renders_metrics_and_categories() -> None:
    """The rendered report contains the metric table and category breakdown."""
    report = evaluate_predictions(load_predictions(SAMPLE))
    text = render_report(report)
    assert "ROUGE-L F1" in text
    assert "Keyword coverage" in text
    assert "password_reset" in text
    assert "(10 examples)" in text


def test_missing_field_raises(tmp_path: Path) -> None:
    """A predictions row without a required field is rejected with location info."""
    bad = tmp_path / "bad.jsonl"
    row = {"id": "x", "category": "vpn", "instruction": "i", "reference": "r", "base_output": "b"}
    bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finetuned_output"):
        load_predictions(bad)


def test_empty_file_raises(tmp_path: Path) -> None:
    """An empty predictions file raises instead of returning a hollow report."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no prediction rows"):
        load_predictions(empty)
