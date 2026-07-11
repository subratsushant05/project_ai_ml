"""Evaluation layer: from-scratch text metrics and a base-vs-finetuned harness."""

from qlora_tune.evaluation.harness import EvalReport, evaluate_predictions, render_report
from qlora_tune.evaluation.metrics import (
    RougeScore,
    exact_match,
    keyword_coverage,
    rouge_l,
)

__all__ = [
    "EvalReport",
    "RougeScore",
    "evaluate_predictions",
    "exact_match",
    "keyword_coverage",
    "render_report",
    "rouge_l",
]
