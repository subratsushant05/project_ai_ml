"""Evaluation harness: compare base vs fine-tuned outputs on a predictions file.

The harness is decoupled from inference: any process (a GPU box, an API, a
notebook) writes a predictions JSONL with base and fine-tuned outputs per
example, and this module scores it on CPU.

Expected JSONL fields per line:
    ``id``, ``category``, ``instruction``, ``reference``,
    ``base_output``, ``finetuned_output``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from qlora_tune.evaluation.metrics import exact_match, keyword_coverage, rouge_l

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    "id",
    "category",
    "instruction",
    "reference",
    "base_output",
    "finetuned_output",
)

SYSTEMS = ("base", "finetuned")


@dataclass(frozen=True, slots=True)
class SystemScores:
    """Aggregate metric scores for one system (base or fine-tuned).

    Attributes:
        rouge_l_f1: Mean ROUGE-L F1 across examples.
        exact_match: Mean exact-match rate.
        keyword_coverage: Mean keyword-coverage score.
    """

    rouge_l_f1: float
    exact_match: float
    keyword_coverage: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Full evaluation result for a predictions file.

    Attributes:
        n_examples: Number of scored examples.
        scores: Per-system aggregate scores, keyed ``"base"``/``"finetuned"``.
        per_category: Per-category mean ROUGE-L F1, keyed by system.
    """

    n_examples: int
    scores: dict[str, SystemScores]
    per_category: dict[str, dict[str, float]]


def load_predictions(path: str | Path) -> list[dict[str, str]]:
    """Load and validate a predictions JSONL file.

    Args:
        path: Path to the predictions file.

    Returns:
        List of prediction rows.

    Raises:
        ValueError: If a row is missing required fields or the file is empty.
    """
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing field(s) {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prediction rows found")
    logger.info("Loaded %d prediction rows from %s", len(rows), path)
    return rows


def evaluate_predictions(rows: list[dict[str, str]]) -> EvalReport:
    """Score base and fine-tuned outputs against references.

    Args:
        rows: Prediction rows (see module docstring for the schema).

    Returns:
        An :class:`EvalReport` with aggregate and per-category scores.
    """
    metric_sums = {s: {"rouge": 0.0, "em": 0.0, "kw": 0.0} for s in SYSTEMS}
    cat_sums: dict[str, dict[str, list[float]]] = {s: {} for s in SYSTEMS}

    for row in rows:
        for system in SYSTEMS:
            output = row[f"{system}_output"]
            reference = row["reference"]
            r = rouge_l(output, reference)
            metric_sums[system]["rouge"] += r.f1
            metric_sums[system]["em"] += exact_match(output, reference)
            metric_sums[system]["kw"] += keyword_coverage(output, reference)
            cat_sums[system].setdefault(row["category"], []).append(r.f1)

    n = len(rows)
    scores = {
        system: SystemScores(
            rouge_l_f1=metric_sums[system]["rouge"] / n,
            exact_match=metric_sums[system]["em"] / n,
            keyword_coverage=metric_sums[system]["kw"] / n,
        )
        for system in SYSTEMS
    }
    per_category = {
        system: {cat: sum(vals) / len(vals) for cat, vals in sorted(cat_sums[system].items())}
        for system in SYSTEMS
    }
    return EvalReport(n_examples=n, scores=scores, per_category=per_category)


def render_report(report: EvalReport) -> str:
    """Render an :class:`EvalReport` as a terminal-friendly table.

    Args:
        report: Evaluation report.

    Returns:
        Multi-line string with aggregate scores, deltas and a per-category
        ROUGE-L breakdown.
    """
    base, tuned = report.scores["base"], report.scores["finetuned"]
    rows = [
        ("ROUGE-L F1", base.rouge_l_f1, tuned.rouge_l_f1),
        ("Exact match", base.exact_match, tuned.exact_match),
        ("Keyword coverage", base.keyword_coverage, tuned.keyword_coverage),
    ]
    lines = [
        "=" * 62,
        f"EVALUATION REPORT ({report.n_examples} examples)",
        "=" * 62,
        f"{'Metric':<20}{'Base':>10}{'Fine-tuned':>12}{'Delta':>10}",
        "-" * 62,
    ]
    for name, b, t in rows:
        lines.append(f"{name:<20}{b:>10.3f}{t:>12.3f}{t - b:>+10.3f}")
    lines.append("-" * 62)
    lines.append("Per-category ROUGE-L F1 (base -> fine-tuned):")
    for cat in report.per_category["base"]:
        b = report.per_category["base"][cat]
        t = report.per_category["finetuned"][cat]
        lines.append(f"  {cat:<22}{b:.3f} -> {t:.3f}")
    lines.append("=" * 62)
    return "\n".join(lines)
