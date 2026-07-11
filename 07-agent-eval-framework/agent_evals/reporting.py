"""Report writers: results.json, Markdown, and self-contained HTML."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from agent_evals.html_report import render_html
from agent_evals.models import EvalResult

logger = logging.getLogger(__name__)

METRIC_LABELS: dict[str, str] = {
    "tool_selection": "Tool selection (F1)",
    "tool_order": "Tool call order",
    "efficiency": "Trajectory efficiency",
    "answer_correctness": "Answer correctness",
    "cost_latency": "Cost / latency",
}


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def write_results_json(results: Mapping[str, EvalResult], path: str | Path) -> None:
    """Write full per-case and aggregate results as JSON.

    Args:
        results: Eval results keyed by agent name.
        path: Destination ``results.json`` path.
    """
    payload = {name: result.model_dump() for name, result in results.items()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


def comparison_table(results: Mapping[str, EvalResult]) -> str:
    """Render the side-by-side agent comparison as a Markdown table.

    Args:
        results: Eval results keyed by agent name.

    Returns:
        A GitHub-flavored Markdown table (metrics as rows, agents as columns).
    """
    agents = list(results)
    metrics = list(next(iter(results.values())).mean_scores) if results else []
    header = "| Metric | " + " | ".join(agents) + " |"
    divider = "|---" * (len(agents) + 1) + "|"
    rows = [header, divider]
    for metric in metrics:
        cells = [f"{results[a].mean_scores.get(metric, 0.0):.3f}" for a in agents]
        rows.append(f"| {_label(metric)} | " + " | ".join(cells) + " |")
    rows.append(
        "| **Overall score** | "
        + " | ".join(f"**{results[a].overall_score:.3f}**" for a in agents) + " |"
    )
    rows.append(
        "| Total cost (USD) | " + " | ".join(f"{results[a].total_cost_usd:.4f}" for a in agents) + " |"
    )
    rows.append("| Total tokens | " + " | ".join(str(results[a].total_tokens) for a in agents) + " |")
    rows.append("| Latency p50 (s) | " + " | ".join(f"{results[a].latency_p50_s:.2f}" for a in agents) + " |")
    rows.append("| Latency p95 (s) | " + " | ".join(f"{results[a].latency_p95_s:.2f}" for a in agents) + " |")
    return "\n".join(rows)


def render_markdown(results: Mapping[str, EvalResult]) -> str:
    """Render the full Markdown report.

    Args:
        results: Eval results keyed by agent name.

    Returns:
        Markdown text with the comparison table and per-agent case tables.
    """
    dataset = next(iter(results.values())).dataset_name if results else "dataset"
    lines = [
        "# Agent evaluation report",
        "",
        f"Dataset: `{dataset}` | Agents: {', '.join(results)} | "
        f"Cases per agent: {len(next(iter(results.values())).case_results) if results else 0}",
        "",
        "## Side-by-side comparison",
        "",
        comparison_table(results),
        "",
    ]
    for name, result in results.items():
        lines += [f"## Per-case results: {name}", ""]
        metrics = list(result.mean_scores)
        header = "| Case | Tools called | " + " | ".join(_label(m) for m in metrics) + " |"
        lines += [header, "|---" * (len(metrics) + 2) + "|"]
        for case in result.case_results:
            scores = " | ".join(f"{case.metrics[m].score:.3f}" for m in metrics)
            tools = ", ".join(case.tool_sequence) or "-"
            lines.append(f"| {case.case_id} | {tools} | {scores} |")
        lines.append("")
    return "\n".join(lines)


def write_reports(results: Mapping[str, EvalResult], output_dir: str | Path) -> list[Path]:
    """Write results.json, report.md, and report.html to a directory.

    Args:
        results: Eval results keyed by agent name.
        output_dir: Directory to create/write into.

    Returns:
        Paths of the three written files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    md_path = out / "report.md"
    html_path = out / "report.html"
    write_results_json(results, json_path)
    md_path.write_text(render_markdown(results), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    html_path.write_text(render_html(results), encoding="utf-8")
    logger.info("Wrote %s", html_path)
    return [json_path, md_path, html_path]
