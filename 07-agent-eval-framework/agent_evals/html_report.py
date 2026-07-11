"""Self-contained HTML report with an inline-SVG grouped bar chart.

No external assets: styles are inline CSS custom properties (light + dark),
the chart is hand-built SVG with native ``<title>`` hover tooltips, and the
full comparison table doubles as the accessible table view of the chart.
"""

from __future__ import annotations

import html
from collections.abc import Mapping

from agent_evals.models import EvalResult

# Validated categorical palette (light, dark) -- assigned in fixed order.
_SERIES = [("#2a78d6", "#3987e5"), ("#1baf7a", "#199e70"),
           ("#eda100", "#c98500"), ("#4a3aa7", "#9085e9")]

_METRIC_LABELS = {
    "tool_selection": "Tool selection",
    "tool_order": "Tool order",
    "efficiency": "Efficiency",
    "answer_correctness": "Answer correctness",
    "cost_latency": "Cost / latency",
}

_CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; }
}
* { box-sizing: border-box; }
body { margin:0; padding:32px 16px; background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 12px; }
.sub { color:var(--ink-2); margin:0 0 24px; }
.card { background:var(--surface); border:1px solid var(--grid);
  border-radius:10px; padding:20px; margin-bottom:24px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th { text-align:left; color:var(--ink-2); font-weight:600; }
th, td { padding:7px 10px; border-bottom:1px solid var(--grid); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
th.num { text-align:right; }
tr.total td { font-weight:600; border-top:2px solid var(--axis); }
.legend { display:flex; gap:18px; margin:0 0 10px; font-size:13px; color:var(--ink-2); }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.swatch { width:10px; height:10px; border-radius:3px; display:inline-block; }
svg text { font:12px system-ui,-apple-system,"Segoe UI",sans-serif; }
.answer { color:var(--ink-2); font-size:13px; }
"""


def _bar_path(x: float, y: float, width: float, baseline: float, radius: float = 4) -> str:
    """SVG path for a bar with a rounded data-end anchored to the baseline."""
    r = min(radius, width / 2, max(baseline - y, 0.0))
    return (
        f"M{x:.1f},{baseline:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"L{x + width - r:.1f},{y:.1f} Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} "
        f"L{x + width:.1f},{baseline:.1f} Z"
    )


def _chart_svg(results: Mapping[str, EvalResult]) -> str:
    """Grouped bar chart of mean metric scores, one bar per agent."""
    agents = list(results)[: len(_SERIES)]
    metrics = list(next(iter(results.values())).mean_scores)
    width, height = 840, 300
    left, right, top, bottom = 44, 12, 16, 44
    plot_w, baseline = width - left - right, height - bottom
    plot_h = baseline - top
    group_w = plot_w / max(len(metrics), 1)
    bar_w = min(36.0, (group_w - 16) / max(len(agents), 1) - 2)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Mean metric scores by agent" style="width:100%;height:auto">'
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = baseline - tick * plot_h
        stroke = "var(--axis)" if tick == 0.0 else "var(--grid)"
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="var(--muted)">{tick:g}</text>'
        )
    for gi, metric in enumerate(metrics):
        group_x = left + gi * group_w
        total_bars_w = len(agents) * bar_w + (len(agents) - 1) * 2
        start = group_x + (group_w - total_bars_w) / 2
        for ai, agent in enumerate(agents):
            score = results[agent].mean_scores.get(metric, 0.0)
            bar_h = max(score * plot_h, 1.0)
            x = start + ai * (bar_w + 2)
            y = baseline - bar_h
            label = html.escape(_METRIC_LABELS.get(metric, metric))
            parts.append(
                f'<path d="{_bar_path(x, y, bar_w, baseline)}" class="s{ai}">'
                f"<title>{html.escape(agent)} - {label}: {score:.3f}</title></path>"
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                f'fill="var(--ink-2)">{score:.2f}</text>'
            )
        label = html.escape(_METRIC_LABELS.get(metric, metric))
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{baseline + 18:.1f}" '
            f'text-anchor="middle" fill="var(--muted)">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _series_css(agents: list[str]) -> str:
    """Per-series fill colors for light and dark surfaces."""
    light = "".join(f".s{i}{{fill:{_SERIES[i][0]}}}" for i in range(len(agents)))
    dark = "".join(f".s{i}{{fill:{_SERIES[i][1]}}}" for i in range(len(agents)))
    return f"{light}@media (prefers-color-scheme: dark){{{dark}}}"


def _legend(agents: list[str]) -> str:
    items = "".join(
        f'<span><i class="swatch" style="background:{_SERIES[i][0]}"></i>{html.escape(a)}</span>'
        for i, a in enumerate(agents)
    )
    return f'<div class="legend">{items}</div>'


def _comparison_rows(results: Mapping[str, EvalResult]) -> str:
    agents = list(results)
    metrics = list(next(iter(results.values())).mean_scores)
    rows = []
    for metric in metrics:
        cells = "".join(f'<td class="num">{results[a].mean_scores.get(metric, 0.0):.3f}</td>' for a in agents)
        rows.append(f"<tr><td>{html.escape(_METRIC_LABELS.get(metric, metric))}</td>{cells}</tr>")
    extras = [
        ("Overall score", [f"{results[a].overall_score:.3f}" for a in agents], "total"),
        ("Total cost (USD)", [f"{results[a].total_cost_usd:.4f}" for a in agents], ""),
        ("Total tokens", [f"{results[a].total_tokens:,}" for a in agents], ""),
        ("Latency p50 (s)", [f"{results[a].latency_p50_s:.2f}" for a in agents], ""),
        ("Latency p95 (s)", [f"{results[a].latency_p95_s:.2f}" for a in agents], ""),
    ]
    for label, values, cls in extras:
        cells = "".join(f'<td class="num">{v}</td>' for v in values)
        rows.append(f'<tr class="{cls}"><td>{label}</td>{cells}</tr>')
    return "".join(rows)


def _case_table(result: EvalResult) -> str:
    metrics = list(result.mean_scores)
    head = "".join(f'<th class="num">{html.escape(_METRIC_LABELS.get(m, m))}</th>' for m in metrics)
    rows = []
    for case in result.case_results:
        cells = "".join(f'<td class="num">{case.metrics[m].score:.3f}</td>' for m in metrics)
        tools = html.escape(", ".join(case.tool_sequence) or "-")
        answer = html.escape(case.final_answer)
        rows.append(
            f"<tr><td>{html.escape(case.case_id)}</td><td>{tools}</td>{cells}</tr>"
            f'<tr><td></td><td colspan="{len(metrics) + 1}" class="answer">{answer}</td></tr>'
        )
    return (
        f"<table><thead><tr><th>Case</th><th>Tools called</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(results: Mapping[str, EvalResult]) -> str:
    """Render the full self-contained HTML report.

    Args:
        results: Eval results keyed by agent name.

    Returns:
        A complete HTML document (inline CSS, inline SVG, no external assets).
    """
    agents = list(results)
    dataset = next(iter(results.values())).dataset_name if results else "dataset"
    n_cases = len(next(iter(results.values())).case_results) if results else 0
    agent_cols = "".join(f'<th class="num">{html.escape(a)}</th>' for a in agents)
    sections = "".join(
        f"<h2>Per-case results: {html.escape(name)}</h2><div class='card'>{_case_table(result)}</div>"
        for name, result in results.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent evaluation report</title>
<style>{_CSS}{_series_css(agents)}</style></head>
<body><main>
<h1>Agent evaluation report</h1>
<p class="sub">Dataset <strong>{html.escape(dataset)}</strong> - {n_cases} cases per agent - \
higher is better (all scores in [0, 1])</p>
<div class="card">{_legend(agents)}{_chart_svg(results)}</div>
<h2>Side-by-side comparison</h2>
<div class="card"><table><thead><tr><th>Metric</th>{agent_cols}</tr></thead>
<tbody>{_comparison_rows(results)}</tbody></table></div>
{sections}
</main></body></html>
"""
