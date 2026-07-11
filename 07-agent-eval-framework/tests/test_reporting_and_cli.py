"""Report writers and CLI entry points (offline, end to end)."""

import json
from pathlib import Path

from agent_evals import demo, run
from agent_evals.agents import GoodAgent, SloppyAgent
from agent_evals.models import Dataset
from agent_evals.reporting import comparison_table, write_reports
from agent_evals.runner import Runner

DATASET = Dataset.from_jsonl(Path(__file__).resolve().parent.parent / "datasets" / "basic.jsonl")


def test_write_reports_produces_all_three_files(tmp_path: Path) -> None:
    results = Runner().compare({"GoodAgent": GoodAgent(), "SloppyAgent": SloppyAgent()}, DATASET)
    paths = write_reports(results, tmp_path / "out")
    names = {p.name for p in paths}
    assert names == {"results.json", "report.md", "report.html"}
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert set(payload) == {"GoodAgent", "SloppyAgent"}
    assert len(payload["GoodAgent"]["case_results"]) == 10
    markdown = (tmp_path / "out" / "report.md").read_text()
    assert "| Metric | GoodAgent | SloppyAgent |" in markdown
    html = (tmp_path / "out" / "report.html").read_text()
    assert "<svg" in html and "GoodAgent" in html and "</html>" in html


def test_comparison_table_has_metric_rows() -> None:
    results = Runner().compare({"GoodAgent": GoodAgent()}, DATASET)
    table = comparison_table(results)
    for label in ("Tool selection", "Tool call order", "Trajectory efficiency",
                  "Answer correctness", "Cost / latency", "Overall score"):
        assert label in table


def test_demo_cli_end_to_end(tmp_path: Path, capsys) -> None:
    exit_code = demo.main(["--output", str(tmp_path / "reports")])
    assert exit_code == 0
    assert (tmp_path / "reports" / "report.html").exists()
    out = capsys.readouterr().out
    assert "GoodAgent" in out and "SloppyAgent" in out


def test_run_cli_with_agent_factory_spec(tmp_path: Path) -> None:
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "basic.jsonl"
    exit_code = run.main([
        "--dataset", str(dataset_path),
        "--agent", "agent_evals.agents:good_agent",
        "--agent", "agent_evals.agents:sloppy_agent",
        "--output", str(tmp_path / "reports"),
    ])
    assert exit_code == 0
    payload = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert set(payload) == {"GoodAgent", "SloppyAgent"}
