"""Demo entry point: evaluate both bundled agents and write all reports.

Usage:
    python -m agent_evals.demo [--output reports] [--dataset path.jsonl]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from agent_evals.agents import GoodAgent, SloppyAgent
from agent_evals.models import Dataset
from agent_evals.reporting import comparison_table, write_reports
from agent_evals.runner import Runner

logger = logging.getLogger(__name__)

BUNDLED_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "basic.jsonl"


def main(argv: list[str] | None = None) -> int:
    """Run the demo evaluation and print the comparison table.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BUNDLED_DATASET,
                        help="Path to a JSONL dataset (default: bundled basic.jsonl)")
    parser.add_argument("--output", type=Path, default=Path("reports"),
                        help="Directory for results.json / report.md / report.html")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dataset = Dataset.from_jsonl(args.dataset)
    runner = Runner()
    results = runner.compare({"GoodAgent": GoodAgent(), "SloppyAgent": SloppyAgent()}, dataset)
    paths = write_reports(results, args.output)

    print(f"\nDataset: {dataset.name} ({len(dataset)} cases)\n")
    print(comparison_table(results))
    print("\nReports written:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
