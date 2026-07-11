"""General CLI: evaluate any agent factory against any JSONL dataset.

Usage:
    python -m agent_evals.run --dataset datasets/basic.jsonl \\
        --agent agent_evals.agents:good_agent [--output reports] [--judge offline]

The ``--agent`` target is ``module.path:factory`` where ``factory()`` returns
a callable mapping an input string to a Trajectory.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from agent_evals.judge import load_judge
from agent_evals.models import Dataset
from agent_evals.reporting import comparison_table, write_reports
from agent_evals.runner import AgentFn, Runner

logger = logging.getLogger(__name__)


def load_agent(spec: str) -> AgentFn:
    """Import an agent factory from a ``module:factory`` spec and call it.

    Args:
        spec: Dotted module path and factory name, e.g.
            ``agent_evals.agents:good_agent``.

    Returns:
        The agent callable produced by the factory.

    Raises:
        ValueError: If the spec is not of the form ``module:factory``.
    """
    if ":" not in spec:
        raise ValueError(f"Agent spec must be 'module:factory', got {spec!r}")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    return factory()


def main(argv: list[str] | None = None) -> int:
    """Evaluate one or more agents on a dataset and write reports.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to a JSONL dataset")
    parser.add_argument("--agent", action="append", required=True, dest="agents",
                        help="module:factory (repeatable to compare agents)")
    parser.add_argument("--output", type=Path, default=Path("reports"),
                        help="Directory for results.json / report.md / report.html")
    parser.add_argument("--judge", default=None,
                        help="Judge backend: offline | openai | anthropic (default: env/offline)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dataset = Dataset.from_jsonl(args.dataset)
    runner = Runner(judge=load_judge(args.judge) if args.judge else None)
    agents = {}
    for spec in args.agents:
        agent = load_agent(spec)
        name = getattr(agent, "name", spec.split(":", 1)[1])
        agents[name] = agent
    results = runner.compare(agents, dataset)
    paths = write_reports(results, args.output)

    print(f"\nDataset: {dataset.name} ({len(dataset)} cases)\n")
    print(comparison_table(results))
    print("\nReports written:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
