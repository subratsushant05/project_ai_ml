"""Command-line demo: run the full research pipeline offline.

Usage::

    python -m agent_researcher.demo "How do transformer models work?"

The demo streams per-node progress, pauses at the human approval gate
(auto-approving so the run is non-interactive), resumes the same
checkpointed thread, and writes the final report to ``report.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m agent_researcher.demo",
        description="Run the multi-agent research pipeline (offline by default).",
    )
    parser.add_argument("question", help="Research question to investigate")
    parser.add_argument(
        "--output",
        default="report.md",
        help="Path for the generated markdown report (default: report.md)",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Checkpoint thread id (default: a fresh random id)",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip the human approval gate instead of auto-approving it",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser


def _describe(node: str, update: dict[str, Any]) -> str:
    """Summarize a node's state update for progress output."""
    if node == "planner":
        plan = update.get("plan", [])
        lines = [f"planned {len(plan)} sub-questions"]
        lines += [f"    {i}. {q}" for i, q in enumerate(plan, start=1)]
        return "\n".join(lines)
    if node == "researcher":
        findings = update.get("findings", [])
        sources = sum(len(f.sources) for f in findings)
        return f"gathered {len(findings)} findings from {sources} sources"
    if node == "approval_gate":
        return "approved" if update.get("approved") else "rejected"
    if node == "writer":
        revision = update.get("revision_count", 0)
        return f"draft v{revision + 1} written ({len(update.get('draft', ''))} chars)"
    if node == "critic":
        critique = update.get("critique")
        if critique is None:
            return "no critique produced"
        return f"scored {critique.score:g}/10 ({len(critique.feedback)} issue(s))"
    return "done"


def _stream(graph: Any, payload: Any, config: dict[str, Any]) -> bool:
    """Stream one graph run, printing progress; return True if interrupted."""
    interrupted = False
    for chunk in graph.stream(payload, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupted = True
                continue
            print(f"[{node}] {_describe(node, update or {})}")
    return interrupted


def main(argv: list[str] | None = None) -> int:
    """Run the demo end to end.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # langgraph 0.6 + langchain-core 0.3 emit a pending-deprecation notice on
    # import. langchain-core installs its own warning filters when imported,
    # so import it first, then register the ignore so demo output stays clean.
    with warnings.catch_warnings():
        import langchain_core  # noqa: F401

        warnings.filterwarnings("ignore", message="The default value of")
        from langgraph.types import Command

        from agent_researcher.config import Settings
        from agent_researcher.graph import build_graph

    settings = Settings(require_approval=not args.no_approval)
    graph = build_graph(settings=settings)
    thread_id = args.thread_id or f"demo-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Question : {args.question}")
    print(f"Thread   : {thread_id}")
    print(f"Providers: model={settings.model_provider}, "
          f"search={settings.search_provider}\n")

    interrupted = _stream(graph, {"question": args.question}, config)
    if interrupted:
        print("[approval_gate] interrupted -- awaiting human approval "
              "(checkpoint saved)")
        print(f"[approval_gate] demo auto-approves; resuming thread {thread_id!r}")
        _stream(graph, Command(resume=True), config)

    final = graph.get_state(config).values
    draft = final.get("draft", "")
    if not draft:
        print("\nNo report was produced (approval rejected?).", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.write_text(draft, encoding="utf-8")
    critique = final.get("critique")
    print()
    if critique is not None:
        print(f"Final quality score : {critique.score:g}/10")
    print(f"Revisions performed : {final.get('revision_count', 0)}")
    print(f"Report written to   : {output} ({len(draft)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
