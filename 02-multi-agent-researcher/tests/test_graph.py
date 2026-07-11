"""End-to-end graph tests: happy path, revision loop, approval interrupt."""

from __future__ import annotations

import operator
from typing import Any, get_type_hints

import pytest
from langgraph.types import Command

from agent_researcher.config import Settings
from agent_researcher.graph import build_graph
from agent_researcher.offline_llm import OfflineChatModel
from agent_researcher.state import ResearchState, collect_citations

QUESTION = "How do transformer models work?"


class ScriptedWriterModel:
    """Offline model whose writer output is overridden per revision.

    Args:
        drafts: Draft to emit per writer call; the last entry repeats.
    """

    def __init__(self, drafts: list[str]) -> None:
        self._inner = OfflineChatModel()
        self._drafts = drafts
        self._calls = 0

    def invoke(self, system: str, user: str) -> str:
        """Delegate to the offline model except for writer calls."""
        if "technical writer" in system.lower():
            draft = self._drafts[min(self._calls, len(self._drafts) - 1)]
            self._calls += 1
            return draft
        return self._inner.invoke(system, user)


GOOD_DRAFT = (
    "# Research Report: Q\n\n## Findings\n\n### A?\n\nAnswer [1].\n\n"
    "### B?\n\nAnswer [2].\n\n## Limitations and Further Research\n\n"
    "Scope note.\n\n## References\n\n[1] T - u\n[2] T2 - u2\n" + "filler " * 60
)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def test_graph_runs_end_to_end_offline(settings: Settings) -> None:
    """The full pipeline produces an accepted, citation-backed report."""
    graph = build_graph(settings=settings)
    final = graph.invoke({"question": QUESTION}, _config("e2e"))

    assert len(final["plan"]) == 3
    assert len(final["findings"]) == 3
    assert final["critique"].score >= settings.quality_threshold
    draft = final["draft"]
    assert "## References" in draft
    assert "[1]" in draft


def test_offline_run_performs_exactly_one_revision(settings: Settings) -> None:
    """First offline draft scores below threshold, so one revision happens."""
    graph = build_graph(settings=settings)
    final = graph.invoke({"question": QUESTION}, _config("one-rev"))
    assert final["revision_count"] == 1
    assert "## Limitations" in final["draft"]


def test_bad_draft_triggers_exactly_one_bounded_revision(
    settings: Settings,
) -> None:
    """A persistently bad writer is cut off after max_revisions, not looped."""
    model = ScriptedWriterModel(drafts=["Too short."])
    graph = build_graph(model=model, settings=settings)
    final = graph.invoke({"question": QUESTION}, _config("bad-writer"))
    assert final["revision_count"] == settings.max_revisions == 1
    assert final["critique"].score < settings.quality_threshold


def test_good_first_draft_skips_revision(settings: Settings) -> None:
    """A draft that satisfies the critic is accepted with zero revisions."""
    model = ScriptedWriterModel(drafts=[GOOD_DRAFT])
    graph = build_graph(model=model, settings=settings)
    final = graph.invoke({"question": QUESTION}, _config("good-writer"))
    assert final["revision_count"] == 0
    assert final["critique"].score >= settings.quality_threshold


def test_approval_interrupt_pauses_then_resumes_same_thread(
    settings: Settings,
) -> None:
    """With approval required, the graph checkpoints and resumes by thread."""
    gated = settings.model_copy(update={"require_approval": True})
    graph = build_graph(settings=gated)
    config = _config("hitl-approve")

    paused = graph.invoke({"question": QUESTION}, config)
    assert "__interrupt__" in paused
    assert "draft" not in paused
    snapshot = graph.get_state(config)
    assert "approval_gate" in snapshot.next

    final = graph.invoke(Command(resume=True), config)
    assert final["approved"] is True
    assert final["draft"].startswith("# Research Report:")


def test_rejected_approval_ends_without_draft(settings: Settings) -> None:
    """A human rejection routes the graph to END before the writer runs."""
    gated = settings.model_copy(update={"require_approval": True})
    graph = build_graph(settings=gated)
    config = _config("hitl-reject")

    graph.invoke({"question": QUESTION}, config)
    final = graph.invoke(Command(resume=False), config)
    assert final["approved"] is False
    assert "draft" not in final


def test_report_references_cover_all_finding_sources(settings: Settings) -> None:
    """Every cited source appears in the final references section."""
    graph = build_graph(settings=settings)
    final = graph.invoke({"question": QUESTION}, _config("refs"))
    references = final["draft"].split("## References", maxsplit=1)[1]
    for citation in collect_citations(final["findings"]):
        assert citation.url in references
        assert f"[{citation.index}]" in references


def test_findings_use_additive_reducer() -> None:
    """State accumulates findings instead of overwriting them."""
    hints = get_type_hints(ResearchState, include_extras=True)
    assert operator.add in hints["findings"].__metadata__


def test_state_accumulates_across_nodes(settings: Settings) -> None:
    """Later nodes see earlier keys; nothing is dropped along the way."""
    graph = build_graph(settings=settings)
    final = graph.invoke({"question": QUESTION}, _config("accumulate"))
    assert set(final) >= {
        "question",
        "plan",
        "findings",
        "draft",
        "critique",
        "revision_count",
        "approved",
    }
    assert final["question"] == QUESTION


def test_missing_question_fails_fast(settings: Settings) -> None:
    """Invoking without a question raises instead of hallucinating one."""
    graph = build_graph(settings=settings)
    with pytest.raises(KeyError):
        graph.invoke({}, _config("no-question"))
