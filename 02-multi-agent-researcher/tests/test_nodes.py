"""Tests for individual agent nodes and the critic routing function."""

from __future__ import annotations

import pytest

from agent_researcher.config import Settings
from agent_researcher.nodes import ResearchNodes
from agent_researcher.offline_llm import OfflineChatModel
from agent_researcher.search import OfflineSearchTool
from agent_researcher.state import Critique, ResearchState, collect_citations


@pytest.fixture()
def nodes(
    model: OfflineChatModel,
    search_tool: OfflineSearchTool,
    settings: Settings,
) -> ResearchNodes:
    """Nodes bound to offline providers."""
    return ResearchNodes(model=model, search_tool=search_tool, settings=settings)


@pytest.fixture()
def researched_state(nodes: ResearchNodes) -> ResearchState:
    """State after planner and researcher have run."""
    state: ResearchState = {"question": "How do transformer models work?"}
    state.update(nodes.plan(state))
    state.update(nodes.research(state))
    return state


def test_planner_node_sets_plan_and_resets_loop_state(nodes: ResearchNodes) -> None:
    """Planner writes the plan and initializes the revision loop."""
    update = nodes.plan({"question": "How do transformer models work?"})
    assert len(update["plan"]) == 3
    assert update["revision_count"] == 0
    assert update["critique"] is None


def test_researcher_builds_findings_with_global_citations(
    researched_state: ResearchState,
) -> None:
    """Each finding is grounded and citation numbers are report-global."""
    findings = researched_state["findings"]
    assert len(findings) == len(researched_state["plan"])
    for finding in findings:
        assert finding.sources, f"no sources for {finding.sub_question!r}"
        assert any(f"[{c.index}]" in finding.answer for c in finding.sources)
    citations = collect_citations(findings)
    indices = [c.index for c in citations]
    assert indices == list(range(1, len(indices) + 1))
    urls = [c.url for c in citations]
    assert len(urls) == len(set(urls))


def test_writer_first_draft_has_structure_and_citations(
    nodes: ResearchNodes, researched_state: ResearchState
) -> None:
    """The first draft carries inline citations and a references section."""
    update = nodes.write(researched_state)
    draft = update["draft"]
    assert update["revision_count"] == 0
    assert draft.startswith("# Research Report:")
    assert "## References" in draft
    assert "[1]" in draft
    assert draft.count("### ") == len(researched_state["plan"])


def test_writer_revision_increments_count_and_addresses_critique(
    nodes: ResearchNodes, researched_state: ResearchState
) -> None:
    """A critique-driven rewrite bumps the count and adds the missing section."""
    researched_state.update(nodes.write(researched_state))
    researched_state["critique"] = Critique(
        score=7.0, feedback=["Add a section on limitations and further research."]
    )
    update = nodes.write(researched_state)
    assert update["revision_count"] == 1
    assert "## Limitations" in update["draft"]


def test_critic_node_parses_score_into_critique(
    nodes: ResearchNodes, researched_state: ResearchState
) -> None:
    """The critic's raw text is parsed into a structured Critique."""
    researched_state.update(nodes.write(researched_state))
    update = nodes.critique(researched_state)
    critique = update["critique"]
    assert isinstance(critique, Critique)
    assert 0.0 <= critique.score <= 10.0
    assert critique.feedback


def test_route_after_critic_enforces_revision_budget(nodes: ResearchNodes) -> None:
    """Low scores revise until the bound; good scores accept immediately."""
    low = Critique(score=5.0, feedback=["weak"])
    high = Critique(score=9.0, feedback=[])
    assert nodes.route_after_critic({"critique": low, "revision_count": 0}) == "revise"
    assert nodes.route_after_critic({"critique": low, "revision_count": 1}) == "accept"
    assert nodes.route_after_critic({"critique": high, "revision_count": 0}) == "accept"
    assert nodes.route_after_critic({"critique": None, "revision_count": 1}) == "accept"


def test_approval_gate_auto_approves_when_disabled(nodes: ResearchNodes) -> None:
    """With the flag off, the gate approves without interrupting."""
    assert nodes.approve({"question": "Q?"}) == {"approved": True}
