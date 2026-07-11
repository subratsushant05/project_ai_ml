"""Agent nodes, routing functions, and LLM-output parsers for the graph."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from langgraph.types import interrupt

from agent_researcher import prompts
from agent_researcher.state import (
    Citation,
    Critique,
    Finding,
    ResearchState,
    collect_citations,
)

if TYPE_CHECKING:
    from agent_researcher.config import Settings
    from agent_researcher.llm import ChatModel
    from agent_researcher.search import SearchResult, SearchTool

logger = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"^(?:\d+\s*[.)]\s*|[-*]\s+)")
_SCORE_RE = re.compile(r"score\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_plan(text: str) -> list[str]:
    """Extract sub-questions from a planner completion.

    Numbered or bulleted lines are preferred; if none exist, bare lines
    ending in ``?`` are used so chatty model output still parses.

    Args:
        text: Raw planner output.

    Returns:
        Deduplicated sub-questions in their original order.
    """
    bulleted: list[str] = []
    bare_questions: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _BULLET_RE.match(line)
        candidate = line[match.end() :].strip() if match else line
        if not candidate:
            continue
        if match and candidate not in bulleted:
            bulleted.append(candidate)
        elif not match and candidate.endswith("?") and candidate not in bare_questions:
            bare_questions.append(candidate)
    return bulleted or bare_questions


def parse_score(text: str) -> float | None:
    """Extract a ``SCORE: <number>`` value from a critic completion.

    Args:
        text: Raw critic output.

    Returns:
        The score clamped to ``[0, 10]``, or ``None`` if absent.
    """
    match = _SCORE_RE.search(text)
    if match is None:
        return None
    return min(max(float(match.group(1)), 0.0), 10.0)


def parse_feedback(text: str) -> list[str]:
    """Extract the bullet list following ``FEEDBACK:`` from critic output."""
    feedback: list[str] = []
    capturing = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("FEEDBACK"):
            capturing = True
            continue
        if capturing and line.startswith(("-", "*")):
            feedback.append(line.lstrip("-* ").strip())
    return feedback


class ResearchNodes:
    """The four agents plus routing logic, bound to concrete providers.

    Args:
        model: Chat model used by every agent.
        search_tool: Search backend used by the researcher.
        settings: Pipeline configuration (thresholds, bounds, flags).
    """

    def __init__(
        self, model: ChatModel, search_tool: SearchTool, settings: Settings
    ) -> None:
        self._model = model
        self._search = search_tool
        self._settings = settings

    def plan(self, state: ResearchState) -> dict[str, Any]:
        """Planner: break the question into sub-questions."""
        user = prompts.PLANNER_USER.format(
            question=state["question"], n=self._settings.num_sub_questions
        )
        raw = self._model.invoke(prompts.PLANNER_SYSTEM, user)
        plan = parse_plan(raw)[: self._settings.num_sub_questions]
        if not plan:
            raise ValueError(f"Planner produced no parseable sub-questions: {raw!r}")
        logger.info("Planner produced %d sub-questions", len(plan))
        return {"plan": plan, "revision_count": 0, "critique": None}

    def research(self, state: ResearchState) -> dict[str, Any]:
        """Researcher: answer each sub-question from retrieved sources."""
        registry: dict[str, Citation] = {}
        findings: list[Finding] = []
        for sub_question in state["plan"]:
            results = self._search.search(
                sub_question, top_k=self._settings.search_top_k
            )
            citations = [self._register(registry, result) for result in results]
            user = prompts.RESEARCHER_USER.format(
                sub_question=sub_question, sources=prompts.format_sources(citations)
            )
            answer = self._model.invoke(prompts.RESEARCHER_SYSTEM, user)
            findings.append(
                Finding(sub_question=sub_question, answer=answer, sources=citations)
            )
            logger.info(
                "Researcher answered %r with %d sources", sub_question, len(citations)
            )
        return {"findings": findings}

    @staticmethod
    def _register(registry: dict[str, Citation], result: SearchResult) -> Citation:
        """Assign a stable, report-wide citation number to a search result."""
        if result.url not in registry:
            registry[result.url] = Citation(
                index=len(registry) + 1,
                title=result.title,
                url=result.url,
                snippet=result.snippet,
            )
        return registry[result.url]

    def approve(self, state: ResearchState) -> dict[str, Any]:
        """Approval gate: optionally pause for a human before drafting."""
        if not self._settings.require_approval:
            return {"approved": True}
        decision = interrupt(
            {
                "message": "Approve the research plan and findings before drafting?",
                "plan": state.get("plan", []),
                "findings_count": len(state.get("findings", [])),
            }
        )
        logger.info("Approval gate resumed with decision=%r", decision)
        return {"approved": bool(decision)}

    def write(self, state: ResearchState) -> dict[str, Any]:
        """Writer: synthesize (or revise) the markdown report."""
        findings = state.get("findings", [])
        citations = collect_citations(findings)
        critique = state.get("critique")
        user = prompts.WRITER_USER.format(
            question=state["question"],
            findings=prompts.format_findings(findings),
            references=prompts.format_references(citations),
            critique=prompts.format_critique(critique),
        )
        draft = self._model.invoke(prompts.WRITER_SYSTEM, user)
        revision_count = state.get("revision_count", 0) + (1 if critique else 0)
        logger.info(
            "Writer produced draft #%d (%d chars)", revision_count + 1, len(draft)
        )
        return {"draft": draft, "revision_count": revision_count}

    def critique(self, state: ResearchState) -> dict[str, Any]:
        """Critic: score the draft and list actionable issues."""
        user = prompts.CRITIC_USER.format(
            question=state["question"], draft=state.get("draft", "")
        )
        raw = self._model.invoke(prompts.CRITIC_SYSTEM, user)
        score = parse_score(raw)
        if score is None:
            logger.warning("Critic output had no parseable score; treating as 0")
            score = 0.0
        critique = Critique(score=score, feedback=parse_feedback(raw))
        logger.info(
            "Critic scored the draft %.1f/10 (%d issues)",
            critique.score,
            len(critique.feedback),
        )
        return {"critique": critique}

    def route_after_approval(self, state: ResearchState) -> Literal["write", "end"]:
        """Route to the writer when approved, otherwise finish early."""
        return "write" if state.get("approved") else "end"

    def route_after_critic(self, state: ResearchState) -> Literal["revise", "accept"]:
        """Bounded revision loop: revise at most ``max_revisions`` times."""
        critique = state.get("critique")
        score = critique.score if critique else 0.0
        if score >= self._settings.quality_threshold:
            return "accept"
        if state.get("revision_count", 0) >= self._settings.max_revisions:
            logger.info("Revision budget exhausted; accepting draft as-is")
            return "accept"
        return "revise"
