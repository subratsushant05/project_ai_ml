"""Typed graph state and the structured records that flow through it."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single source citation, numbered globally across the report.

    Attributes:
        index: 1-based citation number shared by inline markers and the
            references section.
        title: Source document title.
        url: Source location.
        snippet: Short excerpt used as evidence.
    """

    index: int = Field(ge=1)
    title: str
    url: str
    snippet: str


class Finding(BaseModel):
    """The researcher's citation-backed answer to one sub-question.

    Attributes:
        sub_question: The sub-question this finding answers.
        answer: Concise answer with inline ``[n]`` citation markers.
        sources: Citations backing the answer.
    """

    sub_question: str
    answer: str
    sources: list[Citation] = Field(default_factory=list)


class Critique(BaseModel):
    """The critic's structured review of a draft.

    Attributes:
        score: Overall quality score from 0 (unusable) to 10 (excellent).
        feedback: Concrete, actionable issues found in the draft.
    """

    score: float = Field(ge=0.0, le=10.0)
    feedback: list[str] = Field(default_factory=list)


class ResearchState(TypedDict, total=False):
    """Shared state for the research graph.

    ``findings`` uses an additive reducer so parallel or repeated research
    steps accumulate instead of overwriting each other. Every other key is
    replaced by the most recent node update.

    Keys:
        question: The user's research question.
        plan: Sub-questions produced by the Planner.
        findings: Citation-backed answers produced by the Researcher.
        draft: Current markdown report produced by the Writer.
        critique: Latest review produced by the Critic.
        revision_count: Number of revisions the Writer has performed.
        approved: Whether the human approval gate passed.
    """

    question: str
    plan: list[str]
    findings: Annotated[list[Finding], operator.add]
    draft: str
    critique: Critique | None
    revision_count: int
    approved: bool


def collect_citations(findings: list[Finding]) -> list[Citation]:
    """Return the unique citations across ``findings``, ordered by index.

    Args:
        findings: Findings whose sources should be merged.

    Returns:
        Deduplicated citations sorted by their global citation number.
    """
    seen: dict[int, Citation] = {}
    for finding in findings:
        for citation in finding.sources:
            seen.setdefault(citation.index, citation)
    return [seen[index] for index in sorted(seen)]
