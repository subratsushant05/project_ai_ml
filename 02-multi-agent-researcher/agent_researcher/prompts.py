"""Prompt templates and formatting helpers shared by every agent node.

The structured markers (``QUESTION:``, ``SOURCES:``, ...) serve two
purposes: they keep prompts unambiguous for hosted LLMs, and they give the
deterministic :class:`~agent_researcher.offline_llm.OfflineChatModel` a
stable format to parse. The ``*_TAIL`` constants are the closing
instructions of each user prompt and double as section terminators for the
offline parser, so the two can never drift apart.
"""

from __future__ import annotations

from agent_researcher.state import Citation, Critique, Finding

PLANNER_SYSTEM = (
    "You are a research planner. Break the research question into focused, "
    "independently answerable sub-questions. Respond with a numbered list "
    "only, one sub-question per line."
)
PLANNER_USER = (
    "QUESTION: {question}\n\n"
    "Produce exactly {n} sub-questions that together answer the question."
)

RESEARCHER_SYSTEM = (
    "You are a research analyst. Answer the sub-question using only the "
    "numbered sources provided, citing them inline as [n]. Be concise and "
    "factual; do not invent information."
)
RESEARCHER_TAIL = "Write a short, citation-backed answer."
RESEARCHER_USER = (
    "SUB-QUESTION: {sub_question}\n\nSOURCES:\n{sources}\n\n" + RESEARCHER_TAIL
)

WRITER_SYSTEM = (
    "You are a technical writer. Produce a structured markdown research "
    "report with an overview, one subsection per sub-question, key "
    "takeaways, and a numbered references section. Preserve every inline "
    "[n] citation marker."
)
WRITER_TAIL = (
    "Write the full report in markdown. If the critique is not 'None', "
    "address every point it raises."
)
WRITER_USER = (
    "QUESTION: {question}\n\nFINDINGS:\n{findings}\n\n"
    "REFERENCES:\n{references}\n\nCRITIQUE:\n{critique}\n\n" + WRITER_TAIL
)

CRITIC_SYSTEM = (
    "You are a critical reviewer. Score the draft from 0 to 10 and list "
    "concrete, actionable issues. Respond with a line 'SCORE: <number>' "
    "followed by 'FEEDBACK:' and one issue per '-' bullet."
)
CRITIC_TAIL = "Review the draft."
CRITIC_USER = "QUESTION: {question}\n\nDRAFT:\n{draft}\n\n" + CRITIC_TAIL


def format_sources(citations: list[Citation]) -> str:
    """Render citations as numbered source blocks for the researcher prompt.

    Args:
        citations: Sources retrieved for one sub-question.

    Returns:
        One ``[n] title | url`` block per source with its snippet, or a
        fixed sentinel when nothing was retrieved.
    """
    if not citations:
        return "No sources found."
    return "\n\n".join(
        f"[{c.index}] {c.title} | {c.url}\n{c.snippet}" for c in citations
    )


def format_findings(findings: list[Finding]) -> str:
    """Render findings as markdown subsections for the writer prompt."""
    return "\n\n".join(f"### {f.sub_question}\n\n{f.answer}" for f in findings)


def format_references(citations: list[Citation]) -> str:
    """Render the global citation list as numbered reference lines."""
    return "\n".join(f"[{c.index}] {c.title} - {c.url}" for c in citations)


def format_critique(critique: Critique | None) -> str:
    """Render a critique for the writer prompt (``None`` on the first pass)."""
    if critique is None:
        return "None"
    lines = [f"Score: {critique.score:g}/10"]
    lines.extend(f"- {issue}" for issue in critique.feedback)
    return "\n".join(lines)
