"""Deterministic, rule-based chat model for offline runs and tests.

``OfflineChatModel`` implements the same ``invoke(system, user)`` interface
as the hosted adapters, but produces its output from templates plus the
structured content embedded in each prompt (search snippets, findings,
critique bullets). It performs no network calls and is fully deterministic,
which keeps the demo reproducible and the test suite fast.
"""

from __future__ import annotations

import re

from agent_researcher import prompts

_LEAD_IN_RE = re.compile(
    r"^(?:how|what|why|when|where|which)\s+"
    r"(?:do|does|did|is|are|can|could|should|would|will|might)\s+",
    re.IGNORECASE,
)
_TRAILING_VERB_RE = re.compile(
    r"\s+(?:work|works|function|functions|operate|operates)\s*$", re.IGNORECASE
)
_SOURCE_HEADER_RE = re.compile(r"^\[(\d+)\]\s*(.+?)\s*\|\s*(\S+)\s*$")
_PLAN_COUNT_RE = re.compile(r"exactly (\d+) sub-questions")

_SUB_QUESTION_TEMPLATES = (
    "What are the core principles behind {topic}?",
    "What are the main real-world applications of {topic}?",
    "What are the key challenges and limitations of {topic}?",
    "What future developments are expected for {topic}?",
)


def _between(text: str, start: str, end: str | None = None) -> str:
    """Return the stripped substring of ``text`` between two markers."""
    lo = text.find(start)
    if lo == -1:
        return ""
    lo += len(start)
    hi = text.find(end, lo) if end is not None else -1
    return (text[lo:hi] if hi != -1 else text[lo:]).strip()


def _first_line_field(text: str, name: str) -> str:
    """Extract a single-line ``NAME: value`` field from ``text``."""
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` with a trailing period."""
    head = re.split(r"(?<=\.)\s+", text.strip(), maxsplit=1)[0].rstrip(".")
    return f"{head}."


def _topic(question: str) -> str:
    """Reduce a research question to its noun-phrase topic."""
    topic = question.strip().rstrip("?!. ").strip()
    topic = _LEAD_IN_RE.sub("", topic)
    topic = _TRAILING_VERB_RE.sub("", topic)
    topic = re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.IGNORECASE)
    return topic or question.strip().rstrip("?")


class OfflineChatModel:
    """Rule-based chat model that emulates each agent role deterministically.

    The role is inferred from a distinctive phrase in the system prompt, so
    the same nodes work unchanged against hosted providers.
    """

    def invoke(self, system: str, user: str) -> str:
        """Produce a deterministic completion for the given prompt pair.

        Args:
            system: System prompt identifying the agent role.
            user: Structured user prompt built from ``prompts`` templates.

        Returns:
            The generated text for that role.

        Raises:
            ValueError: If the system prompt matches no known agent role.
        """
        role = system.lower()
        if "research planner" in role:
            return self._plan(user)
        if "research analyst" in role:
            return self._answer(user)
        if "technical writer" in role:
            return self._write(user)
        if "critical reviewer" in role:
            return self._review(user)
        raise ValueError("OfflineChatModel received an unrecognized system prompt")

    def _plan(self, user: str) -> str:
        """Break the question into templated sub-questions."""
        question = _first_line_field(user, "QUESTION")
        count_match = _PLAN_COUNT_RE.search(user)
        count = int(count_match.group(1)) if count_match else 3
        count = max(1, min(count, len(_SUB_QUESTION_TEMPLATES)))
        topic = _topic(question)
        return "\n".join(
            f"{i}. {template.format(topic=topic)}"
            for i, template in enumerate(_SUB_QUESTION_TEMPLATES[:count], start=1)
        )

    def _answer(self, user: str) -> str:
        """Compose a citation-backed answer from the provided sources."""
        sources = _between(user, "SOURCES:\n", f"\n\n{prompts.RESEARCHER_TAIL}")
        sentences: list[str] = []
        for block in sources.split("\n\n"):
            lines = block.strip().splitlines()
            if not lines:
                continue
            header = _SOURCE_HEADER_RE.match(lines[0])
            if header is None:
                continue
            index = header.group(1)
            snippet = " ".join(line.strip() for line in lines[1:])
            if snippet:
                sentences.append(f"{_first_sentence(snippet).rstrip('.')} [{index}].")
        if not sentences:
            return "No relevant sources were available to answer this sub-question."
        return " ".join(sentences)

    def _write(self, user: str) -> str:
        """Assemble the markdown report from findings and references."""
        question = _first_line_field(user, "QUESTION")
        findings = _between(user, "FINDINGS:\n", "\nREFERENCES:")
        references = _between(user, "REFERENCES:\n", "\nCRITIQUE:")
        critique = _between(user, "CRITIQUE:\n", f"\n\n{prompts.WRITER_TAIL}")
        section_count = findings.count("### ")
        reference_count = sum(1 for line in references.splitlines() if line.strip())

        parts = [
            f"# Research Report: {question}",
            "",
            "## Overview",
            "",
            f'This report addresses the question "{question}". The analysis is '
            f"organized around {section_count} sub-questions and draws on "
            f"{reference_count} distinct sources, cited inline by number and "
            "listed under References.",
            "",
            "## Findings",
            "",
            findings,
            "",
            "## Key Takeaways",
            "",
            self._takeaways(findings),
        ]
        if critique and critique != "None":
            parts += ["", "## Limitations and Further Research", ""]
            parts.append(self._limitations(critique))
        parts += ["", "## References", "", references, ""]
        return "\n".join(parts)

    @staticmethod
    def _takeaways(findings: str) -> str:
        """Derive one bullet per finding from its opening sentence."""
        bullets = []
        for chunk in findings.split("### ")[1:]:
            lines = chunk.strip().splitlines()
            body = " ".join(line for line in lines[1:] if line.strip())
            if body:
                bullets.append(f"- {_first_sentence(body)}")
        return "\n".join(bullets) or "- No findings were available."

    @staticmethod
    def _limitations(critique: str) -> str:
        """Write a limitations section that addresses the critique bullets."""
        issues = [
            line.strip().lstrip("- ").strip()
            for line in critique.splitlines()
            if line.strip().startswith("-")
        ]
        text = (
            "This report synthesizes a small curated corpus rather than an "
            "exhaustive literature search, so coverage is indicative rather "
            "than complete. Findings should be validated against primary "
            "sources before high-stakes use."
        )
        if issues:
            text += "\n\nReviewer feedback addressed in this revision:\n"
            text += "\n".join(f"- {issue}" for issue in issues)
        return text

    def _review(self, user: str) -> str:
        """Score the draft against a fixed rubric and list issues."""
        draft = _between(user, "DRAFT:\n", f"\n\n{prompts.CRITIC_TAIL}")
        issues: list[tuple[str, int]] = []
        if "## References" not in draft:
            issues.append(("Add a numbered references section.", 3))
        if draft.count("### ") < 2:
            issues.append(
                ("Organize findings into one subsection per sub-question.", 2)
            )
        if not re.search(r"\[\d+\]", draft):
            issues.append(("Cite sources inline using [n] markers.", 2))
        if "## Limitations" not in draft:
            issues.append(("Add a section on limitations and further research.", 3))
        if len(draft) < 400:
            issues.append(("Expand the report; it is too brief to be useful.", 1))

        score = max(0, 10 - sum(weight for _, weight in issues))
        feedback = "\n".join(f"- {issue}" for issue, _ in issues)
        return f"SCORE: {score}\nFEEDBACK:\n{feedback}"
