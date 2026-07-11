"""LLM-as-judge behind a small interface, with an offline default.

The default :class:`OfflineJudge` is deterministic and dependency-free so the
whole eval suite runs in CI with no API keys. Remote judges (OpenAI,
Anthropic) are imported lazily and selected via ``AGENT_EVALS_JUDGE``.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_evals.settings import Settings

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on "
    "or that the their there this to was were what which who will with would "
    "you your not does did do".split()
)

_RUBRIC = (
    "You are grading an AI agent's answer.\n"
    "Question: {question}\n"
    "Reference answer: {expected}\n"
    "Agent answer: {actual}\n\n"
    "Score the agent answer for factual agreement with the reference on a "
    "scale from 0.0 (contradicts or misses it) to 1.0 (fully equivalent). "
    "Reply with exactly two lines:\nSCORE: <float>\nRATIONALE: <one sentence>"
)


class JudgeVerdict(BaseModel):
    """A judge's assessment of one answer.

    Attributes:
        score: Quality score in ``[0, 1]``.
        rationale: Short human-readable justification.
    """

    score: float
    rationale: str


@runtime_checkable
class Judge(Protocol):
    """Anything that can grade an answer against a reference."""

    name: str

    def evaluate(self, question: str, expected: str, actual: str) -> JudgeVerdict:
        """Grade ``actual`` against ``expected`` for the given question."""
        ...


def _keywords(text: str) -> list[str]:
    """Extract ordered, de-duplicated content keywords from text."""
    tokens = re.findall(r"[a-z0-9.]+", text.lower())
    seen: list[str] = []
    for tok in tokens:
        tok = tok.strip(".")
        if not tok or tok in _STOPWORDS or (len(tok) < 3 and not tok.isdigit()):
            continue
        if tok not in seen:
            seen.append(tok)
    return seen


class OfflineJudge:
    """Deterministic rubric-based judge: keyword coverage of the reference.

    The score is the fraction of content keywords from the expected answer
    that appear in the agent's answer. It is intentionally simple -- the point
    is a stable, offline, explainable baseline that never flakes in CI.
    """

    name = "offline"

    def evaluate(self, question: str, expected: str, actual: str) -> JudgeVerdict:
        """Grade by keyword coverage.

        Args:
            question: The original task (unused, kept for interface parity).
            expected: Reference answer.
            actual: Agent answer.

        Returns:
            Verdict with coverage score and a rationale naming missed terms.
        """
        keywords = _keywords(expected)
        if not keywords:
            score = 1.0 if actual.strip() else 0.0
            return JudgeVerdict(score=score, rationale="No reference keywords; graded on non-empty answer.")
        actual_tokens = set(_keywords(actual))
        actual_lower = actual.lower()
        hits = [k for k in keywords if k in actual_tokens or k in actual_lower]
        missed = [k for k in keywords if k not in hits]
        score = len(hits) / len(keywords)
        rationale = f"Covered {len(hits)}/{len(keywords)} reference keywords."
        if missed:
            rationale += " Missing: " + ", ".join(missed[:6]) + "."
        return JudgeVerdict(score=round(score, 4), rationale=rationale)


def _parse_verdict(text: str) -> JudgeVerdict:
    """Parse a ``SCORE:`` / ``RATIONALE:`` reply from a remote judge."""
    score_match = re.search(r"SCORE:\s*([01](?:\.\d+)?)", text)
    rationale_match = re.search(r"RATIONALE:\s*(.+)", text)
    score = float(score_match.group(1)) if score_match else 0.0
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()[:200]
    return JudgeVerdict(score=min(max(score, 0.0), 1.0), rationale=rationale)


class OpenAIJudge:
    """LLM judge backed by the OpenAI API (lazy import, needs an API key)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # deferred: optional dependency

        self._client = OpenAI()
        self._model = model

    def evaluate(self, question: str, expected: str, actual: str) -> JudgeVerdict:
        """Grade via a single chat completion against the rubric."""
        prompt = _RUBRIC.format(question=question, expected=expected, actual=actual)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return _parse_verdict(response.choices[0].message.content or "")


class AnthropicJudge:
    """LLM judge backed by the Anthropic API (lazy import, needs an API key)."""

    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        from anthropic import Anthropic  # deferred: optional dependency

        self._client = Anthropic()
        self._model = model

    def evaluate(self, question: str, expected: str, actual: str) -> JudgeVerdict:
        """Grade via a single message against the rubric."""
        prompt = _RUBRIC.format(question=question, expected=expected, actual=actual)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return _parse_verdict(text)


def load_judge(name: str | None = None, settings: Settings | None = None) -> Judge:
    """Instantiate a judge by name, defaulting to settings/env config.

    Args:
        name: ``offline`` | ``openai`` | ``anthropic``. ``None`` reads
            ``AGENT_EVALS_JUDGE`` (default ``offline``).
        settings: Optional settings instance (mainly for tests).

    Returns:
        A ready-to-use judge.

    Raises:
        ValueError: If the name is not a known judge backend.
    """
    settings = settings or Settings()
    name = name or settings.judge
    logger.debug("Loading judge backend: %s", name)
    if name == "offline":
        return OfflineJudge()
    if name == "openai":
        return OpenAIJudge(model=settings.judge_model)
    if name == "anthropic":
        return AnthropicJudge(model=settings.judge_model)
    raise ValueError(f"Unknown judge backend: {name!r} (expected offline|openai|anthropic)")
