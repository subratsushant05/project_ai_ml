"""Trajectory metrics: each one is a small class implementing ``Metric``.

Every metric returns a :class:`~agent_evals.models.MetricResult` with a
normalized score in ``[0, 1]`` plus a details dict for debugging, so metrics
compose uniformly in the runner, in reports, and in CI thresholds.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Protocol, runtime_checkable

from agent_evals.judge import Judge
from agent_evals.models import MetricResult, StepType, TestCase, Trajectory
from agent_evals.settings import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class Metric(Protocol):
    """Anything that can score a trajectory against a test case."""

    name: str

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        """Score one trajectory against one test case."""
        ...


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Edit distance between two token sequences.

    Args:
        a: First sequence (e.g. expected tool names).
        b: Second sequence (e.g. actual tool names).

    Returns:
        Minimum number of insertions, deletions, and substitutions.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, start=1):
        curr = [i]
        for j, tok_b in enumerate(b, start=1):
            cost = 0 if tok_a == tok_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


class ToolSelectionAccuracy:
    """Did the agent call the right *set* of tools? Score is the F1.

    Precision/recall are computed on the set of distinct tool names versus
    ``expected_tools``. Order and repetition are ignored here on purpose --
    they are covered by :class:`ToolCallOrder` and
    :class:`TrajectoryEfficiency`.
    """

    name = "tool_selection"

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        """Score set overlap between expected and actual tools."""
        expected = set(case.expected_tools)
        actual = set(trajectory.tool_sequence)
        if not expected and not actual:
            return MetricResult(
                metric=self.name,
                score=1.0,
                details={"precision": 1.0, "recall": 1.0, "f1": 1.0},
            )
        tp = len(expected & actual)
        precision = tp / len(actual) if actual else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return MetricResult(
            metric=self.name,
            score=round(f1, 4),
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
            },
        )


class ToolCallOrder:
    """Did the agent call tools in the right *order*?

    Score is ``1 - edit_distance / max(len(expected), len(actual))`` between
    the reference tool sequence and the actual one, so 1.0 is an exact
    sequence match and 0.0 shares nothing.
    """

    name = "tool_order"

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        """Score normalized edit distance between tool sequences."""
        expected = case.reference_sequence
        actual = trajectory.tool_sequence
        if not expected and not actual:
            return MetricResult(metric=self.name, score=1.0, details={"edit_distance": 0})
        distance = levenshtein(expected, actual)
        denom = max(len(expected), len(actual))
        score = 1.0 - distance / denom
        return MetricResult(
            metric=self.name,
            score=round(score, 4),
            details={
                "edit_distance": distance,
                "expected_sequence": list(expected),
                "actual_sequence": list(actual),
            },
        )


class TrajectoryEfficiency:
    """Did the agent get there without wandering?

    Score is ``reference_steps / actual_steps`` (capped at 1.0), where
    ``reference_steps`` is the length of the ideal tool sequence. Loops and
    redundant calls inflate ``actual_steps`` and drag the score down; exact
    duplicate calls (same tool, same arguments) are also surfaced in details.
    """

    name = "efficiency"

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        """Score step count against the reference minimum."""
        reference_steps = len(case.reference_sequence)
        calls = trajectory.tool_calls
        actual_steps = len(calls)
        signatures = [(c.name, tuple(sorted((c.arguments or {}).items()))) for c in calls]
        redundant = actual_steps - len(set(signatures))
        if actual_steps == 0:
            score = 1.0 if reference_steps == 0 else 0.0
        elif reference_steps == 0:
            score = 0.0  # did work where none was needed
        else:
            score = min(1.0, reference_steps / actual_steps)
        return MetricResult(
            metric=self.name,
            score=round(score, 4),
            details={
                "reference_steps": reference_steps,
                "actual_steps": actual_steps,
                "redundant_calls": redundant,
            },
        )


class AnswerCorrectness:
    """Is the final answer right?

    Exact match (case/whitespace-insensitive) scores 1.0 outright. Otherwise
    the score is a fuzzy ratio (``difflib.SequenceMatcher``), blended 50/50
    with a judge verdict when a judge is supplied.
    """

    name = "answer_correctness"

    def __init__(self, judge: Judge | None = None) -> None:
        self._judge = judge

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        """Score the final answer against the expected answer."""
        expected = case.expected_answer.strip()
        actual = trajectory.final_answer.strip()
        exact = expected.casefold() == actual.casefold() and bool(expected)
        fuzzy = SequenceMatcher(None, expected.casefold(), actual.casefold()).ratio()
        details: dict[str, object] = {"exact": exact, "fuzzy": round(fuzzy, 4)}
        if exact:
            return MetricResult(metric=self.name, score=1.0, details=details)
        score = fuzzy
        if self._judge is not None:
            verdict = self._judge.evaluate(case.input, expected, actual)
            details["judge"] = self._judge.name
            details["judge_score"] = verdict.score
            details["judge_rationale"] = verdict.rationale
            score = 0.5 * fuzzy + 0.5 * verdict.score
        return MetricResult(metric=self.name, score=round(score, 4), details=details)


class CostLatency:
    """How expensive and how slow was the run?

    Cost is computed from per-model prices (``Settings.prices``); latency is
    trajectory wall time. Both are mapped to ``[0, 1]`` with a smooth decay
    ``exp(-value / budget)`` so cheaper/faster is strictly better, and the
    score is their mean. Raw values live in details for reporting.
    """

    name = "cost_latency"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    def cost_usd(self, trajectory: Trajectory) -> float:
        """Dollar cost of all LLM calls in a trajectory."""
        total = 0.0
        for step in trajectory.steps:
            if step.type is not StepType.LLM_CALL:
                continue
            price = self._settings.price_for(step.model)
            total += step.prompt_tokens / 1e6 * price.prompt_per_1m
            total += step.completion_tokens / 1e6 * price.completion_per_1m
        return total

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        """Score cost and latency against configured budgets."""
        cost = self.cost_usd(trajectory)
        latency = trajectory.wall_time_s
        cost_score = math.exp(-cost / self._settings.cost_budget_usd)
        latency_score = math.exp(-latency / self._settings.latency_budget_s)
        score = (cost_score + latency_score) / 2
        return MetricResult(
            metric=self.name,
            score=round(score, 4),
            details={
                "cost_usd": round(cost, 6),
                "total_tokens": trajectory.total_tokens,
                "wall_time_s": round(latency, 3),
                "cost_budget_usd": self._settings.cost_budget_usd,
                "latency_budget_s": self._settings.latency_budget_s,
            },
        )


def default_metrics(judge: Judge | None = None, settings: Settings | None = None) -> list[Metric]:
    """The standard metric suite, in report order.

    Args:
        judge: Optional judge for :class:`AnswerCorrectness`.
        settings: Optional settings for :class:`CostLatency` pricing.

    Returns:
        Fresh instances of all five bundled metrics.
    """
    return [
        ToolSelectionAccuracy(),
        ToolCallOrder(),
        TrajectoryEfficiency(),
        AnswerCorrectness(judge=judge),
        CostLatency(settings=settings),
    ]
