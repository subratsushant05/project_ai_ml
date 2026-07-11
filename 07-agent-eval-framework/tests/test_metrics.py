"""Metric unit tests against hand-computed toy cases."""

import math

import pytest

from agent_evals.metrics import (
    AnswerCorrectness,
    CostLatency,
    ToolCallOrder,
    ToolSelectionAccuracy,
    TrajectoryEfficiency,
    levenshtein,
)
from agent_evals.models import Step, StepType, TestCase, Trajectory
from agent_evals.settings import Settings


def make_trajectory(tools: list[str], answer: str = "") -> Trajectory:
    """Build a minimal trajectory with the given tool-call sequence."""
    steps = [
        Step(type=StepType.TOOL_CALL, timestamp=float(i), name=name, arguments={"input": "x"})
        for i, name in enumerate(tools)
    ]
    steps.append(Step(type=StepType.FINAL_ANSWER, timestamp=float(len(tools)), content=answer))
    return Trajectory(input="q", steps=steps)


def test_tool_selection_precision_recall_hand_computed() -> None:
    # expected {a, b, c}; actual {a, b, d}: tp=2, precision=2/3, recall=2/3.
    case = TestCase(id="t", input="q", expected_tools=["a", "b", "c"])
    result = ToolSelectionAccuracy().score(case, make_trajectory(["a", "b", "d"]))
    assert result.details["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert result.details["recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert result.score == pytest.approx(2 / 3, abs=1e-3)  # F1 of equal P/R
    assert result.details["missing"] == ["c"]
    assert result.details["unexpected"] == ["d"]


def test_tool_selection_perfect_and_empty() -> None:
    case = TestCase(id="t", input="q", expected_tools=["a"])
    assert ToolSelectionAccuracy().score(case, make_trajectory(["a", "a"])).score == 1.0
    empty = TestCase(id="t", input="q", expected_tools=[])
    assert ToolSelectionAccuracy().score(empty, make_trajectory([])).score == 1.0
    assert ToolSelectionAccuracy().score(case, make_trajectory([])).score == 0.0


def test_levenshtein_hand_computed() -> None:
    assert levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1  # substitution
    assert levenshtein(["a", "b"], ["a", "b", "c"]) == 1  # insertion
    assert levenshtein(["a", "b", "c"], []) == 3  # deletions
    # abcd -> bad: delete "a", keep "b", substitute "c" -> "a", keep "d".
    assert levenshtein(["a", "b", "c", "d"], ["b", "a", "d"]) == 2


def test_tool_order_normalization() -> None:
    case = TestCase(id="t", input="q", expected_tools=["a", "b"])
    assert ToolCallOrder().score(case, make_trajectory(["a", "b"])).score == 1.0
    # distance 2 (swap = 2 substitutions? no: b,a vs a,b -> 2 subs) over max len 2.
    swapped = ToolCallOrder().score(case, make_trajectory(["b", "a"]))
    assert swapped.score == pytest.approx(1 - levenshtein(["a", "b"], ["b", "a"]) / 2)
    # prefix noise: expected [a], actual [x, a] -> distance 1, denom 2 -> 0.5.
    single = TestCase(id="t", input="q", expected_tools=["a"])
    assert ToolCallOrder().score(single, make_trajectory(["x", "a"])).score == 0.5


def test_tool_order_uses_reference_trajectory_when_present() -> None:
    case = TestCase(
        id="t", input="q", expected_tools=["a", "b"], reference_trajectory=["a", "a", "b"]
    )
    assert ToolCallOrder().score(case, make_trajectory(["a", "a", "b"])).score == 1.0


def test_efficiency_hand_computed() -> None:
    case = TestCase(id="t", input="q", expected_tools=["a", "b"])
    assert TrajectoryEfficiency().score(case, make_trajectory(["a", "b"])).score == 1.0
    # 4 calls for a 2-step reference -> 0.5, with 1 exact duplicate flagged.
    result = TrajectoryEfficiency().score(case, make_trajectory(["a", "a", "x", "b"]))
    assert result.score == 0.5
    assert result.details["redundant_calls"] == 1
    # No tools expected but tools called -> 0; nothing expected, nothing done -> 1.
    empty = TestCase(id="t", input="q", expected_tools=[])
    assert TrajectoryEfficiency().score(empty, make_trajectory(["a"])).score == 0.0
    assert TrajectoryEfficiency().score(empty, make_trajectory([])).score == 1.0


def test_answer_correctness_exact_and_fuzzy() -> None:
    case = TestCase(id="t", input="q", expected_answer="The answer is 42.")
    metric = AnswerCorrectness()
    exact = metric.score(case, make_trajectory([], "the answer is 42."))
    assert exact.score == 1.0 and exact.details["exact"] is True
    fuzzy = metric.score(case, make_trajectory([], "The answer is 43."))
    assert 0.5 < fuzzy.score < 1.0 and fuzzy.details["exact"] is False
    assert metric.score(case, make_trajectory([], "")).score < 0.2


def test_cost_latency_hand_computed_pricing() -> None:
    settings = Settings()
    metric = CostLatency(settings=settings)
    steps = [
        Step(type=StepType.LLM_CALL, timestamp=0.0, model="sim-large",
             prompt_tokens=1000, completion_tokens=500),
        Step(type=StepType.FINAL_ANSWER, timestamp=2.0, content="done"),
    ]
    trajectory = Trajectory(input="q", steps=steps)
    # sim-large: $2.50/1M prompt, $10.00/1M completion.
    expected_cost = 1000 / 1e6 * 2.50 + 500 / 1e6 * 10.00
    assert metric.cost_usd(trajectory) == pytest.approx(expected_cost)
    result = metric.score(TestCase(id="t", input="q"), trajectory)
    assert result.details["cost_usd"] == pytest.approx(expected_cost, abs=1e-6)
    assert result.details["wall_time_s"] == pytest.approx(2.0)
    expected_score = (
        math.exp(-expected_cost / settings.cost_budget_usd)
        + math.exp(-2.0 / settings.latency_budget_s)
    ) / 2
    assert result.score == pytest.approx(expected_score, abs=1e-3)


def test_cost_latency_monotonic_in_cost() -> None:
    metric = CostLatency(settings=Settings())
    case = TestCase(id="t", input="q")

    def traj(tokens: int) -> Trajectory:
        return Trajectory(input="q", steps=[
            Step(type=StepType.LLM_CALL, timestamp=0.0, model="sim-large",
                 prompt_tokens=tokens, completion_tokens=tokens),
            Step(type=StepType.FINAL_ANSWER, timestamp=1.0, content="x"),
        ])

    assert metric.score(case, traj(100)).score > metric.score(case, traj(10_000)).score
