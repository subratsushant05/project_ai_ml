"""Runner aggregation, bundled agents, and the CI threshold helper."""

from pathlib import Path

import pytest

from agent_evals.agents import GoodAgent, SloppyAgent
from agent_evals.models import (
    Dataset,
    MetricResult,
    Step,
    StepType,
    TestCase,
    Trajectory,
)
from agent_evals.runner import Runner, percentile
from agent_evals.testing import assert_agent_passes

DATASET = Dataset.from_jsonl(Path(__file__).resolve().parent.parent / "datasets" / "basic.jsonl")

# Thresholds GoodAgent clears comfortably and SloppyAgent misses.
THRESHOLDS = {
    "tool_selection": 0.95,
    "tool_order": 0.95,
    "efficiency": 0.95,
    "answer_correctness": 0.9,
    "cost_latency": 0.85,
}


class _ConstantMetric:
    """Test double returning a fixed score per case id."""

    name = "constant"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, case: TestCase, trajectory: Trajectory) -> MetricResult:
        return MetricResult(metric=self.name, score=self._scores[case.id])


def _echo_agent(question: str) -> Trajectory:
    return Trajectory(input=question, steps=[
        Step(type=StepType.FINAL_ANSWER, timestamp=0.0, content=question),
    ])


def test_percentile_hand_computed() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([3.0], 95) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_runner_aggregates_mean_scores() -> None:
    dataset = Dataset(name="toy", cases=[
        TestCase(id="a", input="qa"), TestCase(id="b", input="qb"),
    ])
    runner = Runner(metrics=[_ConstantMetric({"a": 1.0, "b": 0.5})])
    result = runner.evaluate(_echo_agent, dataset, agent_name="echo")
    assert result.agent_name == "echo"
    assert len(result.case_results) == 2
    assert result.mean_scores == {"constant": 0.75}
    assert result.overall_score == 0.75


def test_agents_are_deterministic() -> None:
    for agent_cls in (GoodAgent, SloppyAgent):
        first = agent_cls()(DATASET.cases[0].input)
        second = agent_cls()(DATASET.cases[0].input)
        assert first.model_dump() == second.model_dump()


def test_good_agent_beats_sloppy_on_every_metric() -> None:
    runner = Runner()
    results = runner.compare(
        {"good": GoodAgent(), "sloppy": SloppyAgent()}, DATASET
    )
    good, sloppy = results["good"], results["sloppy"]
    for metric in runner.metric_names:
        assert good.mean_scores[metric] > sloppy.mean_scores[metric], metric
    assert good.total_cost_usd < sloppy.total_cost_usd
    assert good.total_tokens < sloppy.total_tokens
    assert good.latency_p50_s < sloppy.latency_p50_s


def test_assert_agent_passes_good_agent() -> None:
    result = assert_agent_passes(GoodAgent(), DATASET, THRESHOLDS)
    assert result.mean_scores["tool_selection"] == 1.0


def test_assert_agent_fails_sloppy_agent() -> None:
    with pytest.raises(AssertionError, match="failed .* threshold"):
        assert_agent_passes(SloppyAgent(), DATASET, THRESHOLDS)


def test_assert_agent_rejects_unknown_metric_threshold() -> None:
    with pytest.raises(KeyError, match="unknown metric"):
        assert_agent_passes(GoodAgent(), DATASET, {"nonexistent": 0.5})
