"""Data model round-trips and offline judge behavior."""

from pathlib import Path

import pytest

from agent_evals.judge import OfflineJudge, load_judge
from agent_evals.models import Dataset, Step, StepType, TestCase, Trajectory


def test_dataset_jsonl_roundtrip(tmp_path: Path) -> None:
    dataset = Dataset(
        name="rt",
        cases=[
            TestCase(id="a", input="q1", expected_tools=["t1"], expected_answer="a1"),
            TestCase(id="b", input="q2", expected_tools=["t1", "t2"],
                     expected_answer="a2", reference_trajectory=["t1", "t1", "t2"]),
        ],
    )
    path = tmp_path / "rt.jsonl"
    dataset.to_jsonl(path)
    loaded = Dataset.from_jsonl(path)
    assert loaded.name == "rt"
    assert [c.model_dump() for c in loaded.cases] == [c.model_dump() for c in dataset.cases]
    assert loaded.cases[0].reference_sequence == ["t1"]  # falls back to expected_tools
    assert loaded.cases[1].reference_sequence == ["t1", "t1", "t2"]


def test_bundled_dataset_loads() -> None:
    path = Path(__file__).resolve().parent.parent / "datasets" / "basic.jsonl"
    dataset = Dataset.from_jsonl(path)
    assert len(dataset) == 10
    assert all(case.expected_tools and case.expected_answer for case in dataset)


def test_trajectory_derived_properties() -> None:
    steps = [
        Step(type=StepType.LLM_CALL, timestamp=10.0, prompt_tokens=100, completion_tokens=20),
        Step(type=StepType.TOOL_CALL, timestamp=10.5, name="calculator", arguments={"input": "1+1"}),
        Step(type=StepType.TOOL_RESULT, timestamp=10.8, name="calculator", content="2"),
        Step(type=StepType.LLM_CALL, timestamp=11.5, prompt_tokens=150, completion_tokens=30),
        Step(type=StepType.FINAL_ANSWER, timestamp=11.6, content="2"),
    ]
    trajectory = Trajectory(input="q", steps=steps)
    assert trajectory.tool_sequence == ["calculator"]
    assert trajectory.final_answer == "2"
    assert trajectory.prompt_tokens == 250
    assert trajectory.total_tokens == 300
    assert trajectory.wall_time_s == pytest.approx(1.6)


def test_offline_judge_is_deterministic() -> None:
    judge = OfflineJudge()
    args = ("What is X?", "Paris is the capital of France.", "I believe the capital is Paris.")
    first = judge.evaluate(*args)
    for _ in range(5):
        again = judge.evaluate(*args)
        assert again.score == first.score
        assert again.rationale == first.rationale


def test_offline_judge_keyword_coverage_hand_computed() -> None:
    judge = OfflineJudge()
    # Keywords: paris, capital, france -> answer covers 2 of 3.
    verdict = judge.evaluate("q", "Paris is the capital of France.", "The capital is Paris.")
    assert verdict.score == pytest.approx(2 / 3, abs=1e-3)
    assert "france" in verdict.rationale.lower()
    assert judge.evaluate("q", "Paris is the capital of France.",
                          "Paris is the capital of France.").score == 1.0
    assert judge.evaluate("q", "Paris is the capital of France.", "no idea").score == 0.0


def test_load_judge_defaults_offline_and_rejects_unknown() -> None:
    assert isinstance(load_judge(), OfflineJudge)
    assert isinstance(load_judge("offline"), OfflineJudge)
    with pytest.raises(ValueError, match="Unknown judge"):
        load_judge("banana")
