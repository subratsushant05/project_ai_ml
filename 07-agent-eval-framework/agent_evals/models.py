"""Core data model for agent trajectories, datasets, and eval results.

Everything in the framework flows through three shapes:

* :class:`Trajectory` -- what an agent actually did (ordered ``Step`` list).
* :class:`Dataset` / :class:`TestCase` -- what it *should* have done.
* :class:`EvalResult` -- how the two compare, per case and aggregated.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StepType(StrEnum):
    """The four kinds of events that can appear in a trajectory."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"


class Step(BaseModel):
    """A single event in an agent trajectory.

    Attributes:
        type: What kind of event this is.
        timestamp: Unix timestamp (seconds) when the event finished.
        name: Tool name for ``tool_call`` / ``tool_result`` steps.
        arguments: Tool-call arguments, if any.
        content: Free-form payload (LLM text, tool output, final answer).
        prompt_tokens: Prompt tokens consumed (``llm_call`` steps).
        completion_tokens: Completion tokens produced (``llm_call`` steps).
        model: Model identifier used for pricing (``llm_call`` steps).
    """

    type: StepType
    timestamp: float
    name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    content: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str | None = None


class Trajectory(BaseModel):
    """An ordered record of everything an agent did for one input.

    Attributes:
        input: The task the agent was given.
        steps: Ordered list of events.
        agent_name: Human-readable name of the agent that produced this.
    """

    input: str
    steps: list[Step] = Field(default_factory=list)
    agent_name: str = "agent"

    @property
    def tool_calls(self) -> list[Step]:
        """All ``tool_call`` steps, in order."""
        return [s for s in self.steps if s.type is StepType.TOOL_CALL]

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered tool names, e.g. ``["weather", "calculator"]``."""
        return [s.name or "" for s in self.tool_calls]

    @property
    def final_answer(self) -> str:
        """Content of the last ``final_answer`` step, or ``""``."""
        for step in reversed(self.steps):
            if step.type is StepType.FINAL_ANSWER:
                return step.content or ""
        return ""

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens across all LLM calls."""
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens across all LLM calls."""
        return sum(s.completion_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def wall_time_s(self) -> float:
        """Elapsed seconds between the first and last step."""
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].timestamp - self.steps[0].timestamp


class TestCase(BaseModel):
    """One row of an eval dataset.

    Attributes:
        id: Stable identifier used in reports.
        input: The task given to the agent under test.
        expected_tools: Tools a correct agent should call (set semantics).
        expected_answer: Reference answer for correctness scoring.
        reference_trajectory: Optional *ordered* tool sequence of an ideal
            run. Falls back to ``expected_tools`` order when absent.
    """

    __test__ = False  # not a pytest test class, despite the name

    id: str
    input: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_answer: str = ""
    reference_trajectory: list[str] | None = None

    @property
    def reference_sequence(self) -> list[str]:
        """Ordered ideal tool sequence for order/efficiency metrics."""
        if self.reference_trajectory is not None:
            return self.reference_trajectory
        return self.expected_tools


class Dataset(BaseModel):
    """A collection of test cases, serialized as JSONL on disk.

    Attributes:
        name: Dataset name shown in reports.
        cases: The test cases.
    """

    name: str = "dataset"
    cases: list[TestCase] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[TestCase]:  # type: ignore[override]
        return iter(self.cases)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> Dataset:
        """Load a dataset from a JSONL file (one ``TestCase`` per line).

        Args:
            path: Path to the ``.jsonl`` file.
            name: Optional dataset name; defaults to the file stem.

        Returns:
            The parsed dataset.
        """
        path = Path(path)
        cases = [
            TestCase.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(name=name or path.stem, cases=cases)

    def to_jsonl(self, path: str | Path) -> None:
        """Write the dataset as JSONL, one compact JSON object per line.

        Args:
            path: Destination file path (parent dirs must exist).
        """
        lines = [
            json.dumps(case.model_dump(exclude_none=True), ensure_ascii=False)
            for case in self.cases
        ]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class MetricResult(BaseModel):
    """Outcome of one metric on one case.

    Attributes:
        metric: Metric name.
        score: Normalized score in ``[0, 1]`` (1 is best).
        details: Metric-specific breakdown (precision, distances, ...).
    """

    metric: str
    score: float
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """All metric results for a single test case.

    Attributes:
        case_id: The test case id.
        input: The task text (repeated for standalone reports).
        final_answer: What the agent answered.
        tool_sequence: Ordered tools the agent called.
        metrics: Metric results keyed by metric name.
        total_tokens: Tokens used on this case.
        cost_usd: Dollar cost of this case.
        wall_time_s: Wall-clock seconds for this case.
    """

    case_id: str
    input: str
    final_answer: str
    tool_sequence: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    total_tokens: int = 0
    cost_usd: float = 0.0
    wall_time_s: float = 0.0


class EvalResult(BaseModel):
    """Aggregate outcome of evaluating one agent on one dataset.

    Attributes:
        agent_name: Name of the agent under test.
        dataset_name: Name of the dataset.
        case_results: Per-case breakdown.
        mean_scores: Mean metric score across cases, keyed by metric name.
        total_cost_usd: Summed dollar cost across cases.
        total_tokens: Summed token usage across cases.
        latency_p50_s: Median per-case wall time.
        latency_p95_s: 95th-percentile per-case wall time.
    """

    agent_name: str
    dataset_name: str
    case_results: list[CaseResult] = Field(default_factory=list)
    mean_scores: dict[str, float] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0

    @property
    def overall_score(self) -> float:
        """Unweighted mean of the per-metric means."""
        if not self.mean_scores:
            return 0.0
        return sum(self.mean_scores.values()) / len(self.mean_scores)
