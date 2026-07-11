"""Evaluation runner: apply a metric suite to an agent over a dataset.

The agent under test is *any* callable ``str -> Trajectory`` -- your real
agent, a replayed log, or one of the bundled examples.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence

from agent_evals.judge import Judge, load_judge
from agent_evals.metrics import CostLatency, Metric, default_metrics
from agent_evals.models import CaseResult, Dataset, EvalResult, Trajectory
from agent_evals.settings import Settings

logger = logging.getLogger(__name__)

AgentFn = Callable[[str], Trajectory]


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of a sequence.

    Args:
        values: Sample values (need not be sorted).
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile, or ``0.0`` for an empty sequence.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


class Runner:
    """Evaluates agents against datasets with a configurable metric suite."""

    def __init__(
        self,
        metrics: Sequence[Metric] | None = None,
        judge: Judge | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            metrics: Metric suite; defaults to the five bundled metrics.
            judge: Judge for answer correctness; defaults to the backend
                selected by ``AGENT_EVALS_JUDGE`` (offline unless set).
            settings: Framework settings (pricing, budgets).
        """
        self._settings = settings or Settings()
        self._judge = judge if judge is not None else load_judge(settings=self._settings)
        self._metrics: list[Metric] = list(
            metrics if metrics is not None
            else default_metrics(judge=self._judge, settings=self._settings)
        )
        self._pricer = CostLatency(settings=self._settings)

    @property
    def metric_names(self) -> list[str]:
        """Names of the configured metrics, in report order."""
        return [m.name for m in self._metrics]

    def evaluate(self, agent: AgentFn, dataset: Dataset, agent_name: str | None = None) -> EvalResult:
        """Evaluate one agent on every case of a dataset.

        Args:
            agent: Callable mapping an input string to a trajectory.
            dataset: The test cases to run.
            agent_name: Display name; falls back to the agent's ``name``
                attribute or class name.

        Returns:
            Aggregated :class:`EvalResult` with per-case breakdowns.
        """
        name = agent_name or getattr(agent, "name", type(agent).__name__)
        logger.info("Evaluating %s on %s (%d cases)", name, dataset.name, len(dataset))
        started = time.perf_counter()
        case_results: list[CaseResult] = []
        for case in dataset:
            trajectory = agent(case.input)
            metric_results = {m.name: m.score(case, trajectory) for m in self._metrics}
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    input=case.input,
                    final_answer=trajectory.final_answer,
                    tool_sequence=trajectory.tool_sequence,
                    metrics=metric_results,
                    total_tokens=trajectory.total_tokens,
                    cost_usd=round(self._pricer.cost_usd(trajectory), 6),
                    wall_time_s=round(trajectory.wall_time_s, 3),
                )
            )
        wall_times = [c.wall_time_s for c in case_results]
        mean_scores = {
            metric: round(
                sum(c.metrics[metric].score for c in case_results) / max(len(case_results), 1), 4
            )
            for metric in self.metric_names
        }
        result = EvalResult(
            agent_name=name,
            dataset_name=dataset.name,
            case_results=case_results,
            mean_scores=mean_scores,
            total_cost_usd=round(sum(c.cost_usd for c in case_results), 6),
            total_tokens=sum(c.total_tokens for c in case_results),
            latency_p50_s=round(percentile(wall_times, 50), 3),
            latency_p95_s=round(percentile(wall_times, 95), 3),
        )
        logger.info(
            "Finished %s in %.2fs (overall %.3f)",
            name, time.perf_counter() - started, result.overall_score,
        )
        return result

    def compare(self, agents: Mapping[str, AgentFn], dataset: Dataset) -> dict[str, EvalResult]:
        """Evaluate several agents on the same dataset.

        Args:
            agents: Mapping of display name to agent callable.
            dataset: The shared dataset.

        Returns:
            Results keyed by agent name, in input order.
        """
        return {name: self.evaluate(agent, dataset, agent_name=name) for name, agent in agents.items()}
