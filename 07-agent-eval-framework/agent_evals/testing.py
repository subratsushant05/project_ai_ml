"""Pytest-friendly helper for gating agents on eval thresholds in CI."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from agent_evals.models import Dataset, EvalResult
from agent_evals.runner import AgentFn, Runner

logger = logging.getLogger(__name__)


def assert_agent_passes(
    agent: AgentFn,
    dataset: Dataset,
    thresholds: Mapping[str, float],
    agent_name: str | None = None,
    runner: Runner | None = None,
) -> EvalResult:
    """Assert that an agent's mean metric scores meet minimum thresholds.

    Designed for use inside a pytest test so agent regressions fail CI::

        def test_agent_quality():
            assert_agent_passes(
                my_agent,
                Dataset.from_jsonl("datasets/basic.jsonl"),
                thresholds={"tool_selection": 0.9, "answer_correctness": 0.8},
            )

    Args:
        agent: Callable mapping an input string to a trajectory.
        dataset: Test cases to evaluate on.
        thresholds: Minimum acceptable mean score per metric name. Metrics
            not listed are reported but not gated.
        agent_name: Optional display name for error messages.
        runner: Optional preconfigured runner (custom metrics/judge).

    Returns:
        The full :class:`EvalResult` for further inspection.

    Raises:
        AssertionError: If any gated metric's mean score is below its
            threshold. The message lists every failing metric.
        KeyError: If a threshold references a metric the runner did not run.
    """
    runner = runner or Runner()
    result = runner.evaluate(agent, dataset, agent_name=agent_name)
    failures: list[str] = []
    for metric, minimum in thresholds.items():
        if metric not in result.mean_scores:
            raise KeyError(
                f"Threshold references unknown metric {metric!r}; "
                f"available: {sorted(result.mean_scores)}"
            )
        score = result.mean_scores[metric]
        if score < minimum:
            failures.append(f"{metric}: {score:.3f} < required {minimum:.3f}")
    if failures:
        raise AssertionError(
            f"Agent {result.agent_name!r} failed {len(failures)} threshold(s) "
            f"on dataset {result.dataset_name!r}:\n  " + "\n  ".join(failures)
        )
    logger.info("Agent %s passed all %d thresholds", result.agent_name, len(thresholds))
    return result
