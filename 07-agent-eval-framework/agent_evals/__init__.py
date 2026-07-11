"""agent_evals: a framework for evaluating LLM agent trajectories.

Public API: the data model, the metric suite, the judge interface, the
runner, and the CI threshold helper.
"""

from agent_evals.judge import Judge, JudgeVerdict, OfflineJudge, load_judge
from agent_evals.metrics import (
    AnswerCorrectness,
    CostLatency,
    Metric,
    ToolCallOrder,
    ToolSelectionAccuracy,
    TrajectoryEfficiency,
    default_metrics,
)
from agent_evals.models import (
    CaseResult,
    Dataset,
    EvalResult,
    MetricResult,
    Step,
    StepType,
    TestCase,
    Trajectory,
)
from agent_evals.runner import Runner
from agent_evals.settings import Settings
from agent_evals.testing import assert_agent_passes

__version__ = "0.1.0"

__all__ = [
    "AnswerCorrectness",
    "CaseResult",
    "CostLatency",
    "Dataset",
    "EvalResult",
    "Judge",
    "JudgeVerdict",
    "Metric",
    "MetricResult",
    "OfflineJudge",
    "Runner",
    "Settings",
    "Step",
    "StepType",
    "TestCase",
    "ToolCallOrder",
    "ToolSelectionAccuracy",
    "Trajectory",
    "TrajectoryEfficiency",
    "assert_agent_passes",
    "default_metrics",
    "load_judge",
    "__version__",
]
