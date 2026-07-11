"""Two deterministic example agents over the bundled fake toolset.

``GoodAgent`` routes each question to the minimal tool plan and answers
precisely. ``SloppyAgent`` reaches the same neighborhood but wastes calls
(leading web_search, duplicated first tool), burns more tokens on a pricier
model, and answers vaguely -- a realistic "worse baseline" for demos and CI.
"""

from __future__ import annotations

import logging
import re

from agent_evals.models import Step, StepType, Trajectory
from agent_evals.tools import TOOLS, WEATHER_FIXTURES, best_search_doc

logger = logging.getLogger(__name__)

_BASE_TS = 1_700_000_000.0
_MATH_RE = re.compile(r"\(?\d[\d\s.+\-*/()]*\d\)?|\d")


class _Recorder:
    """Builds a trajectory with a simulated, deterministic clock."""

    def __init__(self, task: str, agent_name: str) -> None:
        self._ts = _BASE_TS
        self.steps: list[Step] = []
        self._task = task
        self._agent_name = agent_name

    def _add(self, step: Step, duration_s: float) -> None:
        self._ts += duration_s
        step.timestamp = self._ts
        self.steps.append(step)

    def llm_call(self, content: str, model: str, prompt: int, completion: int, dur: float) -> None:
        step = Step(
            type=StepType.LLM_CALL, timestamp=0.0, content=content,
            model=model, prompt_tokens=prompt, completion_tokens=completion,
        )
        self._add(step, dur)

    def tool(self, name: str, argument: str, call_dur: float, result_dur: float) -> str:
        """Record a tool_call/tool_result pair and return the tool output."""
        self._add(Step(type=StepType.TOOL_CALL, timestamp=0.0, name=name,
                       arguments={"input": argument}), call_dur)
        output = TOOLS[name](argument)
        self._add(Step(type=StepType.TOOL_RESULT, timestamp=0.0, name=name,
                       content=output), result_dur)
        return output

    def finish(self, answer: str) -> Trajectory:
        self._add(Step(type=StepType.FINAL_ANSWER, timestamp=0.0, content=answer), 0.01)
        return Trajectory(input=self._task, steps=self.steps, agent_name=self._agent_name)


def _cities_in(question: str) -> list[str]:
    """Fixture cities mentioned in the question, in order of appearance."""
    lower = question.lower()
    found = [(lower.find(city), city) for city in WEATHER_FIXTURES if city in lower]
    return [city for pos, city in sorted(found) if pos >= 0]


def _plan(question: str) -> tuple[list[tuple[str, str]], str]:
    """Route a question to a minimal tool plan.

    Args:
        question: The user task.

    Returns:
        Tuple of (ordered ``(tool, argument)`` calls, answer template). The
        template may contain ``{r0}``, ``{r1}``, ... for tool results.
    """
    lower = question.lower()
    cities = _cities_in(question)
    if "doubled" in lower and cities:
        city = cities[0]
        temp = WEATHER_FIXTURES[city]["temp_c"]
        return (
            [("weather", city), ("calculator", f"{temp} * 2")],
            f"If the temperature in {city.title()} doubled it would be {{r1}} C.",
        )
    if "sum of the temperatures" in lower and len(cities) >= 2:
        temps = [WEATHER_FIXTURES[c]["temp_c"] for c in cities]
        calls = [("weather", c) for c in cities]
        calls.append(("calculator", " + ".join(str(t) for t in temps)))
        names = " and ".join(c.title() for c in cities)
        return calls, f"The sum of the temperatures in {names} is {{r{len(cities)}}} C."
    if "weather" in lower and cities:
        city = cities[0]
        conditions = WEATHER_FIXTURES[city]["conditions"]
        temp = WEATHER_FIXTURES[city]["temp_c"]
        return [("weather", city)], f"It is {temp} C and {conditions} in {city.title()}."
    math_match = _MATH_RE.search(question)
    if math_match and any(op in question for op in "+-*/"):
        expr = math_match.group(0).strip()
        return [("calculator", expr)], f"{expr} = {{r0}}."
    return [("web_search", question)], best_search_doc(question)["answer"]


class GoodAgent:
    """Rule-based agent that executes the minimal plan and answers exactly."""

    name = "GoodAgent"

    def __call__(self, question: str) -> Trajectory:
        """Run the agent on one question.

        Args:
            question: The user task.

        Returns:
            A deterministic trajectory ending in a precise final answer.
        """
        rec = _Recorder(question, self.name)
        calls, template = _plan(question)
        rec.llm_call(f"Plan: {[name for name, _ in calls]}", "sim-small", 180, 40, 0.35)
        results = [rec.tool(name, arg, 0.02, 0.12) for name, arg in calls]
        rec.llm_call("Compose final answer", "sim-small", 220, 60, 0.35)
        answer = template.format(**{f"r{i}": r for i, r in enumerate(results)})
        logger.debug("%s answered %r with tools %s", self.name, answer, [c[0] for c in calls])
        return rec.finish(answer)


class SloppyAgent:
    """Wasteful agent: redundant tools, duplicate calls, vague answers.

    It always opens with an unnecessary ``web_search``, calls its first real
    tool twice, thinks out loud on a pricier model, and hedges its answer --
    strictly worse than :class:`GoodAgent` on every bundled metric.
    """

    name = "SloppyAgent"

    def __call__(self, question: str) -> Trajectory:
        """Run the agent on one question.

        Args:
            question: The user task.

        Returns:
            A deterministic but inefficient trajectory with a vague answer.
        """
        rec = _Recorder(question, self.name)
        calls, template = _plan(question)
        rec.llm_call("Hmm, let me search around first.", "sim-large", 420, 130, 0.9)
        rec.tool("web_search", question, 0.03, 0.25)
        rec.llm_call("Maybe I should try some tools.", "sim-large", 520, 140, 0.9)
        results = []
        for index, (name, arg) in enumerate(calls):
            if index == 0 and name != "web_search":
                rec.tool(name, arg, 0.03, 0.25)  # duplicate first call
            results.append(rec.tool(name, arg, 0.03, 0.25))
        rec.llm_call("Wrapping up, roughly.", "sim-large", 640, 160, 0.9)
        precise = template.format(**{f"r{i}": r for i, r in enumerate(results)})
        answer = _vague(precise)
        logger.debug("%s answered %r", self.name, answer)
        return rec.finish(answer)


def _vague(answer: str) -> str:
    """Degrade a precise answer into a hedged half-answer."""
    words = answer.rstrip(".").split()
    kept = words[: max(2, (2 * len(words)) // 3)]
    return "Hard to say for sure, but possibly " + " ".join(kept) + ", I did not verify."


def good_agent() -> GoodAgent:
    """Factory for the CLI: ``--agent agent_evals.agents:good_agent``."""
    return GoodAgent()


def sloppy_agent() -> SloppyAgent:
    """Factory for the CLI: ``--agent agent_evals.agents:sloppy_agent``."""
    return SloppyAgent()
