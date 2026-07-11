"""Runtime configuration via environment variables (``AGENT_EVALS_*``).

Examples:
    Switch the judge and tighten the cost budget without touching code::

        export AGENT_EVALS_JUDGE=openai
        export AGENT_EVALS_COST_BUDGET_USD=0.002
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    """Per-model pricing in USD per 1M tokens.

    Attributes:
        prompt_per_1m: Dollars per 1M prompt tokens.
        completion_per_1m: Dollars per 1M completion tokens.
    """

    prompt_per_1m: float
    completion_per_1m: float


DEFAULT_PRICES: dict[str, ModelPrice] = {
    # Simulated models used by the bundled example agents.
    "sim-small": ModelPrice(prompt_per_1m=0.15, completion_per_1m=0.60),
    "sim-large": ModelPrice(prompt_per_1m=2.50, completion_per_1m=10.00),
    # A few real-world entries so external trajectories price out of the box.
    "gpt-4o-mini": ModelPrice(prompt_per_1m=0.15, completion_per_1m=0.60),
    "gpt-4o": ModelPrice(prompt_per_1m=2.50, completion_per_1m=10.00),
    "claude-3-5-haiku": ModelPrice(prompt_per_1m=0.80, completion_per_1m=4.00),
    "claude-sonnet-4": ModelPrice(prompt_per_1m=3.00, completion_per_1m=15.00),
}


class Settings(BaseSettings):
    """Framework settings, overridable via ``AGENT_EVALS_*`` env vars.

    Attributes:
        judge: Which judge backend to use: ``offline`` | ``openai`` |
            ``anthropic``.
        judge_model: Model name for the remote judges.
        cost_budget_usd: Characteristic per-case cost used to normalize the
            cost score (``exp(-cost / budget)``).
        latency_budget_s: Characteristic per-case latency used to normalize
            the latency score (``exp(-latency / budget)``).
        prices: Price table keyed by model name.
        fuzzy_pass_threshold: Fuzzy-match ratio treated as "correct" in
            report summaries.
    """

    model_config = SettingsConfigDict(env_prefix="AGENT_EVALS_")

    judge: str = "offline"
    judge_model: str = "gpt-4o-mini"
    cost_budget_usd: float = 0.005
    latency_budget_s: float = 5.0
    prices: dict[str, ModelPrice] = DEFAULT_PRICES
    fuzzy_pass_threshold: float = 0.8

    def price_for(self, model: str | None) -> ModelPrice:
        """Look up pricing for a model, falling back to ``sim-small``.

        Args:
            model: Model identifier from a trajectory step (may be ``None``).

        Returns:
            The matching :class:`ModelPrice`, or the cheapest default when
            the model is unknown -- unknown models should never inflate cost.
        """
        if model is not None and model in self.prices:
            return self.prices[model]
        return self.prices["sim-small"]
