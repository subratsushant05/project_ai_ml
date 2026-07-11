"""Runtime configuration loaded from environment variables.

All knobs use the ``AGENT_`` prefix (see ``.env.example``), so the same
code runs fully offline by default and against hosted providers when the
corresponding variables and API keys are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_corpus_path() -> Path:
    """Return the bundled corpus path relative to the repository root."""
    return Path(__file__).resolve().parent.parent / "sample_data" / "corpus.json"


class Settings(BaseSettings):
    """Environment-driven settings for the research pipeline.

    Attributes:
        model_provider: Chat model backend. ``offline`` is deterministic and
            needs no network; ``openai``/``anthropic`` are created lazily.
        model_name: Optional provider-specific model name override.
        search_provider: Search backend. ``offline`` scores a bundled JSON
            corpus; ``tavily`` performs live web search.
        corpus_path: Location of the offline search corpus.
        search_top_k: Number of sources retrieved per sub-question.
        num_sub_questions: Number of sub-questions the planner produces.
        quality_threshold: Minimum critic score (0-10) to accept a draft.
        max_revisions: Upper bound on Writer revisions after critique.
        require_approval: Pause at a human approval gate before writing.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_provider: Literal["offline", "openai", "anthropic"] = "offline"
    model_name: str = ""
    search_provider: Literal["offline", "tavily"] = "offline"
    corpus_path: Path = Field(default_factory=_default_corpus_path)
    search_top_k: int = Field(default=3, ge=1, le=10)
    num_sub_questions: int = Field(default=3, ge=1, le=4)
    quality_threshold: float = Field(default=8.0, ge=0.0, le=10.0)
    max_revisions: int = Field(default=1, ge=0, le=5)
    require_approval: bool = False
