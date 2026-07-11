"""Shared pytest fixtures. Everything runs offline and deterministically."""

from __future__ import annotations

import sys
from pathlib import Path

import langchain_core  # noqa: F401  (installs its warning filters up front)
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_researcher.config import Settings  # noqa: E402
from agent_researcher.offline_llm import OfflineChatModel  # noqa: E402
from agent_researcher.search import OfflineSearchTool  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    """Fully explicit offline settings, independent of ambient env/.env."""
    return Settings(
        _env_file=None,
        model_provider="offline",
        search_provider="offline",
        corpus_path=PROJECT_ROOT / "sample_data" / "corpus.json",
        search_top_k=3,
        num_sub_questions=3,
        quality_threshold=8.0,
        max_revisions=1,
        require_approval=False,
    )


@pytest.fixture()
def search_tool(settings: Settings) -> OfflineSearchTool:
    """Offline search tool over the bundled corpus."""
    return OfflineSearchTool(settings.corpus_path)


@pytest.fixture()
def model() -> OfflineChatModel:
    """Deterministic offline chat model."""
    return OfflineChatModel()
