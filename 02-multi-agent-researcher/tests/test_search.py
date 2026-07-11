"""Tests for the offline search tool: ranking, limits, determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_researcher.search import OfflineSearchTool, SearchTool, create_search_tool


def test_ranking_prefers_topical_match(search_tool: OfflineSearchTool) -> None:
    """The solar-cost document must rank first for a solar-cost query."""
    results = search_tool.search("falling cost of solar photovoltaic modules")
    assert results
    assert results[0].id == "re-solar-costs"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(search_tool: OfflineSearchTool) -> None:
    """No more than top_k results are returned."""
    assert len(search_tool.search("electric vehicle batteries", top_k=2)) == 2
    assert len(search_tool.search("electric vehicle batteries", top_k=1)) == 1


def test_no_match_returns_empty_list(search_tool: OfflineSearchTool) -> None:
    """Queries with no corpus overlap return an empty list, not junk."""
    assert search_tool.search("xylophone zeppelin quokka") == []


def test_search_is_deterministic(search_tool: OfflineSearchTool) -> None:
    """Identical queries return identical rankings on every call."""
    first = [r.id for r in search_tool.search("transformer attention models")]
    second = [r.id for r in search_tool.search("transformer attention models")]
    assert first == second
    assert first[0].startswith("tf-")


def test_missing_corpus_raises() -> None:
    """A helpful error is raised when the corpus file is absent."""
    with pytest.raises(FileNotFoundError, match="corpus"):
        OfflineSearchTool(Path("/nonexistent/corpus.json"))


def test_factory_returns_offline_tool(settings) -> None:
    """The factory honors the offline provider and satisfies the protocol."""
    tool = create_search_tool(settings)
    assert isinstance(tool, OfflineSearchTool)
    assert isinstance(tool, SearchTool)
