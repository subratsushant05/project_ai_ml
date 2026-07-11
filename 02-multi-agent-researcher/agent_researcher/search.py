"""Search tool abstraction with an offline, deterministic default.

``OfflineSearchTool`` ranks a small bundled JSON corpus with simple keyword
scoring so the whole pipeline runs without network access. A Tavily adapter
is provided for live web search and imports its client lazily, keeping the
dependency optional.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_researcher.config import Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=\.)\s+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "behind", "by", "can", "do",
    "does", "for", "from", "how", "in", "is", "it", "its", "key", "main",
    "of", "on", "or", "that", "the", "their", "this", "to", "what", "when",
    "where", "which", "why", "with",
})


class SearchResult(BaseModel):
    """A single ranked search hit.

    Attributes:
        id: Stable document identifier.
        title: Document title.
        url: Document location.
        snippet: Short excerpt suitable for prompting and citations.
        score: Relevance score (higher is better).
    """

    id: str
    title: str
    url: str
    snippet: str
    score: float = Field(ge=0.0)


@runtime_checkable
class SearchTool(Protocol):
    """Minimal interface every search backend implements."""

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return up to ``top_k`` results ranked by relevance."""
        ...


def _tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords, and strip trivial plurals."""
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def _snippet(body: str, sentences: int = 2) -> str:
    """Return the first ``sentences`` sentences of ``body``."""
    return " ".join(_SENTENCE_RE.split(body.strip())[:sentences])


class OfflineSearchTool:
    """Keyword search over a bundled JSON corpus of short documents.

    Scoring is a term-frequency sum with a 2x boost for title matches;
    ties break on document id, so rankings are fully deterministic.

    Args:
        corpus_path: Path to a JSON file shaped like
            ``{"documents": [{"id", "title", "url", "body"}, ...]}``.

    Raises:
        FileNotFoundError: If ``corpus_path`` does not exist.
    """

    def __init__(self, corpus_path: Path | str) -> None:
        path = Path(corpus_path)
        if not path.is_file():
            raise FileNotFoundError(f"Search corpus not found: {path}")
        documents = json.loads(path.read_text(encoding="utf-8"))["documents"]
        self._index: list[tuple[dict[str, str], Counter[str], Counter[str]]] = [
            (doc, Counter(_tokenize(doc["title"])), Counter(_tokenize(doc["body"])))
            for doc in documents
        ]
        logger.debug("Loaded %d documents from %s", len(self._index), path)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Rank corpus documents against ``query``.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return.

        Returns:
            Matching documents with ``score > 0``, best first.
        """
        terms = set(_tokenize(query))
        scored: list[tuple[float, dict[str, str]]] = []
        for doc, title_counts, body_counts in self._index:
            score = float(sum(2 * title_counts[t] + body_counts[t] for t in terms))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [
            SearchResult(
                id=doc["id"],
                title=doc["title"],
                url=doc["url"],
                snippet=_snippet(doc["body"]),
                score=score,
            )
            for score, doc in scored[:top_k]
        ]


class TavilySearchTool:
    """Live web search backed by the Tavily API (optional dependency).

    Args:
        api_key: Tavily API key; falls back to ``TAVILY_API_KEY``.

    Raises:
        ImportError: If ``tavily-python`` is not installed.
    """

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from tavily import TavilyClient
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "TavilySearchTool requires the optional dependency "
                "'tavily-python'. Install it with: pip install tavily-python"
            ) from exc
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Query Tavily and adapt results to :class:`SearchResult`."""
        response = self._client.search(query, max_results=top_k)
        return [
            SearchResult(
                id=item.get("url", f"tavily-{position}"),
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:400],
                score=float(item.get("score", 0.0)),
            )
            for position, item in enumerate(response.get("results", []), start=1)
        ]


def create_search_tool(settings: Settings) -> SearchTool:
    """Build the search tool selected by ``settings.search_provider``.

    Args:
        settings: Application settings.

    Returns:
        A ready-to-use search tool.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    provider = settings.search_provider
    if provider == "offline":
        return OfflineSearchTool(settings.corpus_path)
    if provider == "tavily":
        return TavilySearchTool()
    raise ValueError(f"Unknown search provider: {provider!r}")
