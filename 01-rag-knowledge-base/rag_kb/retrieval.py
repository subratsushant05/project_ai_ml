"""Hybrid retrieval: dense cosine search + sparse BM25, fused with RRF."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag_kb.embeddings import Embedder
from rag_kb.schemas import Chunk
from rag_kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk together with its fused relevance score."""

    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into alphanumeric tokens.

    Args:
        text: Text to tokenize.

    Returns:
        Tokens in order of appearance; empty for texts with no word characters.
    """
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse several rankings of item ids with weighted reciprocal rank fusion.

    Each item's fused score is ``sum(weight / (k + rank))`` over the rankings
    that contain it, where ``rank`` is 1-based.

    Args:
        rankings: One ranked list of item ids per retriever (best first).
        weights: One non-negative weight per ranking.
        k: Rank-offset constant; larger values flatten rank differences.

    Returns:
        ``(item_id, fused_score)`` pairs sorted by score descending, ties
        broken by item id for determinism.

    Raises:
        ValueError: If ``rankings`` and ``weights`` differ in length.
    """
    if len(rankings) != len(weights):
        raise ValueError(
            f"got {len(rankings)} ranking(s) but {len(weights)} weight(s)"
        )
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """Combines dense vector search and BM25 keyword search via RRF.

    The BM25 index is built once from the store's current chunks; create a
    new retriever after ingesting more documents.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        fusion_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_pool: int = 20,
    ) -> None:
        """Build the retriever and its BM25 index.

        Args:
            store: Vector store holding the corpus.
            embedder: Embedder used for queries (must match the store's dim).
            fusion_weight: Dense weight in [0, 1]; sparse gets the remainder.
            rrf_k: Rank-offset constant for reciprocal rank fusion.
            candidate_pool: Candidates fetched per retriever before fusion.
        """
        self._store = store
        self._embedder = embedder
        self._fusion_weight = fusion_weight
        self._rrf_k = rrf_k
        self._candidate_pool = candidate_pool
        self._chunks = store.chunks
        self._bm25 = (
            BM25Okapi([tokenize(chunk.text) for chunk in self._chunks])
            if self._chunks
            else None
        )

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Retrieve the ``top_k`` chunks most relevant to ``query``.

        Args:
            query: Natural-language query.
            top_k: Maximum number of chunks to return.

        Returns:
            Scored chunks in descending fused-score order; empty if the
            corpus is empty.
        """
        if not self._chunks or top_k <= 0:
            return []
        pool = max(self._candidate_pool, top_k)
        query_vector = self._embedder.embed_one(query)
        dense = [i for i, _ in self._store.search(query_vector, pool)]
        sparse = self._sparse_ranking(query, pool)
        fused = reciprocal_rank_fusion(
            [dense, sparse],
            [self._fusion_weight, 1.0 - self._fusion_weight],
            k=self._rrf_k,
        )
        logger.debug("Query %r fused %d candidate(s)", query, len(fused))
        return [
            ScoredChunk(chunk=self._chunks[index], score=score)
            for index, score in fused[:top_k]
        ]

    def _sparse_ranking(self, query: str, pool: int) -> list[int]:
        """Rank chunk indices by BM25 score for ``query`` (best first)."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        order = np.argsort(-scores, kind="stable")[:pool]
        return [int(i) for i in order if scores[i] > 0.0]
