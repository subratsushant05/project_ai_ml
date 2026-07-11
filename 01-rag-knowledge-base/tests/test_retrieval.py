"""Tests for BM25 + dense hybrid retrieval and reciprocal rank fusion."""

import pytest

from rag_kb.embeddings import HashingEmbedder
from rag_kb.retrieval import HybridRetriever, reciprocal_rank_fusion, tokenize
from rag_kb.schemas import Chunk
from rag_kb.vector_store import VectorStore


def _build_retriever(texts: list[str], **kwargs) -> HybridRetriever:
    embedder = HashingEmbedder(dim=256)
    store = VectorStore(dim=256)
    chunks = [Chunk(text=t, source=f"doc{i}.md") for i, t in enumerate(texts)]
    store.add(embedder.embed(texts), chunks)
    return HybridRetriever(store, embedder, **kwargs)


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, World! v2.0") == ["hello", "world", "v2", "0"]
    assert tokenize("...") == []


def test_rrf_prefers_items_ranked_high_in_both_lists() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 3]], [0.5, 0.5], k=10)
    order = [item for item, _ in fused]
    # Items 1 and 2 (ranks 1+2) must beat item 3 (ranks 3+3).
    assert set(order[:2]) == {1, 2}
    assert order[2] == 3


def test_rrf_weight_extremes_reproduce_single_ranking() -> None:
    dense, sparse = [0, 1, 2], [2, 1, 0]
    dense_only = reciprocal_rank_fusion([dense, sparse], [1.0, 0.0], k=60)
    assert [item for item, _ in dense_only] == dense
    sparse_only = reciprocal_rank_fusion([dense, sparse], [0.0, 1.0], k=60)
    assert [item for item, _ in sparse_only] == sparse


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1, 2]], [0.5, 0.5])


def test_hybrid_retrieval_surfaces_exact_keyword_match() -> None:
    retriever = _build_retriever(
        [
            "The zephyr subsystem controls wind compensation.",
            "General notes about weather and climate patterns.",
            "Unrelated notes about database migrations.",
        ]
    )
    results = retriever.retrieve("zephyr subsystem", top_k=2)
    assert results
    assert results[0].chunk.text.startswith("The zephyr subsystem")
    assert results[0].score >= results[-1].score


def test_hybrid_retrieval_is_deterministic() -> None:
    texts = [f"topic {i} with shared words about systems" for i in range(10)]
    first = _build_retriever(texts).retrieve("shared systems", top_k=5)
    second = _build_retriever(texts).retrieve("shared systems", top_k=5)
    assert [r.chunk.text for r in first] == [r.chunk.text for r in second]


def test_retrieve_on_empty_store_returns_nothing() -> None:
    embedder = HashingEmbedder(dim=64)
    retriever = HybridRetriever(VectorStore(dim=64), embedder)
    assert retriever.retrieve("anything", top_k=3) == []
