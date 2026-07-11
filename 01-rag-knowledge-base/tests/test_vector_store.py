"""Tests for the in-memory vector store and its persistence."""

import numpy as np
import pytest

from rag_kb.schemas import Chunk
from rag_kb.vector_store import VectorStore


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, source="test.md")


def test_search_returns_most_similar_first() -> None:
    store = VectorStore(dim=3)
    store.add(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]),
        [_chunk("x-axis"), _chunk("y-axis"), _chunk("diagonal")],
    )
    results = store.search(np.array([1.0, 0.1, 0.0]), top_k=2)
    assert [index for index, _ in results] == [0, 2]
    assert results[0][1] > results[1][1]


def test_search_empty_store_returns_nothing() -> None:
    store = VectorStore(dim=4)
    assert store.search(np.ones(4), top_k=5) == []


def test_add_rejects_mismatched_shapes() -> None:
    store = VectorStore(dim=4)
    with pytest.raises(ValueError):
        store.add(np.ones((1, 3)), [_chunk("wrong dim")])
    with pytest.raises(ValueError):
        store.add(np.ones((2, 4)), [_chunk("one chunk, two vectors")])


def test_save_load_roundtrip(tmp_path) -> None:
    store = VectorStore(dim=8)
    rng = np.random.default_rng(seed=7)
    vectors = rng.normal(size=(5, 8))
    chunks = [
        Chunk(text=f"chunk {i}", source="doc.md", section="S", position=i)
        for i in range(5)
    ]
    store.add(vectors, chunks)
    store.save(tmp_path / "index")

    loaded = VectorStore.load(tmp_path / "index")
    assert len(loaded) == 5
    assert loaded.dim == 8
    assert loaded.chunks == chunks
    query = rng.normal(size=8)
    assert loaded.search(query, top_k=3) == store.search(query, top_k=3)


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        VectorStore(dim=0)
