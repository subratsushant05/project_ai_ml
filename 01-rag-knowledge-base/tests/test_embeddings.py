"""Tests for the offline hashing embedder."""

import numpy as np

from rag_kb.embeddings import HashingEmbedder


def test_embeddings_are_deterministic_across_instances() -> None:
    texts = ["retrieval augmented generation", "vector databases store embeddings"]
    first = HashingEmbedder(dim=256).embed(texts)
    second = HashingEmbedder(dim=256).embed(texts)
    np.testing.assert_array_equal(first, second)


def test_embeddings_shape_dtype_and_unit_norm() -> None:
    embedder = HashingEmbedder(dim=128)
    vectors = embedder.embed(["one text", "another text", "a third text"])
    assert vectors.shape == (3, 128)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_batch_returns_empty_matrix() -> None:
    vectors = HashingEmbedder(dim=64).embed([])
    assert vectors.shape == (0, 64)


def test_similar_texts_score_higher_than_dissimilar() -> None:
    embedder = HashingEmbedder(dim=512)
    vectors = embedder.embed(
        [
            "the cat sat on the mat",
            "a cat sat upon a mat",
            "quarterly financial revenue forecast",
        ]
    )
    similar = float(vectors[0] @ vectors[1])
    dissimilar = float(vectors[0] @ vectors[2])
    assert similar > dissimilar
