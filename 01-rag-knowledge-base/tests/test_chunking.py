"""Tests for fixed-size and sentence-aware chunking."""

import pytest

from rag_kb.chunking import chunk_document, fixed_size_chunks, sentence_chunks
from rag_kb.config import ChunkStrategy, Settings
from rag_kb.schemas import Document


def test_fixed_chunks_overlap_and_coverage() -> None:
    text = "abcdefghij" * 10  # 100 characters
    chunks = fixed_size_chunks(text, size=40, overlap=10)
    assert chunks[0] == text[:40]
    assert chunks[1] == text[30:70]  # step = size - overlap = 30
    assert chunks[-1].endswith(text[-1])


def test_fixed_chunks_empty_and_whitespace_text() -> None:
    assert fixed_size_chunks("", size=10, overlap=2) == []
    assert fixed_size_chunks("   \n\t  ", size=10, overlap=2) == []


def test_fixed_chunks_text_shorter_than_size() -> None:
    assert fixed_size_chunks("short text", size=100, overlap=20) == ["short text"]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-5, 0), (10, 10), (10, 15), (10, -1)],
)
def test_fixed_chunks_invalid_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        fixed_size_chunks("some text", size=size, overlap=overlap)


def test_sentence_chunks_pack_whole_sentences_greedily() -> None:
    text = "First sentence here. Second one follows. Third is last."
    chunks = sentence_chunks(text, size=45)
    assert chunks == ["First sentence here. Second one follows.", "Third is last."]


def test_sentence_chunks_respect_size_unless_single_long_sentence() -> None:
    short = "Tiny. " * 20
    for chunk in sentence_chunks(short, size=30):
        assert len(chunk) <= 30
    long_sentence = "x" * 200 + "."
    assert sentence_chunks(long_sentence, size=50) == [long_sentence]


def test_sentence_chunks_empty_text() -> None:
    assert sentence_chunks("", size=100) == []
    assert sentence_chunks("   ", size=100) == []


def test_chunk_document_copies_metadata_and_numbers_positions() -> None:
    settings = Settings(
        _env_file=None,
        chunk_strategy=ChunkStrategy.FIXED,
        chunk_size=30,
        chunk_overlap=5,
    )
    document = Document(
        text="alpha beta gamma delta epsilon zeta eta theta iota kappa",
        source="handbook.md",
        section="Greek",
    )
    chunks = chunk_document(document, settings)
    assert len(chunks) > 1
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "handbook.md" and c.section == "Greek" for c in chunks)
