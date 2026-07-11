"""Tests for the document loaders."""

from pathlib import Path

import pytest

from rag_kb.ingestion import load_path


def test_markdown_loader_splits_sections(corpus_dir: Path) -> None:
    documents = load_path(corpus_dir / "cats.md")
    sections = [doc.section for doc in documents]
    assert sections == ["Cats", "Diet"]
    assert all(doc.source == "cats.md" for doc in documents)
    assert "obligate carnivores" in documents[1].text


def test_text_loader_returns_single_document(corpus_dir: Path) -> None:
    documents = load_path(corpus_dir / "dogs.txt")
    assert len(documents) == 1
    assert documents[0].source == "dogs.txt"
    assert documents[0].section is None


def test_directory_loading_is_recursive_and_sorted(corpus_dir: Path) -> None:
    nested = corpus_dir / "nested"
    nested.mkdir()
    (nested / "birds.txt").write_text("Birds can fly.", encoding="utf-8")
    documents = load_path(corpus_dir)
    assert [doc.source for doc in documents] == [
        "cats.md",
        "cats.md",
        "dogs.txt",
        "birds.txt",
    ]


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "does-not-exist")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    weird = tmp_path / "data.csv"
    weird.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_path(weird)
