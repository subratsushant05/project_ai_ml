"""Shared fixtures for the rag_kb test suite."""

from pathlib import Path

import pytest

from rag_kb.config import Settings
from rag_kb.pipeline import RAGPipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture()
def settings() -> Settings:
    """Deterministic offline settings, isolated from the host environment."""
    return Settings(_env_file=None)


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """A tiny corpus with one markdown and one text file."""
    (tmp_path / "cats.md").write_text(
        "# Cats\n\nCats are small felines. They purr when content.\n\n"
        "## Diet\n\nCats are obligate carnivores and eat meat.\n",
        encoding="utf-8",
    )
    (tmp_path / "dogs.txt").write_text(
        "Dogs are loyal companions. Dogs enjoy long walks and fetch.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def pipeline(settings: Settings) -> RAGPipeline:
    """A fresh offline pipeline."""
    return RAGPipeline(settings)
