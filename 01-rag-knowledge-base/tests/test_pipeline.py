"""End-to-end tests for the RAG pipeline (offline and deterministic)."""

from pathlib import Path

from rag_kb.config import Settings
from rag_kb.pipeline import RAGPipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_sample_corpus_query_cites_the_right_document() -> None:
    pipeline = RAGPipeline(Settings(_env_file=None))
    result = pipeline.ingest(SAMPLE_DIR)
    assert result.documents >= 5
    assert result.chunks >= result.documents

    response = pipeline.query("How do I roll back a bad deployment?")
    assert "[1]" in response.answer
    assert response.citations[0].source == "03-deployments.md"
    markers = [citation.marker for citation in response.citations]
    assert markers == list(range(1, len(markers) + 1))


def test_citation_markers_in_answer_map_to_citation_list(
    pipeline: RAGPipeline, corpus_dir: Path
) -> None:
    pipeline.ingest(corpus_dir)
    response = pipeline.query("Do dogs enjoy walks?", top_k=3)
    for citation in response.citations:
        assert f"[{citation.marker}]" in response.answer


def test_index_save_and_load_roundtrip(
    pipeline: RAGPipeline, corpus_dir: Path, tmp_path: Path
) -> None:
    pipeline.ingest(corpus_dir)
    before = pipeline.query("What do cats eat?").answer
    pipeline.save_index(tmp_path / "index")

    fresh = RAGPipeline(Settings(_env_file=None))
    fresh.load_index(tmp_path / "index")
    assert len(fresh.store) == len(pipeline.store)
    assert fresh.query("What do cats eat?").answer == before
