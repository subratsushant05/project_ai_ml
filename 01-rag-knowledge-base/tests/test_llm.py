"""Tests for the offline extractive answer synthesizer."""

from rag_kb.llm import OfflineLLM
from rag_kb.schemas import Chunk


def _chunk(text: str, source: str) -> Chunk:
    return Chunk(text=text, source=source)


def test_citation_markers_cover_every_context_in_order() -> None:
    contexts = [
        _chunk("Deploys are canary based.", "deploy.md"),
        _chunk("Rollbacks take minutes.", "deploy.md"),
        _chunk("Incidents need a commander.", "incidents.md"),
    ]
    answer = OfflineLLM().generate("How do deploys work?", contexts)
    lines = answer.splitlines()
    assert len(lines) == len(contexts)
    for marker, line in enumerate(lines, start=1):
        assert line.endswith(f"[{marker}]")


def test_no_context_yields_explicit_fallback() -> None:
    answer = OfflineLLM().generate("Anything?", [])
    assert "No relevant context" in answer


def test_most_relevant_sentence_is_extracted() -> None:
    chunk = _chunk(
        "The office has plants. Rollback redeploys the previous release. "
        "Lunch is at noon.",
        "deploy.md",
    )
    answer = OfflineLLM(max_sentences_per_context=1).generate(
        "How does rollback work?", [chunk]
    )
    assert "Rollback redeploys the previous release." in answer
    assert "plants" not in answer
    assert "Lunch" not in answer


def test_generate_is_deterministic() -> None:
    contexts = [_chunk("Alpha beta gamma. Delta epsilon.", "a.md")]
    first = OfflineLLM().generate("alpha?", contexts)
    second = OfflineLLM().generate("alpha?", contexts)
    assert first == second
