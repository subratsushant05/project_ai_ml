"""Offline CLI demo: ingest the bundled sample corpus and answer questions.

Run from the project root with::

    python -m rag_kb.demo

Everything runs locally and deterministically; no API keys are required.
"""

import logging
from pathlib import Path

from rag_kb.pipeline import RAGPipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

EXAMPLE_QUERIES = (
    "How do I roll back a bad deployment?",
    "What is the review turnaround expectation for pull requests?",
    "What should I do first when a production incident is declared?",
)


def main() -> None:
    """Ingest the sample corpus and print answers to the example queries."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    pipeline = RAGPipeline()
    result = pipeline.ingest(SAMPLE_DIR)
    print(f"\nIngested {result.documents} documents into {result.chunks} chunks.\n")
    for question in EXAMPLE_QUERIES:
        response = pipeline.query(question)
        print(f"Q: {response.question}")
        print(f"A: {response.answer}")
        print("Sources:")
        for citation in response.citations:
            location = f" > {citation.section}" if citation.section else ""
            print(f"  [{citation.marker}] {citation.source}{location}")
        print()


if __name__ == "__main__":
    main()
