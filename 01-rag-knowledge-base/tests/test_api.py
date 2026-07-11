"""Tests for the FastAPI endpoints using TestClient (fully offline)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_kb.api import create_app
from rag_kb.pipeline import RAGPipeline


@pytest.fixture()
def client(pipeline: RAGPipeline) -> TestClient:
    return TestClient(create_app(pipeline))


def test_health_reports_empty_index(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks_indexed": 0}


def test_ingest_then_query_returns_cited_answer(
    client: TestClient, corpus_dir: Path
) -> None:
    ingest = client.post("/ingest", json={"path": str(corpus_dir)})
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["documents"] == 3 and body["chunks"] >= 3

    query = client.post("/query", json={"question": "What do cats eat?", "top_k": 2})
    assert query.status_code == 200
    payload = query.json()
    assert "[1]" in payload["answer"]
    assert len(payload["citations"]) <= 2
    assert payload["citations"][0]["source"] == "cats.md"

    health = client.get("/health")
    assert health.json()["chunks_indexed"] == body["chunks"]


def test_ingest_missing_path_returns_404(client: TestClient) -> None:
    response = client.post("/ingest", json={"path": "/nonexistent/nowhere"})
    assert response.status_code == 404


def test_query_before_ingest_returns_409(client: TestClient) -> None:
    response = client.post("/query", json={"question": "anything?"})
    assert response.status_code == 409


def test_empty_question_fails_validation(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
