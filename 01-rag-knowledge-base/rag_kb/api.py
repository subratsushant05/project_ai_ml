"""FastAPI application exposing the RAG pipeline over HTTP."""

import logging

from fastapi import FastAPI, HTTPException, Request

from rag_kb import __version__
from rag_kb.pipeline import RAGPipeline
from rag_kb.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        pipeline: Pipeline to serve; a fresh one is built from environment
            configuration when omitted (useful for tests to inject their own).

    Returns:
        The configured application, with the pipeline on ``app.state``.
    """
    app = FastAPI(
        title="rag_kb",
        version=__version__,
        description="Offline-first RAG knowledge base with hybrid retrieval.",
    )
    app.state.pipeline = pipeline or RAGPipeline()

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report service liveness and index size."""
        return HealthResponse(
            status="ok", chunks_indexed=len(request.app.state.pipeline.store)
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: Request, body: IngestRequest) -> IngestResponse:
        """Ingest a file or directory from the server's filesystem."""
        try:
            return request.app.state.pipeline.ingest(body.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, body: QueryRequest) -> QueryResponse:
        """Answer a question from the indexed corpus."""
        pipeline: RAGPipeline = request.app.state.pipeline
        if len(pipeline.store) == 0:
            raise HTTPException(
                status_code=409,
                detail="The index is empty; ingest documents via /ingest first.",
            )
        return pipeline.query(body.question, body.top_k)

    return app


app = create_app()
