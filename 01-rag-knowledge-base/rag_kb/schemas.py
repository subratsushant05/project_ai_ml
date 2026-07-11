"""Pydantic models shared across the package and the HTTP API."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A loaded source document, or one page/section of it.

    Attributes:
        text: Raw text content.
        source: File name (or path) the text came from.
        section: Markdown heading the text belongs to, if any.
        page: 1-based page number for paginated formats (PDF), if any.
    """

    text: str
    source: str
    section: str | None = None
    page: int | None = None


class Chunk(BaseModel):
    """A retrievable slice of a document.

    Attributes:
        text: Chunk text content.
        source: File name the chunk came from.
        section: Markdown heading the chunk belongs to, if any.
        page: 1-based page number for paginated formats, if any.
        position: 0-based index of the chunk within its document.
    """

    text: str
    source: str
    section: str | None = None
    page: int | None = None
    position: int = 0


class Citation(BaseModel):
    """Maps an inline answer marker such as ``[1]`` to its source.

    Attributes:
        marker: 1-based citation number used inline in the answer.
        source: File name of the cited document.
        section: Section heading of the cited chunk, if any.
        page: Page number of the cited chunk, if any.
    """

    marker: int = Field(ge=1)
    source: str
    section: str | None = None
    page: int | None = None


class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``."""

    path: str = Field(min_length=1, description="File or directory to ingest.")


class IngestResponse(BaseModel):
    """Result of an ingestion run."""

    documents: int
    chunks: int


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryResponse(BaseModel):
    """Answer with its supporting citations."""

    question: str
    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    chunks_indexed: int
