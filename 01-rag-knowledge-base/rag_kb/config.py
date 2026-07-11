"""Application configuration loaded from environment variables.

All settings use the ``RAG_KB_`` prefix, e.g. ``RAG_KB_CHUNK_SIZE=500``.
A local ``.env`` file is honoured if present.
"""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SENTENCE = "sentence"


class EmbedderKind(str, Enum):
    """Available embedding backends."""

    HASH = "hash"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"


class LLMKind(str, Enum):
    """Available answer-synthesis providers."""

    OFFLINE = "offline"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Runtime configuration for the RAG pipeline.

    Attributes:
        chunk_strategy: How documents are split into chunks.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Character overlap between consecutive fixed-size chunks.
        embedder: Which embedding backend to use.
        embedding_dim: Dimensionality of the hashed embedder.
        st_model: Model name for the SentenceTransformers backend.
        openai_embedding_model: Model name for the OpenAI embedding backend.
        top_k: Default number of chunks returned per query.
        candidate_pool: Candidates fetched per retriever before rank fusion.
        fusion_weight: Weight of the dense retriever in RRF (0 = sparse only,
            1 = dense only).
        rrf_k: Rank-offset constant used by reciprocal rank fusion.
        llm_provider: Which answer-synthesis provider to use.
        openai_chat_model: Chat model for the OpenAI provider.
        anthropic_model: Model for the Anthropic provider.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_KB_", env_file=".env", extra="ignore"
    )

    chunk_strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)

    embedder: EmbedderKind = EmbedderKind.HASH
    embedding_dim: int = Field(default=512, ge=8)
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    top_k: int = Field(default=4, ge=1)
    candidate_pool: int = Field(default=20, ge=1)
    fusion_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

    llm_provider: LLMKind = LLMKind.OFFLINE
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-5"
