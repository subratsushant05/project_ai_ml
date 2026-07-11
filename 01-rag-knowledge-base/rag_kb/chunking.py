"""Chunking strategies: fixed-size with overlap, and sentence-aware packing."""

import logging
import re
from collections.abc import Iterable

from rag_kb.config import ChunkStrategy, Settings
from rag_kb.schemas import Chunk, Document

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def fixed_size_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into fixed-size character windows with overlap.

    Args:
        text: Text to split. Whitespace-only input yields no chunks.
        size: Maximum characters per chunk. Must be positive.
        overlap: Characters shared between consecutive chunks. Must satisfy
            ``0 <= overlap < size``.

    Returns:
        Stripped, non-empty chunk strings in document order.

    Raises:
        ValueError: If ``size`` or ``overlap`` is out of range.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap}")
    if not text.strip():
        return []

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step
    return chunks


def sentence_chunks(text: str, size: int) -> list[str]:
    """Pack whole sentences greedily into chunks of at most ``size`` characters.

    Sentences are never split; a single sentence longer than ``size`` becomes
    its own (oversized) chunk rather than being truncated.

    Args:
        text: Text to split. Whitespace-only input yields no chunks.
        size: Target maximum characters per chunk. Must be positive.

    Returns:
        Chunk strings in document order.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    length = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if buffer else 0)
        if buffer and length + extra > size:
            chunks.append(" ".join(buffer))
            buffer, length = [], 0
            extra = len(sentence)
        buffer.append(sentence)
        length += extra
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def chunk_document(document: Document, settings: Settings) -> list[Chunk]:
    """Split one document into chunks using the configured strategy.

    Args:
        document: Source document; its metadata is copied onto every chunk.
        settings: Chunking configuration.

    Returns:
        Chunks with ``position`` numbered from 0 within the document.
    """
    if settings.chunk_strategy is ChunkStrategy.FIXED:
        pieces = fixed_size_chunks(
            document.text, settings.chunk_size, settings.chunk_overlap
        )
    else:
        pieces = sentence_chunks(document.text, settings.chunk_size)
    return [
        Chunk(
            text=piece,
            source=document.source,
            section=document.section,
            page=document.page,
            position=position,
        )
        for position, piece in enumerate(pieces)
    ]


def chunk_documents(documents: Iterable[Document], settings: Settings) -> list[Chunk]:
    """Chunk a collection of documents, preserving input order.

    Args:
        documents: Documents to split.
        settings: Chunking configuration.

    Returns:
        All chunks, concatenated in document order.
    """
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, settings))
    logger.debug("Produced %d chunk(s)", len(chunks))
    return chunks
