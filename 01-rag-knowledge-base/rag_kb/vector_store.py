"""In-memory vector store with cosine similarity and disk persistence."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rag_kb.schemas import Chunk

logger = logging.getLogger(__name__)

_VECTORS_FILE = "vectors.npz"
_CHUNKS_FILE = "chunks.json"


class VectorStore:
    """Stores chunk vectors in a numpy matrix and searches by cosine similarity.

    Vectors are L2-normalized on insertion, so cosine similarity reduces to a
    dot product at query time.
    """

    def __init__(self, dim: int) -> None:
        """Create an empty store.

        Args:
            dim: Dimensionality of stored vectors. Must be positive.

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    @property
    def dim(self) -> int:
        """Dimensionality of stored vectors."""
        return self._dim

    @property
    def chunks(self) -> list[Chunk]:
        """The stored chunks, in insertion order."""
        return list(self._chunks)

    def __len__(self) -> int:
        """Number of stored chunks."""
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        """Append vectors and their chunks to the store.

        Args:
            vectors: Array of shape ``(n, dim)``; normalized defensively.
            chunks: The ``n`` chunks corresponding to the vector rows.

        Raises:
            ValueError: If shapes are inconsistent with the store.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"expected vectors of shape (n, {self._dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"got {vectors.shape[0]} vector(s) for {len(chunks)} chunk(s)"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._vectors = np.vstack([self._vectors, vectors / norms])
        self._chunks.extend(chunks)
        logger.debug("Store now holds %d chunk(s)", len(self._chunks))

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Find the most similar stored vectors to ``query``.

        Args:
            query: Vector of shape ``(dim,)``.
            top_k: Maximum number of results.

        Returns:
            ``(index, cosine_similarity)`` pairs sorted by similarity
            descending (ties broken by index). Empty if the store is empty.
        """
        if len(self._chunks) == 0 or top_k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        similarities = self._vectors @ query
        order = np.argsort(-similarities, kind="stable")[:top_k]
        return [(int(i), float(similarities[i])) for i in order]

    def save(self, directory: str | Path) -> None:
        """Persist vectors (npz) and chunk metadata (json) to a directory.

        Args:
            directory: Target directory; created if missing.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / _VECTORS_FILE, vectors=self._vectors)
        payload = {"dim": self._dim, "chunks": [c.model_dump() for c in self._chunks]}
        (directory / _CHUNKS_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d chunk(s) to %s", len(self._chunks), directory)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a store previously written by :meth:`save`.

        Args:
            directory: Directory containing the persisted files.

        Returns:
            A populated store.

        Raises:
            FileNotFoundError: If the persisted files are missing.
        """
        directory = Path(directory)
        payload = json.loads((directory / _CHUNKS_FILE).read_text(encoding="utf-8"))
        with np.load(directory / _VECTORS_FILE) as data:
            vectors = data["vectors"]
        store = cls(dim=int(payload["dim"]))
        store._vectors = np.asarray(vectors, dtype=np.float32)
        store._chunks = [Chunk.model_validate(c) for c in payload["chunks"]]
        logger.info("Loaded %d chunk(s) from %s", len(store), directory)
        return store
