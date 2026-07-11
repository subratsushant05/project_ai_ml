"""Embedding backends behind a common :class:`Embedder` interface.

The default :class:`HashingEmbedder` is fully offline and deterministic.
SentenceTransformers and OpenAI backends are imported lazily and only when
explicitly selected via configuration.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from rag_kb.config import EmbedderKind, Settings

logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized copies of ``vectors`` (zero rows kept)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


class Embedder(ABC):
    """Maps text to fixed-size, L2-normalized dense vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed. May be empty.

        Returns:
            A float32 array of shape ``(len(texts), dim)`` with unit-norm rows
            (all-zero rows for texts without recognizable tokens).
        """

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text and return a 1-D vector of shape ``(dim,)``."""
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic offline embedder: hashed bag-of-words with sublinear TF.

    Token counts are hashed into a fixed-size vector (feature hashing),
    dampened with ``log1p`` (TF-IDF-style sublinear term frequency), and
    L2-normalized. No fitting, files, or network access is required, and the
    output depends only on the input text and ``dim``.
    """

    def __init__(self, dim: int = 512) -> None:
        """Create the embedder.

        Args:
            dim: Number of hash buckets, i.e. the embedding dimensionality.
        """
        self._dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm=None
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """See :meth:`Embedder.embed`."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        counts = self._vectorizer.transform(list(texts)).toarray()
        return _l2_normalize(np.log1p(counts))


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a local SentenceTransformers model (optional)."""

    def __init__(self, model_name: str) -> None:
        """Load the model, importing the optional dependency lazily.

        Args:
            model_name: Hugging Face model identifier.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'sentence-transformers' backend requires "
                "`pip install sentence-transformers`, or set RAG_KB_EMBEDDER=hash."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """See :meth:`Embedder.embed`."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI embeddings API (optional)."""

    def __init__(self, model_name: str, dim: int) -> None:
        """Create the client, importing the optional dependency lazily.

        Args:
            model_name: OpenAI embedding model name.
            dim: Requested embedding dimensionality.

        Raises:
            ImportError: If ``openai`` is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'openai' backend requires `pip install openai`, "
                "or set RAG_KB_EMBEDDER=hash."
            ) from exc
        self._client = OpenAI()
        self._model_name = model_name
        self._dim = dim

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """See :meth:`Embedder.embed`."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        response = self._client.embeddings.create(
            model=self._model_name, input=list(texts), dimensions=self._dim
        )
        vectors = np.asarray([item.embedding for item in response.data])
        return _l2_normalize(vectors)


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``settings.embedder``.

    Args:
        settings: Runtime configuration.

    Returns:
        A ready-to-use embedder instance.
    """
    if settings.embedder is EmbedderKind.SENTENCE_TRANSFORMERS:
        logger.info("Using SentenceTransformers embedder: %s", settings.st_model)
        return SentenceTransformerEmbedder(settings.st_model)
    if settings.embedder is EmbedderKind.OPENAI:
        logger.info("Using OpenAI embedder: %s", settings.openai_embedding_model)
        return OpenAIEmbedder(settings.openai_embedding_model, settings.embedding_dim)
    logger.info("Using offline hashing embedder (dim=%d)", settings.embedding_dim)
    return HashingEmbedder(settings.embedding_dim)
