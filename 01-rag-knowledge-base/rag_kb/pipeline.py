"""High-level pipeline wiring ingestion, retrieval, and answer synthesis."""

import logging
from pathlib import Path

from rag_kb.chunking import chunk_documents
from rag_kb.config import Settings
from rag_kb.embeddings import create_embedder
from rag_kb.ingestion import load_path
from rag_kb.llm import create_llm_provider
from rag_kb.retrieval import HybridRetriever
from rag_kb.schemas import Citation, IngestResponse, QueryResponse
from rag_kb.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: ingest files, then answer cited questions."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Assemble the pipeline from configuration.

        Args:
            settings: Runtime configuration; read from the environment when
                omitted.
        """
        self.settings = settings or Settings()
        self.embedder = create_embedder(self.settings)
        self.store = VectorStore(self.embedder.dim)
        self.llm = create_llm_provider(self.settings)
        self._retriever: HybridRetriever | None = None

    def ingest(self, path: str | Path) -> IngestResponse:
        """Load, chunk, embed, and index a file or directory.

        Args:
            path: File or directory to ingest.

        Returns:
            Counts of loaded documents and indexed chunks.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        documents = load_path(path)
        chunks = chunk_documents(documents, self.settings)
        if chunks:
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            self.store.add(vectors, chunks)
            self._retriever = None  # BM25 index must be rebuilt.
        logger.info(
            "Ingested %s: %d document(s), %d chunk(s)",
            path,
            len(documents),
            len(chunks),
        )
        return IngestResponse(documents=len(documents), chunks=len(chunks))

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        """Answer a question from the indexed corpus, with citations.

        Args:
            question: Natural-language question.
            top_k: Number of chunks to retrieve; defaults to the configured
                value.

        Returns:
            The answer plus one citation per retrieved chunk; marker ``[i]``
            in the answer refers to citation ``i``.
        """
        k = top_k if top_k is not None else self.settings.top_k
        results = self._get_retriever().retrieve(question, k)
        contexts = [result.chunk for result in results]
        answer = self.llm.generate(question, contexts)
        citations = [
            Citation(
                marker=marker,
                source=chunk.source,
                section=chunk.section,
                page=chunk.page,
            )
            for marker, chunk in enumerate(contexts, start=1)
        ]
        return QueryResponse(question=question, answer=answer, citations=citations)

    def save_index(self, directory: str | Path) -> None:
        """Persist the vector index to ``directory``."""
        self.store.save(directory)

    def load_index(self, directory: str | Path) -> None:
        """Replace the current index with one persisted by :meth:`save_index`.

        Raises:
            ValueError: If the stored dimensionality does not match the
                configured embedder.
        """
        store = VectorStore.load(directory)
        if store.dim != self.embedder.dim:
            raise ValueError(
                f"index dim {store.dim} does not match embedder dim "
                f"{self.embedder.dim}"
            )
        self.store = store
        self._retriever = None

    def _get_retriever(self) -> HybridRetriever:
        """Return the retriever, rebuilding it after any ingestion."""
        if self._retriever is None:
            self._retriever = HybridRetriever(
                self.store,
                self.embedder,
                fusion_weight=self.settings.fusion_weight,
                rrf_k=self.settings.rrf_k,
                candidate_pool=self.settings.candidate_pool,
            )
        return self._retriever
