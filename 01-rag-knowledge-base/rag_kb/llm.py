"""Answer-synthesis providers behind a common :class:`LLMProvider` interface.

The default :class:`OfflineLLM` is extractive and fully deterministic: it
selects the sentences from each retrieved chunk that best match the question
and stitches them together with inline citation markers. OpenAI and Anthropic
providers are imported lazily and only when explicitly selected.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_kb.config import LLMKind, Settings
from rag_kb.retrieval import tokenize
from rag_kb.schemas import Chunk

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the "
    "this to was we what when where which who why with you your".split()
)


class LLMProvider(ABC):
    """Turns a question plus retrieved context chunks into a cited answer."""

    @abstractmethod
    def generate(self, question: str, contexts: Sequence[Chunk]) -> str:
        """Produce an answer that cites contexts as ``[1]``, ``[2]``, ...

        Args:
            question: The user's question.
            contexts: Retrieved chunks, most relevant first. Marker ``[i]``
                must refer to ``contexts[i - 1]``.

        Returns:
            The answer text with inline citation markers.
        """


class OfflineLLM(LLMProvider):
    """Deterministic extractive synthesizer requiring no model or network.

    For each context chunk, the sentences sharing the most non-stopword terms
    with the question are extracted (in original order) and suffixed with that
    chunk's citation marker.
    """

    def __init__(self, max_sentences_per_context: int = 2) -> None:
        """Create the synthesizer.

        Args:
            max_sentences_per_context: Sentence budget per context chunk.
        """
        self._max_sentences = max_sentences_per_context

    def generate(self, question: str, contexts: Sequence[Chunk]) -> str:
        """See :meth:`LLMProvider.generate`."""
        if not contexts:
            return "No relevant context was found in the knowledge base."
        terms = {t for t in tokenize(question) if t not in _STOPWORDS}
        parts = [
            f"{self._extract(chunk.text, terms)} [{marker}]"
            for marker, chunk in enumerate(contexts, start=1)
        ]
        return "\n".join(parts)

    def _extract(self, text: str, terms: set[str]) -> str:
        """Pick the sentences of ``text`` most relevant to ``terms``."""
        sentences = [
            " ".join(s.split()) for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()
        ]
        if not sentences:
            return text.strip()
        scored = [
            (len(terms.intersection(tokenize(sentence))), position, sentence)
            for position, sentence in enumerate(sentences)
        ]
        best = sorted(scored, key=lambda item: (-item[0], item[1]))
        picked = [item for item in best[: self._max_sentences] if item[0] > 0]
        if not picked:
            picked = [scored[0]]
        picked.sort(key=lambda item: item[1])
        return " ".join(sentence for _, _, sentence in picked)


def _build_prompt(question: str, contexts: Sequence[Chunk]) -> str:
    """Format a grounded-answering prompt shared by the cloud providers."""
    blocks = "\n\n".join(
        f"[{marker}] (source: {chunk.source}) {chunk.text}"
        for marker, chunk in enumerate(contexts, start=1)
    )
    return (
        "Answer the question using only the numbered context below. "
        "Cite supporting passages inline as [1], [2], etc. "
        "If the context is insufficient, say so.\n\n"
        f"Context:\n{blocks}\n\nQuestion: {question}\n\nAnswer:"
    )


class OpenAILLM(LLMProvider):
    """Answer synthesis via the OpenAI chat completions API (optional)."""

    def __init__(self, model_name: str) -> None:
        """Create the client, importing the optional dependency lazily.

        Raises:
            ImportError: If ``openai`` is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'openai' provider requires `pip install openai`, "
                "or set RAG_KB_LLM_PROVIDER=offline."
            ) from exc
        self._client = OpenAI()
        self._model_name = model_name

    def generate(self, question, contexts):  # pragma: no cover - network
        """See :meth:`LLMProvider.generate`."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": _build_prompt(question, contexts)}],
        )
        return response.choices[0].message.content or ""


class AnthropicLLM(LLMProvider):
    """Answer synthesis via the Anthropic messages API (optional)."""

    def __init__(self, model_name: str) -> None:
        """Create the client, importing the optional dependency lazily.

        Raises:
            ImportError: If ``anthropic`` is not installed.
        """
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'anthropic' provider requires `pip install anthropic`, "
                "or set RAG_KB_LLM_PROVIDER=offline."
            ) from exc
        self._client = Anthropic()
        self._model_name = model_name

    def generate(self, question, contexts):  # pragma: no cover - network
        """See :meth:`LLMProvider.generate`."""
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": _build_prompt(question, contexts)}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the provider selected by ``settings.llm_provider``.

    Args:
        settings: Runtime configuration.

    Returns:
        A ready-to-use provider instance.
    """
    if settings.llm_provider is LLMKind.OPENAI:
        logger.info("Using OpenAI LLM provider: %s", settings.openai_chat_model)
        return OpenAILLM(settings.openai_chat_model)
    if settings.llm_provider is LLMKind.ANTHROPIC:
        logger.info("Using Anthropic LLM provider: %s", settings.anthropic_model)
        return AnthropicLLM(settings.anthropic_model)
    logger.info("Using offline extractive LLM provider")
    return OfflineLLM()
