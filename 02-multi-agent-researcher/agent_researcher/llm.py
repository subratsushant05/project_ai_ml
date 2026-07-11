"""Chat model abstraction and the provider factory.

Every agent node talks to a :class:`ChatModel` -- a structural protocol with
a single ``invoke(system, user) -> str`` method. The default provider is
the deterministic offline model; hosted providers (OpenAI, Anthropic) are
imported lazily inside the factory so they remain optional dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent_researcher.offline_llm import OfflineChatModel

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from agent_researcher.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatModel(Protocol):
    """Minimal chat interface every provider adapter implements."""

    def invoke(self, system: str, user: str) -> str:
        """Return the model's completion for a system/user prompt pair."""
        ...


class LangChainChatAdapter:
    """Adapt any LangChain ``BaseChatModel`` to the :class:`ChatModel` protocol.

    Args:
        chat_model: A configured LangChain chat model instance.
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model

    def invoke(self, system: str, user: str) -> str:
        """Send the prompt pair and return the response text."""
        message = self._chat_model.invoke([("system", system), ("human", user)])
        content = message.content
        if isinstance(content, list):  # multimodal responses
            content = " ".join(
                part if isinstance(part, str) else str(part.get("text", ""))
                for part in content
            )
        return str(content)


def create_chat_model(settings: Settings) -> ChatModel:
    """Build the chat model selected by ``settings.model_provider``.

    Hosted providers are imported lazily so the base install stays free of
    ``langchain-openai`` / ``langchain-anthropic``.

    Args:
        settings: Application settings.

    Returns:
        A ready-to-use chat model.

    Raises:
        ImportError: If a hosted provider's package is not installed.
        ValueError: If the provider name is not recognized.
    """
    provider = settings.model_provider
    if provider == "offline":
        logger.debug("Using deterministic OfflineChatModel")
        return OfflineChatModel()
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "model_provider='openai' requires the optional dependency "
                "'langchain-openai'. Install it with: pip install langchain-openai"
            ) from exc
        model_name = settings.model_name or "gpt-4o-mini"
        return LangChainChatAdapter(ChatOpenAI(model=model_name, temperature=0))
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "model_provider='anthropic' requires the optional dependency "
                "'langchain-anthropic'. Install it with: "
                "pip install langchain-anthropic"
            ) from exc
        model_name = settings.model_name or "claude-sonnet-4-5"
        return LangChainChatAdapter(ChatAnthropic(model=model_name, temperature=0))
    raise ValueError(f"Unknown model provider: {provider!r}")
