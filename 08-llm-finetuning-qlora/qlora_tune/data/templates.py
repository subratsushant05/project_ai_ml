"""Explicit chat-template implementations (no transformers dependency).

Writing these out by hand keeps the light layer dependency-free and makes the
exact training-text format visible and unit-testable. Both templates render a
(system, user, assistant) triple into the single training string an SFT
trainer would see.

References:
    * Llama 3 family: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    * ChatML (used by Qwen and others): ``<|im_start|>role\\n...<|im_end|>``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from qlora_tune.data.records import Example

DEFAULT_SYSTEM_PROMPT = (
    "You are an experienced IT helpdesk agent. Given a support ticket, "
    "reply with a clear, numbered resolution."
)

TemplateName = Literal["llama3", "chatml"]
Formatter = Callable[[str, str, str], str]


def format_llama3(system: str, user: str, assistant: str) -> str:
    """Render one conversation turn in the Llama 3 instruct format.

    Args:
        system: System prompt text.
        user: User message (the ticket).
        assistant: Assistant answer (the resolution) used as the SFT target.

    Returns:
        The full training string, including ``<|begin_of_text|>`` and the
        closing ``<|eot_id|>`` after the assistant message.
    """
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
    )


def format_chatml(system: str, user: str, assistant: str) -> str:
    """Render one conversation turn in ChatML (Qwen-style) format.

    Args:
        system: System prompt text.
        user: User message (the ticket).
        assistant: Assistant answer (the resolution) used as the SFT target.

    Returns:
        The full training string with ``<|im_start|>``/``<|im_end|>`` markers
        and a trailing newline after the final ``<|im_end|>``.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )


_FORMATTERS: dict[str, Formatter] = {
    "llama3": format_llama3,
    "chatml": format_chatml,
}


def get_formatter(name: str) -> Formatter:
    """Look up a chat-template formatter by name.

    Args:
        name: ``"llama3"`` or ``"chatml"``.

    Returns:
        The formatter callable ``(system, user, assistant) -> str``.

    Raises:
        KeyError: If the template name is unknown.
    """
    try:
        return _FORMATTERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown chat template {name!r}; available: {sorted(_FORMATTERS)}"
        ) from None


def format_examples(
    examples: list[Example],
    template: TemplateName = "llama3",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[str]:
    """Format a list of examples into training strings.

    Args:
        examples: Instruction examples.
        template: Which chat template to use.
        system_prompt: System prompt prepended to every conversation.

    Returns:
        One formatted training string per example, in order.
    """
    formatter = get_formatter(template)
    return [formatter(system_prompt, ex.instruction, ex.response) for ex in examples]
