"""Tests for the hand-written chat-template implementations."""

from __future__ import annotations

import pytest

from qlora_tune.data.records import Example
from qlora_tune.data.templates import (
    format_chatml,
    format_examples,
    format_llama3,
    get_formatter,
)


def test_llama3_template_exact_string() -> None:
    """The Llama 3 format matches the published prompt layout exactly."""
    result = format_llama3("Be helpful.", "My VPN is down.", "Restart the client.")
    expected = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\nBe helpful.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nMy VPN is down.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nRestart the client.<|eot_id|>"
    )
    assert result == expected


def test_chatml_template_exact_string() -> None:
    """The ChatML format matches the Qwen-style layout exactly."""
    result = format_chatml("Be helpful.", "My VPN is down.", "Restart the client.")
    expected = (
        "<|im_start|>system\nBe helpful.<|im_end|>\n"
        "<|im_start|>user\nMy VPN is down.<|im_end|>\n"
        "<|im_start|>assistant\nRestart the client.<|im_end|>\n"
    )
    assert result == expected


def test_get_formatter_lookup_and_unknown() -> None:
    """Known names resolve to the right function; unknown names raise KeyError."""
    assert get_formatter("llama3") is format_llama3
    assert get_formatter("chatml") is format_chatml
    with pytest.raises(KeyError):
        get_formatter("alpaca")


def test_format_examples_uses_instruction_and_response() -> None:
    """Batch formatting embeds each example's fields in order."""
    examples = [
        Example(id="T-1", category="vpn", instruction="Ticket one", response="Fix one"),
        Example(id="T-2", category="vpn", instruction="Ticket two", response="Fix two"),
    ]
    out = format_examples(examples, template="chatml", system_prompt="S")
    assert len(out) == 2
    assert "Ticket one" in out[0] and "Fix one" in out[0]
    assert out[1].startswith("<|im_start|>system\nS<|im_end|>\n")
