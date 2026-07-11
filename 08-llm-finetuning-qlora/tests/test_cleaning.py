"""Tests for PII scrubbing, deduplication and length filtering."""

from __future__ import annotations

from qlora_tune.data.cleaning import clean_examples, scrub_pii
from qlora_tune.data.generator import generate_dataset
from qlora_tune.data.records import Example


def _example(
    idx: int, instruction: str, response: str = "1. Did the thing and closed it."
) -> Example:
    return Example(id=f"T-{idx}", category="vpn", instruction=instruction, response=response)


def test_scrub_email_addresses() -> None:
    """Emails are replaced with the [EMAIL] token."""
    text = "Contact me at jane.doe+it@example-corp.com for details."
    assert scrub_pii(text) == "Contact me at [EMAIL] for details."


def test_scrub_phone_numbers_multiple_formats() -> None:
    """Common US-style phone formats are replaced with [PHONE]."""
    assert scrub_pii("Call +1-555-123-4567 now") == "Call [PHONE] now"
    assert scrub_pii("Call (555) 123-4567 now") == "Call [PHONE] now"
    assert scrub_pii("Call 555.123.4567 now") == "Call [PHONE] now"


def test_scrub_leaves_ordinary_text_alone() -> None:
    """Version numbers, error codes and small numbers are not false positives."""
    text = "Ubuntu 22.04 fails with error 0x8007 after MTU 1350 on port 443."
    assert scrub_pii(text) == text


def test_dedupe_keeps_first_occurrence() -> None:
    """Duplicate instructions (case/whitespace-insensitive) are removed."""
    examples = [
        _example(1, "My VPN drops every few minutes at home."),
        _example(2, "my vpn   drops every few minutes at home."),
        _example(3, "A completely different unique ticket about the VPN client."),
    ]
    cleaned, report = clean_examples(examples)
    assert [ex.id for ex in cleaned] == ["T-1", "T-3"]
    assert report.duplicates_removed == 1


def test_dedupe_collapses_pii_only_variants() -> None:
    """Two tickets differing only in contact details collapse after scrubbing."""
    base = "The VPN client disconnects every few minutes when I work from home."
    examples = [
        _example(1, base + " Reach me at a.11@example-corp.com."),
        _example(2, base + " Reach me at b.22@example-corp.com."),
    ]
    cleaned, report = clean_examples(examples)
    assert len(cleaned) == 1
    assert report.pii_scrubbed == 2


def test_length_filter_drops_out_of_range() -> None:
    """Examples outside the [min_chars, max_chars] window are dropped."""
    examples = [
        _example(1, "too short", response="x"),
        _example(2, "A reasonable ticket describing a broken VPN connection in detail."),
        _example(3, "long " * 2000),
    ]
    cleaned, report = clean_examples(examples, min_chars=40, max_chars=4000)
    assert [ex.id for ex in cleaned] == ["T-2"]
    assert report.length_filtered == 2


def test_pipeline_scrubs_generated_dataset() -> None:
    """After cleaning the generated dataset, no email addresses survive."""
    cleaned, report = clean_examples(generate_dataset())
    assert report.pii_scrubbed > 0
    assert not any("@" in ex.instruction for ex in cleaned)
    assert all("[EMAIL]" not in ex.response for ex in cleaned)
