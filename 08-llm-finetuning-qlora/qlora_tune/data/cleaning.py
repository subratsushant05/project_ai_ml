"""Dataset cleaning: PII scrubbing, deduplication and length filtering."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace

from qlora_tune.data.records import Example

logger = logging.getLogger(__name__)

# Deliberately conservative patterns: false negatives are safer to reason
# about than mangled text, and the corpus is synthetic English helpdesk prose.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?![\w-])"
)

EMAIL_TOKEN = "[EMAIL]"
PHONE_TOKEN = "[PHONE]"


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Summary of what a cleaning pass changed.

    Attributes:
        input_count: Number of examples before cleaning.
        output_count: Number of examples after cleaning.
        duplicates_removed: Examples dropped by deduplication.
        length_filtered: Examples dropped by the length filter.
        pii_scrubbed: Examples in which at least one PII match was replaced.
    """

    input_count: int
    output_count: int
    duplicates_removed: int
    length_filtered: int
    pii_scrubbed: int


def scrub_pii(text: str) -> str:
    """Replace email addresses and phone numbers with placeholder tokens.

    Args:
        text: Arbitrary text possibly containing contact details.

    Returns:
        Text with emails replaced by ``[EMAIL]`` and phone numbers by
        ``[PHONE]``.
    """
    text = _EMAIL_RE.sub(EMAIL_TOKEN, text)
    text = _PHONE_RE.sub(PHONE_TOKEN, text)
    return text


def _normalize_key(text: str) -> str:
    """Normalize text for duplicate detection (case/whitespace-insensitive)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_examples(
    examples: list[Example],
    min_chars: int = 40,
    max_chars: int = 4000,
) -> tuple[list[Example], CleaningReport]:
    """Run the full cleaning pipeline: scrub PII, dedupe, filter by length.

    PII scrubbing runs first so that two tickets differing only in contact
    details collapse to one entry during deduplication. Deduplication keys on
    the normalized instruction text and keeps the first occurrence.

    Args:
        examples: Raw examples.
        min_chars: Minimum combined instruction+response length to keep.
        max_chars: Maximum combined instruction+response length to keep.

    Returns:
        Tuple of (cleaned examples, :class:`CleaningReport`).
    """
    scrubbed: list[Example] = []
    pii_count = 0
    for ex in examples:
        new_instruction = scrub_pii(ex.instruction)
        new_response = scrub_pii(ex.response)
        if new_instruction != ex.instruction or new_response != ex.response:
            pii_count += 1
        scrubbed.append(replace(ex, instruction=new_instruction, response=new_response))

    seen: set[str] = set()
    deduped: list[Example] = []
    for ex in scrubbed:
        key = _normalize_key(ex.instruction)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    duplicates_removed = len(scrubbed) - len(deduped)

    kept = [
        ex for ex in deduped if min_chars <= len(ex.instruction) + len(ex.response) <= max_chars
    ]
    length_filtered = len(deduped) - len(kept)

    report = CleaningReport(
        input_count=len(examples),
        output_count=len(kept),
        duplicates_removed=duplicates_removed,
        length_filtered=length_filtered,
        pii_scrubbed=pii_count,
    )
    logger.info(
        "Cleaning: %d -> %d (dupes=%d, length=%d, pii=%d)",
        report.input_count,
        report.output_count,
        duplicates_removed,
        length_filtered,
        pii_count,
    )
    return kept, report
