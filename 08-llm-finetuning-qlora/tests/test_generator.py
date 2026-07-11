"""Tests for the synthetic dataset generator."""

from __future__ import annotations

from collections import Counter

from qlora_tune.data.generator import CATEGORIES, generate_dataset


def test_generation_is_deterministic() -> None:
    """The same seed must yield an identical dataset, element for element."""
    a = generate_dataset(seed=13)
    b = generate_dataset(seed=13)
    assert a == b


def test_different_seeds_differ() -> None:
    """Different seeds should produce different instruction texts."""
    a = generate_dataset(seed=13)
    b = generate_dataset(seed=14)
    assert [ex.instruction for ex in a] != [ex.instruction for ex in b]


def test_size_and_category_balance() -> None:
    """Default generation yields 300 examples, 60 per category."""
    examples = generate_dataset()
    assert len(examples) == 300
    counts = Counter(ex.category for ex in examples)
    assert set(counts) == set(CATEGORIES)
    assert all(n == 60 for n in counts.values())


def test_ids_are_unique_and_fields_populated() -> None:
    """Every example has a unique id and non-empty instruction/response."""
    examples = generate_dataset()
    ids = [ex.id for ex in examples]
    assert len(set(ids)) == len(ids)
    assert all(ex.instruction.strip() and ex.response.strip() for ex in examples)


def test_pii_fraction_zero_produces_no_contacts() -> None:
    """With pii_fraction=0 no ticket embeds an email address."""
    examples = generate_dataset(pii_fraction=0.0)
    assert not any("@" in ex.instruction for ex in examples)
