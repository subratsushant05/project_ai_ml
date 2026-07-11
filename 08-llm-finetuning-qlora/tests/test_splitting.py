"""Tests for stratified train/val/test splitting."""

from __future__ import annotations

from collections import Counter

import pytest

from qlora_tune.data.generator import generate_dataset
from qlora_tune.data.splitting import stratified_split


def test_split_proportions_per_category() -> None:
    """Each category is split 80/10/10 (up to rounding of 60 items)."""
    examples = generate_dataset(pii_fraction=0.0)
    splits = stratified_split(examples, train_frac=0.8, val_frac=0.1, seed=13)
    for name, expected in [("train", 48), ("val", 6), ("test", 6)]:
        counts = Counter(ex.category for ex in splits[name])
        assert all(n == expected for n in counts.values()), (name, counts)


def test_split_is_disjoint_and_complete() -> None:
    """Every example lands in exactly one split."""
    examples = generate_dataset(pii_fraction=0.0)
    splits = stratified_split(examples)
    all_ids = [ex.id for split in splits.values() for ex in split]
    assert len(all_ids) == len(examples)
    assert set(all_ids) == {ex.id for ex in examples}


def test_split_is_deterministic() -> None:
    """The same seed yields the same assignment."""
    examples = generate_dataset(pii_fraction=0.0)
    a = stratified_split(examples, seed=7)
    b = stratified_split(examples, seed=7)
    assert {k: [e.id for e in v] for k, v in a.items()} == {
        k: [e.id for e in v] for k, v in b.items()
    }


def test_invalid_fractions_raise() -> None:
    """Fractions that leave no test split (or are out of range) raise."""
    examples = generate_dataset(pii_fraction=0.0)
    with pytest.raises(ValueError):
        stratified_split(examples, train_frac=0.9, val_frac=0.1)
    with pytest.raises(ValueError):
        stratified_split(examples, train_frac=0.0, val_frac=0.1)
