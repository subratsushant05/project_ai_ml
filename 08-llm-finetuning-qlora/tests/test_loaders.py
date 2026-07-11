"""Tests for JSONL/CSV loading and saving."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_tune.data.generator import generate_dataset
from qlora_tune.data.loaders import load_examples, save_examples


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """Examples survive a JSONL save/load cycle unchanged."""
    examples = generate_dataset(n_per_category=3)
    path = tmp_path / "data.jsonl"
    save_examples(examples, path)
    loaded = load_examples(path)
    assert [ex.to_dict() for ex in loaded] == [ex.to_dict() for ex in examples]


def test_csv_roundtrip_core_fields(tmp_path: Path) -> None:
    """CSV round-trips the four core fields (meta is JSONL-only)."""
    examples = generate_dataset(n_per_category=3)
    path = tmp_path / "data.csv"
    save_examples(examples, path)
    loaded = load_examples(path)
    assert len(loaded) == len(examples)
    assert loaded[0].id == examples[0].id
    assert loaded[0].instruction == examples[0].instruction


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    """Anything other than .jsonl/.csv is rejected."""
    with pytest.raises(ValueError, match="Unsupported extension"):
        save_examples([], tmp_path / "data.parquet")


def test_missing_required_field_raises(tmp_path: Path) -> None:
    """Rows missing a required field fail with the row index in the message."""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "1", "category": "vpn"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_examples(path)
