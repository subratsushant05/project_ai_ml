"""Tests for the pydantic TrainingConfig schema and YAML round-tripping."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from qlora_tune.config import LoraSettings, TrainingConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("name", ["llama-3.1-8b.yaml", "qwen2.5-0.5b-smoke-test.yaml"])
def test_bundled_configs_are_valid(name: str) -> None:
    """Both shipped example configs load and validate."""
    config = TrainingConfig.from_yaml(CONFIG_DIR / name)
    assert config.quantization.load_in_4bit is True
    assert config.lora.alpha == 2 * config.lora.r


def test_disallowed_lora_rank_rejected() -> None:
    """Ranks outside the allowed power-of-two set fail validation."""
    with pytest.raises(ValidationError, match="allowed set"):
        LoraSettings(r=7, alpha=14)


def test_unknown_target_module_rejected() -> None:
    """A typo'd target module name fails validation with a helpful message."""
    with pytest.raises(ValidationError, match="Unknown target module"):
        LoraSettings(target_modules=["q_proj", "qq_proj"])


def test_alpha_convention_warning(caplog: pytest.LogCaptureFixture) -> None:
    """alpha != 2*r is legal but logs a warning about the convention."""
    with caplog.at_level(logging.WARNING, logger="qlora_tune.config"):
        LoraSettings(r=16, alpha=8)
    assert any("alpha=2*r" in rec.message for rec in caplog.records)


def test_extra_fields_forbidden() -> None:
    """Misspelled top-level keys are rejected instead of silently ignored."""
    with pytest.raises(ValidationError):
        TrainingConfig(model_name="m", max_seq_len=512)  # type: ignore[call-arg]


def test_bad_numeric_ranges_rejected() -> None:
    """Out-of-range numerics (dropout, lr) fail validation."""
    with pytest.raises(ValidationError):
        LoraSettings(dropout=1.5)
    with pytest.raises(ValidationError):
        TrainingConfig(model_name="m", optimizer={"learning_rate": -1.0})


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """to_yaml followed by from_yaml reproduces the config exactly."""
    original = TrainingConfig.from_yaml(CONFIG_DIR / "llama-3.1-8b.yaml")
    out = tmp_path / "roundtrip.yaml"
    original.to_yaml(out)
    reloaded = TrainingConfig.from_yaml(out)
    assert reloaded == original


def test_effective_batch_size() -> None:
    """Effective batch size is micro-batch times accumulation steps."""
    config = TrainingConfig(
        model_name="m", per_device_train_batch_size=4, gradient_accumulation_steps=8
    )
    assert config.effective_batch_size == 32
