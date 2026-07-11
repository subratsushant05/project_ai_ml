"""Tests for the analytic LoRA parameter and VRAM math."""

from __future__ import annotations

from pathlib import Path

import pytest

from qlora_tune.config import TrainingConfig
from qlora_tune.planning import (
    KNOWN_ARCHITECTURES,
    build_plan,
    lora_trainable_params,
    render_plan,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_lora_params_llama31_8b_attention_only_hand_computed() -> None:
    """Llama 3.1 8B, r=16, q/k/v/o adapters.

    Per layer: q 16*(4096+4096)=131072, k 16*(4096+1024)=81920,
    v 81920, o 131072 -> 425,984. Times 32 layers = 13,631,488.
    """
    dims = KNOWN_ARCHITECTURES["meta-llama/Llama-3.1-8B-Instruct"]
    n = lora_trainable_params(dims, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    assert n == 13_631_488


def test_lora_params_qwen_05b_hand_computed() -> None:
    """Qwen2.5 0.5B, r=8, q/k/v/o adapters.

    Per layer: q 8*(896+896)=14336, k 8*(896+128)=8192, v 8192,
    o 14336 -> 45,056. Times 24 layers = 1,081,344.
    """
    dims = KNOWN_ARCHITECTURES["Qwen/Qwen2.5-0.5B-Instruct"]
    n = lora_trainable_params(dims, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    assert n == 1_081_344


def test_lora_params_scale_linearly_with_rank() -> None:
    """Doubling r exactly doubles the trainable parameter count."""
    dims = KNOWN_ARCHITECTURES["meta-llama/Llama-3.1-8B-Instruct"]
    targets = ["q_proj", "v_proj", "gate_proj"]
    assert lora_trainable_params(dims, 32, targets) == 2 * lora_trainable_params(dims, 16, targets)


def test_build_plan_for_bundled_config() -> None:
    """The 8B example config yields a plan with all-module adapter count.

    q/k/v/o as above plus gate 16*(4096+14336)=294912, up 294912,
    down 294912 per layer -> 1,310,720 per layer -> 41,943,040 total.
    """
    config = TrainingConfig.from_yaml(CONFIG_DIR / "llama-3.1-8b.yaml")
    plan = build_plan(config)
    assert plan.trainable_params == 41_943_040
    assert 0 < plan.trainable_pct < 1


def test_vram_estimate_is_plausible_for_24gb_gpu() -> None:
    """The 8B QLoRA estimate lands well under 24 GiB but above 4-bit weights alone."""
    config = TrainingConfig.from_yaml(CONFIG_DIR / "llama-3.1-8b.yaml")
    plan = build_plan(config)
    assert plan.vram.base_weights_gib < plan.vram.total_gib < 24
    assert plan.vram.total_gib > 4


def test_unknown_model_raises_with_supported_list() -> None:
    """Planning refuses unknown models instead of downloading configs."""
    config = TrainingConfig(model_name="some-org/unknown-model")
    with pytest.raises(KeyError, match="supports"):
        build_plan(config)


def test_render_plan_mentions_key_numbers() -> None:
    """The rendered plan includes the trainable count and total VRAM."""
    config = TrainingConfig.from_yaml(CONFIG_DIR / "llama-3.1-8b.yaml")
    plan = build_plan(config)
    text = render_plan(plan)
    assert "41,943,040" in text
    assert "TOTAL" in text
    assert config.model_name in text
