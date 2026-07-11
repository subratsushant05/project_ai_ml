"""Training configuration schema (pydantic v2) with YAML load/save.

The config layer is deliberately light: it validates everything a training
run needs *before* any heavy library is imported, so misconfigurations fail
in seconds on a laptop instead of minutes into a GPU job.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

ALLOWED_LORA_RANKS: frozenset[int] = frozenset({4, 8, 16, 32, 64, 128})
KNOWN_TARGET_MODULES: frozenset[str] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


class LoraSettings(BaseModel):
    """LoRA adapter hyperparameters.

    Attributes:
        r: Adapter rank; restricted to commonly used powers of two.
        alpha: LoRA scaling numerator (effective scale is ``alpha / r``).
        dropout: Dropout applied to the LoRA branch during training.
        target_modules: Linear submodules to wrap with adapters.
    """

    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @field_validator("r")
    @classmethod
    def _check_rank(cls, v: int) -> int:
        if v not in ALLOWED_LORA_RANKS:
            raise ValueError(f"r={v} not in allowed set {sorted(ALLOWED_LORA_RANKS)}")
        return v

    @field_validator("target_modules")
    @classmethod
    def _check_targets(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_modules must not be empty")
        unknown = sorted(set(v) - KNOWN_TARGET_MODULES)
        if unknown:
            raise ValueError(
                f"Unknown target module(s) {unknown}; known: {sorted(KNOWN_TARGET_MODULES)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("target_modules contains duplicates")
        return v

    @model_validator(mode="after")
    def _warn_alpha_convention(self) -> LoraSettings:
        if self.alpha != 2 * self.r:
            logger.warning(
                "lora.alpha=%d deviates from the common alpha=2*r convention (r=%d -> alpha=%d)",
                self.alpha,
                self.r,
                2 * self.r,
            )
        return self


class QuantizationSettings(BaseModel):
    """4-bit quantization settings mirroring ``BitsAndBytesConfig``.

    Attributes:
        load_in_4bit: Whether to quantize base weights to 4-bit (QLoRA).
        quant_type: 4-bit data type; ``nf4`` is the QLoRA-paper default.
        double_quant: Quantize the quantization constants too (saves ~0.4 bit/param).
        compute_dtype: Dtype used for matmuls on dequantized weights.
    """

    model_config = ConfigDict(extra="forbid")

    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class OptimizerSettings(BaseModel):
    """Optimizer and LR-schedule settings.

    Attributes:
        learning_rate: Peak learning rate.
        scheduler: LR schedule shape.
        warmup_ratio: Fraction of total steps used for linear warmup.
        weight_decay: Decoupled weight decay.
        name: Trainer optimizer identifier (paged variants avoid OOM spikes).
    """

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=2e-4, gt=0.0, le=1e-2)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.0, ge=0.0, le=0.3)
    name: str = "paged_adamw_8bit"


class TrainingConfig(BaseModel):
    """Full QLoRA fine-tuning run configuration.

    Attributes:
        model_name: Hugging Face model id of the base model.
        chat_template: Template used to serialize conversations.
        lora: LoRA adapter settings.
        quantization: 4-bit quantization settings.
        optimizer: Optimizer/schedule settings.
        max_seq_length: Tokenizer truncation / packing length.
        packing: Pack multiple short examples into each sequence.
        per_device_train_batch_size: Micro-batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate before an update.
        num_train_epochs: Training epochs.
        gradient_checkpointing: Trade compute for activation memory.
        logging_steps: Trainer logging interval.
        eval_steps: Evaluation interval (steps).
        output_dir: Where checkpoints and the adapter are written.
        seed: Global seed for reproducibility.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    chat_template: Literal["llama3", "chatml"] = "llama3"
    lora: LoraSettings = Field(default_factory=LoraSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    optimizer: OptimizerSettings = Field(default_factory=OptimizerSettings)
    max_seq_length: int = Field(default=2048, ge=128, le=32768)
    packing: bool = True
    per_device_train_batch_size: int = Field(default=1, ge=1, le=512)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=1024)
    num_train_epochs: float = Field(default=2.0, gt=0.0, le=100.0)
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    output_dir: str = "outputs/adapter"
    seed: int = 13

    @property
    def effective_batch_size(self) -> int:
        """Tokens-independent effective batch size per optimizer step."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load and validate a config from a YAML file.

        Args:
            path: Path to the YAML config.

        Returns:
            Validated :class:`TrainingConfig`.

        Raises:
            pydantic.ValidationError: If the file contents are invalid.
            FileNotFoundError: If the file does not exist.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at top level")
        config = cls.model_validate(raw)
        logger.info("Loaded config from %s (model=%s)", path, config.model_name)
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to a YAML file (round-trips with ``from_yaml``).

        Args:
            path: Destination path; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Saved config to %s", path)
