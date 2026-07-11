"""Dry-run planning: analytic LoRA parameter counts and VRAM estimates.

Everything here is plain arithmetic over a table of known architecture
dimensions, so a training plan can be produced and unit-tested without
downloading a model or importing torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qlora_tune.config import TrainingConfig

logger = logging.getLogger(__name__)

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ModelDims:
    """Transformer dimensions needed for analytic parameter/VRAM math.

    Attributes:
        hidden_size: Model (residual stream) width.
        num_layers: Number of decoder blocks.
        num_heads: Attention query heads.
        num_kv_heads: Key/value heads (GQA).
        head_dim: Per-head dimension.
        intermediate_size: MLP inner width.
        vocab_size: Tokenizer vocabulary size.
        total_params: Total base-model parameter count.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    total_params: int


# Dimensions from the models' published config.json files.
# ModelDims(hidden, layers, heads, kv_heads, head_dim, intermediate, vocab, total)
KNOWN_ARCHITECTURES: dict[str, ModelDims] = {
    "meta-llama/Llama-3.1-8B-Instruct": (
        ModelDims(4096, 32, 32, 8, 128, 14336, 128256, 8_030_261_248)
    ),
    "meta-llama/Llama-3.2-1B-Instruct": (
        ModelDims(2048, 16, 32, 8, 64, 8192, 128256, 1_235_814_400)
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        ModelDims(3072, 28, 24, 8, 128, 8192, 128256, 3_212_749_824)
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": (
        ModelDims(4096, 32, 32, 8, 128, 14336, 32768, 7_248_023_552)
    ),
    "Qwen/Qwen2.5-0.5B-Instruct": ModelDims(896, 24, 14, 2, 64, 4864, 151936, 494_032_768),
    "Qwen/Qwen2.5-7B-Instruct": ModelDims(3584, 28, 28, 4, 128, 18944, 152064, 7_615_616_512),
}


def module_shapes(dims: ModelDims) -> dict[str, tuple[int, int]]:
    """Return (in_features, out_features) for each LoRA-targetable module.

    Args:
        dims: Architecture dimensions.

    Returns:
        Mapping from module name to its linear layer shape.
    """
    h = dims.hidden_size
    q_out = dims.num_heads * dims.head_dim
    kv_out = dims.num_kv_heads * dims.head_dim
    i = dims.intermediate_size
    return {
        "q_proj": (h, q_out),
        "k_proj": (h, kv_out),
        "v_proj": (h, kv_out),
        "o_proj": (q_out, h),
        "gate_proj": (h, i),
        "up_proj": (h, i),
        "down_proj": (i, h),
    }


def lora_trainable_params(dims: ModelDims, r: int, target_modules: list[str]) -> int:
    """Compute LoRA trainable parameters analytically.

    Each adapted ``in -> out`` linear layer gains two matrices: A with shape
    ``(r, in)`` and B with shape ``(out, r)``, i.e. ``r * (in + out)`` params,
    repeated in every decoder layer. LoRA adds no bias terms.

    Args:
        dims: Architecture dimensions.
        r: LoRA rank.
        target_modules: Modules to adapt (subset of :func:`module_shapes`).

    Returns:
        Total trainable parameter count.

    Raises:
        KeyError: If a target module is not recognized.
    """
    shapes = module_shapes(dims)
    per_layer = 0
    for name in target_modules:
        fan_in, fan_out = shapes[name]
        per_layer += r * (fan_in + fan_out)
    return per_layer * dims.num_layers


@dataclass(frozen=True, slots=True)
class VramEstimate:
    """Rough VRAM budget for one training GPU, in GiB.

    Attributes:
        base_weights_gib: 4-bit (or 16-bit) base model weights.
        adapter_gib: Adapter weights, gradients and Adam moments.
        activations_gib: Activations with gradient checkpointing assumed.
        overhead_gib: CUDA context / allocator / framework slack.
        total_gib: Sum of all components.
    """

    base_weights_gib: float
    adapter_gib: float
    activations_gib: float
    overhead_gib: float
    total_gib: float


def estimate_vram(dims: ModelDims, config: TrainingConfig, trainable_params: int) -> VramEstimate:
    """Estimate peak training VRAM from first principles.

    Assumptions (documented, deliberately rough — expect +/- 20%):
      * 4-bit NF4 base weights cost ~0.55 bytes/param including quantization
        constants (~0.5 with double quantization, slightly more without).
      * Adapters train in fp32 under ``prepare_model_for_kbit_training``:
        weight + grad + two Adam moments = 16 bytes/param.
      * Gradient checkpointing stores ~2 residual-width tensors per layer
        plus one full logits tensor (fp32 upcast for the loss).

    Args:
        dims: Architecture dimensions.
        config: Validated training config.
        trainable_params: Output of :func:`lora_trainable_params`.

    Returns:
        A :class:`VramEstimate` with per-component and total GiB.
    """
    if config.quantization.load_in_4bit:
        bytes_per_weight = 0.55 if not config.quantization.double_quant else 0.52
    else:
        bytes_per_weight = 2.0
    base = dims.total_params * bytes_per_weight / _GIB

    adapter = trainable_params * 16 / _GIB

    tokens = config.per_device_train_batch_size * config.max_seq_length
    act_bytes = 2  # bf16/fp16 activations
    residuals = tokens * dims.hidden_size * act_bytes * dims.num_layers * 2
    if not config.gradient_checkpointing:
        residuals *= 6  # rough multiplier: all intermediate tensors kept
    logits = tokens * dims.vocab_size * (2 + 4)  # bf16 logits + fp32 upcast
    activations = (residuals + logits) / _GIB

    overhead = 1.0
    total = base + adapter + activations + overhead
    return VramEstimate(
        base_weights_gib=round(base, 2),
        adapter_gib=round(adapter, 2),
        activations_gib=round(activations, 2),
        overhead_gib=overhead,
        total_gib=round(total, 2),
    )


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Complete dry-run plan for a training config.

    Attributes:
        config: The validated config the plan was derived from.
        dims: Architecture dimensions used for the math.
        trainable_params: Analytic LoRA trainable parameter count.
        trainable_pct: Trainable params as a percent of total params.
        vram: VRAM estimate breakdown.
    """

    config: TrainingConfig
    dims: ModelDims
    trainable_params: int
    trainable_pct: float
    vram: VramEstimate


def build_plan(config: TrainingConfig) -> TrainingPlan:
    """Build a dry-run plan for a config against the known-architecture table.

    Args:
        config: Validated training config.

    Returns:
        A :class:`TrainingPlan`.

    Raises:
        KeyError: If ``config.model_name`` is not in
            :data:`KNOWN_ARCHITECTURES` (dry-run planning never downloads
            model configs).
    """
    try:
        dims = KNOWN_ARCHITECTURES[config.model_name]
    except KeyError:
        known = "\n  ".join(sorted(KNOWN_ARCHITECTURES))
        raise KeyError(
            f"No architecture entry for {config.model_name!r}. "
            f"Dry-run planning supports:\n  {known}"
        ) from None
    trainable = lora_trainable_params(dims, config.lora.r, config.lora.target_modules)
    vram = estimate_vram(dims, config, trainable)
    return TrainingPlan(
        config=config,
        dims=dims,
        trainable_params=trainable,
        trainable_pct=round(100 * trainable / dims.total_params, 4),
        vram=vram,
    )


def render_plan(plan: TrainingPlan) -> str:
    """Render a plan as a human-readable report block.

    Args:
        plan: Plan produced by :func:`build_plan`.

    Returns:
        Multi-line string suitable for printing to a terminal.
    """
    c, v = plan.config, plan.vram
    quant = c.quantization
    quant_desc = (
        f"4-bit {quant.quant_type} (double_quant={quant.double_quant}, "
        f"compute={quant.compute_dtype})"
        if quant.load_in_4bit
        else "16-bit (no quantization)"
    )
    lines = [
        "=" * 62,
        "QLoRA DRY-RUN TRAINING PLAN",
        "=" * 62,
        f"Model                  {c.model_name}",
        f"  total params         {plan.dims.total_params:,}",
        f"  layers / hidden      {plan.dims.num_layers} / {plan.dims.hidden_size}",
        f"Quantization           {quant_desc}",
        f"LoRA                   r={c.lora.r} alpha={c.lora.alpha} dropout={c.lora.dropout}",
        f"  target modules       {', '.join(c.lora.target_modules)}",
        f"  trainable params     {plan.trainable_params:,} ({plan.trainable_pct:.4f}% of base)",
        f"Sequence length        {c.max_seq_length} (packing={c.packing})",
        f"Batch                  micro={c.per_device_train_batch_size} x "
        f"accum={c.gradient_accumulation_steps} -> effective={c.effective_batch_size}",
        f"Optimizer              {c.optimizer.name} lr={c.optimizer.learning_rate} "
        f"({c.optimizer.scheduler}, warmup={c.optimizer.warmup_ratio})",
        f"Epochs                 {c.num_train_epochs}",
        "-" * 62,
        "Estimated VRAM (single GPU, +/-20%):",
        f"  base weights         {v.base_weights_gib:6.2f} GiB",
        f"  adapter + optimizer  {v.adapter_gib:6.2f} GiB",
        f"  activations          {v.activations_gib:6.2f} GiB",
        f"  framework overhead   {v.overhead_gib:6.2f} GiB",
        f"  TOTAL                {v.total_gib:6.2f} GiB",
        "=" * 62,
    ]
    return "\n".join(lines)
