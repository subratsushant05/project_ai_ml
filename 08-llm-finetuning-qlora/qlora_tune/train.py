"""QLoRA supervised fine-tuning entry point.

Heavy dependencies (torch / transformers / peft / trl) are imported lazily
inside the functions that need them, so ``--dry-run`` — config validation,
analytic trainable-parameter counts and VRAM estimation — works on a machine
with only the light requirements installed.

Usage:
    python -m qlora_tune.train --config configs/llama-3.1-8b.yaml --dry-run
    python -m qlora_tune.train --config configs/llama-3.1-8b.yaml \\
        --train-file data/train.jsonl --val-file data/val.jsonl   # GPU only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from qlora_tune.config import TrainingConfig
from qlora_tune.planning import build_plan, render_plan

logger = logging.getLogger(__name__)


def build_tokenizer(config: TrainingConfig) -> Any:
    """Load the tokenizer for the configured base model (heavy).

    Args:
        config: Validated training config.

    Returns:
        A ``transformers.PreTrainedTokenizerBase``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer had no pad token; using eos (%s)", tokenizer.eos_token)
    return tokenizer


def build_model(config: TrainingConfig) -> Any:
    """Load the 4-bit quantized base model with LoRA adapters attached (heavy).

    Args:
        config: Validated training config.

    Returns:
        A peft-wrapped ``transformers.PreTrainedModel`` ready for SFT.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = getattr(torch, config.quantization.compute_dtype)
    quant_config = None
    if config.quantization.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    if config.quantization.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def run_training(config: TrainingConfig, train_file: Path, val_file: Path | None) -> None:
    """Run SFT training with TRL (heavy; requires a CUDA GPU in practice).

    Args:
        config: Validated training config.
        train_file: JSONL file of formatted training examples.
        val_file: Optional JSONL file of validation examples.
    """
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from qlora_tune.data.templates import DEFAULT_SYSTEM_PROMPT, get_formatter

    tokenizer = build_tokenizer(config)
    model = build_model(config)

    data_files = {"train": str(train_file)}
    if val_file is not None:
        data_files["validation"] = str(val_file)
    dataset = load_dataset("json", data_files=data_files)

    formatter = get_formatter(config.chat_template)

    def to_text(row: dict[str, Any]) -> dict[str, str]:
        return {"text": formatter(DEFAULT_SYSTEM_PROMPT, row["instruction"], row["response"])}

    dataset = dataset.map(to_text)

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        max_length=config.max_seq_length,
        packing=config.packing,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.optimizer.learning_rate,
        lr_scheduler_type=config.optimizer.scheduler,
        warmup_ratio=config.optimizer.warmup_ratio,
        weight_decay=config.optimizer.weight_decay,
        optim=config.optimizer.name,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if val_file is not None else "no",
        eval_steps=config.eval_steps,
        bf16=config.quantization.compute_dtype == "bfloat16",
        fp16=config.quantization.compute_dtype == "float16",
        seed=config.seed,
        report_to="none",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Adapter saved to %s", config.output_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for helpdesk instructions")
    parser.add_argument("--config", type=Path, required=True, help="YAML training config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the analytic training plan; no heavy deps needed",
    )
    parser.add_argument("--train-file", type=Path, help="Formatted train JSONL (real runs)")
    parser.add_argument("--val-file", type=Path, help="Formatted validation JSONL")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = TrainingConfig.from_yaml(args.config)

    if args.dry_run:
        plan = build_plan(config)
        print(render_plan(plan))
        return 0

    if args.train_file is None:
        parser.error("--train-file is required unless --dry-run is set")
    run_training(config, args.train_file, args.val_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
