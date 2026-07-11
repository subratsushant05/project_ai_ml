"""Merge trained LoRA adapters into the base model and export (heavy).

The merged model is a plain ``transformers`` checkpoint that can be served
directly or converted to GGUF for llama.cpp (see the README for the
conversion commands).

Usage:
    python -m qlora_tune.merge --base meta-llama/Llama-3.1-8B-Instruct \\
        --adapter outputs/adapter --out outputs/merged
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    """Load the base model in 16-bit, merge the adapter, and save.

    The base model is loaded *without* quantization: merging into 4-bit
    weights would bake quantization error into the exported checkpoint.

    Args:
        base_model: Hugging Face id of the base model used for training.
        adapter_dir: Directory containing the trained PEFT adapter.
        output_dir: Destination directory for the merged model + tokenizer.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model %s in bf16", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    logger.info("Attaching adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    logger.info("Merging adapter weights into the base model")
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Merged model saved to %s", output_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base", required=True, help="Base model Hugging Face id")
    parser.add_argument("--adapter", type=Path, required=True, help="Trained adapter directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    merge_adapter(args.base, args.adapter, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
