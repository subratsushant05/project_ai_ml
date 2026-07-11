"""qlora_tune: a QLoRA fine-tuning toolkit for instruction-tuning small LLMs.

The package is split into a *light* layer (data, config, planning, evaluation)
that runs anywhere with CPU-only dependencies, and a *heavy* layer (training,
adapter merging) that lazily imports torch / transformers / peft / trl and is
only needed on a GPU machine.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
