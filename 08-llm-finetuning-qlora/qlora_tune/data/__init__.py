"""Data layer: synthetic dataset generation, loading, cleaning, splitting, formatting."""

from qlora_tune.data.cleaning import clean_examples, scrub_pii
from qlora_tune.data.generator import generate_dataset
from qlora_tune.data.loaders import load_examples, save_examples
from qlora_tune.data.records import Example
from qlora_tune.data.splitting import stratified_split
from qlora_tune.data.templates import format_chatml, format_llama3, get_formatter

__all__ = [
    "Example",
    "clean_examples",
    "format_chatml",
    "format_llama3",
    "generate_dataset",
    "get_formatter",
    "load_examples",
    "save_examples",
    "scrub_pii",
    "stratified_split",
]
