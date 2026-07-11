"""End-to-end light-path demo: data -> config -> dry-run plan -> evaluation.

Runs the entire CPU-friendly pipeline with no heavy dependencies:

1. Generate the synthetic helpdesk dataset.
2. Clean it (PII scrub, dedupe, length filter) and split stratified by category.
3. Format the train split with a chat template and write JSONL artifacts.
4. Validate the bundled configs and print a dry-run training plan.
5. Score the bundled sample predictions (base vs fine-tuned) and print the report.

Usage:
    python -m qlora_tune.demo [--out demo_output] [--config configs/....yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from qlora_tune.config import TrainingConfig
from qlora_tune.data.cleaning import clean_examples
from qlora_tune.data.generator import generate_dataset
from qlora_tune.data.loaders import save_examples
from qlora_tune.data.splitting import stratified_split
from qlora_tune.data.templates import format_examples
from qlora_tune.evaluation.harness import evaluate_predictions, load_predictions, render_report
from qlora_tune.planning import build_plan, render_plan

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "llama-3.1-8b.yaml"
SAMPLE_PREDICTIONS = REPO_ROOT / "sample_data" / "sample_predictions.jsonl"


def _section(title: str) -> None:
    """Print a section banner."""
    print(f"\n### {title}\n")


def run_demo(out_dir: Path, config_path: Path, predictions_path: Path) -> int:
    """Run the full light-path pipeline and print a report.

    Args:
        out_dir: Directory for generated artifacts (datasets, formatted text).
        config_path: Training config YAML used for the dry-run plan.
        predictions_path: Predictions JSONL for the evaluation step.

    Returns:
        Process exit code (0 on success).
    """
    _section("1/5 Generate synthetic helpdesk dataset")
    raw = generate_dataset()
    counts = Counter(ex.category for ex in raw)
    print(f"Generated {len(raw)} examples across {len(counts)} categories:")
    for category, n in sorted(counts.items()):
        print(f"  {category:<18} {n}")

    _section("2/5 Clean and split")
    cleaned, report = clean_examples(raw)
    print(
        f"Cleaning: {report.input_count} -> {report.output_count} examples "
        f"(duplicates removed: {report.duplicates_removed}, "
        f"length-filtered: {report.length_filtered}, "
        f"PII scrubbed in: {report.pii_scrubbed})"
    )
    splits = stratified_split(cleaned)
    for name, split in splits.items():
        save_examples(split, out_dir / f"{name}.jsonl")
        print(f"  {name:<6} {len(split):>4} examples -> {out_dir / f'{name}.jsonl'}")

    _section("3/5 Chat-template formatting")
    config = TrainingConfig.from_yaml(config_path)
    formatted = format_examples(splits["train"], template=config.chat_template)
    formatted_path = out_dir / f"train_{config.chat_template}.txt"
    formatted_path.write_text("\n\n".join(formatted), encoding="utf-8")
    print(f"Formatted {len(formatted)} train examples with the "
          f"'{config.chat_template}' template -> {formatted_path}")
    preview = formatted[0][:400]
    print(f"First formatted example (truncated):\n---\n{preview}...\n---")

    _section("4/5 Dry-run training plan")
    print(render_plan(build_plan(config)))

    _section("5/5 Evaluate bundled sample predictions")
    rows = load_predictions(predictions_path)
    print(render_report(evaluate_predictions(rows)))

    print("\nDemo complete. Artifacts written to", out_dir.resolve())
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Run the qlora_tune light-path demo")
    parser.add_argument("--out", type=Path, default=Path("demo_output"), help="Artifact directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training config YAML")
    parser.add_argument(
        "--predictions", type=Path, default=SAMPLE_PREDICTIONS, help="Predictions JSONL to score"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return run_demo(args.out, args.config, args.predictions)


if __name__ == "__main__":
    sys.exit(main())
