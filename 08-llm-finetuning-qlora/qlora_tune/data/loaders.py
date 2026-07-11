"""Reading and writing instruction datasets as JSONL or CSV."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from qlora_tune.data.records import Example

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "category", "instruction", "response")


def load_examples(path: str | Path) -> list[Example]:
    """Load examples from a ``.jsonl`` or ``.csv`` file (by extension).

    Args:
        path: File path ending in ``.jsonl`` or ``.csv``.

    Returns:
        Parsed examples.

    Raises:
        ValueError: On unsupported extension or rows missing required fields.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix == ".csv":
        rows = _read_csv(path)
    else:
        raise ValueError(f"Unsupported extension {path.suffix!r}; expected .jsonl or .csv")

    examples: list[Example] = []
    for i, row in enumerate(rows):
        missing = [f for f in REQUIRED_FIELDS if not row.get(f)]
        if missing:
            raise ValueError(f"{path.name} row {i}: missing required field(s) {missing}")
        examples.append(Example.from_dict(row))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def save_examples(examples: list[Example], path: str | Path) -> None:
    """Save examples to a ``.jsonl`` or ``.csv`` file (by extension).

    Args:
        examples: Examples to write.
        path: Destination path ending in ``.jsonl`` or ``.csv``. Parent
            directories are created as needed.

    Raises:
        ValueError: On unsupported extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    elif path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(REQUIRED_FIELDS))
            writer.writeheader()
            for ex in examples:
                row = ex.to_dict()
                row.pop("meta", None)
                writer.writerow(row)
    else:
        raise ValueError(f"Unsupported extension {path.suffix!r}; expected .jsonl or .csv")
    logger.info("Saved %d examples to %s", len(examples), path)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file with a header row into a list of dicts."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
