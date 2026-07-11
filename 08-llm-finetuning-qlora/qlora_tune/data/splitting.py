"""Stratified train/validation/test splitting."""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from qlora_tune.data.records import Example

logger = logging.getLogger(__name__)


def stratified_split(
    examples: list[Example],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 13,
) -> dict[str, list[Example]]:
    """Split examples into train/val/test, stratified by category.

    Each category is shuffled independently with the given seed and sliced
    according to the requested fractions, so every split preserves the
    category distribution of the full dataset (up to rounding).

    Args:
        examples: Cleaned examples to split.
        train_frac: Fraction of each category assigned to train.
        val_frac: Fraction assigned to validation; the remainder goes to test.
        seed: RNG seed for the per-category shuffles.

    Returns:
        Dict with keys ``"train"``, ``"val"`` and ``"test"``.

    Raises:
        ValueError: If the fractions are out of range or sum to >= 1 leaves
            no room for a test split.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 to leave a test split")

    by_category: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_category[ex.category].append(ex)

    rng = random.Random(seed)
    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for category in sorted(by_category):
        bucket = by_category[category][:]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        splits["train"].extend(bucket[:n_train])
        splits["val"].extend(bucket[n_train : n_train + n_val])
        splits["test"].extend(bucket[n_train + n_val :])

    logger.info(
        "Split %d examples -> train=%d val=%d test=%d",
        len(examples),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits
