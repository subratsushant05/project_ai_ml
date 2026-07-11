"""Text-generation metrics implemented from scratch (no external metric libs).

Includes token-level ROUGE-L (longest common subsequence), normalized exact
match, and a keyword-coverage score tailored to procedural helpdesk answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

_STOPWORDS: frozenset[str] = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "then", "this", "to", "was", "we", "were", "will", "with", "your", "you",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens.

    Args:
        text: Raw text.

    Returns:
        List of lowercase tokens; punctuation is discarded.
    """
    return _TOKEN_RE.findall(text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Uses the classic O(len(a) * len(b)) dynamic program with a rolling row,
    so memory is O(min side).
    """
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


@dataclass(frozen=True, slots=True)
class RougeScore:
    """ROUGE-L precision / recall / F1 triple.

    Attributes:
        precision: LCS length / prediction length.
        recall: LCS length / reference length.
        f1: Harmonic mean of precision and recall.
    """

    precision: float
    recall: float
    f1: float


def rouge_l(prediction: str, reference: str) -> RougeScore:
    """Compute token-level ROUGE-L between a prediction and a reference.

    Args:
        prediction: Model output text.
        reference: Gold answer text.

    Returns:
        :class:`RougeScore`. All fields are 0.0 when either side is empty.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return RougeScore(0.0, 0.0, 0.0)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return RougeScore(precision, recall, f1)


def exact_match(prediction: str, reference: str) -> float:
    """Normalized exact match: 1.0 if token sequences are identical, else 0.0.

    Args:
        prediction: Model output text.
        reference: Gold answer text.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if tokenize(prediction) == tokenize(reference) else 0.0


def keyword_coverage(prediction: str, reference: str) -> float:
    """Fraction of distinct reference content words present in the prediction.

    Content words are reference tokens that are not stopwords and are at
    least 3 characters long. This rewards answers that mention the right
    entities and actions even when phrasing differs — a useful signal for
    procedural helpdesk resolutions.

    Args:
        prediction: Model output text.
        reference: Gold answer text.

    Returns:
        Coverage in [0, 1]; 0.0 if the reference has no content words.
    """
    ref_keywords = {t for t in tokenize(reference) if t not in _STOPWORDS and len(t) >= 3}
    if not ref_keywords:
        return 0.0
    pred_tokens = set(tokenize(prediction))
    return len(ref_keywords & pred_tokens) / len(ref_keywords)
