"""Tests for the from-scratch ROUGE-L, exact-match and keyword-coverage metrics."""

from __future__ import annotations

import pytest

from qlora_tune.evaluation.metrics import exact_match, keyword_coverage, rouge_l, tokenize


def test_rouge_identical_texts() -> None:
    """Identical texts score 1.0 on precision, recall and F1."""
    score = rouge_l("restart the vpn client", "restart the vpn client")
    assert score.precision == score.recall == score.f1 == 1.0


def test_rouge_hand_computed_partial_overlap() -> None:
    """pred='the cat sat' vs ref='the cat sat on the mat'.

    LCS = 3, precision = 3/3 = 1.0, recall = 3/6 = 0.5,
    F1 = 2*1.0*0.5 / 1.5 = 2/3.
    """
    score = rouge_l("the cat sat", "the cat sat on the mat")
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(2 / 3)


def test_rouge_respects_word_order() -> None:
    """LCS is order-sensitive: reversed tokens share only a 1-token subsequence."""
    score = rouge_l("gamma beta alpha", "alpha beta gamma")
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == pytest.approx(1 / 3)
    assert score.f1 == pytest.approx(1 / 3)


def test_rouge_no_overlap_and_empty() -> None:
    """Disjoint or empty inputs score zero without dividing by zero."""
    assert rouge_l("alpha beta", "gamma delta").f1 == 0.0
    assert rouge_l("", "reference text").f1 == 0.0
    assert rouge_l("prediction", "").f1 == 0.0


def test_exact_match_is_normalized() -> None:
    """Exact match ignores case and punctuation but not word changes."""
    assert exact_match("Hello, World!", "hello world") == 1.0
    assert exact_match("hello there world", "hello world") == 0.0


def test_keyword_coverage_hand_computed() -> None:
    """ref keywords {restart, vpn, client, update, firmware}; pred hits 3/5."""
    reference = "Restart the VPN client and update firmware"
    prediction = "Please restart your vpn client"
    assert keyword_coverage(prediction, reference) == pytest.approx(0.6)


def test_tokenize_strips_punctuation_and_lowercases() -> None:
    """Tokenizer keeps alphanumerics and apostrophe contractions only."""
    assert tokenize("Can't reach VPN-server (error 0x8007)!") == [
        "can't",
        "reach",
        "vpn",
        "server",
        "error",
        "0x8007",
    ]
