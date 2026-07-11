"""Tests for the business threshold optimizer."""

from __future__ import annotations

import numpy as np

from churn_pipeline.threshold import optimize_threshold, profit_at_threshold


def test_profit_math_on_toy_case() -> None:
    """Hand-computed profit for a tiny, fully specified case.

    With offer_cost=50, churn_loss=600, save_rate=0.5 a targeted true
    churner is worth 0.5*600-50 = 250 and a targeted non-churner costs 50.
    Threshold 0.5 targets probs {0.9, 0.8, 0.6}: two churners and one
    non-churner -> 2*250 - 1*50 = 450.
    """
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.6, 0.2, 0.1])
    profit = profit_at_threshold(
        y_true, y_prob, threshold=0.5, offer_cost=50, churn_loss=600, offer_save_rate=0.5
    )
    assert profit == 450.0


def test_profit_zero_when_nobody_targeted() -> None:
    """A threshold above every probability yields zero profit."""
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.2, 0.3, 0.4])
    profit = profit_at_threshold(
        y_true, y_prob, threshold=0.99, offer_cost=50, churn_loss=600, offer_save_rate=0.4
    )
    assert profit == 0.0


def test_optimizer_finds_separating_threshold() -> None:
    """On perfectly separated scores the optimum targets exactly churners."""
    y_true = np.array([0] * 50 + [1] * 25)
    y_prob = np.concatenate([np.full(50, 0.1), np.full(25, 0.9)])
    result = optimize_threshold(
        y_true, y_prob, offer_cost=50, churn_loss=600, offer_save_rate=0.4
    )
    assert 0.1 < result.best_threshold <= 0.9
    # 25 churners * (0.4*600 - 50) = 4750, no false positives.
    assert result.best_profit == 25 * (0.4 * 600 - 50)
    assert len(result.thresholds) == len(result.profits)


def test_optimizer_profit_is_curve_maximum() -> None:
    """Reported best profit equals the maximum of the returned curve."""
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=400)
    y_prob = np.clip(0.3 * y_true + rng.uniform(0, 0.7, size=400), 0, 1)
    result = optimize_threshold(
        y_true, y_prob, offer_cost=50, churn_loss=600, offer_save_rate=0.4
    )
    assert result.best_profit == result.profits.max()
    assert 0.0 < result.f1_threshold < 1.0
