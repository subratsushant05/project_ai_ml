"""Business-calibrated decision threshold selection.

A churn model outputs probabilities; the business acts on a binary decision
(send a retention offer or not). The profit-maximising cut-off depends on the
economics of the offer, not on symmetric classification metrics:

* Targeting a true churner: pay ``offer_cost``, and with probability
  ``offer_save_rate`` retain revenue worth ``churn_loss``.
* Targeting a non-churner: pay ``offer_cost`` for nothing.

Profit at threshold ``t`` = ``TP(t) * (save_rate * churn_loss - offer_cost)
- FP(t) * offer_cost``, measured against the do-nothing baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Outcome of a threshold sweep.

    Attributes:
        best_threshold: Probability cut-off that maximises expected profit.
        best_profit: Expected profit at ``best_threshold`` (USD, vs. doing
            nothing).
        thresholds: Swept threshold grid.
        profits: Expected profit at each grid point.
        f1_threshold: Cut-off that maximises F1, for comparison.
    """

    best_threshold: float
    best_profit: float
    thresholds: np.ndarray = field(repr=False)
    profits: np.ndarray = field(repr=False)
    f1_threshold: float = 0.5


def profit_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    offer_cost: float,
    churn_loss: float,
    offer_save_rate: float,
) -> float:
    """Compute expected campaign profit at a single threshold.

    Args:
        y_true: Binary churn labels.
        y_prob: Predicted churn probabilities.
        threshold: Customers with probability >= threshold get the offer.
        offer_cost: Cost of one retention offer.
        churn_loss: Revenue lost when a customer churns.
        offer_save_rate: Probability the offer retains a true churner.

    Returns:
        Expected profit in the same currency as the inputs.
    """
    y_true = np.asarray(y_true)
    targeted = np.asarray(y_prob) >= threshold
    tp = int(np.sum(targeted & (y_true == 1)))
    fp = int(np.sum(targeted & (y_true == 0)))
    return tp * (offer_save_rate * churn_loss - offer_cost) - fp * offer_cost


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    offer_cost: float,
    churn_loss: float,
    offer_save_rate: float,
    grid_size: int = 199,
) -> ThresholdResult:
    """Sweep thresholds and pick the profit-maximising cut-off.

    Args:
        y_true: Binary churn labels of an evaluation set.
        y_prob: Predicted churn probabilities for the same rows.
        offer_cost: Cost of one retention offer.
        churn_loss: Revenue lost when a customer churns.
        offer_save_rate: Probability the offer retains a true churner.
        grid_size: Number of thresholds in (0, 1) to evaluate.

    Returns:
        A :class:`ThresholdResult` with the profit curve and best cut-offs.
    """
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.005, 0.995, grid_size)
    profits = np.array(
        [
            profit_at_threshold(y_true, y_prob, t, offer_cost, churn_loss, offer_save_rate)
            for t in thresholds
        ]
    )
    best_idx = int(np.argmax(profits))

    f1_scores = [f1_score(y_true, (np.asarray(y_prob) >= t).astype(int)) for t in thresholds]
    f1_threshold = float(thresholds[int(np.argmax(f1_scores))])

    result = ThresholdResult(
        best_threshold=float(thresholds[best_idx]),
        best_profit=float(profits[best_idx]),
        thresholds=thresholds,
        profits=profits,
        f1_threshold=f1_threshold,
    )
    logger.info(
        "Profit-optimal threshold %.3f (profit $%.0f) | F1-optimal threshold %.3f",
        result.best_threshold,
        result.best_profit,
        result.f1_threshold,
    )
    return result
