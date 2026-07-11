"""Tests for the pure promotion-policy decision logic."""

from mlops_pipeline.registry import PromotionPolicy, promotion_decision

POLICY = PromotionPolicy(margin=0.01, min_auc=0.70, max_brier=0.20)


def test_first_promotion_allowed_without_incumbent() -> None:
    """A healthy candidate is promoted when no production model exists."""
    decision = promotion_decision({"auc": 0.80, "brier": 0.15}, None, POLICY)
    assert decision.promote
    assert any("first promotion" in r for r in decision.reasons)


def test_better_candidate_is_promoted() -> None:
    """Beating the incumbent by more than the margin promotes."""
    decision = promotion_decision(
        {"auc": 0.85, "brier": 0.14}, {"auc": 0.80, "brier": 0.15}, POLICY
    )
    assert decision.promote


def test_worse_candidate_is_rejected() -> None:
    """A candidate below the incumbent is rejected."""
    decision = promotion_decision(
        {"auc": 0.75, "brier": 0.16}, {"auc": 0.80, "brier": 0.15}, POLICY
    )
    assert not decision.promote


def test_margin_is_respected() -> None:
    """Improvements smaller than the margin do not promote."""
    incumbent = {"auc": 0.800, "brier": 0.15}
    barely_better = promotion_decision({"auc": 0.805, "brier": 0.15}, incumbent, POLICY)
    assert not barely_better.promote
    at_margin = promotion_decision({"auc": 0.810, "brier": 0.15}, incumbent, POLICY)
    assert at_margin.promote


def test_absolute_guardrails_block_promotion() -> None:
    """AUC floor and Brier cap apply even without an incumbent."""
    low_auc = promotion_decision({"auc": 0.65, "brier": 0.15}, None, POLICY)
    assert not low_auc.promote
    bad_calibration = promotion_decision({"auc": 0.90, "brier": 0.25}, None, POLICY)
    assert not bad_calibration.promote
    # ... and also when the candidate would otherwise beat the incumbent.
    both = promotion_decision(
        {"auc": 0.69, "brier": 0.15}, {"auc": 0.60, "brier": 0.18}, POLICY
    )
    assert not both.promote
