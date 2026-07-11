"""Tests for the offline chat model, output parsers, and the factory."""

from __future__ import annotations

import pytest

from agent_researcher import prompts
from agent_researcher.config import Settings
from agent_researcher.llm import ChatModel, create_chat_model
from agent_researcher.nodes import parse_feedback, parse_plan, parse_score
from agent_researcher.offline_llm import OfflineChatModel


def test_offline_planner_emits_parseable_numbered_list(
    model: OfflineChatModel,
) -> None:
    """Planner output parses into exactly n topical sub-questions."""
    user = prompts.PLANNER_USER.format(
        question="How do transformer models work?", n=3
    )
    raw = model.invoke(prompts.PLANNER_SYSTEM, user)
    plan = parse_plan(raw)
    assert len(plan) == 3
    assert all(q.endswith("?") for q in plan)
    assert all("transformer models" in q for q in plan)


def test_parse_plan_handles_mixed_bullet_formats() -> None:
    """Numbering styles and chatter around the list are tolerated."""
    raw = (
        "Here is the plan you asked for:\n\n"
        "1. What is X?\n"
        "2) How is X used?\n"
        "- What limits X?\n"
        "* What limits X?\n"
    )
    assert parse_plan(raw) == ["What is X?", "How is X used?", "What limits X?"]


def test_parse_plan_falls_back_to_bare_questions() -> None:
    """Without bullets, bare question lines are still recovered."""
    raw = "Sure!\nWhat is X?\nHow does X compare to Y?\nThanks."
    assert parse_plan(raw) == ["What is X?", "How does X compare to Y?"]


def test_parse_score_variants() -> None:
    """Score parsing is case-insensitive, clamped, and safe on garbage."""
    assert parse_score("SCORE: 7\nFEEDBACK:\n- x") == 7.0
    assert parse_score("score = 9.5") == 9.5
    assert parse_score("Score: 99") == 10.0
    assert parse_score("no score here") is None


def test_parse_feedback_extracts_bullets() -> None:
    """Only bullets after the FEEDBACK marker are captured."""
    raw = "SCORE: 4\nFEEDBACK:\n- fix citations\n* add sections\nignored line"
    assert parse_feedback(raw) == ["fix citations", "add sections"]


def test_offline_critic_separates_good_from_bad_drafts(
    model: OfflineChatModel,
) -> None:
    """A skeletal draft scores far below a complete, structured one."""
    bad = prompts.CRITIC_USER.format(question="Q?", draft="Too short.")
    bad_score = parse_score(model.invoke(prompts.CRITIC_SYSTEM, bad))
    good_draft = (
        "# Report\n\n## Findings\n\n### A?\n\nAnswer [1].\n\n### B?\n\n"
        "Answer [2].\n\n## Limitations and Further Research\n\nScope note.\n\n"
        "## References\n\n[1] T - u\n[2] T2 - u2\n" + "filler " * 60
    )
    good = prompts.CRITIC_USER.format(question="Q?", draft=good_draft)
    good_score = parse_score(model.invoke(prompts.CRITIC_SYSTEM, good))
    assert bad_score is not None and good_score is not None
    assert bad_score < 8.0 <= good_score


def test_offline_model_rejects_unknown_role(model: OfflineChatModel) -> None:
    """An unrecognized system prompt fails loudly, not silently."""
    with pytest.raises(ValueError, match="unrecognized"):
        model.invoke("You are a pirate.", "QUESTION: Arr?")


def test_factory_returns_offline_model(settings: Settings) -> None:
    """The default factory path yields the deterministic offline model."""
    built = create_chat_model(settings)
    assert isinstance(built, OfflineChatModel)
    assert isinstance(built, ChatModel)


def test_factory_rejects_unknown_provider(settings: Settings) -> None:
    """An unknown provider raises a clear error."""
    broken = settings.model_copy(update={"model_provider": "nope"})
    with pytest.raises(ValueError, match="Unknown model provider"):
        create_chat_model(broken)
