"""The step budget the agent is allowed to see.

An agent that cannot see the limit it is judged against cannot manage it. But
a warning shown too early is worse than none: it buys a better termination rate
by making the agent stop before the work is done.
"""
from __future__ import annotations

from agent.loop import budget_note


def test_early_steps_get_the_count_but_no_pressure():
    note = budget_note(0, 24)
    assert "Step 1 of 24" in note
    assert "low" not in note and "Finish now" not in note


def test_the_agent_is_warned_before_it_is_cut_off():
    assert "Finish now" in budget_note(22, 24)


def test_the_warning_asks_for_an_honest_summary_not_just_a_stop():
    # Otherwise the fix trades a real failure for a false success.
    note = budget_note(23, 24)
    assert "what is not" in note


def test_pressure_arrives_gradually_not_all_at_once():
    stages = [budget_note(i, 24) for i in (0, 8, 17, 23)]
    assert "low" not in stages[0] and "low" not in stages[1]
    assert "low" in stages[2]
    assert "Finish now" in stages[3]


def test_a_tiny_budget_still_warns_rather_than_dividing_to_zero():
    assert "Finish now" in budget_note(0, 2)
