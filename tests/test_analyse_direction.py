"""The sign a table prints has to agree with the header above it.

This exists because it did not. `cluster_boot(a=baseline, b=cond)` returns
mean(cond) - mean(baseline), and the report printed that number under the
header "{baseline} minus {cond}" -- the opposite subtraction. Every row in
every panel was labelled backwards.

Nothing crashed, no number was wrong, and the prose in DECISIONS.md happened
to be written from the computation rather than the header, so it stayed
correct. That is exactly why it survived: an inverted label is invisible
unless you recompute a mean by hand and compare. A reader who trusted the
header would have concluded that removing the search tool *cost* six tool
calls, when removing it saved six.

A wrong number is a bug you find. A wrong label is a bug that finds you, in
front of whoever is reading the table.
"""
from __future__ import annotations

import json

import pytest

from experiments.analyse import (factorial_interaction, fixed_task_contrast,
                                 confirmatory_decision, load, model_calls, report)


@pytest.fixture
def lopsided(tmp_path):
    """`ablated` spends twice the tool calls of `base`, on every task.

    No statistics needed to know the answer: the effect of ablating is +10.
    """
    rows = []
    for task in ("t1", "t2", "t3"):
        for rep in range(3):
            rows.append({"condition": "base", "task": task, "n_tool_calls": 10,
                         "correct": True, "outcome": "finished"})
            rows.append({"condition": "ablated", "task": task, "n_tool_calls": 20,
                         "correct": True, "outcome": "finished"})
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return p


def _panel(capsys, path) -> str:
    rows, key = load(str(path))
    report(rows, key, "base")
    return capsys.readouterr().out


def test_the_header_names_the_subtraction_that_was_actually_done(capsys, lopsided):
    out = _panel(capsys, lopsided)
    assert "ablated minus base" in out
    assert "base minus ablated" not in out, "the header is inverted again"


def test_a_condition_that_spends_more_reports_a_positive_effect(capsys, lopsided):
    """The load-bearing assertion: sign and header agree.

    +10 under "ablated minus base" is the truth. -10 under the same header, or
    +10 under "base minus ablated", are both the bug.
    """
    line = next(ln for ln in _panel(capsys, lopsided).splitlines()
                if "tool calls" in ln)
    assert "+10.00" in line, line


def test_an_interval_that_excludes_zero_is_marked(capsys, lopsided):
    """Every run differs by exactly 10, so no resample can straddle zero."""
    line = next(ln for ln in _panel(capsys, lopsided).splitlines()
                if "tool calls" in ln)
    assert "excludes 0" in line


def test_a_recorded_all_zero_metric_is_not_mistaken_for_missing(capsys, tmp_path):
    rows = []
    for condition in ("base", "ablated"):
        rows.append({"condition": condition, "task": "t1", "rep": 0,
                     "correct": False, "outcome": "finished"})
    p = tmp_path / "zeros.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert "correct" in _panel(capsys, p)


def test_old_architecture_rows_recover_the_omitted_planner_call():
    old = {"arm": "plan_execute", "n_model_calls": 22}
    new = {"arm": "plan_execute", "n_model_calls": 23, "n_planning_calls": 1}
    react = {"arm": "react", "n_model_calls": 20}

    assert model_calls(old) == 23
    assert model_calls(new) == 23
    assert model_calls(react) == 20


def test_factorial_interaction_is_a_difference_in_differences():
    rows = []
    values = {"base": 10, "a": 8, "b": 9, "both": 4}
    for task in ("t1", "t2", "t3"):
        for condition, value in values.items():
            rows.append({"task": task, "condition": condition, "value": value})

    got = factorial_interaction(rows, "condition", "base", "a", "b", "both",
                                lambda r: r["value"], n=200)
    # 4 - 8 - 9 + 10 = -3 on every task; every bootstrap draw is the same.
    assert got == (-3, -3, -3)


def test_fixed_task_inference_resamples_reps_without_resampling_tasks():
    rows = []
    for task, base in (("easy", 10), ("hard", 20)):
        for rep, jitter in enumerate((-1, 0, 1)):
            rows.append({"task": task, "rep": rep, "condition": "a",
                         "value": base + jitter, "outcome": "finished"})
            rows.append({"task": task, "rep": rep, "condition": "b",
                         "value": base + jitter + 2, "outcome": "finished"})

    got = fixed_task_contrast(rows, "condition", {"b": 1, "a": -1},
                              lambda r: r["value"], n=200)
    assert got == (2, 2, 2)


def _decision_document(*, treatment_correct=True, dirty=False, error=False):
    rows = []
    for task in ("t1", "t2", "t3"):
        for rep in range(4):
            rows.append({"task": task, "rep": rep, "condition": "full",
                         "correct": True, "behaviour_frac": 1.0,
                         "input_tokens": 100, "output_tokens": 20,
                         "outcome": "finished"})
            rows.append({"task": task, "rep": rep, "condition": "no_search",
                         "correct": treatment_correct,
                         "behaviour_frac": 1.0 if treatment_correct else 0.0,
                         "input_tokens": 70, "output_tokens": 10,
                         "outcome": "error" if error else "finished"})
    return {"manifest": {"git_dirty": dirty}, "rows": rows}


def test_confirmatory_decision_passes_only_quality_and_cost_together():
    got = confirmatory_decision(_decision_document(), baseline="full",
                                treatment="no_search", boots=100)
    assert got["status"] == "PASS"
    assert got["contrasts"]["correct"]["estimate"] == 0
    assert got["contrasts"]["tokens"]["ci95"][1] < 0


def test_confirmatory_decision_fails_on_quality_loss():
    got = confirmatory_decision(_decision_document(treatment_correct=False),
                                baseline="full", treatment="no_search", boots=100)
    assert got["status"] == "FAIL"
    assert "quality non-inferiority" in got["reasons"][0]


@pytest.mark.parametrize("kwargs", [{"dirty": True}, {"error": True}])
def test_confirmatory_decision_is_invalid_when_provenance_or_infra_fails(kwargs):
    got = confirmatory_decision(_decision_document(**kwargs), baseline="full",
                                treatment="no_search", boots=100)
    assert got["status"] == "INVALID"


def test_confirmatory_decision_rejects_a_run_from_the_wrong_frozen_model():
    doc = _decision_document()
    doc.update(model="snapshot-a", reps=4)
    doc["manifest"]["args"] = {
        "schedule_seed": 7, "max_steps": 18, "max_tokens": 4096,
        "temperature": 1.0, "require_clean": True,
        "require_model_snapshot": True, "fail_if_output_exists": True,
        "unambiguous": False,
    }
    got = confirmatory_decision(
        doc, baseline="full", treatment="no_search", boots=100,
        expected_model="snapshot-b", expected_reps=4,
        expected_tasks=["t1", "t2", "t3"], expected_schedule_seed=7,
        expected_max_steps=18, expected_max_tokens=4096,
        expected_temperature=1.0)
    assert got["status"] == "INVALID"
    assert "model does not match" in got["reasons"][0]
