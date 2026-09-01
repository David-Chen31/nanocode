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

from experiments.analyse import load, report


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
