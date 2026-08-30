"""The alternative architecture must actually be the alternative, and must be fair.

A comparison against a straw man proves nothing, and a straw man is easy to
build by accident -- give the alternative one fewer turn, or hide the repository
layout from its planner, and it loses for reasons that have nothing to do with
control flow. These pin the fairness properties down.
"""
from __future__ import annotations

import json

from agent.llm import FixtureBackend
from agent.loop import AgentConfig
from agent.plan_execute import PLANNER_SYSTEM, PlanExecuteAgent
from agent.workspace import Workspace


def _fixture(tmp_path, turns):
    p = tmp_path / "fx.json"
    p.write_text(json.dumps({"__sequence__": turns}), encoding="utf-8")
    return FixtureBackend(p)


def _call(name, **args):
    return {"id": "t1", "name": name, "arguments": args}


PLAN_THEN_WORK = [
    {"text": "1. Read sol.py\n2. Add the function\n3. Run the tests", "tool_calls": []},
    {"text": "", "tool_calls": [_call("write_file", path="sol.py", content="x = 1\n")]},
    {"text": "", "tool_calls": [_call("finish", summary="done")]},
]


def test_the_plan_is_produced_before_any_tool_runs(tmp_path):
    agent = PlanExecuteAgent(_fixture(tmp_path, PLAN_THEN_WORK),
                             Workspace(tmp_path / "ws"),
                             config=AgentConfig(max_steps=6))
    out = agent.run("add x", task_id="t")
    assert "1." in out.plan and "Read sol.py" in out.plan
    assert out.result.outcome == "finished"


def test_the_plan_reaches_the_executor(tmp_path):
    """Otherwise this is ReAct that wasted a call on a plan nobody read."""

    class Capturing:
        """Records the system prompt of every call, and delegates the rest."""

        def __init__(self, inner):
            self.inner = inner
            self.systems: list[str] = []
            self.name, self.model = inner.name, inner.model

        def complete(self, messages, *, system=None, **kw):
            self.systems.append(system or "")
            return self.inner.complete(messages, system=system, **kw)

    backend = Capturing(_fixture(tmp_path, PLAN_THEN_WORK))
    agent = PlanExecuteAgent(backend, Workspace(tmp_path / "ws"),
                             config=AgentConfig(max_steps=6))
    agent.run("add x", task_id="t")

    assert "numbered plan" in backend.systems[0], "the first call was not the planner"
    assert "Read sol.py" in backend.systems[1], "the executor never saw the plan"


def test_planning_costs_a_step_from_the_same_budget(tmp_path):
    """Both arms must be allowed at most max_steps model calls."""
    agent = PlanExecuteAgent(_fixture(tmp_path, [{"text": "1. do it", "tool_calls": []}]
                                      + [{"text": "", "tool_calls": [_call("list_files")]}] * 10),
                             Workspace(tmp_path / "ws"),
                             config=AgentConfig(max_steps=4))
    out = agent.run("go", task_id="t")
    model_calls = sum(1 for s in out.result.trace.steps if s.kind == "model")
    assert model_calls == 3, "the executor did not give back the planning turn"


def test_the_planning_call_is_counted_in_cost(tmp_path):
    agent = PlanExecuteAgent(
        _fixture(tmp_path, [{"text": "1. plan", "tool_calls": [],
                             "input_tokens": 500, "output_tokens": 50}]
                 + PLAN_THEN_WORK[1:]),
        Workspace(tmp_path / "ws"), config=AgentConfig(max_steps=6))
    out = agent.run("go", task_id="t")
    assert out.result.trace.usage.input_tokens >= 500, \
        "the planning call was left out of the accounting"


def test_the_planner_is_told_the_repository_layout(tmp_path):
    """A planner that cannot see the tree is planning blind -- an unfair arm."""
    ws = Workspace(tmp_path / "ws")
    ws.write("pkg/core.py", "x = 1\n")
    agent = PlanExecuteAgent(_fixture(tmp_path, PLAN_THEN_WORK), ws,
                             config=AgentConfig(max_steps=6))
    # plan_for builds its system prompt from PLANNER_SYSTEM + orientation
    from agent.loop import orientation
    assert "pkg/" in orientation(ws)
    assert "numbered plan" in PLANNER_SYSTEM


def test_the_plan_is_declared_revisable(tmp_path):
    """Blind execution of a pre-read plan would be the straw man."""
    from agent.plan_execute import PLAN_PREFACE
    text = PLAN_PREFACE.format(plan="x")
    assert "guidance rather than a script" in text
    assert "contradicts" in text
