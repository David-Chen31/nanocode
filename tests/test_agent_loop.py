"""End-to-end checks for the agent loop, driven by scripted fixtures."""
from __future__ import annotations

import json

from agent.llm import FixtureBackend
from agent.loop import Agent, AgentConfig
from agent.workspace import Workspace


def _fixture(tmp_path, turns):
    p = tmp_path / "fx.json"
    p.write_text(json.dumps({"__sequence__": turns}), encoding="utf-8")
    return FixtureBackend(p)


def _call(name, **args):
    return {"id": "t1", "name": name, "arguments": args}


def test_agent_writes_file_and_finishes(tmp_path):
    backend = _fixture(tmp_path, [
        {"text": "Creating the file.", "tool_calls": [
            _call("write_file", path="sol.py", content="def add(a, b):\n    return a + b\n")],
         "input_tokens": 100, "output_tokens": 30},
        {"text": "Checking it runs.", "tool_calls": [
            _call("run", command='python -c "import sol; print(sol.add(2,3))"')],
         "input_tokens": 140, "output_tokens": 20},
        {"text": "", "tool_calls": [_call("finish", summary="added add()")],
         "input_tokens": 160, "output_tokens": 10},
    ])
    ws = Workspace(tmp_path / "ws")
    res = Agent(backend, ws, config=AgentConfig(max_steps=6)).run("write add()", task_id="t")

    assert res.outcome == "finished"
    assert res.summary == "added add()"
    assert (ws.root / "sol.py").exists()
    assert res.trace.usage.input_tokens == 400
    assert res.trace.usage.calls == 3
    # unknown model falls back to the default price row: 400*1.0 + 60*5.0 per 1M
    assert res.trace.cost_usd == 0.0007
    kinds = [s.kind for s in res.trace.steps]
    assert kinds.count("model") == 3 and "end" in kinds


def test_ask_user_is_routed_to_the_responder(tmp_path):
    backend = _fixture(tmp_path, [
        {"text": "", "tool_calls": [
            _call("ask_user", question="Return [] or raise on empty input?",
                  options=["return []", "raise ValueError"])]},
        {"text": "", "tool_calls": [_call("finish", summary="done")]},
    ])
    answers = []

    def responder(question, options):
        answers.append(question)
        return options[0]

    ws = Workspace(tmp_path / "ws2")
    res = Agent(backend, ws, config=AgentConfig(max_steps=6), responder=responder).run("task")

    assert len(answers) == 1
    assert res.trace.n_asks == 1
    assert res.asked[0]["answer"] == "return []"


def test_ask_budget_is_enforced(tmp_path):
    turns = [{"text": "", "tool_calls": [_call("ask_user", question=f"q{i}")]} for i in range(4)]
    turns.append({"text": "", "tool_calls": [_call("finish", summary="done")]})
    backend = _fixture(tmp_path, turns)
    ws = Workspace(tmp_path / "ws3")

    res = Agent(backend, ws, config=AgentConfig(max_steps=8, max_asks=2),
                responder=lambda q, o: "yes").run("task")

    assert res.trace.n_asks == 2
    refused = [s for s in res.trace.steps if s.kind == "ask" and s.payload.get("refused")]
    assert len(refused) == 2


def test_no_ask_removes_the_tool(tmp_path):
    ws = Workspace(tmp_path / "ws4")
    backend = _fixture(tmp_path, [{"text": "", "tool_calls": [_call("finish", summary="x")]}])
    agent = Agent(backend, ws, config=AgentConfig(allow_ask=False))
    assert "ask_user" not in agent.tools


def test_edit_file_rejects_ambiguous_match(tmp_path):
    ws = Workspace(tmp_path / "ws5")
    ws.write("a.py", "x = 1\nx = 1\n")
    from agent.tools import build_toolset
    tools = build_toolset(ws)
    out = tools["edit_file"].fn(path="a.py", old="x = 1", new="x = 2")
    assert "not unique" in out


def test_a_bad_call_is_survivable_and_the_agent_recovers(tmp_path):
    """The whole point of validating arguments: the run continues.

    Turn one omits `path`, turn two invents a tool. Neither may end the run --
    the model is told what was wrong and gets to try again, which is what turn
    three does successfully.
    """
    backend = _fixture(tmp_path, [
        {"text": "", "tool_calls": [_call("write_file", content="x = 1\n")]},
        {"text": "", "tool_calls": [_call("grep", pattern="x")]},
        {"text": "", "tool_calls": [_call("write_file", path="sol.py", content="x = 1\n")]},
        {"text": "", "tool_calls": [_call("finish", summary="done")]},
    ])
    ws = Workspace(tmp_path / "ws")
    res = Agent(backend, ws, config=AgentConfig(max_steps=8)).run("write it", task_id="t")

    assert res.outcome == "finished"
    assert (ws.root / "sol.py").read_text(encoding="utf-8") == "x = 1\n"

    errors = [s.payload["error"] for s in res.trace.steps
              if s.kind == "tool" and "error" in s.payload]
    assert len(errors) == 2
    assert "path" in errors[0]
    # The unknown-tool error must name the real tools, or the model cannot
    # correct itself except by guessing again.
    assert "search" in errors[1] and "no such tool" in errors[1]


def test_every_tool_call_still_gets_a_result_after_an_error(tmp_path):
    """An errored call is answered like any other, or the next request is malformed."""
    backend = _fixture(tmp_path, [
        {"text": "", "tool_calls": [_call("read_file")]},
        {"text": "", "tool_calls": [_call("finish", summary="done")]},
    ])
    ws = Workspace(tmp_path / "ws")
    res = Agent(backend, ws, config=AgentConfig(max_steps=4)).run("read it", task_id="t")
    assert res.outcome == "finished"
