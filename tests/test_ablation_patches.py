"""The ablation conditions must actually ablate.

A manipulation that silently fails to manipulate produces a clean null and no
warning. These run offline against the fixture backend, so the claim that
`no_recovery` aborts and `garbled_errors` garbles is checked rather than
asserted in prose.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import agent.workspace as ws_mod
from agent.llm import FixtureBackend
from agent.llm import Usage
from agent.loop import Agent, AgentConfig
from agent.trace import Trace
from agent.workspace import Workspace

_SPEC = importlib.util.spec_from_file_location(
    "ablation", Path(__file__).resolve().parents[1] / "experiments" / "ablation.py")
abl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(abl)

# Turn one omits a required argument; turn two would finish normally.
MALFORMED = [
    {"text": "", "tool_calls": [{"id": "t1", "name": "write_file",
                                 "arguments": {"content": "x = 1\n"}}]},
    {"text": "", "tool_calls": [{"id": "t2", "name": "finish",
                                 "arguments": {"summary": "done"}}]},
]


@pytest.fixture(autouse=True)
def _always_restore():
    yield
    abl.restore()


def _agent(tmp_path, turns):
    p = tmp_path / "fx.json"
    p.write_text(json.dumps({"__sequence__": turns}), encoding="utf-8")
    return Agent(FixtureBackend(p), Workspace(tmp_path / "ws"),
                 config=AgentConfig(max_steps=4))


def test_baseline_recovers_from_a_malformed_call(tmp_path):
    abl.apply_condition("full")
    assert _agent(tmp_path, MALFORMED).run("go").outcome == "finished"


def test_no_recovery_aborts_where_the_old_code_aborted(tmp_path):
    abl.apply_condition("no_recovery")
    agent = _agent(tmp_path, MALFORMED)
    with pytest.raises(abl._AblatedCrash):
        agent.run("go")
    assert agent.last_trace is not None
    assert agent.last_trace.usage.calls == 1, "the partial trajectory was lost"


def test_argument_and_parse_recovery_can_be_ablated_separately(tmp_path):
    import agent.llm as llm_mod

    abl.apply_condition("no_argument_recovery")
    with pytest.raises(abl._AblatedCrash):
        _agent(tmp_path, MALFORMED).run("go")
    assert llm_mod._parse_arguments("{")["__raw__"] == "{"

    abl.restore()
    abl.apply_condition("no_parse_recovery")
    assert _agent(tmp_path, MALFORMED).run("go").outcome == "finished"
    with pytest.raises(json.JSONDecodeError):
        llm_mod._parse_arguments("{")


def test_garbled_errors_really_garbles():
    abl.apply_condition("garbled_errors")
    out = ws_mod._decode("'pytest' 不是内部或外部命令".encode("gbk"))
    assert "\ufffd" in out and "不是" not in out


def test_baseline_decodes_the_same_bytes_correctly():
    abl.apply_condition("full")
    assert "不是内部或外部命令" in ws_mod._decode(
        "'pytest' 不是内部或外部命令".encode("gbk"))


def test_restore_puts_every_patched_symbol_back():
    import agent.llm as llm_mod
    import agent.loop as loop_mod
    abl.apply_condition("no_recovery")
    abl.restore()
    assert ws_mod._decode is abl._REAL_DECODE
    assert loop_mod.check_arguments is abl._REAL_CHECK
    assert llm_mod._parse_arguments is abl._REAL_PARSE


def test_trajectory_features_keep_provider_token_usage():
    trace = Trace(run_id="r", task_id="t", model="fixture", backend="fixture")
    trace.record("model", {"text": "x"}, Usage(120, 30, 1))
    got = abl.trajectory_features(trace)
    assert got["input_tokens"] == 120
    assert got["output_tokens"] == 30
    assert got["total_tokens"] == 150


def test_score_reports_probe_level_fraction(monkeypatch, tmp_path):
    task = abl.load_tasks()[0]
    repo = abl.build_repo(tmp_path, task, "findable")
    # Use minimal stand-ins so the assertion is about score aggregation, not
    # the candidate executor already tested elsewhere.
    class Matrix:
        def __init__(self, tokens):
            self.tokens = [tokens]
            self.invalid = [False]

    matrices = iter([Matrix(["a", "wrong", "c"]), Matrix(["a", "b", "c"])])
    monkeypatch.setattr(abl, "run_candidates", lambda *a, **k: next(matrices))
    got = abl.score(repo, task)
    assert got["correct"] is False
    assert got["behaviour_matches"] == 2
    assert got["behaviour_total"] == 3
    assert got["behaviour_frac"] == pytest.approx(2 / 3)
