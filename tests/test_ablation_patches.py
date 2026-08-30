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
from agent.loop import Agent, AgentConfig
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
    with pytest.raises(abl._AblatedCrash):
        _agent(tmp_path, MALFORMED).run("go")


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
