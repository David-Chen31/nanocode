"""What happens when the model gets a tool call wrong.

These are not edge cases. Truncated JSON, invented argument names and invented
tool names all show up in real traces, and every one of them is recoverable --
provided the agent survives it and the message says what to do instead.
"""
from __future__ import annotations

from agent.llm import RAW_ARGS, _parse_arguments
from agent.tools import build_toolset, check_arguments
from agent.workspace import Workspace


def _tools(tmp_path):
    return build_toolset(Workspace(tmp_path))


def test_valid_arguments_pass(tmp_path):
    t = _tools(tmp_path)["write_file"]
    assert check_arguments(t, {"path": "a.py", "content": "x"}) == ""


def test_missing_required_argument_names_it(tmp_path):
    t = _tools(tmp_path)["write_file"]
    msg = check_arguments(t, {"content": "x"})
    assert "path" in msg and "missing" in msg
    assert "<locals>" not in msg, "the Python closure leaked into the model's view"


def test_the_message_lists_what_the_tool_accepts(tmp_path):
    t = _tools(tmp_path)["edit_file"]
    msg = check_arguments(t, {})
    for name in ("path", "old", "new"):
        assert name in msg


def test_invented_argument_is_reported(tmp_path):
    t = _tools(tmp_path)["read_file"]
    msg = check_arguments(t, {"path": "a.py", "encoding": "utf-8"})
    assert "unexpected" in msg and "encoding" in msg


def test_optional_arguments_may_be_omitted(tmp_path):
    t = _tools(tmp_path)["search"]
    assert check_arguments(t, {"pattern": "x"}) == ""


# -- argument parsing -------------------------------------------------------

def test_well_formed_json_parses():
    assert _parse_arguments('{"path": "a.py"}') == {"path": "a.py"}


def test_empty_arguments_mean_no_arguments():
    assert _parse_arguments("") == {} and _parse_arguments(None) == {}


def test_truncated_json_does_not_raise():
    """A write_file cut off by the token limit must not abort the run."""
    out = _parse_arguments('{"path": "a.py", "content": "def f():\n    ret')
    assert RAW_ARGS in out


def test_a_json_scalar_where_an_object_belongs_is_caught():
    assert RAW_ARGS in _parse_arguments('"just a string"')


def test_unparseable_arguments_produce_actionable_advice(tmp_path):
    t = _tools(tmp_path)["write_file"]
    msg = check_arguments(t, _parse_arguments('{"content": "trunca'))
    assert "not valid JSON" in msg
    assert "cut off" in msg          # names the likely cause
    assert "smaller" in msg          # and what to do about it
