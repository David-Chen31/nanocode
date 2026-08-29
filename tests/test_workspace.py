"""Workspace containment, and the encoding trap under `run`.

The decoding tests are here because the bug they pin was expensive and silent.
An agent whose error messages arrive as replacement characters does not fail --
it retries, and keeps retrying, until the step budget is gone.
"""
from __future__ import annotations

import pytest

from agent.workspace import PathEscape, Workspace, _decode


def test_utf8_output_is_decoded(tmp_path):
    assert _decode("结果: ok".encode("utf-8")) == "结果: ok"


def test_oem_codepage_output_falls_back_instead_of_mangling():
    """What cmd.exe writes when a command does not exist, on a CP936 box.

    Strict UTF-8 cannot decode these bytes. The point of the fallback is that
    the agent gets a sentence it can act on rather than a row of U+FFFD.
    """
    raw = "'pytest' 不是内部或外部命令".encode("gbk")
    out = _decode(raw)
    assert "pytest" in out
    assert "�" not in out, "the message was mangled instead of decoded"


def test_undecodable_bytes_still_return_something(tmp_path):
    out = _decode(b"\xff\xfe\x00 partial ascii \xff")
    assert "partial ascii" in out          # never raises, never loses the ASCII


def test_empty_stream_is_empty_string():
    assert _decode(b"") == ""


def test_a_missing_command_reports_a_readable_error(tmp_path):
    """End to end: the agent must be able to tell 'not found' from 'failed'."""
    res = Workspace(tmp_path).run("nosuchcommand_xyzzy")
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "nosuchcommand_xyzzy" in combined
    assert "�" not in combined


def test_run_captures_child_stdout(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("hello.py", "print('hi from the child')\n")
    assert "hi from the child" in ws.python("hello.py").stdout


def test_timeout_keeps_what_was_printed_first(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("slow.py", "import sys, time\nprint('before the hang', flush=True)\ntime.sleep(30)\n")
    res = ws.python("slow.py", timeout=2.0)
    assert res.timed_out and res.returncode == -1
    assert "before the hang" in res.stdout, "the reason it hung was discarded"


@pytest.mark.parametrize("bad", ["../outside.txt", "a/../../outside.txt"])
def test_paths_cannot_escape(tmp_path, bad):
    with pytest.raises(PathEscape):
        Workspace(tmp_path).resolve(bad)


def test_render_states_the_exit_code_even_when_silent(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("quiet.py", "raise SystemExit(3)\n")
    assert "exit code: 3" in ws.python("quiet.py").render()
