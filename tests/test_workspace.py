"""Workspace containment, and the encoding trap under `run`.

The decoding tests are here because the bug they pin was expensive and silent.
An agent whose error messages arrive as replacement characters does not fail --
it retries, and keeps retrying, until the step budget is gone.
"""
from __future__ import annotations

import sys

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


def test_a_runaway_print_loop_is_killed_not_absorbed(tmp_path):
    """The failure that killed a 432-run study, reproduced small.

    Unbounded capture reads until memory runs out; the 30 second timeout never
    gets a chance to fire because the process is doing exactly what it was
    asked to. The volume bound is what stops it.
    """
    ws = Workspace(tmp_path)
    ws.write("flood.py", "while True:\n    print('x' * 4096)\n")
    res = ws.python("flood.py", timeout=30.0, max_output_bytes=200_000)

    assert res.returncode == -1
    assert not res.timed_out, "it was stopped on volume, not on the clock"
    assert "looping" in res.stderr        # tells the agent what is wrong
    assert len(res.stdout) < 1_000_000    # and did not absorb it all


def test_the_volume_kill_is_fast(tmp_path):
    """It must not wait out the timeout, or a study of many runs never ends."""
    import time as _t
    ws = Workspace(tmp_path)
    ws.write("flood.py", "while True:\n    print('x' * 4096)\n")
    t0 = _t.monotonic()
    ws.python("flood.py", timeout=30.0, max_output_bytes=200_000)
    assert _t.monotonic() - t0 < 10.0


def test_normal_output_is_untouched_by_the_cap(tmp_path):
    ws = Workspace(tmp_path)
    ws.write("chatty.py", "for i in range(50):\n    print('line', i)\n")
    res = ws.python("chatty.py")
    assert "line 0" in res.stdout and "line 49" in res.stdout
    assert "omitted" not in res.stdout and "looping" not in res.stderr
    assert res.returncode == 0


def test_the_cli_console_survives_characters_it_cannot_encode():
    """A check mark in the summary must not turn a finished run into a traceback."""
    import io
    from agent.cli import _survivable_console

    real_out, real_err = sys.stdout, sys.stderr
    fake = io.TextIOWrapper(io.BytesIO(), encoding="cp936")
    try:
        sys.stdout = sys.stderr = fake
        _survivable_console()
        print("summary: 完成 \u2713 \U0001f389")     # what models actually emit
        fake.flush()
    finally:
        sys.stdout, sys.stderr = real_out, real_err

    written = fake.buffer.getvalue().decode("cp936", errors="replace")
    assert "完成" in written, "the Chinese was mangled instead of preserved"
    assert "?" in written, "the unencodable characters were not replaced"
