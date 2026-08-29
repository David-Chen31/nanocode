"""Search: find the match, skip the junk, and never lie about the cap.

The last one is the reason most of these exist. A capped result that does not
say it was capped is worse than no result at all -- the agent reads "3 matches"
as "3 callers" and edits on that basis.
"""
from __future__ import annotations

import pytest

from agent.search import SearchLimits, find_files, search
from agent.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.write("pkg/core.py", "def total(xs):\n    return sum(xs)\n")
    w.write("pkg/helpers.py", "from .core import total\n\ndef twice(xs):\n    return total(xs) * 2\n")
    w.write("tests/test_core.py", "from pkg.core import total\n\ndef test_total():\n    assert total([1]) == 1\n")
    w.write("README.md", "A package with a total() helper.\n")
    return w


def test_search_finds_uses_across_files(ws):
    out = search(ws, r"\btotal\(")
    assert "pkg/helpers.py:4" in out
    assert "tests/test_core.py:4" in out


def test_search_reports_location_not_content(ws):
    """Search says where to look; read_file is what looks."""
    out = search(ws, "def total")
    assert out.count("\n") < 5           # not the file body
    assert "pkg/core.py:1: def total(xs):" in out


def test_glob_restricts_the_walk(ws):
    assert "README.md" not in search(ws, "total", glob="*.py")
    assert "README.md" in search(ws, "total", glob="*.md")


def test_ignore_case_is_off_by_default(ws):
    assert "no matches" in search(ws, "TOTAL")
    assert "no matches" not in search(ws, "TOTAL", ignore_case=True)


def test_invalid_regex_is_an_error_not_a_crash(ws):
    """The model writes these. A bad one is a normal event."""
    out = search(ws, "def total(")
    assert out.startswith("error: invalid regex")


def test_skipped_directories_are_not_walked(ws):
    ws.write(".git/objects/pack.txt", "total total total\n")
    ws.write("node_modules/dep/index.js", "total()\n")
    out = search(ws, "total")
    assert ".git" not in out and "node_modules" not in out


def test_binary_files_are_skipped(ws):
    (ws.root / "blob.dat").write_bytes(b"total\x00total\n")
    assert "blob.dat" not in search(ws, "total")


def test_per_file_cap_does_not_hide_other_files(ws):
    """One huge file must not consume the whole budget."""
    ws.write("noisy.py", "total\n" * 100)
    out = search(ws, "total", limits=SearchLimits(per_file=3, total=60))
    assert out.count("noisy.py:") == 3
    assert "pkg/helpers.py" in out, "a noisy file starved the rest of the repo"


def test_truncation_is_announced_with_a_count(ws):
    ws.write("noisy.py", "total\n" * 100)
    out = search(ws, "total", limits=SearchLimits(per_file=3, total=60))
    assert "not shown" in out
    assert "Narrow" in out


def test_uncapped_result_makes_no_truncation_claim(ws):
    out = search(ws, "def twice")
    assert "not shown" not in out and "not scanned" not in out


def test_long_lines_are_clipped(ws):
    ws.write("wide.py", "x = '" + "y" * 5_000 + "'  # total\n")
    out = search(ws, "total", glob="wide.py", limits=SearchLimits(line_chars=80))
    assert len(out.splitlines()[-1]) < 200


def test_no_match_says_what_was_searched(ws):
    out = search(ws, "nonexistent_symbol")
    assert "nonexistent_symbol" in out and "no matches" in out


def test_search_cannot_escape_the_workspace(ws):
    assert search(ws, "total", path="../..").startswith("error:")


def test_find_files_matches_name_or_path(ws):
    assert "tests/test_core.py" in find_files(ws, "test_*.py")
    assert "pkg/core.py" in find_files(ws, "pkg/*.py")


def test_find_files_reports_nothing_found(ws):
    assert "no files matching" in find_files(ws, "*.rs")


def test_find_files_cannot_escape_the_workspace(ws):
    assert find_files(ws, "*.py", path="../..").startswith("error:")
