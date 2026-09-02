"""Real tasks on a real repository, scored by tests the agent never sees.

The twelve diagnostic tasks in bench/tasks are single functions in a synthetic
package. That gap turned out to matter: gpt-4o-mini scores 0.42 there and 0/4
on real repository tasks, and two agent bugs -- a preamble read as a conclusion,
and no sense of the working directory -- never showed up on the scaffold at all.
So the architecture question gets asked on real code instead.

The repository is nanocode itself: multi-file, with a test suite, and every task
below is a change someone might actually ask for.

WHY THE VERIFIER IS HIDDEN

Each task ships a pytest file written into the workspace only *after* the agent
has stopped. If it were present during the run the agent could read the
assertions and satisfy them literally, which measures reading comprehension
rather than whether the change works. Scoring is therefore two conditions:

    regression   the repository's own suite still passes
    behaviour    the hidden verifier passes

Both are required. Either alone is gameable: an agent that changes nothing
passes the first, and an agent that breaks the build to satisfy the second
should not be credited for it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Not copied into the agent's workspace: version control, caches, the research
# corpus and its results. A smaller copy keeps each run fast and keeps the
# agent's orientation listing readable.
EXCLUDE = {".git", "results", "__pycache__", ".pytest_cache", "docs",
           "experiments", "askoract", ".venv", "venv", "demo_ws"}

# These two test modules import the research packages that EXCLUDE drops, so
# they would error on import and make the staged repository's own suite red
# before the agent had touched anything. A red baseline would poison the
# regression half of the score.
EXCLUDE_FILES = {
    # These import packages deliberately omitted from the compact demo copy.
    # Keep this list explicit: the recording baseline must be green, but the
    # agent should still see every test for the core coding-agent package.
    "tests/test_askoract.py",
    "tests/test_ablation_patches.py",
    "tests/test_analyse_direction.py",
    "tests/test_open_source_adapter.py",
    "tests/test_open_source_container.py",
    "tests/test_open_source_data.py",
    "tests/test_power.py",
    "tests/test_provenance.py",
}


@dataclass
class RepoTask:
    id: str
    prompt: str
    verifier: str                      # pytest source, written in after the run
    touches: tuple[str, ...] = ()      # files a correct answer must change


def stage(root: Path) -> None:
    """Copy the repository into a fresh workspace.

    The destination is skipped explicitly. Staging into a directory inside the
    repository would otherwise walk into the copy being written and recurse
    until the path length blows up -- which it did, with an unhelpful WinError 3.
    """
    root = Path(root).resolve()
    for src in REPO_ROOT.rglob("*"):
        if root == src or root in src.parents:
            continue
        rel = src.relative_to(REPO_ROOT)
        if any(p in EXCLUDE for p in rel.parts) or not src.is_file():
            continue
        if rel.as_posix() in EXCLUDE_FILES:
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


TASKS: list[RepoTask] = [
    RepoTask(
        id="r01_search_max_results",
        prompt="The `search` tool in the agent's toolset gives the caller no way to "
               "cap how many matches come back. Add an optional integer "
               "`max_results` argument to the search tool that limits the total "
               "number of matches returned, and expose it in the tool's JSON schema.",
        touches=("agent/search.py", "agent/tools.py"),
        verifier='''
import tempfile
from agent.tools import build_toolset
from agent.workspace import Workspace


def test_max_results_caps_the_matches():
    ws = Workspace(tempfile.mkdtemp())
    for i in range(12):
        ws.write("m%d.py" % i, "needle here\\n")
    out = build_toolset(ws)["search"].fn(pattern="needle", max_results=3)
    hits = [l for l in out.splitlines() if ":1:" in l]
    assert len(hits) == 3


def test_max_results_is_in_the_schema():
    ws = Workspace(tempfile.mkdtemp())
    assert "max_results" in build_toolset(ws)["search"].parameters["properties"]
'''),
    RepoTask(
        id="r02_read_file_range",
        prompt="The `read_file` tool always returns the whole file. Add optional "
               "integer `start` and `end` arguments (1-based, inclusive) that return "
               "only that range of lines, keeping the existing line numbering. "
               "Expose both in the tool's JSON schema.",
        touches=("agent/tools.py",),
        verifier='''
import tempfile
from agent.tools import build_toolset
from agent.workspace import Workspace


def _tools():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("f.py", "".join("line%d\\n" % i for i in range(1, 11)))
    return build_toolset(ws)


def test_a_range_returns_only_those_lines():
    out = _tools()["read_file"].fn(path="f.py", start=3, end=5)
    assert "line3" in out and "line5" in out
    assert "line2" not in out and "line6" not in out


def test_line_numbers_are_preserved():
    out = _tools()["read_file"].fn(path="f.py", start=4, end=4)
    assert "4" in out.split("line4")[0]


def test_no_range_still_returns_everything():
    out = _tools()["read_file"].fn(path="f.py")
    assert "line1" in out and "line10" in out
'''),
    RepoTask(
        id="r03_run_duration",
        prompt="RunResult in agent/workspace.py does not record how long a command "
               "took. Add a `duration` field (seconds, a float) that Workspace.run "
               "fills in, and include it in RunResult.render() output.",
        touches=("agent/workspace.py",),
        verifier='''
import tempfile
from agent.workspace import Workspace


def test_duration_is_recorded_and_plausible():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("s.py", "import time; time.sleep(0.3)\\n")
    res = ws.python("s.py")
    assert isinstance(res.duration, float)
    assert 0.2 < res.duration < 25.0


def test_duration_appears_in_render():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("q.py", "print('x')\\n")
    assert "duration" in ws.python("q.py").render().lower()
'''),
    RepoTask(
        id="r04_clip_line_boundary",
        prompt="clip_tool_output in agent/context.py cuts text at an exact character "
               "count, which can slice a line in half. Change it so the head and the "
               "tail are cut at line boundaries instead, while still respecting the "
               "character budget as an upper bound.",
        touches=("agent/context.py",),
        verifier='''
from agent.context import ContextPolicy, clip_tool_output


def test_clipping_does_not_split_a_line():
    text = "".join(("x" * 40) + " line %d\\n" % i for i in range(200))
    out, clipped = clip_tool_output(text, ContextPolicy(tool_output_chars=800))
    assert clipped
    originals = set(text.splitlines())
    kept = [l for l in out.splitlines() if l.strip() and "omitted" not in l]
    assert all(l in originals for l in kept), "a line was cut in half"


def test_the_budget_is_still_respected():
    text = "".join("line %d\\n" % i for i in range(5000))
    out, _ = clip_tool_output(text, ContextPolicy(tool_output_chars=1000))
    assert len(out) <= 1400


def test_short_output_is_untouched():
    got = clip_tool_output("a\\nb\\n", ContextPolicy(tool_output_chars=500))
    assert got == ("a\\nb\\n", False)
'''),
    RepoTask(
        id="r05_find_files_exclude",
        prompt="The `find_files` tool can only include paths by glob. Add an optional "
               "`exclude` glob argument that drops matching paths from the result, "
               "and expose it in the tool's JSON schema.",
        touches=("agent/search.py", "agent/tools.py"),
        verifier='''
import tempfile
from agent.tools import build_toolset
from agent.workspace import Workspace


def _tools():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("pkg/core.py", "\\n")
    ws.write("pkg/core_test.py", "\\n")
    ws.write("pkg/util.py", "\\n")
    return build_toolset(ws)


def test_exclude_removes_matching_paths():
    out = _tools()["find_files"].fn(glob="*.py", exclude="*_test.py")
    assert "core.py" in out and "util.py" in out
    assert "core_test.py" not in out


def test_exclude_is_optional_and_in_the_schema():
    tools = _tools()
    assert "exclude" in tools["find_files"].parameters["properties"]
    assert "core_test.py" in tools["find_files"].fn(glob="*.py")
'''),
    RepoTask(
        id="r06_cli_json_flag",
        prompt="Add a --json flag to the agent CLI (agent/cli.py). When passed, the "
               "CLI prints a single JSON object to stdout with keys outcome, summary, "
               "steps and cost_usd, and prints nothing else to stdout.",
        touches=("agent/cli.py",),
        verifier='''
import json, subprocess, sys
from pathlib import Path


def test_json_flag_emits_one_object(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    p = subprocess.run([sys.executable, "-m", "agent.cli", "say hi",
                        "--workspace", str(ws), "--json"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(Path.cwd()))
    payload = json.loads(p.stdout.strip())
    for key in ("outcome", "summary", "steps", "cost_usd"):
        assert key in payload
'''),
    RepoTask(
        id="r07_token_estimate_cache",
        prompt="Conversation.token_estimate in agent/context.py recomputes the "
               "estimate for every message on every call, repeated on every turn. "
               "Add a cache so repeated calls over an unchanged message list do not "
               "recompute, and make sure it is invalidated when a message is added "
               "or evicted.",
        touches=("agent/context.py",),
        verifier='''
import agent.context as ctx

ARGS = chr(123) + chr(34) + "path" + chr(34) + ": " + chr(34) + "m.py" + chr(34) + chr(125)


def test_a_second_call_does_not_recompute():
    """What "cached" means, measured rather than assumed."""
    c = ctx.Conversation(policy=ctx.ContextPolicy())
    for i in range(6):
        c.add({"role": "user", "content": "message number %d" % i})
    seen = []
    real = ctx.estimate_tokens

    def counting(text):
        seen.append(1)
        return real(text)

    ctx.estimate_tokens = counting
    try:
        c.token_estimate()
        first = len(seen)
        c.token_estimate()
        second = len(seen)
    finally:
        ctx.estimate_tokens = real
    assert second == first, "the second call recomputed every message"


def test_repeated_calls_agree():
    c = ctx.Conversation(policy=ctx.ContextPolicy())
    c.add({"role": "user", "content": "hello there"})
    assert c.token_estimate() == c.token_estimate()


def test_adding_a_message_changes_the_estimate():
    c = ctx.Conversation(policy=ctx.ContextPolicy())
    c.add({"role": "user", "content": "hello"})
    first = c.token_estimate()
    c.add({"role": "user", "content": "a much longer second message " * 20})
    assert c.token_estimate() > first, "the cache was not invalidated"


def test_eviction_changes_the_estimate():
    c = ctx.Conversation(policy=ctx.ContextPolicy(max_tokens=200, keep_recent=1))
    c.add({"role": "user", "content": "task"})
    for i in range(8):
        c.add({"role": "assistant", "content": None,
               "tool_calls": [{"id": "c%d" % i, "type": "function",
                               "function": {"name": "read_file", "arguments": ARGS}}]})
        c.note_call("c%d" % i, "read_file", {"path": "m.py"})
        c.add({"role": "tool", "tool_call_id": "c%d" % i, "content": "payload " * 300})
    before = c.token_estimate()
    c.compact()
    assert c.token_estimate() < before, "the cache survived an eviction"
'''),
    RepoTask(
        id="r08_workspace_move",
        prompt="Workspace in agent/workspace.py can read, write and list, but cannot "
               "rename a file. Add a `move(src, dst)` method that renames within the "
               "workspace, raises PathEscape if either path escapes the root, creates "
               "the destination's parent directory when needed, and raises "
               "FileNotFoundError when the source does not exist.",
        touches=("agent/workspace.py",),
        verifier='''
import tempfile
import pytest
from agent.workspace import PathEscape, Workspace


def test_move_renames():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("a.py", "body\\n")
    ws.move("a.py", "sub/b.py")
    assert ws.read("sub/b.py") == "body\\n"
    with pytest.raises(FileNotFoundError):
        ws.read("a.py")


def test_move_refuses_to_escape():
    ws = Workspace(tempfile.mkdtemp())
    ws.write("a.py", "x\\n")
    with pytest.raises(PathEscape):
        ws.move("a.py", "../escaped.py")


def test_missing_source_raises():
    ws = Workspace(tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError):
        ws.move("nope.py", "b.py")
'''),
]
