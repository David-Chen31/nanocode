"""Tool definitions for the coding agent.

`ask_user` is deliberately a normal tool. The whole research question is when the
agent should reach for it, so the gating policy lives outside the tool (see
askoract/voi.py) and the tool itself stays dumb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .llm import RAW_ARGS
from .search import find_files as _find_files
from .search import search as _search
from .workspace import PathEscape, Workspace


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


def check_arguments(tool: Tool, args: dict[str, Any]) -> str:
    """Validate a call against the tool's schema. Returns "" when it is fine.

    This runs before dispatch so that a bad call produces a message written in
    terms the model can act on -- the names the schema declares -- instead of a
    TypeError naming a closure it has never seen. The three failures below are
    the ones that actually occur:

      unparseable   the arguments were not JSON, usually truncated output
      missing       a required field was omitted
      unexpected    an invented field, often a plausible synonym

    Types are deliberately not checked. Coercing "3" to 3 is the sort of
    helpfulness that hides a real disagreement about the interface, and the
    tools already handle their own inputs.
    """
    if RAW_ARGS in args:
        return ("your tool arguments were not valid JSON, so the call could not be "
                "made. This usually means the arguments were cut off. Re-send the "
                "call, splitting a long `content` across several smaller edits if "
                "needed.")
    schema = tool.parameters
    allowed = set(schema.get("properties", {}))
    missing = [k for k in schema.get("required", []) if k not in args]
    unexpected = [k for k in args if k not in allowed]
    parts = []
    if missing:
        parts.append("missing required argument(s): " + ", ".join(sorted(missing)))
    if unexpected:
        parts.append("unexpected argument(s): " + ", ".join(sorted(unexpected)))
    if not parts:
        return ""
    return (f"{tool.name} was called incorrectly -- " + "; ".join(parts)
            + ". It accepts: " + ", ".join(sorted(allowed) or ["(no arguments)"])
            + ". Required: " + (", ".join(sorted(schema.get("required", []))) or "(none)") + ".")


class AskUser(Exception):
    """Raised by the ask_user tool to hand control back to the driver.

    The driver decides who answers: a human, or the oracle user simulator during
    an experiment.
    """

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__(question)
        self.question = question
        self.options = options or []


class Finish(Exception):
    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def build_toolset(ws: Workspace, *, allow_ask: bool = True) -> dict[str, Tool]:
    """Bind the tool implementations to one workspace.

    `allow_ask` is how the never-ask baseline is realised: the tool is simply
    absent from the schema, so the agent cannot fall back on it.
    """

    # Every tool below catches OSError, not only the specific errors that came
    # to mind. Passing a directory where a file belongs raised a bare
    # PermissionError carrying a Windows errno, which reaches the model as noise
    # it cannot act on -- and the failures worth handling are the ones that were
    # not anticipated.
    def list_files(path: str = ".") -> str:
        try:
            entries = ws.listdir(path)
        except NotADirectoryError:
            return f"error: {path} is a file, not a directory. Use read_file for it."
        except FileNotFoundError:
            return f"error: no such directory: {path}"
        except (PathEscape, OSError) as exc:
            return f"error: could not list {path}: {exc}"
        return "\n".join(entries) if entries else "(empty)"

    def read_file(path: str) -> str:
        try:
            if ws.resolve(path).is_dir():
                return f"error: {path} is a directory. Use list_files to see inside it."
            content = ws.read(path)
        except FileNotFoundError:
            return f"error: no such file: {path}"
        except (PathEscape, OSError) as exc:
            return f"error: could not read {path}: {exc}"
        lines = content.splitlines()
        return "\n".join(f"{i:>5}\t{ln}" for i, ln in enumerate(lines, 1)) or "(empty file)"

    def write_file(path: str, content: str) -> str:
        if not path.strip():
            return "error: path is empty"
        try:
            if ws.resolve(path).is_dir():
                return f"error: {path} is a directory, not a file"
            ws.write(path, content)
        except PathEscape as exc:
            return "error: " + str(exc)
        except OSError as exc:
            return f"error: could not write {path}: {exc}"
        return f"wrote {len(content)} bytes to {path}"

    def edit_file(path: str, old: str, new: str) -> str:
        # An empty anchor matches between every pair of characters, so the
        # uniqueness check below rejects it with a match count that explains
        # nothing. Say what is actually wrong.
        if old == "":
            return ("error: `old` is empty. Give the exact text to replace, or "
                    "use write_file to replace the whole file.")
        if old == new:
            # Reporting "edited" for a no-op teaches the agent it made progress
            # when it did not, which is how a loop starts.
            return "error: `old` and `new` are identical, so this would change nothing"
        try:
            content = ws.read(path)
        except FileNotFoundError:
            return f"error: no such file: {path}"
        except (PathEscape, OSError) as exc:
            return f"error: could not read {path}: {exc}"
        n = content.count(old)
        if n == 0:
            return "error: old string not found"
        if n > 1:
            return f"error: old string is not unique ({n} matches); include more context"
        try:
            ws.write(path, content.replace(old, new))
        except OSError as exc:
            return f"error: could not write {path}: {exc}"
        return f"edited {path}"

    def search(pattern: str, path: str = ".", glob: str | None = None,
               ignore_case: bool = False) -> str:
        return _search(ws, pattern, path, glob, ignore_case)

    def find_files(glob: str, path: str = ".") -> str:
        return _find_files(ws, glob, path)

    def run(command: str) -> str:
        return ws.run(command).render()

    def ask_user(question: str, options: list[str] | None = None) -> str:
        raise AskUser(question, options)

    def finish(summary: str) -> str:
        raise Finish(summary)

    tools = [
        Tool("list_files", "List entries in a directory of the workspace.",
             _obj({"path": {"type": "string", "description": "Directory, defaults to '.'"}}, []),
             list_files),
        Tool("read_file", "Read a file with line numbers.",
             _obj({"path": {"type": "string"}}, ["path"]), read_file),
        Tool("write_file", "Create or overwrite a file with the given content.",
             _obj({"path": {"type": "string"}, "content": {"type": "string"}},
                  ["path", "content"]), write_file),
        Tool("edit_file", "Replace one unique occurrence of `old` with `new` in a file.",
             _obj({"path": {"type": "string"}, "old": {"type": "string"},
                   "new": {"type": "string"}}, ["path", "old", "new"]), edit_file),
        # The descriptions are the whole interface for choosing between these
        # two and read_file, so they say what question each answers rather than
        # what each does. A model that cannot tell them apart falls back to
        # opening files one at a time, which is the behaviour they exist to fix.
        Tool("search",
             "Search file contents with a regular expression across the workspace. "
             "Use this to find where something is defined or used before opening "
             "anything. Returns 'path:line: text' for each match, not whole files.",
             _obj({"pattern": {"type": "string",
                               "description": "Python regular expression."},
                   "path": {"type": "string",
                            "description": "Directory to search under, defaults to '.'"},
                   "glob": {"type": "string",
                            "description": "Restrict to matching paths, e.g. '*.py'."},
                   "ignore_case": {"type": "boolean"}},
                  ["pattern"]), search),
        Tool("find_files",
             "List files whose path or name matches a glob, e.g. 'test_*.py'. "
             "Use this to locate a file when you know roughly what it is called.",
             _obj({"glob": {"type": "string"},
                   "path": {"type": "string",
                            "description": "Directory to search under, defaults to '.'"}},
                  ["glob"]), find_files),
        Tool("run", "Run a shell command in the workspace and return its output.",
             _obj({"command": {"type": "string"}}, ["command"]), run),
        Tool("finish", "Declare the task complete and summarise what was done.",
             _obj({"summary": {"type": "string"}}, ["summary"]), finish),
    ]

    if allow_ask:
        tools.append(Tool(
            "ask_user",
            "Ask the user one closed question to resolve an ambiguity in the task. "
            "Use only when the answer would change what you write, and phrase it so "
            "it can be answered by picking one of the given options.",
            _obj({"question": {"type": "string"},
                  "options": {"type": "array", "items": {"type": "string"},
                              "description": "Two or more concrete answers to choose between."}},
                 ["question"]),
            ask_user,
        ))

    return {t.name: t for t in tools}
