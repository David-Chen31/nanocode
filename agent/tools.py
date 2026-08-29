"""Tool definitions for the coding agent.

`ask_user` is deliberately a normal tool. The whole research question is when the
agent should reach for it, so the gating policy lives outside the tool (see
askoract/voi.py) and the tool itself stays dumb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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

    def list_files(path: str = ".") -> str:
        try:
            entries = ws.listdir(path)
        except (FileNotFoundError, NotADirectoryError, PathEscape) as exc:
            return "error: " + str(exc)
        return "\n".join(entries) if entries else "(empty)"

    def read_file(path: str) -> str:
        try:
            content = ws.read(path)
        except (FileNotFoundError, PathEscape) as exc:
            return "error: " + str(exc)
        lines = content.splitlines()
        return "\n".join(f"{i:>5}\t{ln}" for i, ln in enumerate(lines, 1)) or "(empty file)"

    def write_file(path: str, content: str) -> str:
        try:
            ws.write(path, content)
        except PathEscape as exc:
            return "error: " + str(exc)
        return f"wrote {len(content)} bytes to {path}"

    def edit_file(path: str, old: str, new: str) -> str:
        try:
            content = ws.read(path)
        except (FileNotFoundError, PathEscape) as exc:
            return "error: " + str(exc)
        n = content.count(old)
        if n == 0:
            return "error: old string not found"
        if n > 1:
            return f"error: old string is not unique ({n} matches); include more context"
        ws.write(path, content.replace(old, new))
        return f"edited {path}"

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
