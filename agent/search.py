"""Finding things in a repository the agent has not read.

Without this the agent has `list_files` and `read_file`, which means the only way
to find anything is to walk the tree and open files one at a time. That is not a
theoretical limitation: in the trajectory study the agents mostly did not explore
at all, and part of that is because exploring was expensive.

WHY NOT JUST LET IT TYPE `grep`

The agent already has `run`, so it could shell out. Three reasons not to rely on
that:

- `grep` is not present on Windows, and the agent has no reliable way to know
  which platform it is on. A tool that works everywhere beats a command that
  works on the developer's machine.
- Shelling out puts shell quoting between the model and its intent. A pattern
  containing a quote, a space or a `$` becomes a debugging session.
- `grep -rn` over a repository emits unbounded output straight into the context
  window, and it walks `.git` and `node_modules` while it does it.

So this is implemented here, where the output can be capped and the walk can
skip what is never worth reading.

TRUNCATION IS ANNOUNCED, NEVER SILENT

The cap is the part most likely to mislead. An agent that asks for the callers of
a function and gets back three results will conclude there are three callers. So
when results are dropped the reply says how many and suggests narrowing, and the
per-file cap is separate from the global one -- one enormous file should not use
up the whole budget and hide every other match in the repository.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from .workspace import PathEscape, Workspace

# Directories that are never the answer and are expensive to walk. Skipping them
# is not a heuristic about relevance; it is that their contents are build
# products, vendored code, or a database.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             "node_modules", ".venv", "venv", "dist", "build", ".idea", ".vscode",
             ".tox", ".eggs", "site-packages"}

# Text-ish enough to be worth scanning. A file with no extension is still read --
# READMEs, Makefiles and scripts live there -- and the binary check below is what
# actually filters.
SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".o",
                 ".a", ".zip", ".gz", ".tar", ".whl", ".png", ".jpg", ".jpeg",
                 ".gif", ".ico", ".pdf", ".mp4", ".mov", ".woff", ".woff2",
                 ".ttf", ".class", ".jar", ".db", ".sqlite"}

MAX_FILE_BYTES = 2_000_000


@dataclass
class SearchLimits:
    """Caps on what one search may return.

    Per-file and total are separate on purpose: a single generated file with two
    thousand matches would otherwise consume the whole budget and hide the one
    match that mattered somewhere else.
    """

    per_file: int = 5
    total: int = 60
    line_chars: int = 200


def _is_binary(head: bytes) -> bool:
    # A NUL in the first block is the same test `grep` uses, and for the same
    # reason: it is cheap and it is almost never wrong on real source trees.
    return b"\x00" in head


def _walk(ws: Workspace, base: Path, glob: str | None):
    """Yield workspace-relative paths under `base`, skipping the junk."""
    for path in sorted(base.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(ws.root).parts):
            continue
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(ws.root).as_posix()
        if glob and not (fnmatch.fnmatch(rel, glob)
                         or fnmatch.fnmatch(path.name, glob)):
            continue
        yield rel, path


def find_files(ws: Workspace, glob: str, path: str = ".", limit: int = 200) -> str:
    """Locate files by name pattern. Answers 'where does X live'."""
    try:
        base = ws.resolve(path)
    except PathEscape as exc:
        return "error: " + str(exc)
    if not base.is_dir():
        return f"error: {path} is not a directory"

    hits = [rel for rel, _ in _walk(ws, base, glob)]
    if not hits:
        return f"no files matching {glob!r} under {path}"
    shown, extra = hits[:limit], len(hits) - limit
    out = "\n".join(shown)
    if extra > 0:
        out += f"\n... and {extra} more; narrow the pattern to see them"
    return out


def search(ws: Workspace, pattern: str, path: str = ".", glob: str | None = None,
           ignore_case: bool = False, limits: SearchLimits | None = None) -> str:
    """Regex search over file contents. Answers 'who touches X'.

    Returns `path:line: text`, not the surrounding file. Search reports where to
    look; `read_file` is what looks. Keeping those separate is what stops one
    query from pulling half the repository into the context window.
    """
    limits = limits or SearchLimits()
    try:
        # The model writes these, and a malformed one is a normal event rather
        # than an exceptional one. Report it in the same channel as any other
        # tool error so the agent can simply try again.
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"error: invalid regex {pattern!r}: {exc}"
    try:
        base = ws.resolve(path)
    except PathEscape as exc:
        return "error: " + str(exc)
    if not base.is_dir():
        return f"error: {path} is not a directory"

    lines: list[str] = []
    total = 0
    suppressed = 0            # matches inside files that hit the per-file cap
    files_hit = 0
    unvisited = 0             # files never scanned because the total cap was hit

    for rel, p in _walk(ws, base, glob):
        if total >= limits.total:
            unvisited += 1
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = p.read_bytes()
        except OSError:
            continue
        if _is_binary(raw[:8192]):
            continue
        text = raw.decode("utf-8", errors="replace")

        in_file = 0
        for n, line in enumerate(text.splitlines(), 1):
            if not rx.search(line):
                continue
            in_file += 1
            if in_file > limits.per_file or total >= limits.total:
                suppressed += 1
                continue
            body = line.strip()
            if len(body) > limits.line_chars:
                body = body[:limits.line_chars] + " ..."
            lines.append(f"{rel}:{n}: {body}")
            total += 1
        if in_file:
            files_hit += 1

    if not lines:
        where = f" in {glob}" if glob else ""
        return f"no matches for {pattern!r}{where} under {path}"

    out = "\n".join(lines)
    notes = []
    if suppressed:
        notes.append(f"{suppressed} further match(es) not shown")
    if unvisited:
        notes.append(f"{unvisited} file(s) not scanned")
    # The count is the honest part: without it a capped result is
    # indistinguishable from a complete one.
    head = f"{total} match(es) in {files_hit} file(s)"
    if notes:
        head += " -- " + ", ".join(notes) + ". Narrow with `glob` or a tighter pattern."
    return head + "\n" + out
