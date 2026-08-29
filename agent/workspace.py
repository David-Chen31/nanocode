"""A contained working directory the agent is allowed to touch.

Containment is enforced on every path the agent supplies. This is not a security
sandbox -- `run` executes real subprocesses -- it is a blast radius limit so that
a hundred experiment runs cannot walk over the repo.
"""
from __future__ import annotations

import locale
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


class PathEscape(Exception):
    """Raised when an agent-supplied path resolves outside the workspace."""


def _decode(raw: bytes) -> str:
    """Decode one output stream, trying the encodings that actually occur.

    A single shell command has two producers writing into these pipes and they
    do not agree on an encoding. The child process is told to emit UTF-8
    (PYTHONIOENCODING below), but when the command does not exist the child
    never runs and it is the *shell* that writes the error -- in the OEM code
    page, which on this machine is CP936.

    Pinning UTF-8 therefore turns "'pytest' is not recognized" into a row of
    replacement characters, and that is not a cosmetic loss. It was measured:
    an agent that could not read "command not found" spent ten of its fourteen
    steps guessing at variations of a command that was never going to run. The
    error message is the most valuable thing a failed command produces, so it
    is worth two decode attempts to keep it legible.

    Strict UTF-8 first because it is the common case and a mis-decode is
    silent; then the locale codec; then latin-1, which cannot fail and at least
    keeps the ASCII skeleton of the message readable.
    """
    if not raw:
        return ""
    for codec in ("utf-8", locale.getpreferredencoding(False)):
        try:
            return raw.decode(codec)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    def render(self, limit: int = 4000) -> str:
        parts = []
        if self.stdout.strip():
            parts.append("stdout:\n" + self.stdout[:limit])
        if self.stderr.strip():
            parts.append("stderr:\n" + self.stderr[:limit])
        parts.append("exit code: " + str(self.returncode) + (" (timed out)" if self.timed_out else ""))
        return "\n".join(parts)


class Workspace:
    def __init__(self, root: str | Path | None = None, *, ephemeral: bool = False) -> None:
        if root is None:
            root = tempfile.mkdtemp(prefix="aoa-ws-")
            ephemeral = True
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._ephemeral = ephemeral

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise PathEscape(rel + " resolves outside the workspace")
        return p

    def read(self, rel: str, *, max_bytes: int = 200_000) -> str:
        p = self.resolve(rel)
        if not p.exists():
            raise FileNotFoundError(rel)
        return p.read_text(encoding="utf-8", errors="replace")[:max_bytes]

    def write(self, rel: str, content: str) -> None:
        p = self.resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def listdir(self, rel: str = ".") -> list[str]:
        p = self.resolve(rel)
        if not p.is_dir():
            raise NotADirectoryError(rel)
        out = []
        for child in sorted(p.iterdir()):
            out.append(child.name + ("/" if child.is_dir() else ""))
        return out

    def run(self, command: str, *, timeout: float = 30.0) -> RunResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                # Bytes, not text=True. Text mode commits to one codec for the
                # whole call, and the two writers into these pipes do not use
                # the same one -- see _decode. Decoding per stream is the only
                # way to keep both the child's output and the shell's errors
                # readable.
                timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            return RunResult(_decode(proc.stdout), _decode(proc.stderr), proc.returncode)
        except subprocess.TimeoutExpired as exc:
            # Whatever the command managed to print before the clock ran out is
            # usually the reason it hung, so it is kept rather than discarded.
            return RunResult(
                _decode(exc.stdout or b""), _decode(exc.stderr or b""),
                -1,
                timed_out=True,
            )

    def python(self, script_rel: str, *, timeout: float = 30.0) -> RunResult:
        return self.run(f'"{sys.executable}" "{script_rel}"', timeout=timeout)

    def cleanup(self) -> None:
        if self._ephemeral and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
