"""A contained working directory the agent is allowed to touch.

Containment is enforced on every path the agent supplies. This is not a security
sandbox -- `run` executes real subprocesses -- it is a blast radius limit so that
a hundred experiment runs cannot walk over the repo.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


class PathEscape(Exception):
    """Raised when an agent-supplied path resolves outside the workspace."""


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
                text=True,
                # Without an explicit encoding, text mode decodes with the
                # locale's preferred codec -- GBK on a Chinese Windows box --
                # while the child has just been told to write UTF-8. Any
                # non-ASCII byte in the child's output then kills the call.
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            return RunResult(proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as exc:
            return RunResult(
                exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
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
