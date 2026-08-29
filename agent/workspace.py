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
import time
from dataclasses import dataclass
from pathlib import Path


# Per-command output ceiling. Generous enough for a verbose test suite, small
# enough that a runaway print loop is stopped in well under a second.
MAX_OUTPUT_BYTES = 4_000_000


class PathEscape(Exception):
    """Raised when an agent-supplied path resolves outside the workspace."""


def _size(f) -> int:
    # The child writes through its own descriptor, so this file object's
    # position never moves. fstat is what actually sees the growth.
    return os.fstat(f.fileno()).st_size


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Kill the command, and on Windows the shell's children too.

    With shell=True the direct child is cmd.exe; killing it leaves the process
    that is actually producing the output running and orphaned.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _read_capped(f, limit: int) -> tuple[str, bool]:
    """Read a spooled stream back, keeping the head and the tail.

    Same reasoning as the context clipper: the end of a traceback is usually
    the half that names the error.
    """
    size = _size(f)
    f.seek(0)
    if size <= limit:
        return _decode(f.read()), False
    head = f.read(int(limit * 0.6))
    f.seek(size - (limit - len(head)))
    note = "\n\n... [" + str(size - limit) + " bytes omitted] ...\n\n"
    return _decode(head) + note + _decode(f.read()), True


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

    def run(self, command: str, *, timeout: float = 30.0,
            max_output_bytes: int = MAX_OUTPUT_BYTES) -> RunResult:
        """Run a shell command in the workspace, bounded in time AND in volume.

        The volume bound is not paranoia. `capture_output=True` reads each pipe
        with an unbounded `fh.read()`, so a program that loops printing fills
        memory long before a 30 second timeout can fire. A 432-run study died
        that way at run ~320: MemoryError inside subprocess's reader thread,
        taking the completed runs with it.

        So output is spooled to temporary files rather than into memory, and the
        size of those files is checked on the same poll that checks the clock.
        Killing on volume is also better feedback than dying on it -- "output
        limit exceeded" tells the agent its program is looping, which is very
        likely the bug it was asked to fix.
        """
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=self.root,
                stdout=out_f,
                stderr=err_f,
                # Bytes on disk, decoded later. Text mode commits to one codec
                # for the whole call, and the two writers into these streams do
                # not use the same one -- see _decode.
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            deadline = time.monotonic() + timeout
            stopped = ""
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    stopped = "timeout"
                    break
                if _size(out_f) + _size(err_f) > max_output_bytes:
                    stopped = "output"
                    break
                time.sleep(0.02)
            if stopped:
                _kill_tree(proc)

            stdout, err_extra = _read_capped(out_f, max_output_bytes)
            stderr, _ = _read_capped(err_f, max_output_bytes)

        if stopped == "output":
            # Said in the output rather than only in a return code, because the
            # agent reads this and a looping program is usually the actual bug.
            stderr += ("\n[killed: the command produced more than "
                       + str(max_output_bytes) + " bytes of output. "
                       "It is probably looping.]")
        elif err_extra:
            stderr += "\n[stdout was truncated in the middle]"
        # A killed command's exit status was chosen by the kill, not by the
        # program, so reporting it as the program's own code would be a lie.
        # -1 is the harness saying "this did not exit on its own".
        code = -1 if stopped else (proc.returncode if proc.returncode is not None else -1)
        return RunResult(stdout, stderr, code, timed_out=stopped == "timeout")

    def python(self, script_rel: str, *, timeout: float = 30.0,
               max_output_bytes: int = MAX_OUTPUT_BYTES) -> RunResult:
        return self.run(f'"{sys.executable}" "{script_rel}"', timeout=timeout,
                        max_output_bytes=max_output_bytes)

    def cleanup(self) -> None:
        if self._ephemeral and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
