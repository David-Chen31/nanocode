"""Execute candidate implementations on probe inputs and record their behaviour.

This is the piece that makes the uncertainty signal *behavioural* rather than
lexical. Two implementations that differ only in variable names produce the same
behaviour vector; two that differ on what `f([])` does do not. Natural-language
semantic entropy cannot make that distinction on code, which is the whole reason
for executing.

A behaviour token captures more than the return value: an in-place mutation and a
raised exception type are both observable differences a user may care about.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The runner is written to the workspace and executed in a fresh interpreter.
# It flushes one JSON line per probe so that a hang on probe j still leaves the
# results for probes < j recoverable.
RUNNER = r'''
import json, sys, io, traceback, copy

def canon(v, _depth=0):
    """Canonical, type-aware repr so that [1,2] and (1,2) are different tokens."""
    if _depth > 6:
        return "<deep>"
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "bool:" + repr(v)
    if isinstance(v, int):
        return "int:" + repr(v)
    if isinstance(v, float):
        if v != v:
            return "float:nan"
        if v in (float("inf"), float("-inf")):
            return "float:" + repr(v)
        return "float:" + repr(round(v, 9) + 0.0)
    if isinstance(v, str):
        return "str:" + repr(v)
    if isinstance(v, bytes):
        return "bytes:" + repr(v)
    if isinstance(v, list):
        return "list:[" + ",".join(canon(x, _depth + 1) for x in v) + "]"
    if isinstance(v, tuple):
        return "tuple:(" + ",".join(canon(x, _depth + 1) for x in v) + ")"
    if isinstance(v, set):
        return "set:{" + ",".join(sorted(canon(x, _depth + 1) for x in v)) + "}"
    if isinstance(v, frozenset):
        return "frozenset:{" + ",".join(sorted(canon(x, _depth + 1) for x in v)) + "}"
    if isinstance(v, dict):
        items = sorted((canon(k, _depth + 1), canon(x, _depth + 1)) for k, x in v.items())
        return "dict:{" + ",".join(k + ":" + x for k, x in items) + "}"
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return "ndarray:" + canon(v.tolist(), _depth + 1)
        if isinstance(v, _np.generic):
            return canon(v.item(), _depth + 1)
    except Exception:
        pass
    if hasattr(v, "__iter__"):
        try:
            return "iter:[" + ",".join(canon(x, _depth + 1) for x in list(v)[:64]) + "]"
        except Exception:
            pass
    return "obj:" + type(v).__name__


def _short(text, limit=90):
    return text if len(text) <= limit else text[:limit] + "..."


def main():
    cand_path, probes_path, entry = sys.argv[1], sys.argv[2], sys.argv[3]
    probes = json.loads(open(probes_path, encoding="utf-8").read())

    ns = {}
    try:
        src = open(cand_path, encoding="utf-8").read()
        exec(compile(src, cand_path, "exec"), ns)
        fn = ns[entry]
    except Exception as exc:
        # One INVALID token for every probe: the candidate never ran at all.
        for i in range(len(probes)):
            print(json.dumps({"i": i, "b": "INVALID:" + type(exc).__name__,
                              "r": "did not load: " + type(exc).__name__}), flush=True)
        return

    for i, args in enumerate(probes):
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            passed = copy.deepcopy(args)
            try:
                out = fn(*passed)
                token = "ret=" + canon(out)
                shown = "returns " + _short(repr(out))
            except Exception as exc:
                token = "raise=" + type(exc).__name__
                shown = "raises " + type(exc).__name__
            # An in-place mutation of the arguments is an observable behaviour.
            try:
                if canon(passed) != canon(args):
                    token += "|mut=" + canon(passed)
                    shown += " and modifies the arguments to " + _short(repr(passed))
            except Exception:
                pass
            printed = buf.getvalue()
            if printed:
                token += "|out=" + repr(printed[:200])
                shown += " and prints " + _short(repr(printed))
        except BaseException as exc:
            token = "harness_error=" + type(exc).__name__
            shown = "harness error"
        finally:
            sys.stdout = real_stdout
        print(json.dumps({"i": i, "b": token, "r": shown}), flush=True)


main()
'''

INVALID_PREFIX = "INVALID:"
TIMEOUT_TOKEN = "TIMEOUT"


@dataclass
class BehaviourMatrix:
    """rows = candidates, cols = probes, cells = behaviour tokens."""

    tokens: list[list[str]]
    probes: list[list[Any]]
    candidate_ids: list[str]
    invalid: list[bool] = field(default_factory=list)
    # Human-readable rendering of each cell, used to phrase questions. Never
    # used for clustering -- `tokens` is the only thing equality runs on.
    shown: list[list[str]] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.tokens)

    @property
    def n_probes(self) -> int:
        return len(self.probes)

    def valid_rows(self) -> list[int]:
        return [i for i, bad in enumerate(self.invalid) if not bad]

    def subset(self, rows: list[int]) -> "BehaviourMatrix":
        return BehaviourMatrix(
            tokens=[self.tokens[i] for i in rows],
            probes=self.probes,
            candidate_ids=[self.candidate_ids[i] for i in rows],
            invalid=[self.invalid[i] for i in rows],
            shown=[self.shown[i] for i in rows] if self.shown else [],
        )

    def display(self, token: str, probe_index: int) -> str:
        """A readable phrase for one behaviour token at one probe."""
        for i, row in enumerate(self.tokens):
            if row[probe_index] == token and self.shown:
                return self.shown[i][probe_index]
        return token


def run_candidates(
    sources: list[str],
    probes: list[list[Any]],
    entry_point: str,
    *,
    candidate_ids: list[str] | None = None,
    timeout: float = 20.0,
    python: str | None = None,
) -> BehaviourMatrix:
    """Run every candidate against every probe in a fresh subprocess."""
    python = python or sys.executable
    ids = candidate_ids or [f"c{i}" for i in range(len(sources))]
    rows: list[list[str]] = []
    shown_rows: list[list[str]] = []
    invalid: list[bool] = []

    with tempfile.TemporaryDirectory(prefix="aoa-exec-") as td:
        tmp = Path(td)
        runner = tmp / "_runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        probes_path = tmp / "_probes.json"
        probes_path.write_text(json.dumps(probes), encoding="utf-8")

        for idx, src in enumerate(sources):
            cand = tmp / f"_cand{idx}.py"
            cand.write_text(src, encoding="utf-8")
            row = [TIMEOUT_TOKEN] * len(probes)
            shown = ["timed out"] * len(probes)
            try:
                proc = subprocess.run(
                    [python, str(runner), str(cand), str(probes_path), entry_point],
                    capture_output=True, text=True, timeout=timeout,
                    # Text mode defaults to the locale codec, which is not UTF-8
                    # on every machine. Candidate code written by an agent will
                    # eventually print a character that codec cannot represent.
                    encoding="utf-8", errors="replace",
                )
                stdout = proc.stdout
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) \
                    else (exc.stdout or "")

            for line in stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 0 <= rec.get("i", -1) < len(row):
                    row[rec["i"]] = rec["b"]
                    shown[rec["i"]] = rec.get("r", rec["b"])

            rows.append(row)
            shown_rows.append(shown)
            invalid.append(all(t.startswith(INVALID_PREFIX) for t in row))

    return BehaviourMatrix(tokens=rows, probes=probes, candidate_ids=ids,
                           invalid=invalid, shown=shown_rows)
