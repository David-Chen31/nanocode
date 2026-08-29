"""The other half: signals that read the requirement, not the model's output.

Every signal tested so far reads the model's own output distribution -- sampling
spread, token probabilities, self-report -- and all of them sit at chance. This
script tests the complementary family: prompts that ask a model to reason about
the *specification text* itself, never seeing any candidate implementation.

Four of them, in increasing structure:

  direct_ask        how under-specified is this, 0 to 1
  enumerate_gaps    list every behaviour left open; score on how many
  two_readings      give two competent readings that differ, or say IDENTICAL
  discriminating    name one concrete input on which two readings differ

The last is the interesting one. It returns something executable, so it can be
scored the same way the behavioural pipeline was: run the reference on the named
input and check whether it lands on the constraint that was deleted. That makes
it directly comparable to the probe-ranking localisation number.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    python experiments/text_side.py --models gpt-4o-mini claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import Usage, make_backend
from askoract.execute import run_candidates
from bench.harness import _norm
from bench.schema import Task, load_tasks

VARIANTS = ("ambiguous", "full")

PROMPTS: dict[str, str] = {
    "direct_ask": (
        "A programmer was given this task:\n\n{prompt}\n\n"
        "Does the task leave any behaviour unspecified that would change the code "
        "written? Reply with exactly two lines:\n"
        "LINE 1: a number from 0 (fully specified) to 1 (severely under-specified)\n"
        "LINE 2: the single most important unspecified point, under 15 words, or NONE"
    ),
    "enumerate_gaps": (
        "A programmer was given this task:\n\n{prompt}\n\n"
        "List every behaviour the task leaves genuinely open -- cases where two "
        "competent programmers would write different code and both be defensible. "
        "One per line, under 12 words each, no numbering. If there are none, reply "
        "with the single word NONE. Do not list anything the task already states."
    ),
    "two_readings": (
        "A programmer was given this task:\n\n{prompt}\n\n"
        "Give two readings of this task that a competent programmer might hold and "
        "that would produce code differing in observable behaviour. Reply with "
        "exactly two lines, each under 15 words. If the task admits only one "
        "reasonable reading, reply with the single word IDENTICAL."
    ),
    "discriminating": (
        "A programmer was given this task:\n\n{prompt}\n\n"
        "Name one concrete call on which two competent implementations of this task "
        "could return different results. Reply with exactly one line: a Python "
        "argument tuple that could be splatted into the function, for example "
        "([], 2.0) or ('abc', 5). If no such call exists, reply with the single "
        "word NONE."
    ),
}


class Meter:
    def __init__(self) -> None:
        self.usage = Usage()
        self.errors = 0
        self._lock = threading.Lock()

    def add(self, u: Usage) -> None:
        with self._lock:
            self.usage.add(u)

    def fail(self) -> None:
        with self._lock:
            self.errors += 1


def _num(text: str) -> float | None:
    m = re.search(r"(\d*\.?\d+)", text)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def parse_args_tuple(line: str) -> list[Any] | None:
    """Turn the model's argument tuple into probe arguments."""
    line = line.strip().strip("`")
    m = re.search(r"\(.*\)", line, re.S)
    if not m:
        return None
    try:
        val = ast.literal_eval(m.group(0))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(val, tuple):
        val = (val,)
    return [list(v) if isinstance(v, tuple) else v for v in val]


def score_response(kind: str, text: str) -> tuple[float, str | None]:
    """Map each prompt's answer onto a [0,1] uncertainty score plus a point."""
    body = text.strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if kind == "direct_ask":
        return (_num(lines[0]) if lines else 0.0) or 0.0, \
            (lines[1][:120] if len(lines) > 1 else None)
    if kind == "enumerate_gaps":
        if not lines or lines[0].upper().startswith("NONE"):
            return 0.0, None
        items = [ln for ln in lines if not ln.upper().startswith("NONE")]
        # three or more genuinely open points is treated as maximal
        return min(1.0, len(items) / 3.0), (items[0][:120] if items else None)
    if kind == "two_readings":
        if not lines or lines[0].upper().startswith("IDENTICAL"):
            return 0.0, None
        return (1.0 if len(lines) >= 2 else 0.5), (lines[0][:120] if lines else None)
    if kind == "discriminating":
        if not lines or lines[0].upper().startswith("NONE"):
            return 0.0, None
        return 1.0, lines[0][:120]
    raise ValueError(kind)


def ask(backend, kind: str, task: Task, variant: str, meter: Meter) -> dict[str, Any]:
    try:
        r = backend.text(PROMPTS[kind].format(prompt=task.prompt(variant)),
                         temperature=0.0, max_tokens=180)
    except Exception as exc:
        meter.fail()
        return {"score": None, "point": None, "raw": f"ERROR {type(exc).__name__}"}
    meter.add(r.usage)
    score, point = score_response(kind, r.text)
    return {"score": score, "point": point, "raw": r.text.strip()[:400]}


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


def localisation_hit(task: Task, raw: str) -> bool | None:
    """Did the named call actually discriminate the deleted constraint?

    Scored by execution, not string matching: run the reference and a candidate
    reading is irrelevant -- what matters is whether the named arguments are the
    ones the diagnostic set records as revealing the constraint.
    """
    args = parse_args_tuple(raw)
    if args is None:
        return None
    wanted = {_norm(a) for c in task.constraints for a in c.discriminating_args}
    if _norm(args) in wanted:
        return True
    # Accept a call that provokes the same reference behaviour as a recorded
    # discriminating input; different literals can probe the same edge.
    try:
        probes = [args] + [a for c in task.constraints for a in c.discriminating_args]
        rows = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    except Exception:
        return False
    return rows[0] in rows[1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/text_side.json")
    args = ap.parse_args()

    tasks = load_tasks()
    kinds = list(PROMPTS)
    out: dict[str, Any] = {"models": {}}

    for model in args.models:
        backend = make_backend("openai:" + model)
        meter = Meter()
        t0 = time.time()
        jobs = [(k, t, v) for k in kinds for t in tasks for v in VARIANTS]
        print(f"{model}: {len(jobs)} calls ...", flush=True)

        res: dict[tuple[str, str, str], dict[str, Any]] = {}
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(ask, backend, k, t, v, meter): (k, t.id, v)
                    for k, t, v in jobs}
            for fut in as_completed(futs):
                res[futs[fut]] = fut.result()

        rows = {}
        for k in kinds:
            amb = [res[(k, t.id, "ambiguous")]["score"] for t in tasks]
            full = [res[(k, t.id, "full")]["score"] for t in tasks]
            amb = [a for a in amb if a is not None]
            full = [f for f in full if f is not None]
            hits = []
            if k == "discriminating":
                for t in tasks:
                    raw = res[(k, t.id, "ambiguous")]["raw"]
                    hits.append(localisation_hit(t, raw))
            rows[k] = {
                "auroc_h1": round(auroc(amb, full), 4),
                "mean_ambiguous": round(sum(amb) / len(amb), 4) if amb else None,
                "mean_full": round(sum(full) / len(full), 4) if full else None,
                "localisation": (round(sum(1 for h in hits if h) / len(hits), 4)
                                 if hits else None),
                "per_task": {t.id: {v: res[(k, t.id, v)] for v in VARIANTS}
                             for t in tasks},
            }

        out["models"][model] = {
            "signals": rows,
            "calls": meter.usage.calls,
            "cost_usd": round(meter.usage.cost_usd(model), 5),
            "errors": meter.errors,
            "wall_seconds": round(time.time() - t0, 1),
        }
        print(f"  done in {time.time() - t0:.0f}s, ${out['models'][model]['cost_usd']:.4f}, "
              f"{meter.errors} errors")

    print("\n" + "=" * 78)
    print("Text-side signals: AUROC on 'is this prompt under-specified?'")
    print("=" * 78)
    print(f"{'model':<30}" + "".join(f"{k[:13]:>13}" for k in kinds))
    print("-" * 78)
    for model, d in out["models"].items():
        print(f"{model:<30}" +
              "".join(f"{d['signals'][k]['auroc_h1']:>13.3f}" for k in kinds))
    print("\nfor reference, every output-distribution signal sat at 0.32 - 0.60")

    print("\n" + "=" * 78)
    print("Localisation: did 'discriminating' name an input that reveals the")
    print("deleted constraint? (compare: behavioural probe ranking scored 8/12)")
    print("=" * 78)
    for model, d in out["models"].items():
        loc = d["signals"]["discriminating"]["localisation"]
        print(f"{model:<30}{loc:>8.3f}" if loc is not None else f"{model:<30}     n/a")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
