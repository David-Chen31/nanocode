"""Pilot A: does grounding the question in the model's own code fix the crying wolf?

The text-side probes failed in a specific way. Asked "what does this spec leave
open?", models answered in the abstract and always found something -- 12/12
fully specified prompts were judged ambiguous, and one proposed reading directly
contradicted a sentence the prompt stated.

This asks a different question, and the difference is the point: not "what is
open in this spec" but "what did YOUR CODE decide that the spec did not". The
model must point at a decision it actually made, which is checkable against the
text in front of it. If grounding is what was missing, the false-alarm rate on
fully specified prompts should collapse while detection holds.

Scored two ways:
  calibration  -- items claimed on the ambiguous prompt vs the fully specified one
  localisation -- does any claimed item name the sentence that was deleted,
                  judged by a third model that never wrote any of the code

    python experiments/pilot_grounded.py --models gpt-4o-mini claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
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
from bench.schema import Task, load_tasks

VARIANTS = ("ambiguous", "full")

GROUNDED = (
    "You were given this task:\n\n{prompt}\n\n"
    "You wrote this code:\n\n```python\n{code}\n```\n\n"
    "List the behavioural decisions this code makes that the task did NOT "
    "specify -- choices where the task is silent and a different choice would "
    "change observable behaviour. One per line, under 12 words each, no "
    "numbering. Do not list anything the task already states. If the task "
    "settles every behavioural decision the code makes, reply with the single "
    "word NONE."
)

JUDGE = (
    "A programming task deliberately omitted this requirement:\n\n"
    "  \"{constraint}\"\n\n"
    "Someone reviewing a candidate implementation listed the decisions they "
    "thought were left unspecified:\n\n{items}\n\n"
    "Does any listed item refer to the SAME decision as the omitted "
    "requirement? Reply with exactly one word: YES or NO."
)


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


def parse_items(text: str) -> list[str]:
    lines = [ln.strip(" -*•\t") for ln in text.strip().splitlines() if ln.strip()]
    lines = [re.sub(r"^\d+[.)]\s*", "", ln) for ln in lines]
    if not lines or lines[0].upper().startswith("NONE"):
        return []
    return [ln for ln in lines if not ln.upper().startswith("NONE")][:8]


def ask_grounded(backend, task: Task, variant: str, code: str,
                 meter: Meter) -> dict[str, Any]:
    try:
        r = backend.text(GROUNDED.format(prompt=task.prompt(variant), code=code.strip()),
                         temperature=0.0, max_tokens=200)
    except Exception as exc:
        meter.fail()
        return {"items": [], "raw": f"ERROR {type(exc).__name__}", "error": True}
    meter.add(r.usage)
    return {"items": parse_items(r.text), "raw": r.text.strip()[:500], "error": False}


def judge_hit(backend, task: Task, items: list[str], meter: Meter) -> bool | None:
    if not items:
        return False
    listing = "\n".join("- " + i for i in items)
    try:
        r = backend.text(
            JUDGE.format(constraint=task.constraints[0].text, items=listing),
            temperature=0.0, max_tokens=8)
    except Exception:
        meter.fail()
        return None
    meter.add(r.usage)
    return r.text.strip().upper().startswith("YES")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini",
                                                    "claude-haiku-4-5-20251001"])
    ap.add_argument("--judge", default="gpt-4o",
                    help="scores localisation; should not be a generator")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results/pilot_grounded.json")
    args = ap.parse_args()

    tasks = load_tasks()
    judge_backend = make_backend("openai:" + args.judge)
    baseline = json.loads(Path("results/text_side.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"judge": args.judge, "models": {}}

    for model in args.models:
        live_path = Path(f"results/live/{model}.json")
        if not live_path.exists():
            live_path = Path("results/live/gpt-4o-mini.json")
        live = json.loads(live_path.read_text(encoding="utf-8"))
        backend = make_backend("openai:" + model)
        meter = Meter()
        t0 = time.time()

        jobs = [(t, v) for t in tasks for v in VARIANTS]
        print(f"{model}: {len(jobs)} grounded calls + judging ...", flush=True)
        res: dict[tuple[str, str], dict[str, Any]] = {}
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {}
            for t, v in jobs:
                code = live["samples"][t.id][v]["sources"][0]
                futs[ex.submit(ask_grounded, backend, t, v, code, meter)] = (t.id, v)
            for fut in as_completed(futs):
                res[futs[fut]] = fut.result()

        hits: dict[str, bool | None] = {}
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(judge_hit, judge_backend, t,
                              res[(t.id, "ambiguous")]["items"], meter): t.id
                    for t in tasks}
            for fut in as_completed(futs):
                hits[futs[fut]] = fut.result()

        amb_counts = [len(res[(t.id, "ambiguous")]["items"]) for t in tasks]
        full_counts = [len(res[(t.id, "full")]["items"]) for t in tasks]
        amb_fire = sum(1 for c in amb_counts if c > 0)
        full_fire = sum(1 for c in full_counts if c > 0)

        out["models"][model] = {
            "detection_rate": amb_fire / len(tasks),
            "false_alarm_rate": full_fire / len(tasks),
            "mean_items_ambiguous": sum(amb_counts) / len(amb_counts),
            "mean_items_full": sum(full_counts) / len(full_counts),
            "localisation": sum(1 for h in hits.values() if h) / len(tasks),
            "calls": meter.usage.calls,
            "cost_usd": round(meter.usage.cost_usd(model), 5),
            "errors": meter.errors,
            "wall_seconds": round(time.time() - t0, 1),
            "per_task": {t.id: {"ambiguous": res[(t.id, "ambiguous")],
                                "full": res[(t.id, "full")],
                                "localisation_hit": hits.get(t.id)} for t in tasks},
        }
        print(f"  {time.time() - t0:.0f}s, ${out['models'][model]['cost_usd']:.4f}, "
              f"{meter.errors} errors")

    n = len(tasks)
    print("\n" + "=" * 80)
    print("Grounded ('what did YOUR CODE decide?') vs abstract ('what is open?')")
    print("=" * 80)
    print(f"{'model':<28}{'variant':<12}{'fires on':>10}{'fires on':>11}{'localises':>11}")
    print(f"{'':<28}{'':<12}{'ambiguous':>10}{'FULL':>11}{'':>11}")
    print("-" * 80)
    for model, d in out["models"].items():
        base = baseline["models"].get(model, {}).get("signals", {})
        eg = base.get("enumerate_gaps", {})
        if eg:
            print(f"{model:<28}{'abstract':<12}"
                  f"{eg['mean_ambiguous'] * n:>7.0f}/{n}{eg['mean_full'] * n:>8.0f}/{n}"
                  f"{'-':>11}")
        print(f"{model:<28}{'GROUNDED':<12}"
              f"{d['detection_rate'] * n:>7.0f}/{n}{d['false_alarm_rate'] * n:>8.0f}/{n}"
              f"{d['localisation'] * n:>8.0f}/{n}")
    print("\nabstract localisation for reference was 1-2/12; behavioural probes were 8/12")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
