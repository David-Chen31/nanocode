"""Can the codebase supply the sentence the requirement left out?

Every signal tested so far lived inside the model, and every one of them failed.
The published account of why (semantic collapse) says the model commits to one
reading and never signals doubt. That leaves an obvious question nobody in the
related-work sweep appears to have asked directly: the information is not in the
model, but is it in the *repository*?

Three matched arms, identical except for what sits above the function:

  bare        the ambiguous requirement alone
  distractor  + two sibling helpers that resolve nothing relevant
  convention  + two sibling helpers that resolve the analogous decision the way
              the reference does, demonstrated in code and never stated in prose

`distractor` is what separates "the convention was read" from "extra context made
the model more careful". Both padded to the same length with the same neutral
helpers.

Measured by execution, not by inspection:
  coverage        does the sample set contain the reference behaviour at all
  majority right  does the behaviour most samples agree on match the reference

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    python experiments/pilot_repo_context.py --models gpt-4o-mini claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import Usage, make_backend
from askoract.bse import compute_bse
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.contexts import ARMS, context_for
from bench.harness import CALIB_SEED
from bench.schema import Task, load_tasks

SYSTEM = (
    "You are a careful Python programmer. Answer with a single self-contained "
    "Python function definition and nothing else: no explanation, no examples, "
    "no markdown fences. Any imports you need go above the function."
)

WITH_CONTEXT = (
    "You are adding one function to this existing module:\n\n"
    "```python\n{context}```\n\n"
    "Add this function to the module:\n\n{prompt}"
)

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(("import ", "from ", "def ", "@")):
            return "\n".join(lines[i:]).strip() + "\n"
    return body.strip() + "\n"


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


def sample(backend, task: Task, arm: str, idx: int, temperature: float,
           meter: Meter) -> str | None:
    ctx = context_for(task.id, arm)
    prompt = task.prompt_ambiguous if ctx is None else \
        WITH_CONTEXT.format(context=ctx, prompt=task.prompt_ambiguous)
    try:
        r = backend.complete([{"role": "user", "content": prompt}],
                             system=SYSTEM, temperature=temperature,
                             max_tokens=700, seed=2000 + idx)
    except Exception:
        meter.fail()
        return None
    meter.add(r.usage)
    return extract_code(r.text)


def score(task: Task, sources: list[str]) -> dict[str, Any]:
    """Behavioural verdict for one arm of one task."""
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    m = run_candidates(sources, probes, task.entry_point)
    ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    res = compute_bse(m)
    rows = m.valid_rows()
    covered = any(m.tokens[i] == ref for i in rows)
    majority_right = False
    if res.classes:
        idx = m.candidate_ids.index(res.majority_class[0])
        majority_right = m.tokens[idx] == ref
    return {"covered": covered, "majority_right": majority_right,
            "classes": res.n_classes, "bse": res.bse, "invalid": res.n_invalid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5-20251001", "qwen-plus"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/pilot_repo_context.json")
    args = ap.parse_args()

    tasks = load_tasks()
    out: dict[str, Any] = {"k": args.k, "temperature": args.temperature, "models": {}}

    for model in args.models:
        backend = make_backend("openai:" + model)
        meter = Meter()
        t0 = time.time()
        jobs = [(t, arm, i) for t in tasks for arm in ARMS for i in range(args.k)]
        print(f"{model}: {len(jobs)} calls across {len(ARMS)} arms ...", flush=True)

        got: dict[tuple[str, str], list[str]] = defaultdict(list)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(sample, backend, t, arm, i, args.temperature, meter):
                    (t.id, arm) for t, arm, i in jobs}
            for fut in as_completed(futs):
                src = fut.result()
                if src:
                    got[futs[fut]].append(src)

        per_task: dict[str, dict[str, Any]] = {}
        for t in tasks:
            per_task[t.id] = {arm: score(t, got[(t.id, arm)]) for arm in ARMS
                              if got[(t.id, arm)]}

        summary = {}
        n = len(tasks)
        for arm in ARMS:
            cov = sum(per_task[t.id][arm]["covered"] for t in tasks if arm in per_task[t.id])
            maj = sum(per_task[t.id][arm]["majority_right"] for t in tasks
                      if arm in per_task[t.id])
            cls = [per_task[t.id][arm]["classes"] for t in tasks if arm in per_task[t.id]]
            summary[arm] = {"coverage": cov, "majority_right": maj, "n": n,
                            "mean_classes": sum(cls) / max(1, len(cls))}

        out["models"][model] = {
            "summary": summary, "per_task": per_task,
            "calls": meter.usage.calls,
            "cost_usd": round(meter.usage.cost_usd(model), 5),
            "errors": meter.errors,
            "wall_seconds": round(time.time() - t0, 1),
        }
        print(f"  {time.time() - t0:.0f}s, ${out['models'][model]['cost_usd']:.4f}, "
              f"{meter.errors} errors")

    n = len(tasks)
    print("\n" + "=" * 78)
    print("Can sibling functions supply the omitted requirement?")
    print("=" * 78)
    print(f"{'model':<28}{'arm':<13}{'majority right':>16}{'coverage':>11}{'classes':>9}")
    print("-" * 78)
    for model, d in out["models"].items():
        for arm in ARMS:
            s = d["summary"][arm]
            print(f"{model if arm == ARMS[0] else '':<28}{arm:<13}"
                  f"{s['majority_right']:>11}/{n:<4}{s['coverage']:>7}/{n:<3}"
                  f"{s['mean_classes']:>9.2f}")
        print("-" * 78)

    print("\nThe comparison that matters is convention vs distractor: both add the")
    print("same amount of surrounding code, only one carries the decision.")

    # Which tasks flipped, and which kinds of constraint they were.
    print("\nper-task majority-right, first model:")
    first = args.models[0]
    pt = out["models"][first]["per_task"]
    print(f"{'task':<24}" + "".join(f"{a:>13}" for a in ARMS))
    print("-" * 63)
    for t in tasks:
        cells = "".join(f"{('YES' if pt[t.id][a]['majority_right'] else 'no'):>13}"
                        for a in ARMS if a in pt[t.id])
        print(f"{t.id:<24}{cells}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
