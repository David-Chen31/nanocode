"""Draw candidate implementations from a real model.

The offline results in docs/FINDINGS.md rest on hand-written candidate sets, and
that is their biggest weakness: the shape of the disagreement was chosen by the
author. This script replaces those fixtures with actual samples, so every number
can be recomputed on a real sampling distribution without changing anything else
in the pipeline.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    python experiments/sample_live.py --model gpt-4o-mini --k 6

Writes results/live/<model>.json, consumed by:

    python experiments/run_experiments.py --candidates results/live/<model>.json
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

SYSTEM = (
    "You are a careful Python programmer. Answer with a single self-contained "
    "Python function definition and nothing else: no explanation, no examples, "
    "no markdown fences. Any imports you need go above the function."
)

CONFIDENCE_PROMPT = (
    "You were asked to implement this:\n\n{prompt}\n\n"
    "How confident are you that your implementation matches what the requester "
    "actually wanted, including behaviour they did not spell out? Reply with a "
    "single number between 0 and 1 and nothing else."
)

# Naive temperature sampling turns out to produce almost no behavioural
# diversity on function-level tasks (1.08 distinct behaviours per task at
# T=1.0, 1.17 at T=1.6). If disagreement is going to be measurable it has to be
# induced on purpose. This nudge asks each sample to resolve whatever the task
# left open differently, without ever telling the model what was left open.
DIVERSIFY = (
    "\n\nIf the task leaves any behaviour unspecified -- edge cases, ordering, "
    "what to do with degenerate input -- pick a reasonable resolution and commit "
    "to it. Variant {i} of {k}: choose a resolution that a different competent "
    "programmer might plausibly have chosen, not necessarily the most obvious one."
)

# The obvious baseline the execution machinery has to beat: just ask the model.
DIRECT_ASK = (
    "A programmer was given this task:\n\n{prompt}\n\n"
    "Does the task leave any behaviour unspecified that would change the code "
    "written? Reply with exactly two lines:\n"
    "LINE 1: a number from 0 (fully specified) to 1 (severely under-specified)\n"
    "LINE 2: the single most important unspecified point, under 15 words, or NONE"
)

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Pull the function out of whatever the model wrapped it in."""
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    lines = body.splitlines()
    # Drop leading prose before the first import/def line.
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(("import ", "from ", "def ", "@")):
            return "\n".join(lines[i:]).strip() + "\n"
    return body.strip() + "\n"


def parse_confidence(text: str) -> float | None:
    m = re.search(r"(\d*\.?\d+)", text)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


class Meter:
    """Thread-safe token and call accounting. Cost is reported, always."""

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


def sample_one(backend, task: Task, variant: str, idx: int, temperature: float,
               meter: Meter, *, diversify: bool = False,
               k: int = 6) -> dict[str, Any] | None:
    prompt = task.prompt(variant)
    if diversify:
        prompt += DIVERSIFY.format(i=idx + 1, k=k)
    try:
        r = backend.complete(
            [{"role": "user", "content": prompt}],
            system=SYSTEM, temperature=temperature, max_tokens=700,
            seed=1000 + idx,
        )
    except Exception as exc:
        meter.fail()
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    meter.add(r.usage)
    return {"source": extract_code(r.text), "mean_logprob": r.mean_logprob,
            "raw_len": len(r.text)}


def direct_ask(backend, task: Task, variant: str, meter: Meter) -> dict[str, Any]:
    """Ask the model straight out how under-specified the task is."""
    try:
        r = backend.text(DIRECT_ASK.format(prompt=task.prompt(variant)),
                         temperature=0.0, max_tokens=60)
    except Exception:
        meter.fail()
        return {"score": None, "point": None}
    meter.add(r.usage)
    lines = [ln.strip() for ln in r.text.strip().splitlines() if ln.strip()]
    score = parse_confidence(lines[0]) if lines else None
    point = lines[1][:120] if len(lines) > 1 else None
    return {"score": score, "point": point}


def ask_confidence(backend, task: Task, variant: str, meter: Meter) -> float | None:
    try:
        r = backend.text(CONFIDENCE_PROMPT.format(prompt=task.prompt(variant)),
                         temperature=0.0, max_tokens=12)
    except Exception:
        meter.fail()
        return None
    meter.add(r.usage)
    return parse_confidence(r.text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--k", type=int, default=6, help="candidates per task variant")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-confidence", action="store_true",
                    help="skip the self-reported-confidence baseline")
    ap.add_argument("--diversify", action="store_true",
                    help="induce diversity instead of relying on temperature")
    ap.add_argument("--no-direct-ask", action="store_true",
                    help="skip the direct-ask baseline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    backend = make_backend("openai:" + args.model)
    tasks = load_tasks()
    meter = Meter()
    t0 = time.time()

    jobs = [(t, v, i) for t in tasks for v in VARIANTS for i in range(args.k)]
    print(f"sampling {len(jobs)} candidates from {args.model} "
          f"(T={args.temperature}, {len(tasks)} tasks x {len(VARIANTS)} variants "
          f"x k={args.k})", flush=True)

    results: dict[tuple[str, str], list[dict[str, Any] | None]] = {
        (t.id, v): [None] * args.k for t in tasks for v in VARIANTS}

    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(sample_one, backend, t, v, i, args.temperature, meter,
                             diversify=args.diversify, k=args.k):
                   (t.id, v, i) for t, v, i in jobs}
        done = 0
        for fut in as_completed(futures):
            tid, v, i = futures[fut]
            results[(tid, v)][i] = fut.result()
            done += 1
            if done % 24 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {time.time() - t0:.0f}s  "
                      f"errors={meter.errors}", flush=True)

    confidence: dict[tuple[str, str], float | None] = {}
    if not args.no_confidence:
        print("asking for self-reported confidence ...", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(ask_confidence, backend, t, v, meter): (t.id, v)
                    for t in tasks for v in VARIANTS}
            for fut in as_completed(futs):
                confidence[futs[fut]] = fut.result()

    direct: dict[tuple[str, str], dict[str, Any]] = {}
    if not args.no_direct_ask:
        print("running the direct-ask baseline ...", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(direct_ask, backend, t, v, meter): (t.id, v)
                    for t in tasks for v in VARIANTS}
            for fut in as_completed(futs):
                direct[futs[fut]] = fut.result()

    samples: dict[str, dict[str, Any]] = {}
    dropped = 0
    for t in tasks:
        samples[t.id] = {}
        for v in VARIANTS:
            got = [r for r in results[(t.id, v)] if r and "source" in r]
            dropped += args.k - len(got)
            samples[t.id][v] = {
                "sources": [r["source"] for r in got],
                "mean_logprobs": [r["mean_logprob"] for r in got],
                "self_report": confidence.get((t.id, v)),
                "direct_ask": direct.get((t.id, v), {}).get("score"),
                "direct_ask_point": direct.get((t.id, v), {}).get("point"),
            }

    payload = {
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
        "diversify": args.diversify,
        "n_tasks": len(tasks),
        "requested": len(jobs),
        "dropped": dropped,
        "api_errors": meter.errors,
        "input_tokens": meter.usage.input_tokens,
        "output_tokens": meter.usage.output_tokens,
        "model_calls": meter.usage.calls,
        "cost_usd": round(meter.usage.cost_usd(args.model), 5),
        "wall_seconds": round(time.time() - t0, 1),
        "samples": samples,
    }

    out = Path(args.out or f"results/live/{args.model.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(jobs) - dropped}/{len(jobs)} candidates kept "
          f"({meter.errors} API errors)")
    print(f"tokens {meter.usage.input_tokens}in / {meter.usage.output_tokens}out"
          f"  cost ${payload['cost_usd']:.4f}  {payload['wall_seconds']}s")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
