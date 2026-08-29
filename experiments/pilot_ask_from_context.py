"""Is the ask-decision a property of the repository rather than of the model?

Rounds 1-3 asked the model about its own uncertainty and every signal failed --
the published account (semantic collapse) says the model commits to one reading
and is not uncertain, so there is nothing to read out. Round 4 then showed that
*which* reading it commits to is decided by the surrounding code.

Put those together and the ask-decision stops being introspection. It becomes a
reading-comprehension question about visible text:

    does the surrounding module settle the decision this requirement leaves open?

That has an answer outside the model, which is why it might be answerable where
self-report was not.

THE MEASUREMENT PROBLEM, AND THE FIX

A first pass asked for a bare ASK/PROCEED verdict. Every arm came back ASK at
0.67-1.00, including the arm where round 4 shows the model actually writes the
right function. A fixed decision boundary measures the model's global appetite
for asking -- which is set by how the question is worded -- far more than it
measures context sensitivity.

So the primary statistic here is a *within-task ranking*. The same model sees
the same requirement twice, differing only in the surrounding module, and scores
how much is left unsettled. Whether it scores everything 8 or everything 2 then
does not matter; only whether it separates the two contexts does. Global bias
cancels exactly. The binary verdict is still collected, because the size of that
bias is itself worth reporting.

Two detectors, five arms (round 4's), one identical requirement throughout:

  naive     score how much is unsettled, then decide -- the round-2 self-report
            signal re-run with context, and the control that makes the other one
            mean something
  grounded  enumerate what the requirement leaves open, check each item against
            the module, then score and decide

  bare        no module at all
  distractor  siblings that resolve nothing relevant   -- should score HIGH
  convention  siblings demonstrating the reference reading -- should score LOW
  diluted     the same, pushed away and padded         -- should score LOW
  anti        siblings demonstrating the OPPOSITE of the hidden spec

`anti` has no correct answer available to the detector: the information needed
to catch it is not on the page. That is the point. If anti scores as settled as
convention, a wrong repository convention is followed without a murmur, and no
detector reading only the repository can help.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    py -3 experiments/pilot_ask_from_context.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import Usage, make_backend
from bench.contexts import ARMS, context_for
from bench.schema import Task, load_tasks

SYSTEM = (
    "You are a careful Python programmer working inside an existing codebase. "
    "You are deciding whether to implement a requirement now or to stop and ask "
    "the person who wrote it. Asking has a real cost: it interrupts someone. "
    "Not asking has a real cost too: you may silently ship the wrong behaviour."
)

# The scored quantity is deliberately about what remains open AFTER reading the
# module, not about the requirement alone. The requirement is byte-identical in
# every arm, so a detector judging only the requirement must score every arm the
# same -- which makes "no separation" a meaningful reading rather than noise.
_RUBRIC = """\
Then rate, on a 0-10 scale, how much is still unsettled once you have read \
everything above:

  0  the requirement plus the code you were shown fully determine what this \
function must return on every input
  10 the behaviour on ordinary inputs is genuinely open, and two reasonable \
programmers here would ship functions that disagree

End your answer with exactly these two lines and nothing after them:
UNSETTLED: <a number from 0 to 10>
DECISION: ASK
(or DECISION: PROCEED)
"""

NAIVE = """{module}Here is the requirement for a function you have been asked to add:

{prompt}

In two or three sentences, say what this function will have to do.

""" + _RUBRIC

GROUNDED = """{module}Here is the requirement for a function you have been asked to add:

{prompt}

Work through this in order.

1. List the behavioural decisions this requirement leaves open -- the points where two reasonable programmers could write functions that return different results on some input. Name the concrete input that would tell them apart.
2. Take each one in turn and say whether the code you were shown already settles it. If it does, quote the line that settles it. If you were shown no code, or the code does not bear on that decision, say so plainly.
3. Count only the decisions left unsettled by BOTH the requirement and the code. A decision the code settles is not unsettled, even if the requirement is silent on it.

""" + _RUBRIC

MODULE = "You are working in this existing module:\n\n```python\n{context}```\n\n"

VARIANTS = {"naive": NAIVE, "grounded": GROUNDED}

_DECIDE = re.compile(r"DECISION:\s*\**\s*(ASK|PROCEED)", re.I)
_SCORE = re.compile(r"UNSETTLED:\s*\**\s*(\d+(?:\.\d+)?)", re.I)


def parse(text: str) -> tuple[str | None, float | None]:
    hits = _DECIDE.findall(text)
    decision = hits[-1].upper() if hits else None
    if decision is None:
        # A model that ignored the format still usually ends on the verdict.
        tail = text[-200:].upper()
        a, p = tail.rfind("ASK"), tail.rfind("PROCEED")
        if a != p:
            decision = "ASK" if a > p else "PROCEED"
    shits = _SCORE.findall(text)
    score = None
    if shits:
        try:
            score = max(0.0, min(10.0, float(shits[-1])))
        except ValueError:
            score = None
    return decision, score


class Meter:
    def __init__(self) -> None:
        self.usage = Usage()
        self.errors = 0
        self.unparsed = 0
        self._lock = threading.Lock()

    def add(self, u: Usage) -> None:
        with self._lock:
            self.usage.add(u)

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)


def ask_once(backend, task: Task, arm: str, variant: str, idx: int,
             temperature: float, meter: Meter):
    ctx = context_for(task.id, arm)
    module = "" if ctx is None else MODULE.format(context=ctx)
    prompt = VARIANTS[variant].format(module=module, prompt=task.prompt_ambiguous)
    try:
        r = backend.complete([{"role": "user", "content": prompt}],
                             system=SYSTEM, temperature=temperature,
                             max_tokens=700, seed=3000 + idx)
    except Exception:
        meter.bump("errors")
        return None, None, ""
    meter.add(r.usage)
    d, s = parse(r.text)
    if d is None or s is None:
        meter.bump("unparsed")
    return d, s, r.text


def majority(votes: list[str]) -> str | None:
    if not votes:
        return None
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return "ASK"  # a split panel is not evidence that the repo settled it
    return c[0][0]


def paired_rank(per_task, variant, high_arm, low_arm, tasks) -> tuple[float, int]:
    """Fraction of tasks where high_arm scores strictly above low_arm.

    Ties count 0.5. This is a within-task AUROC: a detector with any fixed
    global offset -- always cautious, always confident -- scores 0.5.
    """
    wins = 0.0
    n = 0
    for t in tasks:
        a = per_task[t.id][variant][high_arm]["score"]
        b = per_task[t.id][variant][low_arm]["score"]
        if a is None or b is None:
            continue
        n += 1
        wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return (wins / n if n else float("nan")), n


TRUTH = {"bare": "ASK", "distractor": "ASK", "convention": "PROCEED",
         "diluted": "PROCEED", "anti": None, "prose": "PROCEED",
         "conflict": "ASK", "conflict_r": "ASK"}
# (should score higher, should score lower). The first is the tight one: same
# code volume, same requirement, only the content of the siblings differs.
PAIRS = [("distractor", "convention"), ("distractor", "diluted"),
         ("bare", "convention"), ("anti", "convention")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5-20251001", "qwen-plus"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/pilot_ask_from_context.json")
    args = ap.parse_args()

    tasks = load_tasks()
    out: dict[str, Any] = {"k": args.k, "temperature": args.temperature,
                           "arms": list(ARMS), "models": {}}

    for model in args.models:
        backend = make_backend("openai:" + model)
        meter = Meter()
        t0 = time.time()
        jobs = [(t, arm, v, i) for t in tasks for arm in ARMS
                for v in VARIANTS for i in range(args.k)]
        print(f"{model}: {len(jobs)} calls ...", flush=True)

        votes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        texts: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(ask_once, backend, t, arm, v, i, args.temperature,
                              meter): (t.id, arm, v) for t, arm, v, i in jobs}
            for fut in as_completed(futs):
                d, s, text = fut.result()
                key = futs[fut]
                if d:
                    votes[key].append(d)
                if s is not None:
                    scores[key].append(s)
                    texts[key].append(text)

        per_task: dict[str, Any] = {}
        for t in tasks:
            per_task[t.id] = {
                v: {arm: {
                    "decision": majority(votes[(t.id, arm, v)]),
                    "score": (statistics.mean(scores[(t.id, arm, v)])
                              if scores[(t.id, arm, v)] else None),
                    "n": len(scores[(t.id, arm, v)]),
                } for arm in ARMS} for v in VARIANTS}

        summary: dict[str, Any] = {}
        for v in VARIANTS:
            rates = {}
            for arm in ARMS:
                cells = [per_task[t.id][v][arm] for t in tasks]
                dec = [c["decision"] for c in cells if c["decision"]]
                sc = [c["score"] for c in cells if c["score"] is not None]
                rates[arm] = {
                    "ask_rate": round(sum(d == "ASK" for d in dec) / max(1, len(dec)), 3),
                    "mean_score": round(statistics.mean(sc), 2) if sc else None,
                    "n": len(dec),
                }
            paired = {}
            for hi, lo in PAIRS:
                auc, n = paired_rank(per_task, v, hi, lo, tasks)
                paired[f"{hi}>{lo}"] = {"auc": round(auc, 3), "n": n}
            summary[v] = {"per_arm": rates, "paired": paired}

        first = tasks[0].id
        transcripts = {arm: (texts[(first, arm, "grounded")][:1] or [""])[0]
                       for arm in ARMS}

        out["models"][model] = {
            "summary": summary, "per_task": per_task,
            "calls": meter.usage.calls,
            "cost_usd": round(meter.usage.cost_usd(model), 5),
            "errors": meter.errors, "unparsed": meter.unparsed,
            "wall_seconds": round(time.time() - t0, 1),
            "transcripts": transcripts,
        }
        print(f"  {time.time() - t0:.0f}s, ${out['models'][model]['cost_usd']:.4f}, "
              f"{meter.errors} errors, {meter.unparsed} unparsed")

    print("\n" + "=" * 84)
    print("Mean 'how much is still unsettled' score, 0-10, over 12 tasks")
    print("=" * 84)
    print(f"{'model':<26}{'detector':<11}" + "".join(f"{a[:10]:>11}" for a in ARMS))
    print("-" * 84)
    for model, d in out["models"].items():
        for v in VARIANTS:
            cells = "".join(f"{d['summary'][v]['per_arm'][a]['mean_score']:>11.2f}"
                            for a in ARMS)
            print(f"{model if v == 'naive' else '':<26}{v:<11}{cells}")
        print("-" * 84)
    print("want: distractor high, convention and diluted low.")
    print("anti is the blind spot -- if it scores as settled as convention, a")
    print("wrong repository convention is followed without a murmur.")

    print("\n" + "=" * 84)
    print("Within-task ranking (ties 0.5). 0.50 = the context made no difference.")
    print("=" * 84)
    head = "".join(f"{hi[:6]}>{lo[:6]}".rjust(18) for hi, lo in PAIRS)
    print(f"{'model':<26}{'detector':<11}{head}")
    print("-" * 84)
    for model, d in out["models"].items():
        for v in VARIANTS:
            cells = "".join(f"{d['summary'][v]['paired'][f'{hi}>{lo}']['auc']:>18.2f}"
                            for hi, lo in PAIRS)
            print(f"{model if v == 'naive' else '':<26}{v:<11}{cells}")
        print("-" * 84)

    print("\n" + "=" * 84)
    print("Binary ASK rate at the model's own threshold -- the bias the ranking removes")
    print("=" * 84)
    print(f"{'model':<26}{'detector':<11}" + "".join(f"{a[:10]:>11}" for a in ARMS))
    print("-" * 84)
    for model, d in out["models"].items():
        for v in VARIANTS:
            cells = "".join(f"{d['summary'][v]['per_arm'][a]['ask_rate']:>11.2f}"
                            for a in ARMS)
            print(f"{model if v == 'naive' else '':<26}{v:<11}{cells}")
        print("-" * 84)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
