"""Split the ask-decision in two, and find out which half is broken.

Deciding whether to ask is two jobs, not one:

  T1  enumerate the behavioural decisions this requirement leaves open
  T2  for a given decision, judge whether the surrounding code settles it

`pilot_ask_from_context.py` measures the two fused together, and the transcripts
show T1 misfiring badly: on the convention arm the detector never named the
decision the task actually turns on, and listed non-numeric input and float
precision instead. Round 3 saw the same thing -- enumeration false-alarmed on
12/12 tasks.

So measure T2 alone. Each task carries `constraints[0].discriminating_args`: an
input on which the reference and the plausible wrong reading return different
things. Handing the detector that input names the decision exactly, without
leaking which answer is right. The question becomes mechanical:

    does everything above determine what this function returns on f(<args>)?

Ground truth, and this is where the arm structure finally pays off:

  bare        OPEN     no code at all
  distractor  OPEN     code, but none of it bears on this input
  convention  SETTLED  the siblings demonstrate it
  diluted     SETTLED  the same, further away
  anti        SETTLED  the siblings demonstrate it too -- and they are wrong

`anti` is scorable here for the first time. On T2 the correct answer is SETTLED,
because the code really does settle it. Round 4 shows the resulting code is
wrong anyway. A detector answering this question perfectly still walks straight
into it, which is the sharpest way to state the blind spot: the failure is not
that the agent misjudged the repository, it is that judging the repository was
never enough.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    py -3 experiments/pilot_settled_oracle.py
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
    "You judge whether the material you have been given determines a specific "
    "piece of behaviour, or leaves it open."
)

PROMPT = """{module}Here is the requirement for a function you have been asked to add:

{prompt}

Consider this one specific call:

    {call}

Question: taken together, does the requirement and any code above determine what \
that call must return? Or could two careful programmers, both working from \
exactly this material, write functions that return different things here?

Do not tell me what the call should return. Judge only whether the material \
above pins it down. Code counts as pinning it down if it establishes a clear \
convention for the analogous situation, even if it never mentions this function.

Give one short sentence of justification -- if you answer SETTLED, name what \
pins it down -- then end with exactly this line and nothing after it:
VERDICT: SETTLED
or
VERDICT: OPEN
"""

MODULE = "You are working in this existing module:\n\n```python\n{context}```\n\n"

_VERDICT = re.compile(r"VERDICT:\s*\**\s*(SETTLED|OPEN)", re.I)

TRUTH = {"bare": "OPEN", "distractor": "OPEN", "convention": "SETTLED",
         "diluted": "SETTLED", "anti": "SETTLED", "prose": "SETTLED",
         "conflict": "OPEN", "conflict_r": "OPEN"}


def render_call(task: Task, args: list[Any]) -> str:
    return f"{task.entry_point}({', '.join(repr(a) for a in args)})"


def parse(text: str) -> str | None:
    hits = _VERDICT.findall(text)
    if hits:
        return hits[-1].upper()
    tail = text[-160:].upper()
    s, o = tail.rfind("SETTLED"), tail.rfind("OPEN")
    if s == o == -1:
        return None
    return "SETTLED" if s > o else "OPEN"


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


def ask_once(backend, task: Task, arm: str, call: str, idx: int,
             temperature: float, meter: Meter):
    ctx = context_for(task.id, arm)
    module = "" if ctx is None else MODULE.format(context=ctx)
    prompt = PROMPT.format(module=module, prompt=task.prompt_ambiguous, call=call)
    try:
        r = backend.complete([{"role": "user", "content": prompt}],
                             system=SYSTEM, temperature=temperature,
                             max_tokens=300, seed=4000 + idx)
    except Exception:
        meter.bump("errors")
        return None, ""
    meter.add(r.usage)
    v = parse(r.text)
    if v is None:
        meter.bump("unparsed")
    return v, r.text


def majority(votes: list[str]) -> str | None:
    if not votes:
        return None
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return "OPEN"  # a split panel is not evidence that the repo settled it
    return c[0][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5-20251001", "qwen-plus"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/pilot_settled_oracle.json")
    args = ap.parse_args()

    tasks = [t for t in load_tasks()
             if t.constraints and t.constraints[0].discriminating_args]
    calls = {t.id: render_call(t, t.constraints[0].discriminating_args[0])
             for t in tasks}
    print(f"{len(tasks)} tasks carry a discriminating input:")
    for t in tasks:
        print(f"  {t.id:<24}{calls[t.id]}")

    out: dict[str, Any] = {"k": args.k, "temperature": args.temperature,
                           "calls": calls, "models": {}}

    for model in args.models:
        backend = make_backend("openai:" + model)
        meter = Meter()
        t0 = time.time()
        jobs = [(t, arm, i) for t in tasks for arm in ARMS for i in range(args.k)]
        print(f"\n{model}: {len(jobs)} calls ...", flush=True)

        votes: dict[tuple[str, str], list[str]] = defaultdict(list)
        texts: dict[tuple[str, str], list[str]] = defaultdict(list)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(ask_once, backend, t, arm, calls[t.id], i,
                              args.temperature, meter): (t.id, arm)
                    for t, arm, i in jobs}
            for fut in as_completed(futs):
                v, text = fut.result()
                if v:
                    votes[futs[fut]].append(v)
                    texts[futs[fut]].append(text)

        per_task = {t.id: {arm: {
            "verdict": majority(votes[(t.id, arm)]),
            "settled_rate": (statistics.mean(
                [x == "SETTLED" for x in votes[(t.id, arm)]])
                if votes[(t.id, arm)] else None),
        } for arm in ARMS} for t in tasks}

        summary = {}
        for arm in ARMS:
            cells = [per_task[t.id][arm] for t in tasks]
            v = [c["verdict"] for c in cells if c["verdict"]]
            r = [c["settled_rate"] for c in cells if c["settled_rate"] is not None]
            summary[arm] = {
                "settled_majority": sum(x == "SETTLED" for x in v),
                "mean_settled_rate": round(statistics.mean(r), 3) if r else None,
                "correct": sum(x == TRUTH[arm] for x in v),
                "n": len(v),
            }

        out["models"][model] = {
            "summary": summary, "per_task": per_task,
            "n_calls": meter.usage.calls,
            "cost_usd": round(meter.usage.cost_usd(model), 5),
            "errors": meter.errors, "unparsed": meter.unparsed,
            "wall_seconds": round(time.time() - t0, 1),
            "transcripts": {arm: (texts[(tasks[0].id, arm)][:1] or [""])[0]
                            for arm in ARMS},
        }
        print(f"  {time.time() - t0:.0f}s, ${out['models'][model]['cost_usd']:.4f}, "
              f"{meter.errors} errors, {meter.unparsed} unparsed")

    n = len(tasks)
    print("\n" + "=" * 84)
    print("T2 alone: handed the right question, can it judge whether the code answers it?")
    print("=" * 84)
    print(f"{'model':<28}" + "".join(f"{a[:10]:>11}" for a in ARMS) + f"{'correct':>10}")
    print(f"{'ground truth':<28}" + "".join(f"{TRUTH[a][:10]:>11}" for a in ARMS))
    print("-" * 84)
    tot = defaultdict(int)
    for model, d in out["models"].items():
        cells = "".join(f"{d['summary'][a]['settled_majority']:>8}/{n:<2}" for a in ARMS)
        corr = sum(d["summary"][a]["correct"] for a in ARMS)
        for a in ARMS:
            tot[a] += d["summary"][a]["correct"]
        print(f"{model:<28}{cells}{corr:>7}/{n * len(ARMS):<3}")
    print("-" * 84)
    print(f"{'per-arm accuracy':<28}"
          + "".join(f"{tot[a] / (n * len(out['models'])):>11.2f}" for a in ARMS))
    print("\ncells are tasks judged SETTLED, out of", n)
    print("anti is scorable here: the code does settle it, so SETTLED is correct.")
    print("Round 4 says the code that follows is wrong anyway -- getting this")
    print("question right is not enough to avoid the failure.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
