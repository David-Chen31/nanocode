"""Pilot B: split the job into the part the model is good at and the part it is not.

The grounded pilot showed a clean decomposition. Asked what its own code decided,
a model names the deleted constraint 10-12 times out of 12 -- better than the
execution-based probe ranking. But it also fires on every fully specified prompt,
and inspecting those false alarms shows why: it lists properties its code has,
including ones the task states outright. "Sorts the list in place" is offered as
an unspecified decision on a prompt whose text is "Sort xs in place and return
None".

So the failure is not introspection. It is entailment: deciding whether a short
claim is settled by a short text, while also generating. Asked inline, as one
instruction among several, it collapses. Asked on its own, one claim at a time,
it is a much smaller problem.

    enumerate what my code decided   ->  model is good at this
    filter to what the spec omits    ->  model fails at this *inline*
                                         does it succeed when isolated?

    python experiments/pilot_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import Usage, make_backend
from bench.schema import load_tasks

VARIANTS = ("ambiguous", "full")

ENTAIL = (
    "TASK TEXT:\n{prompt}\n\n"
    "CLAIM: {item}\n\n"
    "Does the TASK TEXT above already settle the behaviour the CLAIM describes, "
    "either by stating it or by ruling it out? Answer only about what the text "
    "says -- not about what would be sensible. Reply with exactly one word: "
    "SETTLED or OPEN."
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


def entail(backend, prompt: str, item: str, meter: Meter) -> bool | None:
    """True when the model says the task text already settles the claim."""
    try:
        r = backend.text(ENTAIL.format(prompt=prompt, item=item),
                         temperature=0.0, max_tokens=8)
    except Exception:
        meter.fail()
        return None
    meter.add(r.usage)
    return r.text.strip().upper().startswith("SETTLED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5-20251001", "qwen-plus"])
    ap.add_argument("--filters", nargs="+", default=["self", "gpt-4o"],
                    help="'self' = the generator filters its own items")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/pilot_pipeline.json")
    args = ap.parse_args()

    tasks = load_tasks()
    grounded = json.loads(Path("results/pilot_grounded.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"models": {}}

    for gen in args.generators:
        if gen not in grounded["models"]:
            print(f"skipping {gen}: no grounded items on disk")
            continue
        per_task = grounded["models"][gen]["per_task"]

        for filt in args.filters:
            filter_model = gen if filt == "self" else filt
            backend = make_backend("openai:" + filter_model)
            meter = Meter()
            t0 = time.time()

            jobs = [(t, v, item)
                    for t in tasks for v in VARIANTS
                    for item in per_task[t.id][v]["items"]]
            label = f"{gen}  filtered by {filt}"
            print(f"{label}: {len(jobs)} entailment calls ...", flush=True)

            verdict: dict[tuple[str, str, str], bool | None] = {}
            with ThreadPoolExecutor(args.workers) as ex:
                futs = {ex.submit(entail, backend, t.prompt(v), item, meter):
                        (t.id, v, item) for t, v, item in jobs}
                for fut in as_completed(futs):
                    verdict[futs[fut]] = fut.result()

            kept = {(t.id, v): [item for item in per_task[t.id][v]["items"]
                                if verdict.get((t.id, v, item)) is False]
                    for t in tasks for v in VARIANTS}

            amb_fire = sum(1 for t in tasks if kept[(t.id, "ambiguous")])
            full_fire = sum(1 for t in tasks if kept[(t.id, "full")])
            n = len(tasks)
            # A task still localises if the item the judge matched survived the filter.
            still_loc = 0
            for t in tasks:
                if not per_task[t.id]["localisation_hit"]:
                    continue
                if kept[(t.id, "ambiguous")]:
                    still_loc += 1

            out["models"][label] = {
                "generator": gen, "filter": filter_model,
                "detection_rate": amb_fire / n,
                "false_alarm_rate": full_fire / n,
                "mean_items_ambiguous": sum(len(kept[(t.id, "ambiguous")])
                                            for t in tasks) / n,
                "mean_items_full": sum(len(kept[(t.id, "full")]) for t in tasks) / n,
                "localisation_retained": still_loc / n,
                "calls": meter.usage.calls,
                "cost_usd": round(meter.usage.cost_usd(filter_model), 5),
                "errors": meter.errors,
                "kept": {f"{tid}|{v}": items for (tid, v), items in kept.items()},
            }
            print(f"  {time.time() - t0:.0f}s, "
                  f"${out['models'][label]['cost_usd']:.4f}, {meter.errors} errors")

    n = len(tasks)
    print("\n" + "=" * 84)
    print("enumerate (grounded)  ->  filter by entailment against the spec text")
    print("=" * 84)
    print(f"{'pipeline':<44}{'fires on':>10}{'fires on':>11}{'localises':>11}")
    print(f"{'':<44}{'ambiguous':>10}{'FULL':>11}{'':>11}")
    print("-" * 84)
    for gen in args.generators:
        if gen not in grounded["models"]:
            continue
        g = grounded["models"][gen]
        print(f"{gen + '  (no filter)':<44}{g['detection_rate'] * n:>7.0f}/{n}"
              f"{g['false_alarm_rate'] * n:>8.0f}/{n}{g['localisation'] * n:>8.0f}/{n}")
        for filt in args.filters:
            label = f"{gen}  filtered by {filt}"
            if label not in out["models"]:
                continue
            d = out["models"][label]
            print(f"{'   + filter: ' + filt:<44}{d['detection_rate'] * n:>7.0f}/{n}"
                  f"{d['false_alarm_rate'] * n:>8.0f}/{n}"
                  f"{d['localisation_retained'] * n:>8.0f}/{n}")

    print("\nfor reference: every output-distribution signal fired on ~0/12 of both;")
    print("abstract text prompts fired on 12/12 of both and localised 1-2/12.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
