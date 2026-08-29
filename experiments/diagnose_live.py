"""Why do the live results look nothing like the fixture results?

Two candidate explanations, and they have different consequences:

  A. sampling diversity  -- a real model at T=1.0 may emit the same behaviour
     six times, leaving no disagreement for any signal to read
  B. candidate coverage  -- the sampled set may not contain the reference
     behaviour at all, in which case no question can rescue the task

This prints both, per task, for a live sample file next to the fixtures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from askoract.bse import compute_bse
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.harness import CALIB_SEED
from bench.schema import load_tasks


def profile(task, sources):
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    m = run_candidates(sources, probes, task.entry_point)
    ref = run_candidates([task.reference], probes, task.entry_point)
    res = compute_bse(m)
    rows = m.valid_rows()
    covered = any(m.tokens[i] == ref.tokens[0] for i in rows)
    majority_right = False
    if res.classes:
        top_id = res.majority_class[0]
        idx = m.candidate_ids.index(top_id)
        majority_right = m.tokens[idx] == ref.tokens[0]
    return {"n_classes": res.n_classes, "bse": res.bse, "invalid": res.n_invalid,
            "covered": covered, "majority_right": majority_right,
            "n_valid": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="results/live/gpt-4o-mini.json")
    args = ap.parse_args()

    live = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    print(f"model={live['model']}  k={live['k']}  T={live['temperature']}\n")

    tasks = load_tasks()
    hdr = (f"{'task':<22}{'diff':<6}"
           f"{'fixture':>22}{'live (ambiguous)':>26}")
    print(hdr)
    print(f"{'':<28}{'classes  cov  majOK':>22}{'classes  cov  majOK  BSE':>26}")
    print("-" * 76)

    agg = {"live_covered": 0, "live_majority_right": 0, "live_classes": 0,
           "fix_covered": 0, "fix_classes": 0, "n": 0}

    for t in tasks:
        f = profile(t, t.candidates("ambiguous"))
        lv = profile(t, live["samples"][t.id]["ambiguous"]["sources"])
        agg["n"] += 1
        agg["live_covered"] += int(lv["covered"])
        agg["live_majority_right"] += int(lv["majority_right"])
        agg["live_classes"] += lv["n_classes"]
        agg["fix_covered"] += int(f["covered"])
        agg["fix_classes"] += f["n_classes"]
        print(f"{t.id:<22}{t.difficulty:<6}"
              f"{f['n_classes']:>7}{str(f['covered']):>6}{str(f['majority_right']):>7}"
              f"{lv['n_classes']:>11}{str(lv['covered']):>6}"
              f"{str(lv['majority_right']):>7}{lv['bse']:>7.3f}")

    n = agg["n"]
    print()
    print(f"mean behavioural classes per task   fixture {agg['fix_classes'] / n:.2f}"
          f"   live {agg['live_classes'] / n:.2f}")
    print(f"reference behaviour present in set  fixture {agg['fix_covered']}/{n}"
          f"        live {agg['live_covered']}/{n}")
    print(f"majority already correct                              "
          f"live {agg['live_majority_right']}/{n}")
    print()
    print("A question can only help on tasks that are covered but whose majority")
    print("is wrong. Everything else is either already right or unreachable.")
    reachable = agg["live_covered"] - agg["live_majority_right"]
    print(f"  -> reachable by asking, live: {reachable}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
