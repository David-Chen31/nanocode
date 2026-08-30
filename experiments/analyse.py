"""Read any of the sweep result files and report a panel of outcomes.

WHY THIS REPLACED A SINGLE-OUTCOME ANALYSIS

Rounds 8, 9 and 10 each pre-registered end-to-end correctness as the primary
outcome, and each returned a null on it. That looked like "the components do not
matter". It was not. A power calculation on the designs themselves says
correctness could only ever have resolved effects of roughly this size:

    round 8    12 tasks x 3 reps, base .42     32 points
    round 9    12 tasks x 3 reps, base .31     34 points
    round 10    8 tasks x 2 reps, base .94    >70 points

No component of a coding agent plausibly moves end-to-end correctness by 32
points, so those nulls were guaranteed by the design rather than discovered by
it. Round 8's own manipulation check shows why: deleting one sentence from the
requirement moved correctness 33 points, while every component moved under 10.
The task's information content dominates; the machinery does not compete with it.

The machinery instead governs how much work gets done and how runs fail. Those
are continuous, low-variance quantities, and they have an order of magnitude
more power on the same sample. That is why the only significant results in three
rounds were model calls and tokens.

READING THE OUTPUT HONESTLY

PRIMARY is what the pre-registration named. SECONDARY was recorded but not
pre-registered as primary, so it is exploratory: mechanism-motivated rather than
fished for, but exploratory all the same. With five outcomes across three
comparisons, some interval will exclude zero by chance -- treat an effect whose
interval barely clears zero as noise, and weigh the ones that clear it widely.

    py -3 experiments/analyse.py results/ablation.json --baseline full
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

BOOTS = 5000
SEED = 20260830

# The pre-registered outcome, then the mechanism-proximate ones. Each says what
# it is evidence about, because "significant" is meaningless without that.
PRIMARY: list[tuple[str, Callable[[dict], float], str]] = [
    ("correct", lambda r: 1.0 if r.get("correct") else 0.0,
     "did the change work end to end"),
]
GRADED: list[tuple[str, Callable[[dict], float], str]] = [
    ("assertions", lambda r: float(r.get("behaviour_frac") or 0.0),
     "fraction of the hidden verifier that passed"),
]
SECONDARY: list[tuple[str, Callable[[dict], float], str]] = [
    ("model calls", lambda r: float(r.get("n_model_calls") or 0),
     "how much thinking the run cost"),
    ("tool calls", lambda r: float(r.get("n_tool_calls") or 0),
     "how much acting the run cost"),
    ("commands run", lambda r: float(r.get("n_runs") or 0),
     "how often it reached for the shell"),
    ("failed commands", lambda r: float(r.get("n_failed_runs") or 0),
     "how much of that shell work was wasted"),
    ("tokens (k)", lambda r: (float(r.get("input_tokens") or 0)
                              + float(r.get("output_tokens") or 0)) / 1000.0,
     "total context cost"),
    ("crashed", lambda r: 1.0 if r.get("crashed") else 0.0,
     "did the run die instead of finishing"),
]


def load(path: str) -> tuple[list[dict[str, Any]], str]:
    rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
            if r.get("outcome") != "error"]
    # The sweeps name their independent variable differently; both are the same
    # kind of thing, so the analysis should not care which.
    key = "condition" if rows and "condition" in rows[0] else "arm"
    return rows, key


def by_task(rows, key, cond, f) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r[key] == cond:
            out[r["task"]].append(f(r))
    return out


def cluster_boot(a, b, n: int = BOOTS) -> tuple[float, float, float]:
    """Paired difference with a 95% interval, resampling whole tasks.

    Repetitions of one task share a prompt, a repository and a difficulty, so
    resampling trajectories would treat correlated runs as independent and
    report intervals far too narrow. The task was the sampling unit, so the task
    is the resampling unit.
    """
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return float("nan"), float("nan"), float("nan")

    def stat(sample: list[str]) -> float:
        return statistics.mean(statistics.mean(b[t]) - statistics.mean(a[t])
                               for t in sample)

    point = stat(tasks)
    rng = random.Random(SEED)
    draws = sorted(stat([rng.choice(tasks) for _ in tasks]) for _ in range(n))
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def _has(rows, f) -> bool:
    """Skip an outcome the sweep never recorded, rather than printing zeros."""
    return any(f(r) for r in rows)


def report(rows, key, baseline: str) -> None:
    conds = [c for c in dict.fromkeys(r[key] for r in rows) if c != baseline]
    tasks = sorted({r["task"] for r in rows})
    print(f"{len(rows)} runs, {len(tasks)} tasks, baseline = {baseline}\n")

    for label, panel in (("PRIMARY (pre-registered)", PRIMARY),
                         ("GRADED (same outcome, not thrown away as pass/fail)",
                          GRADED),
                         ("SECONDARY (exploratory: recorded, not pre-registered "
                          "as primary)", SECONDARY)):
        panel = [(n, f, w) for n, f, w in panel if _has(rows, f)]
        if not panel:
            continue
        print("=" * 78)
        print(label)
        print("=" * 78)
        for cond in conds:
            print(f"\n  {baseline} minus {cond}")
            for name, f, why in panel:
                p, lo, hi = cluster_boot(by_task(rows, key, baseline, f),
                                         by_task(rows, key, cond, f))
                mark = "" if lo <= 0 <= hi else "  <- excludes 0"
                print(f"    {name:<16}{p:+8.2f}  [{lo:+7.2f}, {hi:+7.2f}]{mark}"
                      f"{'':4}{why if mark else ''}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()
    rows, key = load(args.path)
    report(rows, key, args.baseline)
    print("Intervals are wide because the task count is small. An interval that "
          "contains 0\nmeans this study could not separate them -- not that the "
          "effect is zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
