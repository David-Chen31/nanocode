"""Read results/ablation.json and say what each component was worth.

WHY THE BOOTSTRAP RESAMPLES TASKS AND NOT TRAJECTORIES

Three repetitions of the same task are not three independent observations. They
share a prompt, a repository, and whatever makes that task easy or hard, so
resampling trajectories would treat 36 correlated runs as 36 independent ones
and produce intervals far too narrow. The task is the unit that was sampled, so
the task is the unit that gets resampled -- a cluster bootstrap, with all of a
task's runs moving together.

With 12 tasks the intervals are wide no matter what. That is the honest state
of the evidence and the reason every difference below is reported with one.

    py -3 experiments/analyse_ablation.py
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


def load(path: str) -> list[dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    # Infrastructure failures are excluded, but they are counted out loud in the
    # run log rather than quietly dropped here.
    return [r for r in d["rows"] if r.get("outcome") != "error"]


def by_task(rows: list[dict], cond: str, f: Callable[[dict], float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["condition"] == cond:
            out[r["task"]].append(f(r))
    return out


def cluster_boot(a: dict[str, list[float]], b: dict[str, list[float]] | None,
                 n: int = BOOTS) -> tuple[float, float, float]:
    """Point estimate and a 95% interval, resampling whole tasks.

    When `b` is given the statistic is the paired difference mean(a) - mean(b)
    computed per task, which removes the task's own difficulty from the
    comparison entirely.
    """
    tasks = sorted(set(a) & set(b)) if b is not None else sorted(a)
    if not tasks:
        return float("nan"), float("nan"), float("nan")

    def stat(sample: list[str]) -> float:
        vals = []
        for t in sample:
            ma = statistics.mean(a[t])
            vals.append(ma - statistics.mean(b[t]) if b is not None else ma)
        return statistics.mean(vals)

    point = stat(tasks)
    rng = random.Random(SEED)
    draws = sorted(stat([rng.choice(tasks) for _ in tasks]) for _ in range(n))
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def fmt(p: float, lo: float, hi: float, pct: bool = True) -> str:
    m = 100 if pct else 1
    unit = "" if pct else ""
    return f"{p * m:+6.1f}{unit}  [{lo * m:+6.1f}, {hi * m:+6.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/ablation.json")
    ap.add_argument("--ceiling", default="results/ablation_ceiling.json")
    ap.add_argument("--baseline", default="full",
                    help="Condition every other one is compared against.")
    args = ap.parse_args()

    rows = load(args.path)
    base_name = args.baseline
    conds = sorted({r["condition"] for r in rows}, key=lambda c: c != base_name)
    tasks = sorted({r["task"] for r in rows})
    print(f"{len(rows)} usable trajectories, {len(tasks)} tasks, "
          f"{len(conds)} conditions\n")

    correct = lambda r: 1.0 if r.get("correct") is True else 0.0
    finished = lambda r: 1.0 if r.get("outcome") == "finished" else 0.0
    calls = lambda r: float(r.get("n_model_calls") or 0)

    # ---- trigger rates: the ceiling on every effect -------------------------
    base = [r for r in rows if r["condition"] == base_name]
    print(f"TRIGGER RATES in {base_name} -- no ablation can exceed these")
    for label, key in (("used search", "n_search"),
                       ("malformed tool call", "bad_args"),
                       ("non-UTF-8 command output", "decode_fallback"),
                       ("a command failed", "n_failed_runs")):
        v = sum(1 for r in base if r.get(key)) / len(base) if base else float('nan')
        print(f"  {label:<26}{v:.2f}")

    # ---- levels -------------------------------------------------------------
    print(f"\n{'condition':<17}{'correct':>18}{'finished':>18}{'model calls':>18}")
    print("-" * 71)
    for c in conds:
        pc = cluster_boot(by_task(rows, c, correct), None)
        pf = cluster_boot(by_task(rows, c, finished), None)
        pk = cluster_boot(by_task(rows, c, calls), None)
        print(f"{c:<17}{pc[0]:>8.2f} [{pc[1]:.2f},{pc[2]:.2f}]"
              f"{pf[0]:>8.2f} [{pf[1]:.2f},{pf[2]:.2f}]"
              f"{pk[0]:>8.1f} [{pk[1]:.1f},{pk[2]:.1f}]")

    # ---- paired differences -------------------------------------------------
    print("\nPAIRED DIFFERENCE, full minus ablated, in percentage points")
    print("(a 95% interval containing 0 means this study cannot separate them)")
    for c in conds:
        if c == base_name:
            continue
        print(f"\n  {c}")
        for label, f in (("correct ", correct), ("finished", finished)):
            p, lo, hi = cluster_boot(by_task(rows, base_name, f), by_task(rows, c, f))
            flag = "" if lo <= 0 <= hi else "   <- excludes 0"
            print(f"    {label}  {fmt(p, lo, hi)}{flag}")
        p, lo, hi = cluster_boot(by_task(rows, base_name, calls), by_task(rows, c, calls))
        flag = "" if lo <= 0 <= hi else "   <- excludes 0"
        print(f"    calls     {p:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]{flag}")

    # ---- the ceiling --------------------------------------------------------
    cp = Path(args.ceiling)
    if cp.exists():
        crows = load(str(cp))
        c_ok = cluster_boot(by_task(crows, "full", correct), None)
        b_ok = cluster_boot(by_task(rows, base_name, correct), None)
        print("\nMANIPULATION CHECK -- is the task winnable at all?")
        print(f"  complete requirement   correct {c_ok[0]:.2f} "
              f"[{c_ok[1]:.2f},{c_ok[2]:.2f}]  (n={len(crows)})")
        print(f"  requirement with one   correct {b_ok[0]:.2f} "
              f"[{b_ok[1]:.2f},{b_ok[2]:.2f}]  (n={len(base)})")
        print("  sentence deleted")
        print("\n  The gap between these two is the headroom the ablations were")
        print("  competing for. Failures below that ceiling are the task being")
        print("  under-specified, not the agent's machinery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
