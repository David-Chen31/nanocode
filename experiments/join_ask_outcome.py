"""Does the unsettled score predict actually getting it wrong?

Separating `distractor` from `convention` is necessary but not sufficient. An
ask-signal earns its keep only if firing it correlates with the code that was
about to be wrong. Round 4 already measured, by execution against the reference,
whether each (model, task, arm) cell produced the right behaviour. This joins
that outcome to the score from `pilot_ask_from_context.py`.

No new model calls: both sides are on disk.

  AUROC   P(score on a cell that came out wrong > score on a cell that came out
          right), ties 0.5. 0.50 means the score knows nothing about outcomes.
  budget  sort the cells by score, ask about the top N, and count what that buys:
          wrong cells caught, against right cells interrupted for nothing.

The `anti` arm is included and is the hardest part of the distribution: it is
wrong almost everywhere, and nothing on the page says so.

    py -3 experiments/join_ask_outcome.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTCOME_RUNS = ["results/pilot_repo_context.json",
                "results/pilot_repo_context4.json",
                "results/pilot_repo_context5.json"]
ASK = "results/pilot_ask_from_context.json"


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(a random positive scores above a random negative), ties 0.5."""
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> int:
    outcome: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for p in OUTCOME_RUNS:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for model, x in d["models"].items():
            for tid, arms in x["per_task"].items():
                for arm, r in arms.items():
                    outcome[(model, tid, arm)].append(bool(r["majority_right"]))

    ask = json.loads(Path(ASK).read_text(encoding="utf-8"))
    variants = ["naive", "grounded"]

    print("=" * 84)
    print("Does 'how much is unsettled' predict the code coming out wrong?")
    print("=" * 84)

    rows: dict[str, list[tuple[str, str, str, float, bool]]] = {v: [] for v in variants}
    for model, x in ask["models"].items():
        for tid, per_v in x["per_task"].items():
            for v in variants:
                for arm, cell in per_v[v].items():
                    outs = outcome.get((model, tid, arm))
                    if cell["score"] is None or not outs:
                        continue
                    # Wrong = the behaviour most samples agreed on failed
                    # differential testing in most round-4 runs.
                    wrong = sum(outs) * 2 <= len(outs)
                    rows[v].append((model, tid, arm, cell["score"], wrong))

    print(f"\n{'detector':<11}{'scope':<28}{'AUROC':>8}{'wrong':>8}{'right':>8}{'cells':>8}")
    print("-" * 84)
    for v in variants:
        models = sorted({r[0] for r in rows[v]})
        for scope in ["ALL POOLED"] + models:
            sub = rows[v] if scope == "ALL POOLED" else [r for r in rows[v] if r[0] == scope]
            pos = [r[3] for r in sub if r[4]]
            neg = [r[3] for r in sub if not r[4]]
            print(f"{v if scope == 'ALL POOLED' else '':<11}{scope:<28}"
                  f"{auroc(pos, neg):>8.3f}{len(pos):>8}{len(neg):>8}{len(sub):>8}")
        print("-" * 84)

    # The same question with `anti` removed. anti is wrong by construction and
    # invisible by construction, so it can only drag the AUROC down; showing both
    # separates "the score is weak" from "the score is blind to anti".
    print(f"\n{'detector':<11}{'subset':<28}{'AUROC':>8}{'wrong':>8}{'right':>8}")
    print("-" * 84)
    for v in variants:
        for label, keep in [("all five arms", lambda a: True),
                            ("without anti", lambda a: a != "anti"),
                            ("only anti", lambda a: a == "anti")]:
            sub = [r for r in rows[v] if keep(r[2])]
            pos = [r[3] for r in sub if r[4]]
            neg = [r[3] for r in sub if not r[4]]
            print(f"{v if label == 'all five arms' else '':<11}{label:<28}"
                  f"{auroc(pos, neg):>8.3f}{len(pos):>8}{len(neg):>8}")
        print("-" * 84)

    # What a budget actually buys.
    print("\n" + "=" * 84)
    print("Ask about the top-N highest-scoring cells (pooled, all arms)")
    print("=" * 84)
    print(f"{'detector':<11}{'budget':<10}{'wrong caught':>16}{'right interrupted':>20}"
          f"{'precision':>12}")
    print("-" * 84)
    for v in variants:
        ranked = sorted(rows[v], key=lambda r: -r[3])
        total_wrong = sum(1 for r in ranked if r[4])
        for frac in (0.1, 0.2, 0.3, 0.5):
            n = max(1, int(len(ranked) * frac))
            top = ranked[:n]
            caught = sum(1 for r in top if r[4])
            wasted = n - caught
            base = total_wrong / len(ranked)
            print(f"{v if frac == 0.1 else '':<11}{f'top {int(frac*100)}%':<10}"
                  f"{caught:>10}/{total_wrong:<5}{wasted:>20}"
                  f"{caught / n:>11.2f}  (base rate {base:.2f})")
        print("-" * 84)

    # Per-arm outcome vs score, so the shape is visible rather than summarised.
    print("\n" + "=" * 84)
    print("Per arm: mean score against how often the code was actually right")
    print("=" * 84)
    print(f"{'detector':<11}{'arm':<14}{'mean score':>12}{'cells right':>14}{'cells':>8}")
    print("-" * 84)
    for v in variants:
        arms = ["bare", "distractor", "convention", "diluted", "anti"]
        for arm in arms:
            sub = [r for r in rows[v] if r[2] == arm]
            if not sub:
                continue
            ms = sum(r[3] for r in sub) / len(sub)
            right = sum(1 for r in sub if not r[4])
            print(f"{v if arm == 'bare' else '':<11}{arm:<14}{ms:>12.2f}"
                  f"{right:>10}/{len(sub):<3}{len(sub):>8}")
        print("-" * 84)
    print("A usable signal would put anti near the top of the score column.")
    print("It cannot: nothing on the page distinguishes a right convention from")
    print("a wrong one. That is the blind spot, stated as a number.")

    out: dict[str, Any] = {v: [{"model": m, "task": t, "arm": a, "score": s,
                                "wrong": w} for m, t, a, s, w in rows[v]]
                           for v in variants}
    Path("results/join_ask_outcome.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwritten to results/join_ask_outcome.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
