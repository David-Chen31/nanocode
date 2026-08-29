"""Does the channel matter, or only the information?

arXiv 2607.27250 ablated natural-language context files (AGENTS.md / CLAUDE.md)
on real repositories and reported a null on correctness, having ruled out poor
file quality. Round 4 here carried the same kind of information in sibling code
and moved correctness by +3.1/12. This compares the two channels inside one
harness, on one task set, with matched code volume:

  distractor  siblings that resolve nothing        -- the information is absent
  prose       the rule stated in a module docstring -- present, asserted
  convention  the rule demonstrated in siblings     -- present, shown

Three readings, in increasing order of how much they'd change the story:

  1  the means. prose ~ distractor says the channel is what matters.
  2  per task. Even if the means agree, prose and code may rescue *different*
     tasks -- some rules are easier to say than to show, and vice versa. That
     would mean the two channels are complements, not substitutes.
  3  behavioural classes. Round 4 found collapse persists in every arm (1.0-1.4
     classes); if prose instead raises class count, prose creates hesitation
     where code creates commitment, which is a different mechanism entirely.

    py -3 experiments/analyse_prose.py [results/pilot_prose.json ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.schema import load_tasks

ORDER = ["bare", "distractor", "prose", "convention", "diluted", "anti"]


def main() -> int:
    paths = sys.argv[1:] or ["results/pilot_prose.json"]
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    tasks = [t.id for t in load_tasks()]

    obs: dict[str, list[int]] = defaultdict(list)
    cls: dict[str, list[float]] = defaultdict(list)
    cov: dict[str, list[int]] = defaultdict(list)
    per_task: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for d in runs:
        for model, x in d["models"].items():
            for arm in ORDER:
                s = x["summary"].get(arm)
                if s:
                    obs[arm].append(s["majority_right"])
                    cls[arm].append(s["mean_classes"])
                    cov[arm].append(s["coverage"])
            for tid, arms in x["per_task"].items():
                for arm, r in arms.items():
                    per_task[tid][arm].append(bool(r["majority_right"]))

    arms = [a for a in ORDER if obs[a]]
    n = len(tasks)

    print("=" * 78)
    print("Same information, two channels")
    print("=" * 78)
    print(f"{'arm':<14}{'majority right /12':>20}{'coverage':>11}{'classes':>10}{'obs':>6}")
    print("-" * 78)
    base = statistics.mean(obs["distractor"]) if obs["distractor"] else 0.0
    for a in arms:
        m = statistics.mean(obs[a])
        delta = m - base
        tag = "" if a == "distractor" else f"  ({delta:+.1f} vs distractor)"
        print(f"{a:<14}{m:>15.1f}/12{statistics.mean(cov[a]):>10.1f}"
              f"{statistics.mean(cls[a]):>10.2f}{len(obs[a]):>6}{tag}")
    print("-" * 78)

    if obs["prose"] and obs["convention"]:
        p, c = statistics.mean(obs["prose"]), statistics.mean(obs["convention"])
        print(f"\nprose {p:.1f}  vs  convention {c:.1f}  vs  distractor {base:.1f}")
        if abs(p - base) < 0.75 and c - base > 1.5:
            print("=> The channel matters. Stating the rule does nothing; showing it works.")
            print("   That is a controlled explanation of the published AGENTS.md null.")
        elif abs(p - c) < 0.75 and p - base > 1.5:
            print("=> The channel does NOT matter -- only whether the information is")
            print("   present. C1 weakens to 'context helps', and 2607.27250's null")
            print("   needs a different explanation (their tasks may not turn on a")
            print("   missing behavioural constraint at all).")
        else:
            print("=> Intermediate. Report the gap, do not round it to either story.")

    # Reading 2: do the two channels rescue the same tasks?
    print("\n" + "=" * 78)
    print("Per task: which channel rescues what  (share of runs majority-right)")
    print("=" * 78)
    print(f"{'task':<24}" + "".join(f"{a[:10]:>11}" for a in arms))
    print("-" * 78)

    def frac(tid: str, arm: str) -> float | None:
        v = per_task[tid].get(arm)
        return sum(v) / len(v) if v else None

    only_prose, only_conv, both, neither = [], [], [], []
    for tid in tasks:
        cells = "".join(
            (f"{frac(tid, a):>11.2f}" if frac(tid, a) is not None else f"{'-':>11}")
            for a in arms)
        print(f"{tid:<24}{cells}")
        d, p, c = frac(tid, "distractor"), frac(tid, "prose"), frac(tid, "convention")
        if None in (d, p, c):
            continue
        gp, gc = (p or 0) - (d or 0), (c or 0) - (d or 0)
        if gp > 0.34 and gc > 0.34:
            both.append(tid)
        elif gp > 0.34:
            only_prose.append(tid)
        elif gc > 0.34:
            only_conv.append(tid)
        elif (c or 0) < 0.34 and (p or 0) < 0.34:
            neither.append(tid)
    print("-" * 78)
    print(f"rescued by BOTH channels      {len(both):>2}   {', '.join(both) or '-'}")
    print(f"rescued by CODE only          {len(only_conv):>2}   {', '.join(only_conv) or '-'}")
    print(f"rescued by PROSE only         {len(only_prose):>2}   {', '.join(only_prose) or '-'}")
    print(f"never right in either         {len(neither):>2}   {', '.join(neither) or '-'}")
    print("\nCODE-only tasks are the evidence that demonstration carries something")
    print("assertion does not. PROSE-only tasks, if any, are the reverse and are")
    print("just as interesting -- some rules may be easier to say than to show.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
