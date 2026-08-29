"""Why does a fixed budget spent at random beat a budget ranked by BSE?

Reads results/experiments.json and lines up, per task: the signal score, whether
the majority vote was already right, and whether a question changed anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

res = json.loads(Path("results/experiments.json").read_text(encoding="utf-8"))
h1, h2 = res["h1"], res["h2"]
outcomes = h2["outcomes"]

rows = []
for tid, oc in outcomes.items():
    bse = h1["bse"]["per_task"][tid]["ambiguous"]
    diff = h1["bse"]["per_task"][tid]["difficulty"]
    no_ask, with_ask = oc["success"][0], oc["success"][-1]
    rows.append((tid, diff, bse, no_ask, with_ask, with_ask and not no_ask))

rows.sort(key=lambda r: -r[2])

print(f"{'task':<22}{'diff':<6}{'BSE':>7}{'ok w/o ask':>12}{'ok w/ ask':>11}"
      f"{'question helps':>16}")
print("-" * 76)
for tid, diff, bse, a, b, helps in rows:
    print(f"{tid:<22}{diff:<6}{bse:>7.3f}{str(a):>12}{str(b):>11}{str(helps):>16}")

helped = [r for r in rows if r[5]]
wasted = [r for r in rows if not r[5]]
print()
print(f"questions that change the outcome : {len(helped)}/{len(rows)}")
print(f"  mean BSE where a question helps : "
      f"{sum(r[2] for r in helped) / max(1, len(helped)):.3f}")
print(f"  mean BSE where it is wasted     : "
      f"{sum(r[2] for r in wasted) / max(1, len(wasted)):.3f}")

# Rank correlation between the signal and the thing the budget actually needs.
def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))

print()
print(f"AUROC of BSE for 'is this prompt ambiguous'        : "
      f"{h1['bse']['auroc']:.3f}")
print(f"AUROC of BSE for 'will a question change anything' : "
      f"{auroc([r[2] for r in helped], [r[2] for r in wasted]):.3f}")
print()
print("Those are different targets. H1 asks the first question, H2 needs the second.")
