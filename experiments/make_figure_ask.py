"""The ask-decision read off the repository: what it gets, and what it misses.

A   how open the model calls the task, by arm, once it has read the module --
    with the within-task ranking against `convention` printed above each bar,
    since that is the statistic the model's global bias cannot move
B   the same question with the enumeration step removed: handed the exact
    decision the task turns on, does it judge settledness correctly?
C   the punchline. On both detector axes `convention` and `anti` are the same
    reading. On the axis that matters they are not remotely the same.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK = "#15141C"
SIGNAL = "#A8451D"
GOOD = "#26654A"
MUTED = "#8D8AA1"
FAINT = "#C9C6D6"

ARMS = ["bare", "distractor", "convention", "diluted", "anti"]
NICE = {"bare": "no\nmodule", "distractor": "irrelevant\nsiblings",
        "convention": "convention\nsiblings", "diluted": "pushed\naway",
        "anti": "opposite\nconvention"}
COLOUR = {"bare": MUTED, "distractor": MUTED, "convention": GOOD,
          "diluted": GOOD, "anti": SIGNAL}
TRUTH = {"bare": "OPEN", "distractor": "OPEN", "convention": "SETTLED",
         "diluted": "SETTLED", "anti": "SETTLED"}


def main() -> int:
    ask = json.loads(Path("results/pilot_ask_from_context.json").read_text(encoding="utf-8"))
    joined = json.loads(Path("results/join_ask_outcome.json").read_text(encoding="utf-8"))
    t2 = json.loads(Path("results/pilot_settled_oracle.json").read_text(encoding="utf-8"))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.4, 5.6),
                                        gridspec_kw={"width_ratios": [1.25, 1.1, 1]})
    fig.patch.set_facecolor("white")

    # ---- A: unsettled score by arm, grounded detector ----------------------
    means, spreads = [], []
    for arm in ARMS:
        vals = [d["summary"]["grounded"]["per_arm"][arm]["mean_score"]
                for d in ask["models"].values()]
        vals = [v for v in vals if v is not None]
        means.append(statistics.mean(vals))
        spreads.append(vals)

    axA.bar(range(len(ARMS)), means, 0.62, color=[COLOUR[a] for a in ARMS],
            edgecolor="none", alpha=.9, zorder=2)
    for i, vals in enumerate(spreads):
        axA.scatter([i + 0.24] * len(vals), vals, s=17, color=INK, alpha=.5,
                    zorder=3, linewidth=0)

    for i, arm in enumerate(ARMS):
        if arm == "convention":
            axA.text(i, means[i] + 0.35, "reference", ha="center", fontsize=8,
                     color=MUTED, style="italic")
            continue
        aucs = [d["summary"]["grounded"]["paired"].get(f"{arm}>convention", {}).get("auc")
                for d in ask["models"].values()]
        aucs = [a for a in aucs if a is not None]
        if not aucs:
            continue
        m = statistics.mean(aucs)
        strong = abs(m - 0.5) > 0.15
        axA.text(i, means[i] + 0.35, f"{m:.2f}", ha="center", fontsize=10.5,
                 fontweight="bold", color=INK if strong else SIGNAL)

    axA.set_xticks(range(len(ARMS)))
    axA.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.0)
    axA.set_ylabel('"how much is still unsettled"   (0-10)')
    axA.set_ylim(0, 9.6)
    axA.set_title("A.  The module changes what it calls open",
                  fontsize=11.5, color=INK, pad=26, loc="left")
    axA.text(0, 1.035,
             "numbers above bars = within-task ranking vs the convention arm; 0.50 = context ignored",
             transform=axA.transAxes, fontsize=8.3, color=MUTED)
    axA.text(0.5, -0.19, "bars = mean of 3 model families, dots = each family",
             transform=axA.transAxes, ha="center", fontsize=8.2, color=MUTED)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ---- B: T2 alone, settledness handed the right question ----------------
    n_tasks = len(t2["calls"])
    rates = []
    for arm in ARMS:
        vals = [d["summary"][arm]["settled_majority"] / n_tasks
                for d in t2["models"].values()]
        rates.append(statistics.mean(vals))

    axB.bar(range(len(ARMS)), rates, 0.62, color=[COLOUR[a] for a in ARMS],
            edgecolor="none", alpha=.9, zorder=2)
    for i, arm in enumerate(ARMS):
        vals = [d["summary"][arm]["settled_majority"] / n_tasks
                for d in t2["models"].values()]
        axB.scatter([i + 0.24] * len(vals), vals, s=17, color=INK, alpha=.5,
                    zorder=3, linewidth=0)
        want = 1.0 if TRUTH[arm] == "SETTLED" else 0.0
        axB.hlines(want, i - 0.36, i + 0.36, color=INK, lw=1.6, ls=(0, (3, 2)),
                   zorder=4)
        axB.text(i, 1.045, TRUTH[arm], ha="center", fontsize=7.6, color=INK,
                 fontweight="bold" if TRUTH[arm] == "SETTLED" else "normal")

    axB.set_xticks(range(len(ARMS)))
    axB.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.0)
    axB.set_ylabel("judged SETTLED by the code")
    axB.set_ylim(0, 1.16)
    axB.set_yticks([0, .25, .5, .75, 1.0])
    axB.set_title("B.  Handed the exact question, without enumerating",
                  fontsize=11.5, color=INK, pad=26, loc="left")
    axB.text(0, 1.035, "dashed rule = the correct answer for that arm",
             transform=axB.transAxes, fontsize=8.3, color=MUTED)
    axB.text(0.5, -0.19,
             "anti is correct at SETTLED: the code really does settle it",
             transform=axB.transAxes, ha="center", fontsize=8.2, color=MUTED)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    # ---- C: convention against anti on three axes --------------------------
    by_arm: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for r in joined["grounded"]:
        by_arm[r["arm"]].append((r["score"], r["wrong"]))

    def axis_values(arm: str) -> tuple[float, float, float]:
        score = statistics.mean(s for s, _ in by_arm[arm]) / 10.0
        settled = statistics.mean(d["summary"][arm]["settled_majority"] / n_tasks
                                  for d in t2["models"].values())
        right = statistics.mean(0.0 if w else 1.0 for _, w in by_arm[arm])
        return score, settled, right

    labels = ['called\nunsettled\n(panel A)', 'called\nsettled\n(panel B)',
              'actually\nright\n(round 4)']
    conv, anti = axis_values("convention"), axis_values("anti")
    width = 0.33
    xs = range(3)
    axC.bar([x - width / 2 for x in xs], conv, width, color=GOOD, alpha=.9,
            label="convention siblings", edgecolor="none", zorder=2)
    axC.bar([x + width / 2 for x in xs], anti, width, color=SIGNAL, alpha=.9,
            label="opposite convention", edgecolor="none", zorder=2)
    for x, (a, b) in enumerate(zip(conv, anti)):
        axC.text(x - width / 2, a + 0.022, f"{a:.2f}", ha="center", fontsize=9,
                 color=INK)
        axC.text(x + width / 2, b + 0.022, f"{b:.2f}", ha="center", fontsize=9,
                 color=INK)
        gap = abs(a - b)
        axC.text(x, max(a, b) + 0.105,
                 "same" if gap < 0.08 else f"gap {gap:.2f}",
                 ha="center", fontsize=9.2, color=MUTED if gap < 0.08 else SIGNAL,
                 fontweight="normal" if gap < 0.08 else "bold")

    axC.set_xticks(list(xs))
    axC.set_xticklabels(labels, fontsize=8.4)
    axC.tick_params(axis='x', pad=6)
    axC.set_ylim(0, 1.08)
    axC.set_ylabel("proportion")
    axC.set_title("C.  A wrong convention is invisible",
                  fontsize=11.5, color=INK, pad=26, loc="left")
    axC.text(0, 1.035,
             "both detectors read the two arms identically; the outcomes are not identical",
             transform=axC.transAxes, fontsize=8.3, color=MUTED)
    axC.legend(frameon=False, fontsize=8.6, loc="lower left")
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    fig.tight_layout(w_pad=2.4)
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ask_from_context.{ext}", dpi=170, bbox_inches="tight",
                    facecolor="white")
    print("wrote", out / "ask_from_context.png")
    print(f"  convention {conv}")
    print(f"  anti       {anti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
