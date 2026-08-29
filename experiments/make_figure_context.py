"""The repository-context result.

Left   five arms, mean over three model families and three runs, with the
       individual observations drawn on top so the spread is visible
Right  which tasks the convention rescues, which it corrupts, and the three
       that resist every arm
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.schema import load_tasks

INK = "#15141C"
ACCENT = "#383D80"
SIGNAL = "#A8451D"
GOOD = "#26654A"
MUTED = "#8D8AA1"
LINE = "#D5D2E0"

RUNS = ["results/pilot_repo_context.json",
        "results/pilot_repo_context4.json",
        "results/pilot_repo_context5.json"]
ARMS = ["bare", "distractor", "convention", "diluted", "anti"]
NICE = {"bare": "bare\nrequirement", "distractor": "irrelevant\nsiblings",
        "convention": "convention\nsiblings", "diluted": "convention,\npushed away",
        "anti": "opposite\nconvention"}
COLOUR = {"bare": MUTED, "distractor": MUTED, "convention": GOOD,
          "diluted": GOOD, "anti": SIGNAL}


def main() -> int:
    obs: dict[str, list[int]] = defaultdict(list)
    per_task: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for p in RUNS:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for model, x in d["models"].items():
            for arm in ARMS:
                s = x["summary"].get(arm)
                if s:
                    obs[arm].append(s["majority_right"])
            for tid, arms in x["per_task"].items():
                for arm, r in arms.items():
                    per_task[tid][arm].append(bool(r["majority_right"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("white")

    # ---- left: the arms --------------------------------------------------
    xs = range(len(ARMS))
    means = [sum(obs[a]) / len(obs[a]) for a in ARMS]
    ax1.bar(xs, means, 0.6, color=[COLOUR[a] for a in ARMS], edgecolor="none",
            alpha=.92, zorder=2)
    for i, a in enumerate(ARMS):
        ax1.scatter([i + 0.22] * len(obs[a]), obs[a], s=16, color=INK,
                    alpha=.5, zorder=3, linewidth=0)
        ax1.text(i, means[i] + 0.22, f"{means[i]:.1f}", ha="center",
                 fontsize=11, fontweight="bold", color=INK)

    ax1.axhline(means[0], color=MUTED, lw=1, ls="--", zorder=1)
    ax1.annotate("", xy=(2, means[2] - 0.15), xytext=(2, means[4] + 0.15),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax1.text(2.12, (means[2] + means[4]) / 2,
             f"{means[2] - means[4]:.1f} of 12 tasks\nswing on the convention",
             fontsize=8.8, color=INK, va="center")

    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.6)
    ax1.set_ylabel("tasks where the majority behaviour is correct  (of 12)")
    ax1.set_ylim(0, 12)
    ax1.set_yticks([0, 3, 6, 9, 12])
    ax1.set_title("What sits above the function decides the answer",
                  fontsize=12, color=INK, pad=12)
    ax1.text(0.5, -0.235, "dots = 3 model families x 3 runs   ·   dashed line = bare baseline",
             transform=ax1.transAxes, ha="center", fontsize=8.2, color=MUTED)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---- right: per task -------------------------------------------------
    tasks = [t.id for t in load_tasks()]
    show = ["bare", "distractor", "convention", "anti"]
    ax2.set_xlim(-0.5, len(show) - 0.5)
    ax2.set_ylim(-0.5, len(tasks) - 0.5)

    for yi, tid in enumerate(reversed(tasks)):
        for xi, arm in enumerate(show):
            vals = per_task[tid].get(arm, [])
            frac = sum(vals) / len(vals) if vals else 0.0
            col = GOOD if frac >= 0.6 else (SIGNAL if frac <= 0.4 else MUTED)
            ax2.add_patch(plt.Rectangle((xi - 0.42, yi - 0.42), 0.84, 0.84,
                                        facecolor=col, alpha=0.18 + 0.62 * frac,
                                        edgecolor=LINE, linewidth=.8))
            if vals:
                ax2.text(xi, yi, f"{sum(vals)}/{len(vals)}", ha="center", va="center",
                         fontsize=7.6, color=INK)

    ax2.set_xticks(range(len(show)))
    ax2.set_xticklabels([NICE[a].replace("\n", " ") for a in show], fontsize=8.4)
    ax2.set_yticks(range(len(tasks)))
    ax2.set_yticklabels([t.replace("_", " ") for t in reversed(tasks)], fontsize=8)
    ax2.set_title("Rescued, corrupted, and immovable", fontsize=12, color=INK, pad=12)
    ax2.text(1.5, -0.95, "cell = runs in which the majority behaviour was correct",
             fontsize=8, color=MUTED, ha="center")
    ax2.set_ylim(-1.2, len(tasks) - 0.4)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.tick_params(length=0)

    # Which tasks never come right, in any arm -- read off the data rather than
    # asserted, so the annotation cannot drift away from the numbers.
    for yi, tid in enumerate(reversed(tasks)):
        best = max((sum(v) / len(v)) for a, v in per_task[tid].items() if v)
        if best == 0.0:
            ax2.text(len(show) - 0.32, yi, "never correct", fontsize=7.6,
                     color=SIGNAL, va="center")
        elif per_task[tid].get("bare") and sum(per_task[tid]["bare"]) == 0                 and per_task[tid].get("convention")                 and sum(per_task[tid]["convention"]) == len(per_task[tid]["convention"]):
            ax2.text(len(show) - 0.32, yi, "fully rescued", fontsize=7.6,
                     color=GOOD, va="center")

    fig.tight_layout()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"repo_context.{ext}", dpi=170, bbox_inches="tight",
                    facecolor="white")
    print("wrote", out / "repo_context.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
