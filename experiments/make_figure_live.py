"""The figure the live study argues with.

  left   H1 AUROC per signal, hand-written fixtures against real samples --
         the collapse, with the chance line drawn in
  right  why: behavioural diversity and reference coverage across every
         sampling configuration tried
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK = "#15141C"
ACCENT = "#383D80"
SIGNAL = "#A8451D"
GOOD = "#26654A"
MUTED = "#8D8AA1"

ORDER = ["bse", "distinct_classes", "text_diversity", "token_entropy",
         "self_report", "direct_ask", "random"]
LABEL = {"bse": "BSE", "distinct_classes": "distinct\nclasses",
         "text_diversity": "text\ndiversity", "token_entropy": "token\nentropy",
         "self_report": "self\nreport", "direct_ask": "direct\nask",
         "random": "random"}


def main() -> int:
    fx = json.loads(Path("results/experiments.json").read_text(encoding="utf-8"))["h1"]
    lv = json.loads(Path("results/exp_mini-naive.json").read_text(encoding="utf-8"))["h1"]
    rows = json.loads(Path("results/live_vs_fixture.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")

    # ---- left: the collapse ---------------------------------------------
    names = [n for n in ORDER if n in fx or n in lv]
    xs = range(len(names))
    w = 0.38
    fixture = [fx[n]["auroc"] if n in fx else 0 for n in names]
    live = [lv[n]["auroc"] if n in lv else 0 for n in names]

    b1 = ax1.bar([x - w / 2 for x in xs], fixture, w, color=ACCENT,
                 edgecolor="none", label="hand-written fixtures")
    b2 = ax1.bar([x + w / 2 for x in xs], live, w, color=SIGNAL,
                 edgecolor="none", label="real samples (gpt-4o-mini)")
    for x, n in zip(xs, names):
        if n not in fx:
            ax1.text(x - w / 2, 0.02, "n/a", fontsize=7.5, color=MUTED,
                     ha="center", rotation=90)
    ax1.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax1.text(len(names) - 0.4, 0.515, "chance", color=MUTED, fontsize=8, ha="right")
    ax1.annotate("", xy=(0 + w / 2, live[0] + 0.03), xytext=(0 - w / 2, fixture[0] - 0.03),
                 arrowprops=dict(arrowstyle="->", color=SIGNAL, lw=1.4,
                                 connectionstyle="arc3,rad=-0.35"))
    ax1.text(0.62, 0.72, "0.986 → 0.389", fontsize=9, color=SIGNAL, fontweight="bold")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([LABEL[n] for n in names], fontsize=8.2)
    ax1.set_ylabel("AUROC:  is this prompt under-specified?")
    ax1.set_ylim(0, 1.06)
    ax1.set_title("The fixture result does not replicate", fontsize=11.5,
                  color=INK, pad=10)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---- right: why ------------------------------------------------------
    # A dual axis here reads as if coverage and diversity were comparable
    # magnitudes. They are not, so coverage is annotated as text instead.
    labels = [r["config"].replace("gpt-4o-mini", "mini").replace(" (hand-written)", "")
              for r in rows]
    ys = list(range(len(rows)))[::-1]
    classes = [r["mean_classes"] for r in rows]

    colours = [ACCENT] + [SIGNAL] * (len(rows) - 1)
    ax2.barh(ys, classes, 0.5, color=colours, edgecolor="none")
    for y, r, c in zip(ys, rows, classes):
        cov = f"{r['coverage']}/{r['n']}"
        full = f"{r['coverage_full_prompt']}/{r['n']}" if r["coverage_full_prompt"] else "--"
        ax2.text(c + 0.06, y, f"{c:.2f}", va="center", fontsize=9,
                 color=INK, fontweight="bold")
        ax2.text(3.45, y, f"{cov:>6}", va="center", fontsize=8.6, color=SIGNAL,
                 ha="right", family="monospace")
        ax2.text(4.15, y, f"{full:>6}", va="center", fontsize=8.6, color=GOOD,
                 ha="right", family="monospace")

    ax2.axvline(1.0, color=MUTED, lw=1, ls=":")
    ax2.text(1.05, ys[-1] - 0.62, "1.0 = every sample behaves identically",
             fontsize=7.6, color=MUTED)
    ax2.text(3.45, ys[0] + 0.62, "coverage", fontsize=8, color=SIGNAL, ha="right")
    ax2.text(3.45, ys[0] + 1.0, "ambiguous", fontsize=7.2, color=SIGNAL, ha="right")
    ax2.text(4.15, ys[0] + 0.62, "coverage", fontsize=8, color=GOOD, ha="right")
    ax2.text(4.15, ys[0] + 1.0, "FULL prompt", fontsize=7.2, color=GOOD, ha="right")

    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=8.6)
    ax2.set_xlabel("distinct behaviours per task (out of 6 samples)", fontsize=9)
    ax2.set_xlim(0, 4.3)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_ylim(-0.9, len(rows) - 0.1)
    ax2.set_title("Real models do not disagree", fontsize=11.5, color=INK, pad=22)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fixture_vs_live.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    print("wrote", out / "fixture_vs_live.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
