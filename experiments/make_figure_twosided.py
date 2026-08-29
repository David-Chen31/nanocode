"""The figure that carries the argument.

x = how strongly the signal fires on a FULLY specified prompt (false alarms)
y = how strongly it fires on the same prompt with one sentence deleted

A useful signal sits top-left: loud on the ambiguous prompt, quiet on the
complete one. The diagonal is the useless line -- fires equally on both.

Two clusters appear, and they fail in opposite directions: signals read off the
model's own output sit at the bottom-left and never fire at all; signals that
ask the model to name an ambiguity sit at the top-right and always fire, even on
prompts that settle the question explicitly.
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
OUT = "#383D80"       # output-distribution family
TXT = "#A8451D"       # text-elicitation family
GOOD = "#26654A"
MUTED = "#8D8AA1"


def main() -> int:
    live = json.loads(Path("results/exp_mini-naive.json").read_text(encoding="utf-8"))["h1"]
    txt = json.loads(Path("results/text_side.json").read_text(encoding="utf-8"))
    tm = txt["models"]["gpt-4o-mini"]["signals"]

    pts = []
    for name, label in [("bse", "BSE"), ("distinct_classes", "distinct classes"),
                        ("text_diversity", "text diversity"),
                        ("token_entropy", "token entropy"),
                        ("self_report", "self report")]:
        per = live[name]["per_task"]
        amb = sum(v["ambiguous"] for v in per.values()) / len(per)
        full = sum(v["full"] for v in per.values()) / len(per)
        pts.append((full, amb, label, OUT, live[name]["auroc"]))

    for name, label in [("direct_ask", "direct ask"),
                        ("enumerate_gaps", "enumerate gaps"),
                        ("two_readings", "two readings"),
                        ("discriminating", "discriminating input")]:
        r = tm[name]
        pts.append((r["mean_full"], r["mean_ambiguous"], label, TXT, r["auroc_h1"]))

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    fig.patch.set_facecolor("white")

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.72, 0.685, "fires equally on both\n= no information",
            fontsize=8.5, color=MUTED, rotation=39, ha="center", va="center")

    ax.add_patch(plt.Rectangle((0, 0.70), 0.34, 0.30, facecolor=GOOD,
                               alpha=0.07, zorder=0))
    ax.text(0.02, 0.965, "a useful signal would live here", fontsize=9,
            color=GOOD, style="italic")

    # The output-distribution family collapses into one corner, so its labels
    # go in a magnified inset rather than on top of each other.
    inset = ax.inset_axes([0.10, 0.24, 0.40, 0.31])
    for a in (ax, inset):
        for x, y, label, colour, au in pts:
            a.scatter([x], [y], s=110 if a is ax else 90, color=colour, zorder=3,
                      edgecolor="white", linewidth=1.4)

    for x, y, label, colour, au in pts:
        if colour is OUT or label == "direct ask":
            continue
        dx, dy, ha = 0.022, 0.0, "left"
        if label == "two readings":
            dy = -0.05
        if label == "discriminating input":
            dx, dy, ha = -0.024, 0.03, "right"
        ax.annotate(f"{label}  ({au:.2f})", (x + dx, y + dy), fontsize=8.8,
                    color=INK, ha=ha, va="center", zorder=4)

    inset.set_xlim(-0.012, 0.235)
    inset.set_ylim(-0.035, 0.245)
    inset.plot([-0.035, 0.245], [-0.035, 0.245], color=MUTED, lw=0.9, ls="--", zorder=1)
    offsets = {"direct ask": (0.009, 0.015, "left"),
               "self report": (0.010, 0.008, "left"),
               "text diversity": (-0.010, 0.014, "right"),
               "BSE": (0.011, 0.011, "left"),
               "distinct classes": (0.011, -0.001, "left"),
               "token entropy": (0.006, -0.019, "left")}
    for x, y, label, colour, au in pts:
        if label not in offsets:
            continue
        dx, dy, ha = offsets[label]
        inset.annotate(f"{label} ({au:.2f})", (x + dx, y + dy), fontsize=7.4,
                       color=INK, ha=ha, va="center", zorder=4)
    inset.set_title("magnified", fontsize=7.6, color=MUTED, pad=3)
    inset.tick_params(labelsize=6.5, length=2, colors=MUTED)
    for sp in inset.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(0.8)
    inset.set_facecolor("white")
    ax.indicate_inset_zoom(inset, edgecolor=MUTED, alpha=0.6)

    ax.scatter([], [], s=90, color=OUT, label="read the model's OUTPUT")
    ax.scatter([], [], s=90, color=TXT, label="read the REQUIREMENT text")
    ax.legend(fontsize=9, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.02))

    ax.annotate("blind:\nnever fires,\neven when the\nspec is broken",
                xy=(0.17, 0.06), xytext=(0.56, 0.12), fontsize=8.8, color=OUT,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=OUT, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("cries wolf:\nalways fires, even on\nprompts that settle it",
                xy=(0.97, 0.97), xytext=(0.55, 0.86), fontsize=8.8, color=TXT,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=TXT, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xlabel("mean score on the FULLY SPECIFIED prompt   (false alarms →)",
                  fontsize=10)
    ax.set_ylabel("mean score on the AMBIGUOUS prompt   (detection →)", fontsize=10)
    ax.set_xlim(-0.04, 1.14)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("Two families of signal, failing in opposite directions",
                 fontsize=12.5, color=INK, pad=14)
    ax.text(0.5, -0.145, "parentheses = AUROC on 'is this prompt under-specified?'"
                          "   ·   0.50 is chance",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"two_sided_failure.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    print("wrote", out / "two_sided_failure.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
