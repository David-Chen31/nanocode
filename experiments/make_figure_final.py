"""The result, with its own precondition measured.

A  what the surrounding module does to the code that gets written. Eight arms,
   three model families. The swing is large and it runs in both directions.
B  what the detector makes of the same eight arms. It tracks whether relevant
   code is present, not whether that code is right -- and a module that
   contradicts itself reads as MORE settled than one that says nothing.
C  how often the precondition behind panel A actually holds in real code.
   A funnel over 4000 real modules, ending at sixteen function families.

Panel C is the one that reorders the other two.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK = "#15141C"
SIGNAL = "#A8451D"
GOOD = "#26654A"
MUTED = "#8D8AA1"
FAINT = "#CFCBDC"

ARMS = ["bare", "distractor", "conflict_r", "conflict", "diluted", "prose",
        "convention", "anti"]
NICE = {"bare": "no module", "distractor": "irrelevant siblings",
        "prose": "rule in words", "convention": "rule in code",
        "diluted": "in code, far away", "anti": "opposite rule",
        "conflict": "siblings clash", "conflict_r": "clash, reversed"}
COLOUR = {"bare": MUTED, "distractor": MUTED, "prose": GOOD, "convention": GOOD,
          "diluted": GOOD, "anti": SIGNAL, "conflict": "#B08A3E",
          "conflict_r": "#B08A3E"}


def main() -> int:
    gen = json.loads(Path("results/pilot_conflict_gen.json").read_text(encoding="utf-8"))
    det = json.loads(Path("results/pilot_conflict_det.json").read_text(encoding="utf-8"))
    base = json.loads(Path("results/convention_baserate.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(16.6, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.05], wspace=0.28)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    fig.patch.set_facecolor("white")

    # ---- A: what gets written ---------------------------------------------
    means = {a: statistics.mean(m["summary"][a]["majority_right"]
                                for m in gen["models"].values()) for a in ARMS}
    spread = {a: [m["summary"][a]["majority_right"] for m in gen["models"].values()]
              for a in ARMS}
    axA.bar(range(len(ARMS)), [means[a] for a in ARMS], 0.64,
            color=[COLOUR[a] for a in ARMS], edgecolor="none", alpha=.9, zorder=2)
    for i, a in enumerate(ARMS):
        axA.scatter([i + 0.25] * 3, spread[a], s=16, color=INK, alpha=.5,
                    zorder=3, linewidth=0)
        axA.text(i, means[a] + 0.28, f"{means[a]:.1f}", ha="center", fontsize=9,
                 fontweight="bold", color=INK)
    axA.axhline(means["distractor"], color=MUTED, lw=1, ls="--", zorder=1)
    axA.annotate("", xy=(7.5, means["convention"]), xytext=(7.5, means["anti"]),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    axA.text(7.4, (means["convention"] + means["anti"]) / 2,
             f"{means['convention'] - means['anti']:.1f}\nof 12",
             fontsize=8.6, color=INK, va="center", ha="right", fontweight="bold")
    axA.set_xlim(-0.7, 7.95)
    axA.set_xticks(range(len(ARMS)))
    axA.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.2,
                        rotation=32, ha="right", rotation_mode="anchor")
    axA.set_ylabel("tasks where the majority behaviour is correct  (of 12)")
    axA.set_ylim(0, 11.6)
    axA.set_title("A.  A local precedent decides the answer — either way",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    axA.text(0.5, -0.20, "dashed = no-information baseline   ·   dots = 3 model families",
             transform=axA.transAxes, ha="center", fontsize=8, color=MUTED)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ---- B: what the detector makes of it ---------------------------------
    n_t = len(det["calls"])
    settled = {a: statistics.mean(m["summary"][a]["settled_majority"] / n_t
                                  for m in det["models"].values()) for a in ARMS}
    truth = {"bare": "OPEN", "distractor": "OPEN", "conflict": "OPEN",
             "conflict_r": "OPEN", "prose": "SETTLED", "convention": "SETTLED",
             "diluted": "SETTLED", "anti": "SETTLED"}
    axB.bar(range(len(ARMS)), [settled[a] for a in ARMS], 0.64,
            color=[COLOUR[a] for a in ARMS], edgecolor="none", alpha=.9, zorder=2)
    for i, a in enumerate(ARMS):
        want = 1.0 if truth[a] == "SETTLED" else 0.0
        axB.hlines(want, i - 0.37, i + 0.37, color=INK, lw=1.5, ls=(0, (3, 2)),
                   zorder=4)
        axB.text(i, settled[a] + 0.028, f"{settled[a]:.2f}", ha="center",
                 fontsize=8.6, color=INK)
    axB.add_patch(Rectangle((1.5, 0), 2.0, 1.16, facecolor=SIGNAL, alpha=.08,
                            zorder=0))
    axB.text(2.5, 1.085, "truth: OPEN", ha="center", fontsize=8.4,
             color=SIGNAL, fontweight="bold")
    axB.set_xticks(range(len(ARMS)))
    axB.set_xticklabels([NICE[a] for a in ARMS], fontsize=8.2,
                        rotation=32, ha="right", rotation_mode="anchor")
    axB.set_ylabel('judged "the code settles it"')
    axB.set_ylim(0, 1.16)
    axB.set_title("B.  The detector reads relevance, not correctness",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    axB.text(0.5, -0.20,
             "dashed rule = the correct answer   ·   a self-contradicting module\n"
             "reads as MORE settled (0.61) than one that says nothing (0.36)",
             transform=axB.transAxes, ha="center", fontsize=8, color=MUTED)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    # ---- C: does the precondition ever hold? ------------------------------
    d1 = base["dims"]["D1_empty"]
    fam = base["family_scope"]
    steps = [
        ("real modules scanned", d1["parsed"], MUTED),
        ("contain one function that\nexplicitly decides the case", d1["modules_any"], MUTED),
        ("contain two, anywhere\nin the file", d1["modules_two_plus"], "#B08A3E"),
        ("two in one function family\n— a convention exists", fam["families_two_plus"], GOOD),
    ]
    top = steps[0][1]
    for i, (label, n, col) in enumerate(steps):
        w = max(n / top, 0.006) * 0.62
        y = -i
        axC.add_patch(Rectangle((0.30, y - 0.30), w, 0.60, facecolor=col,
                                alpha=.85, edgecolor="none"))
        axC.text(0.27, y, label, va="center", ha="right", fontsize=8.2,
                 color=INK if i == 3 else MUTED)
        axC.text(0.96, y, f"{n:,}", va="center", ha="right",
                 fontsize=11.5 if i == 3 else 10,
                 fontweight="bold" if i == 3 else "normal", color=INK)
        if i:
            axC.text(1.0, y, f"{n / top:.1%}", va="center", ha="left",
                     fontsize=8.6, color=SIGNAL if i == 3 else MUTED)

    axC.text(0.30, -3.9,
             f"of those sixteen, {fam['agree']} are self-consistent and "
             f"{fam['disagree']} are not",
             ha="center", fontsize=8.6, color=INK)
    axC.text(0.30, -4.35,
             "so panel A's setting is real — and it is where\nalmost no real code lives",
             ha="center", fontsize=9.4, color=SIGNAL, fontweight="bold")
    axC.set_xlim(-0.62, 1.30)
    axC.set_ylim(-4.7, 0.75)
    axC.axis("off")
    axC.set_title("C.  How often is a local precedent there at all?",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    axC.text(-0.6, 0.45, f"{base['functions']:,} functions across the Python standard\n"
                        "library, numpy, scipy, sklearn, matplotlib, transformers",
             fontsize=8, color=MUTED, ha="left")

    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"final.{ext}", dpi=170, bbox_inches="tight",
                    facecolor="white")
    print("wrote", out / "final.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
