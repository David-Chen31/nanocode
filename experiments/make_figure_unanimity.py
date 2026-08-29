"""The sign flip: what unanimity across model families means, and when.

Left   the flip itself -- precision of the majority behaviour, split by whether
       all five families agreed, under a complete spec and under an omission
Right  why the earlier negative result and this positive one are the same fact:
       a shared prior is systematic, so it is invisible as spread and visible as
       agreement
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
LINE = "#D5D2E0"


def main() -> int:
    d = json.loads(Path("results/pilot_unanimity.json").read_text(encoding="utf-8"))
    amb = d["ambiguous"]["excluding_designed_hard"]
    ful = d["full"]["all_tasks"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    fig.patch.set_facecolor("white")

    # ---- left: the flip --------------------------------------------------
    groups = ["requirement\nSTATED", "one sentence\nOMITTED"]
    unan = [ful["precision_unanimous"], amb["precision_unanimous"]]
    split = [ful["precision_split"], amb["precision_split"]]
    counts_u = [f"{ful['unanimous_correct']}/{ful['n_unanimous']}",
                f"{amb['unanimous_correct']}/{amb['n_unanimous']}"]
    counts_s = [f"{ful['split_correct']}/{ful['n_split']}",
                f"{amb['split_correct']}/{amb['n_split']}"]

    x = [0, 1]
    w = 0.34
    b1 = ax1.bar([i - w / 2 for i in x], unan, w, color=SIGNAL, edgecolor="none",
                 label="all 5 families agreed")
    b2 = ax1.bar([i + w / 2 for i in x], split, w, color=ACCENT, edgecolor="none",
                 label="families split")
    for i, (u, s, cu, cs) in enumerate(zip(unan, split, counts_u, counts_s)):
        ax1.text(i - w / 2, u + 0.03, f"{u:.2f}", ha="center", fontsize=11,
                 fontweight="bold", color=SIGNAL)
        ax1.text(i - w / 2, max(u, 0) + 0.10, cu, ha="center", fontsize=8, color=MUTED)
        ax1.text(i + w / 2, s + 0.03, f"{s:.2f}", ha="center", fontsize=11,
                 fontweight="bold", color=ACCENT)
        ax1.text(i + w / 2, s + 0.10, cs, ha="center", fontsize=8, color=MUTED)

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=10)
    ax1.set_ylabel("majority behaviour matches what the user wanted")
    ax1.set_ylim(0, 1.28)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_title("Unanimity flips sign under omission", fontsize=12, color=INK, pad=12)
    ax1.legend(fontsize=9, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.13))
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---- right: the mechanism -------------------------------------------
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Why both results are the same fact", fontsize=12, color=INK, pad=12)

    def box(y, colour, head, body):
        ax2.add_patch(plt.Rectangle((0.03, y), 0.94, 0.20, facecolor="white",
                                    edgecolor=colour, linewidth=1.3))
        ax2.text(0.06, y + 0.145, head, fontsize=9.6, color=colour, fontweight="bold")
        ax2.text(0.06, y + 0.055, body, fontsize=8.7, color=INK, va="center")

    ax2.text(0.5, 0.94, "the blind spot is a SHARED PRIOR, not uncertainty",
             fontsize=10.4, color=INK, ha="center", style="italic")

    box(0.66, SIGNAL, "systematic  →  invisible from inside",
        "sampling spread 1.08 of 6   ·   logprob −0.005   ·   self-report 0.9\n"
        "every output-distribution signal sat at chance")
    box(0.37, SIGNAL, "systematic  →  ensembling does not average it away",
        "cross-model disagreement as an ambiguity signal: AUROC 0.472\n"
        "more models does not mean more spread when the bias is shared")
    box(0.08, GOOD, "systematic  →  but visible from OUTSIDE, as agreement",
        "independently trained families converging under omission means they\n"
        "are following the prior, not the requirement.  Agreement is the alarm.")

    for y in (0.615, 0.325):
        ax2.annotate("", xy=(0.5, y - 0.045), xytext=(0.5, y),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    fig.tight_layout()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"unanimity_flip.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    print("wrote", out / "unanimity_flip.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
