"""Render the two figures the results argue with.

  left   the headline: each signal's AUROC on the two targets the ask decision
         actually contains, with the chance line drawn in
  right  success rate against the question budget, with tie-break CI bands
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK = "#12211f"
ACCENT = "#0b5c63"
SIGNAL = "#b14a18"
MUTED = "#7d8f90"
THIRD = "#8a6d3b"
ORDER = ["bse", "max_probe_entropy", "distinct_classes", "text_diversity", "random"]
LABEL = {"bse": "BSE", "max_probe_entropy": "max probe H", "distinct_classes": "distinct classes",
         "text_diversity": "text diversity", "random": "random"}


def main() -> int:
    res = json.loads(Path("results/experiments.json").read_text(encoding="utf-8"))
    h1, h2 = res["h1"], res["h2"]
    diag = h2["target_diagnosis"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.patch.set_facecolor("white")

    # ---- left: the two targets ------------------------------------------
    names = [n for n in ORDER if n in diag]
    xs = range(len(names))
    t1 = [h1[n]["auroc"] for n in names]
    t2 = [diag[n]["auroc_T2_question_helps"] for n in names]

    w = 0.36
    ax1.bar([x - w / 2 for x in xs], t1, w, label="T1  is the prompt ambiguous?",
            color=ACCENT, edgecolor="none")
    ax1.bar([x + w / 2 for x in xs], t2, w, label="T2  will a question help?",
            color=SIGNAL, edgecolor="none")
    ax1.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax1.text(len(names) - 0.45, 0.515, "chance", color=MUTED, fontsize=8, ha="right")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([LABEL[n] for n in names], fontsize=9, rotation=18, ha="right")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("The same signal, two different targets", fontsize=11, color=INK, pad=10)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # ---- right: budget curves -------------------------------------------
    styles = {"bse": (ACCENT, "-", 2.0), "random": (SIGNAL, "-", 2.0),
              "text_diversity": (THIRD, "-.", 1.5),
              "distinct_classes": (ACCENT, ":", 1.4),
              "max_probe_entropy": (MUTED, ":", 1.4)}
    for name in ORDER:
        pts = h2["ranked_budget"].get(name)
        if not pts:
            continue
        x = [p["asks_per_task"] for p in pts]
        y = [p["success_rate"] for p in pts]
        colour, ls, lw = styles[name]
        ax2.plot(x, y, ls, color=colour, lw=lw, label=LABEL[name])
        if name in {"bse", "random"}:
            lo = [p["ci95"][0] for p in pts]
            hi = [p["ci95"][1] for p in pts]
            ax2.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)

    ax2.axhline(h2["never_ask_success"], color=MUTED, lw=1, ls="--")
    ax2.text(0.985, h2["never_ask_success"] + 0.015, "never ask", fontsize=8,
             color=MUTED, ha="right")
    ax2.axhline(h2["always_ask_success"], color=MUTED, lw=1, ls="--")
    ax2.text(0.02, h2["always_ask_success"] + 0.012,
             "always ask (perfect-oracle ceiling)", fontsize=8, color=MUTED)
    ax2.set_xlabel("questions allowed per task")
    ax2.set_ylabel("task success rate")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0.35, 1.08)
    ax2.set_title("Spending a fixed question budget", fontsize=11, color=INK, pad=10)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"targets_and_budget.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    print("wrote", out / "targets_and_budget.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
