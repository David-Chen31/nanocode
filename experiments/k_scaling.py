"""Two questions the earlier runs left open.

1. k-scaling. Every result so far used six samples. If the correct reading has
   small but non-zero probability mass, more samples would find it and the whole
   negative result would just be a budget artefact. This sweeps k up to 30 and
   plots coverage and diversity against it.

2. The target that actually matters. "Is this prompt ambiguous" is a proxy. What
   a deployed agent needs to predict is "will the code I am about to deliver be
   wrong". On the live sets that target is not degenerate -- roughly seven of
   twelve tasks deliver wrong code -- so every signal can be scored on it
   directly.

    python experiments/k_scaling.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from askoract.bse import compute_bse
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from askoract.signals import (BSESignal, DistinctClassesSignal, SignalInput,
                              TextDiversitySignal, TokenEntropySignal)
from bench.harness import CALIB_SEED
from bench.schema import load_tasks


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


def analyse(task, sources, logprobs=None) -> dict[str, Any]:
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    m = run_candidates(sources, probes, task.entry_point)
    ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    res = compute_bse(m)
    rows = m.valid_rows()
    covered = any(m.tokens[i] == ref for i in rows)
    majority_right = False
    if res.classes:
        idx = m.candidate_ids.index(res.majority_class[0])
        majority_right = m.tokens[idx] == ref
    inp = SignalInput(task.id, sources, matrix=m, bse=res, mean_logprobs=logprobs)
    return {"bse": res.bse, "classes": res.n_classes, "covered": covered,
            "majority_right": majority_right, "signal_input": inp}


def k_sweep(path: str, ks: list[int]) -> None:
    live = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = load_tasks()
    print(f"k-scaling on {live['model']} (T={live['temperature']}), "
          f"ambiguous prompts only\n")
    print(f"{'k':>4}{'distinct behaviours':>22}{'coverage':>11}{'majority right':>16}")
    print("-" * 55)
    for k in ks:
        cls = cov = maj = 0
        for t in tasks:
            srcs = live["samples"][t.id]["ambiguous"]["sources"][:k]
            if len(srcs) < k:
                continue
            r = analyse(t, srcs)
            cls += r["classes"]
            cov += r["covered"]
            maj += r["majority_right"]
        n = len(tasks)
        print(f"{k:>4}{cls / n:>22.2f}{str(cov) + '/' + str(n):>11}"
              f"{str(maj) + '/' + str(n):>16}")


def wrongness(files: list[str]) -> None:
    """Score every signal on the target a deployed agent actually needs."""
    tasks = load_tasks()
    sigs = [BSESignal(), DistinctClassesSignal(), TextDiversitySignal()]
    print("\n\nTarget: will the code the agent is about to deliver be WRONG?")
    print("(the majority behaviour differs from the reference)\n")
    header = f"{'model':<30}{'n wrong':>9}" + "".join(f"{s.name[:13]:>15}" for s in sigs)
    print(header + f"{'token_entropy':>15}")
    print("-" * len(header + " " * 15))

    for f in files:
        live = json.loads(Path(f).read_text(encoding="utf-8"))
        scores: dict[str, list[float]] = {s.name: [] for s in sigs}
        scores["token_entropy"] = []
        wrong: list[bool] = []
        for t in tasks:
            rec = live["samples"][t.id]["ambiguous"]
            r = analyse(t, rec["sources"][:6], rec.get("mean_logprobs"))
            wrong.append(not r["majority_right"])
            for s in sigs:
                scores[s.name].append(s.score(r["signal_input"]))
            lps = [lp for lp in (rec.get("mean_logprobs") or []) if lp is not None]
            scores["token_entropy"].append(
                TokenEntropySignal().score(r["signal_input"]) if lps else float("nan"))

        row = f"{live['model']:<30}{sum(wrong):>4}/{len(wrong):<4}"
        for name in [s.name for s in sigs] + ["token_entropy"]:
            vals = scores[name]
            if any(v != v for v in vals):
                row += f"{'n/a':>15}"
                continue
            pos = [v for v, w in zip(vals, wrong) if w]
            neg = [v for v, w in zip(vals, wrong) if not w]
            row += f"{auroc(pos, neg):>15.3f}"
        print(row)
    print("\n0.5 = chance. A signal above 0.5 predicts its own errors.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-file", default="results/live/mini-k30.json")
    ap.add_argument("--ks", type=int, nargs="*", default=[2, 4, 6, 10, 15, 20, 30])
    args = ap.parse_args()

    k_sweep(args.k_file, args.ks)
    files = sorted(f for f in glob.glob("results/live/*.json")
                   if not any(x in f for x in ("T16", "diverse", "naive", "k30")))
    wrongness(files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
