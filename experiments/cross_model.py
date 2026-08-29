"""Is the disagreement absent, or only absent inside one model's distribution?

The live study found that sampling one instruction-tuned model six times yields
1.08 distinct behaviours. There are two very different explanations:

  A. the task genuinely admits one reading, and the signal has nothing to find
  B. the model's output distribution is not a posterior over interpretations --
     instruction tuning collapses it onto a single modal reading, so sampling
     measures decoding noise rather than uncertainty about intent

They are distinguished by leaving the model. Different model families were
tuned on different data and carry different priors, so an ensemble of one sample
each is a draw from a genuinely different distribution. If cross-model
disagreement is large where within-model disagreement is nil, explanation B is
right and the method is not dead -- it was reading the wrong distribution.

    python experiments/cross_model.py
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
from bench.harness import CALIB_SEED
from bench.schema import load_tasks

VARIANTS = ("ambiguous", "full")


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


def profile(task, sources) -> dict[str, Any]:
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
    return {"bse": res.bse, "classes": res.n_classes, "covered": covered,
            "majority_right": majority_right}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5,
                    help="candidates per set; both arms use the same k so that "
                         "the normalised entropies are comparable")
    ap.add_argument("--out", default="results/cross_model.json")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob("results/live/*.json")
                   if "T16" not in f and "diverse" not in f and "naive" not in f)
    live = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        live[d["model"]] = d
    models = sorted(live)
    k = min(args.k, len(models))
    print(f"models ({len(models)}): {', '.join(models)}")
    print(f"comparing within-model k={k} against cross-model k={k} "
          f"(one sample per model)\n")

    tasks = load_tasks()
    per_model: dict[str, dict[str, list[dict[str, Any]]]] = {
        m: {v: [] for v in VARIANTS} for m in models}
    cross: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}

    for t in tasks:
        for v in VARIANTS:
            for m in models:
                srcs = live[m]["samples"][t.id][v]["sources"][:k]
                per_model[m][v].append(profile(t, srcs))
            # one sample from each model, in a fixed order
            mixed = [live[m]["samples"][t.id][v]["sources"][0] for m in models[:k]]
            cross[v].append(profile(t, mixed))

    def summarise(label, amb, full):
        return {
            "arm": label,
            "mean_classes": sum(a["classes"] for a in amb) / len(amb),
            "mean_bse": sum(a["bse"] for a in amb) / len(amb),
            "coverage": sum(a["covered"] for a in amb),
            "majority_right": sum(a["majority_right"] for a in amb),
            "reachable": sum(a["covered"] and not a["majority_right"] for a in amb),
            "auroc_h1": auroc([a["bse"] for a in amb], [f["bse"] for f in full]),
            "n": len(amb),
        }

    rows = [summarise(f"within {m}", per_model[m]["ambiguous"], per_model[m]["full"])
            for m in models]
    rows.append(summarise("CROSS-MODEL ensemble", cross["ambiguous"], cross["full"]))

    print(f"{'arm':<34}{'classes':>9}{'BSE':>7}{'cov':>7}{'reach':>7}{'H1 AUROC':>10}")
    print("-" * 76)
    for r in rows[:-1]:
        print(f"{r['arm']:<34}{r['mean_classes']:>9.2f}{r['mean_bse']:>7.3f}"
              f"{str(r['coverage']) + '/' + str(r['n']):>7}{r['reachable']:>7}"
              f"{r['auroc_h1']:>10.3f}")
    print("-" * 76)
    r = rows[-1]
    print(f"{r['arm']:<34}{r['mean_classes']:>9.2f}{r['mean_bse']:>7.3f}"
          f"{str(r['coverage']) + '/' + str(r['n']):>7}{r['reachable']:>7}"
          f"{r['auroc_h1']:>10.3f}")

    wm = rows[:-1]
    print()
    print(f"within-model average   classes {sum(x['mean_classes'] for x in wm) / len(wm):.2f}"
          f"   coverage {sum(x['coverage'] for x in wm) / len(wm):.1f}/{r['n']}"
          f"   reachable {sum(x['reachable'] for x in wm) / len(wm):.1f}"
          f"   H1 {sum(x['auroc_h1'] for x in wm) / len(wm):.3f}")
    print(f"cross-model ensemble   classes {r['mean_classes']:.2f}"
          f"   coverage {r['coverage']}/{r['n']}"
          f"   reachable {r['reachable']}"
          f"   H1 {r['auroc_h1']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
