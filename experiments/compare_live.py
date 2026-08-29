"""The headline comparison: hand-written fixtures against real model samples.

The offline study in docs/FINDINGS.md rests on candidate sets written by the
author. This script measures how far that assumption is from what a real model
actually produces, across every sampling configuration on disk, and runs the
validity check that decides how to read the result:

    coverage under the FULL prompt -- if the model cannot reach the reference
    even when the specification is complete, low coverage under the ambiguous
    prompt says nothing about ambiguity and everything about a too-strict
    reference.

    python experiments/compare_live.py
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


def profile_set(task, sources) -> dict[str, Any]:
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
    return {"classes": res.n_classes, "bse": res.bse, "covered": covered,
            "majority_right": majority_right}


def summarise(label: str, per_task: list[dict[str, Any]],
              full_cov: int | None = None) -> dict[str, Any]:
    n = len(per_task)
    cov = sum(p["covered"] for p in per_task)
    maj = sum(p["majority_right"] for p in per_task)
    return {
        "config": label,
        "mean_classes": sum(p["classes"] for p in per_task) / n,
        "mean_bse": sum(p["bse"] for p in per_task) / n,
        "coverage": cov,
        "coverage_full_prompt": full_cov,
        "majority_right": maj,
        "reachable_by_asking": cov - maj,
        "n": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", nargs="*", default=None,
                    help="live sample JSONs; default: everything in results/live")
    ap.add_argument("--out", default="results/live_vs_fixture.json")
    args = ap.parse_args()

    tasks = load_tasks()
    files = args.live or sorted(glob.glob("results/live/*.json"))

    rows = [summarise("fixture (hand-written)",
                      [profile_set(t, t.candidates("ambiguous")) for t in tasks])]

    for f in files:
        live = json.loads(Path(f).read_text(encoding="utf-8"))
        label = (f"{live['model']} T={live['temperature']}"
                 + (" +diversify" if live.get("diversify") else ""))
        amb = [profile_set(t, live["samples"][t.id]["ambiguous"]["sources"]) for t in tasks]
        full = [profile_set(t, live["samples"][t.id]["full"]["sources"]) for t in tasks]
        rows.append(summarise(label, amb, full_cov=sum(p["covered"] for p in full)))

    print(f"{'configuration':<34}{'behav.':>8}{'mean':>7}{'cov':>7}{'cov':>7}"
          f"{'maj':>6}{'reach':>7}")
    print(f"{'':<34}{'classes':>8}{'BSE':>7}{'amb':>7}{'full':>7}{'ok':>6}{'able':>7}")
    print("-" * 76)
    for r in rows:
        cf = f"{r['coverage_full_prompt']}/{r['n']}" if r["coverage_full_prompt"] is not None else "-"
        print(f"{r['config']:<34}{r['mean_classes']:>8.2f}{r['mean_bse']:>7.3f}"
              f"{str(r['coverage']) + '/' + str(r['n']):>7}{cf:>7}"
              f"{r['majority_right']:>6}{r['reachable_by_asking']:>7}")

    print()
    print("cov amb / cov full : does the sampled set contain the reference behaviour,")
    print("                     under the ambiguous and the fully specified prompt.")
    print("reachable          : covered but the majority is wrong -- the only tasks")
    print("                     where asking a question can change the outcome.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
