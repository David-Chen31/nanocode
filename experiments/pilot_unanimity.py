"""Is cross-model unanimity an anti-signal, and is the effect real?

The free pilot found that when five model families independently produce the same
behaviour, that behaviour is the one the user wanted only 1 time in 5; when they
split, the majority is right 6 times in 7. That inverts the usual reading of
self-consistency and ensembling, so it deserves two controls before anyone
believes it.

  1. construction confound -- three tasks were deliberately built so the common
     reading is wrong. Does the effect survive dropping them?
  2. specificity -- on the FULLY SPECIFIED version of the same tasks the answer
     is written down, so unanimity should now mean agreement on the right thing.
     If unanimity predicts correctness there and wrongness under omission, the
     claim is about under-specification rather than about ensembles in general.

    python experiments/pilot_unanimity.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.harness import CALIB_SEED
from bench.schema import load_tasks

DESIGNED_HARD = {"t06_parse_range", "t08_round_price", "t11_truncate"}


def consensus(task, sources) -> tuple[int, bool]:
    """How many of the sources share the majority behaviour, and is it right."""
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    m = run_candidates(sources, probes, task.entry_point)
    ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    groups: dict[tuple[str, ...], int] = defaultdict(int)
    for row in m.tokens:
        groups[tuple(row)] += 1
    top = max(groups, key=lambda k: groups[k])
    return groups[top], list(top) == ref


def table(rows: list[tuple[str, int, bool]], n_models: int, title: str) -> dict[str, Any]:
    unan = [r for r in rows if r[1] == n_models]
    split = [r for r in rows if r[1] < n_models]
    ur = sum(1 for r in unan if r[2])
    sr = sum(1 for r in split if r[2])
    up = ur / len(unan) if unan else float("nan")
    sp = sr / len(split) if split else float("nan")
    print(f"\n{title}")
    print(f"  unanimous ({n_models}/{n_models}) : {len(unan):>2} tasks, "
          f"{ur} correct  ->  precision {up:.2f}")
    print(f"  split                : {len(split):>2} tasks, "
          f"{sr} correct  ->  precision {sp:.2f}")
    if unan and split:
        print(f"  lift of unanimity    : {up - sp:+.2f}")
    return {"n_unanimous": len(unan), "unanimous_correct": ur,
            "n_split": len(split), "split_correct": sr,
            "precision_unanimous": up, "precision_split": sp}


def main() -> int:
    live = {}
    for f in sorted(glob.glob("results/live/*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if any(x in f for x in ("T16", "diverse", "naive", "k30")):
            continue
        live[d["model"]] = d
    models = sorted(live)
    tasks = load_tasks()
    print(f"{len(models)} families: {', '.join(models)}")

    out: dict[str, Any] = {"models": models}
    for variant in ("ambiguous", "full"):
        rows = []
        for t in tasks:
            first = [live[m]["samples"][t.id][variant]["sources"][0] for m in models]
            c, right = consensus(t, first)
            rows.append((t.id, c, right))

        print("\n" + "=" * 72)
        print(f"prompt variant: {variant.upper()}")
        print("=" * 72)
        print(f"{'task':<22}{'agreeing':>10}{'majority correct':>19}")
        print("-" * 72)
        for tid, c, right in rows:
            mark = " *" if tid in DESIGNED_HARD else ""
            print(f"{tid + mark:<22}{c:>6}/{len(models)}{('YES' if right else 'NO'):>19}")

        out[variant] = {
            "all_tasks": table(rows, len(models), "all 12 tasks"),
            "excluding_designed_hard": table(
                [r for r in rows if r[0] not in DESIGNED_HARD], len(models),
                "excluding the 3 tasks built so the common reading is wrong (*)"),
        }

    print("\n" + "=" * 72)
    print("reading")
    print("=" * 72)
    amb = out["ambiguous"]["excluding_designed_hard"]
    ful = out["full"]["all_tasks"]
    print(f"under omission, excluding constructed cases : unanimity precision "
          f"{amb['precision_unanimous']:.2f} vs split {amb['precision_split']:.2f}")
    print(f"with the requirement stated                 : unanimity precision "
          f"{ful['precision_unanimous']:.2f} vs split {ful['precision_split']:.2f}")
    print("\nIf unanimity is only an anti-signal in the first row, the claim is about")
    print("under-specification, not about ensembles.")

    Path("results/pilot_unanimity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nwritten to results/pilot_unanimity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
