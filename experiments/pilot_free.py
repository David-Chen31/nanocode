"""Two pilots that cost nothing: they run on samples already on disk.

C. Cross-model agreement as a *correctness* predictor.
   Earlier work asked whether cross-model disagreement detects ambiguity (it did
   not, AUROC 0.472). This is a different question with a different label: when
   five model families independently agree on a behaviour, is that behaviour more
   likely to be the one the user wanted? Agreement-as-correctness is the
   ensemble question; disagreement-as-ambiguity was the uncertainty question.

D. Which kinds of constraint do models drop?
   Every task in the diagnostic set deletes one sentence, and those sentences
   fall into a small number of kinds. Tagging them turns twelve anecdotes into a
   first taxonomy of what a coding agent fails to infer.

    python experiments/pilot_free.py
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

# The kind of decision each deleted sentence settles. Assigned by hand from the
# constraint text, before looking at any model's answers.
KIND = {
    "t01_remove_outliers": "degenerate input",
    "t04_normalize": "degenerate input",
    "t09_split_name": "degenerate input",
    "t10_moving_average": "degenerate input",
    "t02_top_k": "ordering / selection",
    "t07_dedupe": "ordering / selection",
    "t03_chunk": "boundary convention",
    "t06_parse_range": "boundary convention",
    "t11_truncate": "boundary convention",
    "t05_merge_configs": "precedence",
    "t08_round_price": "numeric convention",
    "t12_sort_scores": "mutation contract",
}


def load_models() -> dict[str, Any]:
    out = {}
    for f in sorted(glob.glob("results/live/*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if any(x in f for x in ("T16", "diverse", "naive", "k30")):
            continue
        out[d["model"]] = d
    return out


def behaviour_rows(task, sources) -> tuple[list[list[str]], list[str]]:
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    m = run_candidates(sources, probes, task.entry_point)
    ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    return m.tokens, ref


def main() -> int:
    live = load_models()
    models = sorted(live)
    tasks = load_tasks()
    print(f"models: {', '.join(models)}\n")

    # ---------------------------------------------------------------- pilot C
    print("=" * 78)
    print("C. Does cross-model AGREEMENT predict correctness?")
    print("=" * 78)
    print(f"{'task':<22}{'kind':<21}{'models agreeing':>16}{'agreed = right?':>17}")
    print("-" * 78)

    agree_right = agree_wrong = split_right = split_wrong = 0
    per_kind: dict[str, list[bool]] = defaultdict(list)

    for t in tasks:
        first = [live[m]["samples"][t.id]["ambiguous"]["sources"][0] for m in models]
        rows, ref = behaviour_rows(t, first)
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for m, r in zip(models, rows):
            groups[tuple(r)].append(m)
        biggest = max(groups.values(), key=len)
        winner_row = next(k for k, v in groups.items() if v is biggest)
        consensus = len(biggest)
        right = list(winner_row) == ref

        if consensus == len(models):
            if right:
                agree_right += 1
            else:
                agree_wrong += 1
        else:
            if right:
                split_right += 1
            else:
                split_wrong += 1

        per_kind[KIND[t.id]].append(right)
        print(f"{t.id:<22}{KIND[t.id]:<21}{consensus:>10}/{len(models)}"
              f"{('YES' if right else 'NO'):>17}")

    n = len(tasks)
    unan = agree_right + agree_wrong
    print()
    print(f"unanimous across all {len(models)} families : {unan}/{n} tasks, "
          f"of which {agree_right} correct  ->  precision {agree_right / max(1, unan):.2f}")
    print(f"split                                : {split_right + split_wrong}/{n} tasks, "
          f"of which {split_right} correct  ->  precision "
          f"{split_right / max(1, split_right + split_wrong):.2f}")
    print()
    if unan and (split_right + split_wrong):
        lift = agree_right / unan - split_right / (split_right + split_wrong)
        print(f"agreement lift over split: {lift:+.2f}")
        print("A positive lift means unanimity is worth something as a correctness")
        print("signal even though disagreement was worthless as an ambiguity signal.")

    # ---------------------------------------------------------------- pilot D
    print("\n" + "=" * 78)
    print("D. Which kinds of constraint do models fail to infer?")
    print("=" * 78)
    print(f"{'constraint kind':<24}{'tasks':>7}{'majority correct':>19}")
    print("-" * 78)

    by_kind: dict[str, list[bool]] = defaultdict(list)
    for t in tasks:
        for m in models:
            srcs = live[m]["samples"][t.id]["ambiguous"]["sources"][:6]
            rows, ref = behaviour_rows(t, srcs)
            groups: dict[tuple[str, ...], int] = defaultdict(int)
            for r in rows:
                groups[tuple(r)] += 1
            top = max(groups, key=lambda k: groups[k])
            by_kind[KIND[t.id]].append(list(top) == ref)

    order = sorted(by_kind, key=lambda k: sum(by_kind[k]) / len(by_kind[k]))
    for kind in order:
        vals = by_kind[kind]
        ntask = len({t.id for t in tasks if KIND[t.id] == kind})
        print(f"{kind:<24}{ntask:>7}{sum(vals)}/{len(vals)} = "
              f"{sum(vals) / len(vals):.2f}".rjust(19))
    print("\n(denominator = tasks of that kind x 5 model families)")

    out = {
        "pilot_C": {"unanimous": unan, "unanimous_correct": agree_right,
                    "split": split_right + split_wrong, "split_correct": split_right,
                    "n_tasks": n, "models": models},
        "pilot_D": {k: {"correct": sum(v), "total": len(v)} for k, v in by_kind.items()},
        "kinds": KIND,
    }
    Path("results/pilot_free.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwritten to results/pilot_free.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
