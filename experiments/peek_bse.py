"""Quick look at BSE on each task pair. Not an experiment -- a sanity check."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from askoract.bse import compute_bse
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.schema import load_tasks


def main() -> None:
    tasks = load_tasks()
    print(f"{'task':<22} {'diff':<5} {'BSE(amb)':>9} {'BSE(full)':>10} {'delta':>7}  top-probe")
    print("-" * 100)
    for t in tasks:
        probes = filter_probes(synthesize_probes(t.seed_args, n=24, seed=0), t.precondition)
        row = []
        for variant in ("ambiguous", "full"):
            m = run_candidates(t.candidates(variant), probes, t.entry_point)
            row.append(compute_bse(m))
        amb, full = row
        top = amb.top_probe
        desc = f"{t.entry_point}{tuple(top.args)!r} -> {len(top.behaviours)} behaviours" if top else "-"
        print(f"{t.id:<22} {t.difficulty:<5} {amb.bse:>9.3f} {full.bse:>10.3f} "
              f"{amb.bse - full.bse:>7.3f}  {desc}")


if __name__ == "__main__":
    main()
