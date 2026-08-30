"""What could these designs have detected? The backbone of round 11.

Rounds 8 to 10 each pre-registered end-to-end correctness and each returned a
null. Before reading that as "the components do not matter", the honest question
is whether the design could have found an effect had one existed. It could not:
a binary outcome over a dozen task-clusters resolves only very large
differences, and no component of a coding agent moves end-to-end correctness
that far.

Two estimators are run because a single one can be wrong in ways that are hard
to see. The cluster bootstrap is the same procedure the analyses use, so it
reports the power of the actual decision rule; the normal approximation is
independent of it and much faster. They agree to within a couple of points,
which is why the number is quoted at all.

    py -3 experiments/power.py
"""
from __future__ import annotations

import argparse
import math
import random
import statistics

DESIGNS = [
    ("round 8    12 tasks x 3 reps, base .42", 0.42, 3, 12),
    ("round 9    12 tasks x 3 reps, base .31", 0.31, 3, 12),
    ("round 10    8 tasks x 2 reps, base .94", 0.94, 2, 8),
    (None, 0, 0, 0),
    ("if instead 24 tasks x 2 reps, base .50", 0.50, 2, 24),
    ("if instead 48 tasks x 2 reps, base .50", 0.50, 2, 48),
    ("if instead 24 tasks x 2 reps, base .94", 0.94, 2, 24),
]


def _sample(base: float, eff: float, reps: int, n: int, rng: random.Random):
    """One simulated study: per-task means for a control and a treated arm."""
    draw = lambda p: statistics.mean(1.0 if rng.random() < p else 0.0
                                     for _ in range(reps))
    a = [draw(base) for _ in range(n)]
    b = [draw(min(1.0, base + eff)) for _ in range(n)]
    return a, b


def detect_normal(base, eff, reps, n, rng) -> bool:
    a, b = _sample(base, eff, reps, n, rng)
    d = [y - x for x, y in zip(a, b)]
    m = statistics.mean(d)
    se = statistics.pstdev(d) / math.sqrt(len(d) - 1) if len(d) > 1 else 0.0
    return m != 0.0 if se == 0 else abs(m) > 1.96 * se


def detect_bootstrap(base, eff, reps, n, rng, boots=300) -> bool:
    """The decision rule the analyses actually use: resample whole tasks."""
    a, b = _sample(base, eff, reps, n, rng)
    idx = range(n)
    stat = lambda s: statistics.mean(b[i] - a[i] for i in s)
    draws = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(boots))
    lo, hi = draws[int(0.025 * boots)], draws[int(0.975 * boots)]
    return not (lo <= 0 <= hi)


def mde(base, reps, n, detector, trials, seed=11) -> float | None:
    """Smallest true effect the design finds 80% of the time."""
    rng = random.Random(seed)
    for eff in (i / 100 for i in range(2, 71, 2)):
        hits = sum(detector(base, eff, reps, n, rng) for _ in range(trials))
        if hits / trials >= 0.80:
            return eff
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--slow", action="store_true",
                    help="Also run the cluster bootstrap (minutes, not seconds).")
    args = ap.parse_args()

    print("MINIMUM DETECTABLE EFFECT on a binary correctness outcome")
    print("(the smallest real difference the design finds 80% of the time)\n")
    head = f"  {'design':<42}{'normal':>9}"
    if args.slow:
        head += f"{'bootstrap':>12}"
    print(head)
    for label, base, reps, n in DESIGNS:
        if label is None:
            print()
            continue
        fmt = lambda m: f"{m * 100:.0f} pts" if m else ">70 pts"
        line = f"  {label:<42}{fmt(mde(base, reps, n, detect_normal, args.trials)):>9}"
        if args.slow:
            slow = mde(base, reps, n, detect_bootstrap, max(200, args.trials // 4))
            line += f"{fmt(slow):>12}"
        print(line)

    print("\nNo component of a coding agent plausibly moves end-to-end correctness")
    print("by 30 points, so rounds 8-10's nulls on that outcome were guaranteed by")
    print("the design. See docs/FINDINGS.md, round 11.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
