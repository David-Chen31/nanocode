"""A deliberately simple warning about power on binary correctness.

Rounds 8 to 10 each pre-registered end-to-end correctness and each returned a
null. Before reading that as "the components do not matter", the honest question
is whether the design could resolve a practically relevant effect. This simple
simulation shows that a binary outcome over a dozen clusters is weak, but it is
not a calibrated model of the studies themselves.

Both detectors share the same homogeneous Bernoulli generator: there is no task
heterogeneity, arm correlation, or seed-pairing structure. Agreement between
them checks implementation sensitivity, not the assumptions of the generator.
Read the output as a warning and sensitivity calculation, never as proof that a
null result was guaranteed.

    py -3 experiments/power.py
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from statistics import NormalDist

DESIGNS = [
    ("round 8    12 tasks x 3 reps, base .42", 0.42, 3, 12),
    ("round 9    12 tasks x 3 reps, base .31", 0.31, 3, 12),
    ("round 10    8 tasks x 2 reps, base .94", 0.94, 2, 8),
    (None, 0, 0, 0),
    ("if instead 24 tasks x 2 reps, base .50", 0.50, 2, 24),
    ("if instead 48 tasks x 2 reps, base .50", 0.50, 2, 48),
    ("if instead 24 tasks x 2 reps, base .94", 0.94, 2, 24),
]


def _logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def _inv_logit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _sample(base: float, eff: float, reps: int, n: int, rng: random.Random,
            *, task_sd: float = 0.0, pairing: float = 0.0):
    """One simulated paired study under explicit sensitivity assumptions.

    `task_sd` is logit-scale heterogeneity in baseline task difficulty.
    `pairing` is the probability that a paired repetition uses the same uniform
    draw in both arms (a transparent common-random-number dependence model).
    Neither is claimed to be estimated precisely by the small existing runs.
    """
    a, b = [], []
    for _ in range(n):
        p0 = _inv_logit(_logit(base) + rng.gauss(0, task_sd))
        p1 = min(1.0, p0 + eff)
        av, bv = [], []
        for _ in range(reps):
            ua = rng.random()
            ub = ua if rng.random() < pairing else rng.random()
            av.append(1.0 if ua < p0 else 0.0)
            bv.append(1.0 if ub < p1 else 0.0)
        a.append(statistics.mean(av))
        b.append(statistics.mean(bv))
    return a, b


def detect_normal(base, eff, reps, n, rng, **sample_kw) -> bool:
    a, b = _sample(base, eff, reps, n, rng, **sample_kw)
    d = [y - x for x, y in zip(a, b)]
    m = statistics.mean(d)
    se = statistics.pstdev(d) / math.sqrt(len(d) - 1) if len(d) > 1 else 0.0
    return m != 0.0 if se == 0 else abs(m) > 1.96 * se


def detect_bootstrap(base, eff, reps, n, rng, boots=300, **sample_kw) -> bool:
    """The decision rule the analyses actually use: resample whole tasks."""
    a, b = _sample(base, eff, reps, n, rng, **sample_kw)
    idx = range(n)
    stat = lambda s: statistics.mean(b[i] - a[i] for i in s)
    draws = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(boots))
    lo, hi = draws[int(0.025 * boots)], draws[int(0.975 * boots)]
    return not (lo <= 0 <= hi)


def effect_grid(base: float) -> list[float]:
    """Feasible positive additive effects, never values beyond success=1."""
    top = min(0.70, max(0.0, 1.0 - base))
    return [i / 100 for i in range(2, int(top * 100 + 1e-9) + 1, 2)]


def mde(base, reps, n, detector, trials, seed=11, **sample_kw) -> float | None:
    """Smallest true effect the design finds 80% of the time."""
    rng = random.Random(seed)
    for eff in effect_grid(base):
        hits = sum(detector(base, eff, reps, n, rng, **sample_kw)
                   for _ in range(trials))
        if hits / trials >= 0.80:
            return eff
    return None


def false_positive_rate(base, reps, n, detector, trials, seed=19,
                        **sample_kw) -> float:
    rng = random.Random(seed)
    return sum(detector(base, 0.0, reps, n, rng, **sample_kw)
               for _ in range(trials)) / trials


def wilson_interval(successes: int, total: int,
                    confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval for the pilot's paired discordance probability."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes and total must describe a non-empty sample")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def required_paired_blocks(discordance: float, margin: float = 0.05,
                           power: float = 0.80,
                           confidence: float = 0.95) -> int:
    """Normal-approximation blocks for paired non-inferiority at true delta 0.

    A paired binary contrast is -1, 0 or +1. At zero mean its variance is the
    discordance probability. The confirmatory rule uses the lower end of a
    *two-sided* confidence interval, hence z_(1-alpha/2), not a more permissive
    one-sided critical value.
    """
    if not 0 <= discordance <= 1:
        raise ValueError("discordance must be between 0 and 1")
    if not 0 < margin < 1 or not 0 < power < 1 or not 0 < confidence < 1:
        raise ValueError("margin, power and confidence must be between 0 and 1")
    z_ci = NormalDist().inv_cdf(0.5 + confidence / 2)
    z_power = NormalDist().inv_cdf(power)
    return max(1, math.ceil(discordance * ((z_ci + z_power) / margin) ** 2))


def simulate_paired_noninferiority(tasks: int, reps: int, discordance: float,
                                   margin: float = 0.05, *, trials: int = 300,
                                   boots: int = 300, seed: int = 20260901) -> float:
    """Power of the actual stratified repetition-bootstrap decision rule.

    The true treatment difference is zero. Discordant paired blocks split
    evenly between treatment wins and losses; concordant blocks contribute 0.
    Tasks stay fixed while repetitions are resampled within task, matching
    `experiments.analyse.fixed_task_contrast`.
    """
    if tasks <= 0 or reps <= 0 or trials <= 0 or boots <= 0:
        raise ValueError("tasks, reps, trials and boots must be positive")
    rng = random.Random(seed)
    passes = 0
    cutoff = int(0.025 * boots)
    for _ in range(trials):
        effects: list[list[float]] = []
        for _task in range(tasks):
            row = []
            for _rep in range(reps):
                u = rng.random()
                row.append(-1.0 if u < discordance / 2
                           else (1.0 if u < discordance else 0.0))
            effects.append(row)
        draws = []
        for _boot in range(boots):
            per_task = [statistics.mean(rng.choice(row) for _ in row)
                        for row in effects]
            draws.append(statistics.mean(per_task))
        draws.sort()
        passes += draws[cutoff] > -margin
    return passes / trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--slow", action="store_true",
                    help="Also run the cluster bootstrap (minutes, not seconds).")
    ap.add_argument("--sensitivity", action="store_true",
                    help="Also vary task heterogeneity and within-pair dependence.")
    ap.add_argument("--noninferiority", action="store_true",
                    help="Design the paired fixed-task non-inferiority study.")
    ap.add_argument("--pilot-discordant", type=int, default=6)
    ap.add_argument("--pilot-pairs", type=int, default=36)
    ap.add_argument("--ni-tasks", type=int, default=12)
    ap.add_argument("--ni-margin", type=float, default=0.05)
    ap.add_argument("--ni-reps", type=int, default=None,
                    help="Repetitions to simulate (default: conservative design).")
    ap.add_argument("--ni-boots", type=int, default=300)
    args = ap.parse_args()

    print("MINIMUM DETECTABLE EFFECT on a binary correctness outcome")
    print("(the smallest real difference the design finds 80% of the time)\n")
    assumptions = [("homogeneous, independent", 0.0, 0.0)]
    if args.sensitivity:
        assumptions += [("heterogeneous, independent", 0.8, 0.0),
                        ("heterogeneous, paired", 0.8, 0.6)]

    def fmt(m, base):
        if m is not None:
            return f"{m * 100:.0f} pts"
        return f"none <= {(1 - base) * 100:.0f}"

    for assumption, task_sd, pairing in assumptions:
        print(f"\nASSUMPTION: {assumption}  "
              f"(task_sd={task_sd}, pairing={pairing})")
        head = f"  {'design':<42}{'alpha-N':>9}{'MDE-N':>12}"
        if args.slow:
            head += f"{'alpha-B':>10}{'MDE-B':>10}"
        print(head)
        for label, base, reps, n in DESIGNS:
            if label is None:
                print()
                continue
            kw = {"task_sd": task_sd, "pairing": pairing}
            alpha_n = false_positive_rate(base, reps, n, detect_normal,
                                          args.trials, **kw)
            mde_n = mde(base, reps, n, detect_normal, args.trials, **kw)
            line = f"  {label:<42}{alpha_n:>9.3f}{fmt(mde_n, base):>12}"
            if args.slow:
                slow_trials = max(200, args.trials // 4)
                alpha_b = false_positive_rate(base, reps, n, detect_bootstrap,
                                              slow_trials, **kw)
                mde_b = mde(base, reps, n, detect_bootstrap, slow_trials, **kw)
                line += f"{alpha_b:>10.3f}{fmt(mde_b, base):>10}"
            print(line)

    print("\nThese sensitivity simulations are a warning that the binary")
    print("outcome is underpowered, not a calibrated power analysis of the actual")
    print("paired heterogeneous tasks. See docs/FINDINGS.md, round 11.")
    if args.noninferiority:
        q = args.pilot_discordant / args.pilot_pairs
        q_lo, q_hi = wilson_interval(args.pilot_discordant, args.pilot_pairs)
        point_blocks = required_paired_blocks(q, margin=args.ni_margin)
        conservative_blocks = required_paired_blocks(q_hi, margin=args.ni_margin)
        conservative_reps = math.ceil(conservative_blocks / args.ni_tasks)
        reps = args.ni_reps or conservative_reps
        sim_trials = max(100, min(args.trials, 500))
        sim_power = simulate_paired_noninferiority(
            args.ni_tasks, reps, q_hi, args.ni_margin,
            trials=sim_trials, boots=args.ni_boots)
        print("\nPAIRED NON-INFERIORITY DESIGN")
        print(f"  pilot discordance       {args.pilot_discordant}/{args.pilot_pairs} "
              f"= {q:.3f}  (Wilson 95% [{q_lo:.3f}, {q_hi:.3f}])")
        print(f"  margin / CI / power     {args.ni_margin:.3f} / two-sided 95% / 80%")
        print(f"  blocks at point q       {point_blocks}")
        print(f"  blocks at upper q       {conservative_blocks}")
        print(f"  conservative reps       {conservative_reps} per task")
        print(f"  simulated reps          {reps} per task")
        print(f"  bootstrap power at q_hi {sim_power:.3f} "
              f"({sim_trials} trials, {args.ni_boots} bootstraps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
