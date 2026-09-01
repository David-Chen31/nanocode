"""Read any of the sweep result files and report a panel of outcomes.

WHY THIS REPLACED A SINGLE-OUTCOME ANALYSIS

Rounds 8, 9 and 10 each pre-registered end-to-end correctness as the primary
outcome, and each returned a null on it. That looked like "the components do not
matter". It was not. A homogeneous Bernoulli sensitivity calculation warned
that correctness would resolve only very large effects under its assumptions:

    round 8    12 tasks x 3 reps, base .42    ~30-34 points
    round 9    12 tasks x 3 reps, base .31    ~34 points
    round 10    8 tasks x 2 reps, base .94     no feasible positive effect

(Both detectors share one generator, so their agreement does not validate its
assumptions. See experiments/power.py.)

The results therefore show severe underprecision, not that nulls were guaranteed.
Round 8's own manipulation check shows the scale problem: deleting one sentence from the
requirement moved correctness 33 points, while every component moved under 10.
The task's information content dominates; the machinery does not compete with it.

The machinery instead governs how much work gets done and how runs fail. Those
are continuous, lower-variance quantities in these data. That is why the only
clearly separated results in three rounds were model calls and tokens.

READING THE OUTPUT HONESTLY

PRIMARY is what the pre-registration named. SECONDARY was recorded but not
pre-registered as primary, so it is exploratory: mechanism-motivated rather than
fished for, but exploratory all the same. With five outcomes across three
comparisons, some interval will exclude zero by chance -- treat an effect whose
interval barely clears zero as noise, and weigh the ones that clear it widely.

    py -3 experiments/analyse.py results/ablation.json --baseline full
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

BOOTS = 5000
SEED = 20260830

# The pre-registered outcome, then the mechanism-proximate ones. Each says what
# it is evidence about, because "significant" is meaningless without that.
PRIMARY: list[tuple[str, Callable[[dict], float], str]] = [
    ("correct", lambda r: 1.0 if r.get("correct") else 0.0,
     "did the change work end to end"),
]
GRADED: list[tuple[str, Callable[[dict], float], str]] = [
    ("assertions", lambda r: float(r.get("behaviour_frac") or 0.0),
     "fraction of the hidden verifier that passed"),
]
def model_calls(r: dict) -> float:
    """Total model calls, repairing the original architecture result schema.

    Old `architecture.json` rows counted only trace kind="model" and omitted
    the one kind="plan" call. New rows record `n_planning_calls` and already
    include it in `n_model_calls`, so the compatibility adjustment is applied
    only to the historical rows.
    """
    n = float(r.get("n_model_calls") or 0)
    if r.get("arm") == "plan_execute" and "n_planning_calls" not in r:
        n += 1.0
    return n


SECONDARY: list[tuple[str, Callable[[dict], float], str]] = [
    ("finished", lambda r: 1.0 if r.get("outcome") == "finished" else 0.0,
     "did the loop terminate explicitly"),
    ("model calls", model_calls,
     "how much thinking the run cost"),
    ("tool calls", lambda r: float(r.get("n_tool_calls") or 0),
     "how much acting the run cost"),
    ("commands run", lambda r: float(r.get("n_runs") or 0),
     "how often it reached for the shell"),
    ("failed commands", lambda r: float(r.get("n_failed_runs") or 0),
     "how much of that shell work was wasted"),
    ("tokens (k)", lambda r: (float(r.get("input_tokens") or 0)
                              + float(r.get("output_tokens") or 0)) / 1000.0,
     "total context cost"),
    ("crashed", lambda r: 1.0 if r.get("crashed") else 0.0,
     "did the run die instead of finishing"),
]


def load(path: str) -> tuple[list[dict[str, Any]], str]:
    rows = [r for r in load_document(path)["rows"]
            if r.get("outcome") != "error"]
    # The sweeps name their independent variable differently; both are the same
    # kind of thing, so the analysis should not care which.
    key = "condition" if rows and "condition" in rows[0] else "arm"
    return rows, key


def load_document(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def by_task(rows, key, cond, f) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r[key] == cond:
            out[r["task"]].append(f(r))
    return out


def cluster_boot(a, b, n: int = BOOTS) -> tuple[float, float, float]:
    """Paired difference with a 95% interval, resampling whole tasks.

    Repetitions of one task share a prompt, a repository and a difficulty, so
    resampling trajectories would treat correlated runs as independent and
    report intervals far too narrow. The task was the sampling unit, so the task
    is the resampling unit.
    """
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return float("nan"), float("nan"), float("nan")

    def stat(sample: list[str]) -> float:
        return statistics.mean(statistics.mean(b[t]) - statistics.mean(a[t])
                               for t in sample)

    point = stat(tasks)
    rng = random.Random(SEED)
    draws = sorted(stat([rng.choice(tasks) for _ in tasks]) for _ in range(n))
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def factorial_interaction(rows, key: str, baseline: str, factor_a: str,
                          factor_b: str, both: str, f,
                          n: int = BOOTS) -> tuple[float, float, float]:
    """Cluster-bootstrap a 2x2 difference in differences.

    The statistic is `both - factor_a - factor_b + baseline`, computed on
    within-task means before tasks are resampled. Comparing which individual
    arms have intervals excluding zero is not a test of this interaction.
    """
    cells = {c: by_task(rows, key, c, f)
             for c in (baseline, factor_a, factor_b, both)}
    tasks = sorted(set.intersection(*(set(v) for v in cells.values())))
    if not tasks:
        return float("nan"), float("nan"), float("nan")

    effects = {
        t: (statistics.mean(cells[both][t])
            - statistics.mean(cells[factor_a][t])
            - statistics.mean(cells[factor_b][t])
            + statistics.mean(cells[baseline][t]))
        for t in tasks
    }
    point = statistics.mean(effects.values())
    rng = random.Random(SEED)
    draws = sorted(statistics.mean(effects[rng.choice(tasks)] for _ in tasks)
                   for _ in range(n))
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def fixed_task_contrast(rows, key: str, coefficients: dict[str, float], f,
                        n: int = BOOTS) -> tuple[float, float, float]:
    """Inference over stochastic repetitions, conditional on fixed tasks.

    Tasks are never resampled. Repetition blocks are resampled within each
    task, preserving equal task weights and all treatment cells in a block.
    This answers a different estimand from `cluster_boot`, which treats tasks
    as the sampled clusters of a task population.
    """
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row[key] in coefficients and row.get("outcome") != "error":
            values[(row["task"], int(row.get("rep", 0)), row[key])].append(f(row))
    tasks = sorted({task for task, _, _ in values})
    complete: dict[str, list[int]] = {}
    for task in tasks:
        reps = sorted({rep for t, rep, _ in values if t == task})
        complete[task] = [rep for rep in reps
                          if all((task, rep, cell) in values for cell in coefficients)]
    complete = {task: reps for task, reps in complete.items() if reps}
    if not complete:
        return float("nan"), float("nan"), float("nan")

    def block_effect(task: str, rep: int) -> float:
        return sum(coef * statistics.mean(values[(task, rep, cell)])
                   for cell, coef in coefficients.items())

    point = statistics.mean(
        statistics.mean(block_effect(task, rep) for rep in reps)
        for task, reps in complete.items())
    rng = random.Random(SEED)
    draws = []
    for _ in range(n):
        per_task = []
        for task, reps in complete.items():
            sampled = [rng.choice(reps) for _ in reps]
            per_task.append(statistics.mean(block_effect(task, rep) for rep in sampled))
        draws.append(statistics.mean(per_task))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def _has(rows: list[dict[str, Any]], name: str) -> bool:
    """Distinguish an absent metric from a valid metric that is zero everywhere."""
    fields = {
        "correct": ("correct",),
        "assertions": ("behaviour_frac",),
        "finished": ("outcome",),
        "model calls": ("n_model_calls",),
        "tool calls": ("n_tool_calls",),
        "commands run": ("n_runs",),
        "failed commands": ("n_failed_runs",),
        "tokens (k)": ("input_tokens", "output_tokens"),
        "crashed": ("crashed",),
    }
    required = fields[name]
    return bool(rows) and all(all(field in row for field in required) for row in rows)


def report(rows, key, baseline: str, *, fixed_tasks: bool = False,
           boots: int = BOOTS) -> None:
    conds = [c for c in dict.fromkeys(r[key] for r in rows) if c != baseline]
    tasks = sorted({r["task"] for r in rows})
    estimand = ("fixed tasks; repetition blocks resampled within task"
                if fixed_tasks else "task-population sensitivity; tasks resampled")
    print(f"{len(rows)} runs, {len(tasks)} tasks, baseline = {baseline}")
    print(f"inference = {estimand}\n")

    for label, panel in (("PRIMARY (pre-registered)", PRIMARY),
                         ("GRADED (same outcome, not thrown away as pass/fail)",
                          GRADED),
                         ("SECONDARY (exploratory: recorded, not pre-registered "
                          "as primary)", SECONDARY)):
        panel = [(n, f, w) for n, f, w in panel if _has(rows, n)]
        if not panel:
            continue
        print("=" * 78)
        print(label)
        print("=" * 78)
        for cond in conds:
            # `cluster_boot(a=baseline, b=cond)` returns mean(cond) - mean(baseline),
            # so the header has to name the ablated condition first. It used to read
            # "{baseline} minus {cond}", the exact opposite of every number under it:
            # search *raises* tool calls by 6.17 and the table announced a saving of
            # 6.17. The prose in DECISIONS.md was read off the computation and is
            # right; the header was the liar. Pinned by a test now.
            print(f"\n  {cond} minus {baseline}   (the effect of ablating)")
            for name, f, why in panel:
                if fixed_tasks:
                    p, lo, hi = fixed_task_contrast(
                        rows, key, {cond: 1.0, baseline: -1.0}, f, n=boots)
                else:
                    p, lo, hi = cluster_boot(by_task(rows, key, baseline, f),
                                             by_task(rows, key, cond, f), n=boots)
                mark = "" if lo <= 0 <= hi else "  <- excludes 0"
                print(f"    {name:<16}{p:+8.2f}  [{lo:+7.2f}, {hi:+7.2f}]{mark}"
                      f"{'':4}{why if mark else ''}")
        print()


def report_interaction(rows, key: str, baseline: str,
                       factor_a: str, factor_b: str, both: str, *,
                       fixed_tasks: bool = False, boots: int = BOOTS) -> None:
    print("=" * 78)
    print("FACTORIAL INTERACTION (exploratory unless pre-registered)")
    print(f"  {both} - {factor_a} - {factor_b} + {baseline}")
    print("=" * 78)
    for panel in (PRIMARY, GRADED, SECONDARY):
        for name, f, _ in panel:
            if not _has(rows, name):
                continue
            if fixed_tasks:
                p, lo, hi = fixed_task_contrast(
                    rows, key, {both: 1.0, factor_a: -1.0,
                                factor_b: -1.0, baseline: 1.0}, f, n=boots)
            else:
                p, lo, hi = factorial_interaction(
                    rows, key, baseline, factor_a, factor_b, both, f, n=boots)
            mark = "" if lo <= 0 <= hi else "  <- excludes 0"
            print(f"  {name:<18}{p:+8.2f}  [{lo:+7.2f}, {hi:+7.2f}]{mark}")
    print()


def _complete_paired_rows(rows: list[dict[str, Any]], key: str,
                          baseline: str, treatment: str) -> tuple[list[dict[str, Any]],
                                                                  int, int]:
    """Keep only complete task x repetition blocks for a two-arm decision."""
    relevant = [r for r in rows if r.get(key) in (baseline, treatment)]
    blocks: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in relevant:
        if row.get("outcome") == "error":
            continue
        blocks[(row["task"], int(row.get("rep", 0)))][row[key]].append(row)
    complete = []
    for cells in blocks.values():
        if all(len(cells[c]) == 1 for c in (baseline, treatment)):
            complete.extend(cells[baseline] + cells[treatment])
    total_blocks = len({(r["task"], int(r.get("rep", 0))) for r in relevant})
    return complete, len(complete) // 2, total_blocks


def confirmatory_decision(document: dict[str, Any], *, baseline: str,
                          treatment: str, margin: float = 0.05,
                          infra_threshold: float = 0.05,
                          boots: int = BOOTS,
                          expected_model: str | None = None,
                          expected_reps: int | None = None,
                          expected_tasks: list[str] | None = None,
                          expected_schedule_seed: int | None = None,
                          expected_max_steps: int | None = None,
                          expected_max_tokens: int | None = None,
                          expected_temperature: float | None = None) -> dict[str, Any]:
    """Fail-closed cost-quality decision for the pre-registered search study."""
    raw = list(document.get("rows") or [])
    key = "condition" if raw and "condition" in raw[0] else "arm"
    relevant = [r for r in raw if r.get(key) in (baseline, treatment)]
    reasons: list[str] = []

    manifest = document.get("manifest") or {}
    if manifest.get("git_dirty") is not False:
        reasons.append("manifest does not prove a clean git worktree")
    run_args = manifest.get("args") or {}
    if expected_model is not None and document.get("model") != expected_model:
        reasons.append("model does not match the frozen protocol")
    if expected_reps is not None and document.get("reps") != expected_reps:
        reasons.append("repetition count does not match the frozen protocol")
    if expected_tasks is not None:
        got_tasks = {r.get("task") for r in raw}
        if got_tasks != set(expected_tasks):
            reasons.append("task set does not match the frozen protocol")
    frozen_args = {
        "schedule_seed": expected_schedule_seed,
        "max_steps": expected_max_steps,
        "max_tokens": expected_max_tokens,
        "temperature": expected_temperature,
    }
    for name, expected in frozen_args.items():
        if expected is not None and run_args.get(name) != expected:
            reasons.append(f"{name} does not match the frozen protocol")
    if expected_model is not None:
        if set(r.get(key) for r in raw) != {baseline, treatment}:
            reasons.append("conditions do not match the frozen two-arm protocol")
        for guard in ("require_clean", "require_model_snapshot",
                      "fail_if_output_exists"):
            if run_args.get(guard) is not True:
                reasons.append(f"run did not enable {guard}")
        if run_args.get("unambiguous") is not False:
            reasons.append("prompt variant does not match the frozen protocol")
        if expected_tasks is not None and expected_reps is not None:
            expected_rows = len(expected_tasks) * expected_reps * 2
            if len(raw) != expected_rows:
                reasons.append("row count does not match the frozen schedule")

    infra_rates = {}
    for cond in (baseline, treatment):
        cell = [r for r in relevant if r.get(key) == cond]
        errors = sum(r.get("outcome") == "error" for r in cell)
        infra_rates[cond] = errors / len(cell) if cell else 1.0
        if infra_rates[cond] > infra_threshold:
            reasons.append(f"{cond} infrastructure error rate exceeds threshold")

    paired, complete_blocks, total_blocks = _complete_paired_rows(
        relevant, key, baseline, treatment)
    if not complete_blocks:
        reasons.append("no complete paired blocks")
    required = ("correct", "behaviour_frac", "input_tokens", "output_tokens")
    missing = [name for name in required
               if any(name not in row for row in paired)]
    if missing:
        reasons.append("missing confirmatory fields: " + ", ".join(missing))

    result: dict[str, Any] = {
        "status": "INVALID" if reasons else "PENDING",
        "baseline": baseline, "treatment": treatment,
        "margin": margin, "infra_threshold": infra_threshold,
        "infrastructure_error_rates": infra_rates,
        "complete_blocks": complete_blocks, "total_blocks": total_blocks,
        "excluded_blocks": total_blocks - complete_blocks,
        "reasons": reasons,
    }
    if reasons:
        return result

    contrasts = {}
    metrics = {
        "correct": lambda r: 1.0 if r.get("correct") else 0.0,
        "behaviour_frac": lambda r: float(r["behaviour_frac"]),
        "tokens": lambda r: (float(r["input_tokens"])
                             + float(r["output_tokens"])),
    }
    for name, f in metrics.items():
        point, lo, hi = fixed_task_contrast(
            paired, key, {treatment: 1.0, baseline: -1.0}, f, n=boots)
        contrasts[name] = {"estimate": point, "ci95": [lo, hi]}
    result["contrasts"] = contrasts

    quality_ok = (contrasts["correct"]["ci95"][0] > -margin
                  and contrasts["behaviour_frac"]["ci95"][0] > -margin)
    cost_ok = contrasts["tokens"]["ci95"][1] < 0
    if quality_ok and cost_ok:
        result["status"] = "PASS"
        result["reasons"] = ["quality is non-inferior and token use is lower"]
    else:
        result["status"] = "FAIL"
        result["reasons"] = []
        if not quality_ok:
            result["reasons"].append("quality non-inferiority was not established")
        if not cost_ok:
            result["reasons"].append("lower token use was not established")
    return result


def print_confirmatory_decision(result: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"CONFIRMATORY DECISION: {result['status']}")
    print("=" * 78)
    print(f"  complete blocks  {result['complete_blocks']}/{result['total_blocks']}")
    for cond, rate in result["infrastructure_error_rates"].items():
        print(f"  infra {cond:<18}{rate:.1%}")
    for name, item in result.get("contrasts", {}).items():
        lo, hi = item["ci95"]
        print(f"  {name:<20}{item['estimate']:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--interaction", nargs=3,
                    metavar=("FACTOR_A", "FACTOR_B", "BOTH"),
                    help="Report the 2x2 interaction BOTH-A-B+baseline.")
    ap.add_argument("--fixed-tasks", action="store_true",
                    help="Keep tasks fixed; resample repetition blocks within task.")
    ap.add_argument("--boots", type=int, default=BOOTS,
                    help="Bootstrap draws (default: 5000).")
    ap.add_argument("--noninferiority-margin", type=float, default=None,
                    help="Run the fail-closed confirmatory two-arm decision.")
    ap.add_argument("--treatment", default=None,
                    help="Treatment/ablation name for the confirmatory decision.")
    ap.add_argument("--infra-threshold", type=float, default=0.05,
                    help="Maximum infrastructure error rate per condition.")
    ap.add_argument("--decision-json", default=None,
                    help="Optional path for the machine-readable verdict.")
    ap.add_argument("--expected-model", default=None)
    ap.add_argument("--expected-reps", type=int, default=None)
    ap.add_argument("--expected-tasks", nargs="+", default=None)
    ap.add_argument("--expected-schedule-seed", type=int, default=None)
    ap.add_argument("--expected-max-steps", type=int, default=None)
    ap.add_argument("--expected-max-tokens", type=int, default=None)
    ap.add_argument("--expected-temperature", type=float, default=None)
    args = ap.parse_args()
    rows, key = load(args.path)
    report(rows, key, args.baseline, fixed_tasks=args.fixed_tasks, boots=args.boots)
    if args.interaction:
        report_interaction(rows, key, args.baseline, *args.interaction,
                           fixed_tasks=args.fixed_tasks, boots=args.boots)
    if args.noninferiority_margin is not None:
        if not 0 < args.noninferiority_margin < 1:
            ap.error("--noninferiority-margin must be between 0 and 1")
        if not args.treatment:
            ap.error("--treatment is required with --noninferiority-margin")
        if not args.fixed_tasks:
            ap.error("confirmatory non-inferiority requires --fixed-tasks")
        expected = (args.expected_model, args.expected_reps, args.expected_tasks,
                    args.expected_schedule_seed, args.expected_max_steps,
                    args.expected_max_tokens, args.expected_temperature)
        if any(value is None for value in expected):
            ap.error("confirmatory decision requires every --expected-* protocol field")
        decision = confirmatory_decision(
            load_document(args.path), baseline=args.baseline,
            treatment=args.treatment, margin=args.noninferiority_margin,
            infra_threshold=args.infra_threshold, boots=args.boots,
            expected_model=args.expected_model, expected_reps=args.expected_reps,
            expected_tasks=args.expected_tasks,
            expected_schedule_seed=args.expected_schedule_seed,
            expected_max_steps=args.expected_max_steps,
            expected_max_tokens=args.expected_max_tokens,
            expected_temperature=args.expected_temperature)
        print_confirmatory_decision(decision)
        if args.decision_json:
            target = Path(args.decision_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(decision, indent=1), encoding="utf-8")
        print("Intervals are conditional on the fixed tasks and quantify model "
              "repetition uncertainty.")
        return {"PASS": 0, "FAIL": 2, "INVALID": 3}[decision["status"]]
    print("Intervals are wide because the task count is small. An interval that "
          "contains 0\nmeans this study could not separate them -- not that the "
          "effect is zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
