"""The three experiments, run offline over the diagnostic set.

  H1  Can a signal tell an under-specified prompt from a fully specified one,
      without seeing the answer?             -> AUROC over the 12 paired variants
  LOC Does it point at the right ambiguity?  -> precision@1 over the top probe
  H2  Under a fixed budget of questions, does ranking tasks by the signal beat
      spending the same questions at random? -> success rate vs interruptions

Everything here is deterministic given the seeds in bench/harness.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from askoract.bse import compute_bse
from askoract.policy import AskOrActPolicy
from askoract.signals import ConstantSignal, Signal, SignalInput, offline_signals
from bench.harness import OracleUser, TaskArtifacts, build_artifacts
from bench.schema import load_tasks

VARIANTS = ("ambiguous", "full")


# ---------------------------------------------------------------- statistics

def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUROC. Ties score half, which matters here: with only six
    samples per task, BSE takes a small number of distinct values."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def paired_accuracy(pairs: list[tuple[float, float]]) -> float:
    """Fraction of tasks where the ambiguous variant scores above its own fully
    specified twin. Ties count half."""
    if not pairs:
        return float("nan")
    return sum(1.0 if a > f else (0.5 if a == f else 0.0) for a, f in pairs) / len(pairs)


def bootstrap_ci(values: list[float], fn, *, n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(fn(sample))
    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return (lo, hi)


# ---------------------------------------------------------------- setup

def collect(n_probes: int, n_eval: int,
            live: dict[str, Any] | None = None) -> dict[str, dict[str, TaskArtifacts]]:
    """Execute everything once. This is the only expensive step.

    `live` swaps the hand-written fixture candidates for samples drawn from a
    real model. Nothing else about the pipeline changes, which is the point: the
    fixture and live runs are directly comparable.
    """
    out: dict[str, dict[str, TaskArtifacts]] = {}
    samples = (live or {}).get("samples", {})
    for t in load_tasks():
        out[t.id] = {}
        for v in VARIANTS:
            rec = samples.get(t.id, {}).get(v) if live else None
            out[t.id][v] = build_artifacts(
                t, v, n_probes=n_probes, n_eval=n_eval,
                sources=rec["sources"] if rec else None,
                mean_logprobs=rec.get("mean_logprobs") if rec else None,
                self_report=rec.get("self_report") if rec else None,
                direct_ask=rec.get("direct_ask") if rec else None,
            )
    return out


def score_of(sig: Signal, art: TaskArtifacts) -> float:
    return sig.score(SignalInput(task_id=art.task.id, sources=art.sources,
                                 matrix=art.matrix, bse=art.bse,
                                 mean_logprobs=art.mean_logprobs,
                                 self_report=art.self_report,
                                 direct_ask=art.direct_ask))


# ---------------------------------------------------------------- H1

def experiment_h1(cache, signals: list[Signal]) -> dict[str, Any]:
    rows = {}
    for sig in signals:
        amb, full, pairs, per_task = [], [], [], {}
        for tid, variants in cache.items():
            a = score_of(sig, variants["ambiguous"])
            f = score_of(sig, variants["full"])
            amb.append(a)
            full.append(f)
            pairs.append((a, f))
            per_task[tid] = {"ambiguous": round(a, 4), "full": round(f, 4),
                             "difficulty": variants["ambiguous"].task.difficulty}
        deltas = [a - f for a, f in pairs]
        lo, hi = bootstrap_ci(deltas, lambda s: sum(s) / len(s))
        rows[sig.name] = {
            "auroc": round(auroc(amb, full), 4),
            "paired_accuracy": round(paired_accuracy(pairs), 4),
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "delta_ci95": [round(lo, 4), round(hi, 4)],
            "per_task": per_task,
        }
    return rows


# ---------------------------------------------------------------- localisation

def experiment_localisation(cache) -> dict[str, Any]:
    """Does the top-ranked probe name the constraint that was removed?

    Reported with and without the precondition filter, because that filter turned
    out to matter more than any other single choice in the pipeline.
    """
    from askoract.execute import run_candidates
    from askoract.probes import synthesize_probes
    from bench.harness import CALIB_SEED, _norm

    hit_filtered, hit_raw, detail = 0, 0, {}
    for tid, variants in cache.items():
        art = variants["ambiguous"]
        wanted = art.constraint_probe_indices()
        top = art.bse.top_probe
        ok = bool(top and top.index in wanted)
        hit_filtered += int(ok)

        t = art.task
        raw_probes = synthesize_probes(t.seed_args, n=len(art.probes) + 8, seed=CALIB_SEED)
        raw_matrix = run_candidates(t.candidates("ambiguous"), raw_probes, t.entry_point)
        raw_top = compute_bse(raw_matrix).top_probe
        raw_wanted = {_norm(a) for c in t.constraints for a in c.discriminating_args}
        raw_ok = bool(raw_top and _norm(raw_probes[raw_top.index]) in raw_wanted)
        hit_raw += int(raw_ok)

        detail[tid] = {
            "hit": ok, "hit_without_precondition": raw_ok,
            "top_probe": top.args if top else None,
            "top_probe_unfiltered": raw_probes[raw_top.index] if raw_top else None,
        }

    n = len(cache)
    return {"precision_at_1": round(hit_filtered / n, 4),
            "precision_at_1_no_precondition": round(hit_raw / n, 4),
            "n_tasks": n, "per_task": detail}


# ---------------------------------------------------------------- H2

def outcome_curve(art: TaskArtifacts, max_asks: int) -> dict[str, Any]:
    """What happens to this task when granted 0, 1, ... max_asks questions.

    Precomputing this collapses every later sweep into arithmetic: once the
    policy has decided to ask, its behaviour does not depend on the threshold, so
    both sweeps below are derived from this one table without re-executing
    anything.
    """
    successes, asks_used, on_c, off_c, rejected = [], [], [], [], []
    for n in range(max_asks + 1):
        oracle = OracleUser(art)
        # A constant-1 signal with threshold 0 forces exactly n questions.
        policy = AskOrActPolicy(ConstantSignal(1.0, "forced"), 0.0, max_asks=n)
        d = policy.decide(art.task.id, art.task.entry_point, art.sources,
                          art.matrix, oracle)
        successes.append(bool(art.is_correct(d.chosen_id)))
        asks_used.append(d.n_asks)
        on_c.append(oracle.on_constraint)
        off_c.append(oracle.off_constraint)
        rejected.append(oracle.rejected)
    return {"success": successes, "asks": asks_used, "on_constraint": on_c,
            "off_constraint": off_c, "rejected": rejected}


def precompute_outcomes(cache, max_asks: int) -> dict[str, dict[str, Any]]:
    return {tid: outcome_curve(v["ambiguous"], max_asks) for tid, v in cache.items()}


def threshold_sweep(cache, outcomes, sig: Signal, max_asks: int) -> list[dict[str, Any]]:
    """Ask about every task whose score clears tau, act on the rest."""
    scores = {tid: score_of(sig, v["ambiguous"]) for tid, v in cache.items()}
    n = len(cache)
    taus = sorted({0.0, 1.01} | {round(s, 6) for s in scores.values()})
    pts, seen = [], set()
    for tau in taus:
        succ = asks = 0
        for tid, oc in outcomes.items():
            k = max_asks if scores[tid] >= tau else 0
            succ += int(oc["success"][k])
            asks += oc["asks"][k]
        key = (round(asks / n, 6), round(succ / n, 6))
        if key not in seen:
            seen.add(key)
            pts.append({"threshold": tau, "asks_per_task": asks / n,
                        "success_rate": succ / n})
    pts.sort(key=lambda p: (p["asks_per_task"], p["success_rate"]))
    return pts


def ranked_sweep(cache, outcomes, sig: Signal, max_asks: int, *,
                 n_perm: int = 500, seed: int = 0) -> list[dict[str, Any]]:
    """Spend a hard budget of B questions on the highest-scoring tasks.

    This is the comparison that matters. A threshold sweep conflates *which*
    tasks to ask about with *how many* questions get spent; fixing the budget
    isolates the only job the signal has, which is to rank the tasks where a
    question changes the answer above the ones where it does not. Ties are broken
    by a random permutation and averaged over `n_perm` draws, because with six
    samples per task the signal takes few distinct values and an arbitrary
    tie-break would otherwise decide the result silently.
    """
    ids = list(cache.keys())
    scores = {tid: score_of(sig, cache[tid]["ambiguous"]) for tid in ids}
    n = len(ids)
    max_budget = n * max_asks
    rng = random.Random(seed)

    curves: list[list[float]] = []
    for _ in range(n_perm):
        order = sorted(ids, key=lambda t: (-scores[t], rng.random()))
        row = []
        for budget in range(max_budget + 1):
            left, succ = budget, 0
            for tid in order:
                oc = outcomes[tid]
                k = 0
                while k < max_asks:
                    step = oc["asks"][k + 1] - oc["asks"][k]
                    if step <= 0 or left < step:
                        break
                    left -= step
                    k += 1
                succ += int(oc["success"][k])
            row.append(succ / n)
        curves.append(row)

    pts = []
    for budget in range(max_budget + 1):
        col = sorted(c[budget] for c in curves)
        pts.append({
            "budget": budget,
            "asks_per_task": budget / n,
            "success_rate": sum(col) / len(col),
            "ci95": [col[int(0.025 * len(col))],
                     col[min(len(col) - 1, int(0.975 * len(col)))]],
        })
    return pts


def question_efficiency(pts: list[dict[str, Any]], at: float = 0.5) -> float:
    """Success gained per question over the first `at` questions per task --
    the steep part of the curve, and the number a practitioner cares about."""
    if len(pts) < 2:
        return float("nan")
    base = pts[0]["success_rate"]
    chosen = None
    for p in pts[1:]:
        if p["asks_per_task"] <= at + 1e-9:
            chosen = p
    chosen = chosen or pts[1]
    return (chosen["success_rate"] - base) / chosen["asks_per_task"]


def diagnose_targets(cache, outcomes, sig: Signal, max_asks: int) -> dict[str, Any]:
    """Separate the two questions the ask decision actually contains.

    T1  is this prompt under-specified?
    T2  will a question change what gets delivered?

    A budget-constrained agent needs T2. A disagreement statistic is built for
    T1. Reporting both AUROCs side by side is what turns the H2 result from
    "the method lost" into a statement about which target the field has been
    measuring. The easy/hard split is reported too, because the direction of the
    T2 result depends on how many confident-but-wrong tasks the set contains --
    and that prevalence is a property of the model, not of the method.
    """
    helped, wasted, strata = [], [], {}
    for tid, oc in outcomes.items():
        art = cache[tid]["ambiguous"]
        score = score_of(sig, art)
        changes = bool(oc["success"][max_asks]) and not bool(oc["success"][0])
        (helped if changes else wasted).append(score)
        strata.setdefault(art.task.difficulty, {"helped": [], "wasted": []})
        strata[art.task.difficulty]["helped" if changes else "wasted"].append(score)

    easy = strata.get("easy", {"helped": [], "wasted": []})
    return {
        "signal": sig.name,
        "auroc_T2_question_helps": round(auroc(helped, wasted), 4),
        "auroc_T2_easy_tasks_only": round(auroc(easy["helped"], easy["wasted"]), 4),
        "mean_score_when_question_helps": round(sum(helped) / max(1, len(helped)), 4),
        "mean_score_when_wasted": round(sum(wasted) / max(1, len(wasted)), 4),
        "n_helped": len(helped), "n_wasted": len(wasted),
    }


def experiment_h2(cache, signals: list[Signal], *, max_asks: int) -> dict[str, Any]:
    outcomes = precompute_outcomes(cache, max_asks)
    extras = [ConstantSignal(1.0, "always_ask"), ConstantSignal(0.0, "never_ask")]

    ranked = {sig.name: ranked_sweep(cache, outcomes, sig, max_asks) for sig in signals}
    thresholds = {sig.name: threshold_sweep(cache, outcomes, sig, max_asks)
                  for sig in list(signals) + extras}

    n = len(cache)
    return {
        "ranked_budget": ranked,
        "threshold_sweep": thresholds,
        "never_ask_success": sum(int(oc["success"][0]) for oc in outcomes.values()) / n,
        "always_ask_success": sum(int(oc["success"][max_asks])
                                  for oc in outcomes.values()) / n,
        "questions_on_constraint": sum(oc["on_constraint"][max_asks]
                                       for oc in outcomes.values()),
        "questions_off_constraint": sum(oc["off_constraint"][max_asks]
                                        for oc in outcomes.values()),
        "answers_rejected": sum(oc["rejected"][max_asks] for oc in outcomes.values()),
        "question_efficiency": {k: round(question_efficiency(v), 4)
                                for k, v in ranked.items()},
        "target_diagnosis": {sig.name: diagnose_targets(cache, outcomes, sig, max_asks)
                             for sig in signals},
        "outcomes": {tid: {"success": oc["success"], "asks": oc["asks"]}
                     for tid, oc in outcomes.items()},
    }


# ---------------------------------------------------------------- reporting

def print_report(res: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("H1  can the signal tell an ambiguous prompt from a specified one?")
    print("=" * 80)
    print(f"{'signal':<20}{'AUROC':>8}{'paired':>9}{'mean d':>9}   {'95% CI on mean d':<22}")
    print("-" * 80)
    for name, r in sorted(res["h1"].items(), key=lambda kv: -kv[1]["auroc"]):
        ci = f"[{r['delta_ci95'][0]:+.3f}, {r['delta_ci95'][1]:+.3f}]"
        print(f"{name:<20}{r['auroc']:>8.3f}{r['paired_accuracy']:>9.3f}"
              f"{r['mean_delta']:>9.3f}   {ci:<22}")

    loc = res["localisation"]
    n = loc["n_tasks"]
    print("\n" + "=" * 80)
    print("LOC does the top-ranked probe name the constraint that was deleted?")
    print("=" * 80)
    print(f"precision@1, precondition filter on  : {loc['precision_at_1']:.3f} "
          f"({round(loc['precision_at_1'] * n)}/{n})")
    print(f"precision@1, filter off  (ablation)  : "
          f"{loc['precision_at_1_no_precondition']:.3f} "
          f"({round(loc['precision_at_1_no_precondition'] * n)}/{n})")

    h2 = res["h2"]
    print("\n" + "=" * 80)
    print("H2  success under a hard budget of questions")
    print("=" * 80)
    print(f"never ask (majority vote)   : {h2['never_ask_success']:.3f}")
    print(f"ask about every task        : {h2['always_ask_success']:.3f}"
          f"   <- ceiling, set by a perfect oracle user")
    total_q = h2["questions_on_constraint"] + h2["questions_off_constraint"]
    print(f"questions landing on the deleted constraint : "
          f"{h2['questions_on_constraint']}/{total_q}")
    print()
    any_curve = next(iter(h2["ranked_budget"].values()))
    budgets = [p["asks_per_task"] for p in any_curve][:8]
    print(f"{'signal':<20}{'eff':>7}    " + "".join(f"{b:>7.2f}" for b in budgets))
    print("-" * 80)
    for name, pts in sorted(h2["ranked_budget"].items(),
                            key=lambda kv: -h2["question_efficiency"].get(kv[0], 0)):
        cells = "".join(f"{p['success_rate']:>7.3f}" for p in pts[:8])
        print(f"{name:<20}{h2['question_efficiency'][name]:>7.2f}    {cells}")
    print("\n(row = success rate; columns = questions allowed per task)")

    print("=" * 80)
    print("H2b which target is the signal actually good at?")
    print("=" * 80)
    print(f"{'signal':<20}{'T1 ambiguous?':>15}{'T2 helps?':>11}{'T2 easy only':>14}")
    print("-" * 80)
    for name, d in h2["target_diagnosis"].items():
        print(f"{name:<20}{res['h1'][name]['auroc']:>15.3f}"
              f"{d['auroc_T2_question_helps']:>11.3f}"
              f"{d['auroc_T2_easy_tasks_only']:>14.3f}")
    bd = h2["target_diagnosis"]["bse"]
    print(f"BSE mean where a question helps: {bd['mean_score_when_question_helps']:.3f}"
          f"  | where wasted: {bd['mean_score_when_wasted']:.3f}"
          f"  ({bd['n_helped']} vs {bd['n_wasted']} tasks)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--eval-probes", type=int, default=40)
    ap.add_argument("--max-asks", type=int, default=2)
    ap.add_argument("--out", default="results/experiments.json")
    ap.add_argument("--candidates", default=None,
                    help="JSON of live model samples from sample_live.py; "
                         "omit to use the hand-written fixtures.")
    args = ap.parse_args()

    live = json.loads(Path(args.candidates).read_text(encoding="utf-8")) \
        if args.candidates else None

    t0 = time.time()
    print("executing candidates ...", flush=True)
    cache = collect(args.probes, args.eval_probes, live)
    n_exec = sum(a.matrix.n_candidates * a.matrix.n_probes
                 + a.eval_matrix.n_candidates * a.eval_matrix.n_probes
                 for v in cache.values() for a in v.values())
    print(f"  {len(cache)} tasks x {len(VARIANTS)} variants, "
          f"{n_exec} candidate-probe executions, {time.time() - t0:.1f}s")

    signals = offline_signals(seed=0)
    if live:
        from askoract.signals import (DirectAskSignal, SelfReportSignal,
                                      TokenEntropySignal)
        sample0 = next(iter(live["samples"].values()))["ambiguous"]
        if any(lp is not None for lp in sample0.get("mean_logprobs") or []):
            signals.append(TokenEntropySignal())
        if sample0.get("self_report") is not None:
            signals.append(SelfReportSignal())
        if sample0.get("direct_ask") is not None:
            signals.append(DirectAskSignal())
    res = {
        "meta": {
            "n_tasks": len(cache),
            "n_probes": args.probes,
            "n_eval_probes": args.eval_probes,
            "max_asks_per_task": args.max_asks,
            "candidate_probe_executions": n_exec,
            "mode": f"live:{live['model']}" if live else "fixture (no live model)",
            "k_samples": live.get("k") if live else 6,
            "temperature": live.get("temperature") if live else None,
        },
        "h1": experiment_h1(cache, signals),
        "localisation": experiment_localisation(cache),
        "h2": experiment_h2(cache, signals, max_asks=args.max_asks),
    }
    res["meta"]["wall_seconds"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(res)
    print(f"\nwritten to {out}  ({res['meta']['wall_seconds']}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
