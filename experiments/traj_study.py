"""Is the ask-decision visible in what the agent did while looking?

Pre-registered in docs/PREREG_trajectory.md before this ran (and disclosing the
three-trajectory pilot that preceded it).

Rounds 1-3 looked for the signal inside the model and found none. Rounds 4-7
looked in the context window: it reads whether relevant code is *present*, never
whether that code is *right*, and the precondition holds in 0.4% of real
modules. One place is left, and it is the one an agent has that a completion
does not -- the trajectory. An agent is not handed context; it goes looking, and
looking can fail.

  findable    the answer is one import away, in pkg/helpers.py
  unfindable  nothing anywhere in the repo answers
  decoy       pkg/helpers.py answers, wrongly

crossed with whether `ask_user` is on the toolbelt at all.

H1  search effort does not separate findable from unfindable
H2  the agent writes its own test from its own misreading, and passes it
H3  the ask rate does not track findability          <- the deliverable

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    py -3 experiments/traj_study.py --models gpt-4o-mini claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import make_backend
from agent.loop import Agent, AgentConfig
from agent.workspace import Workspace
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.harness import CALIB_SEED
from bench.scaffold import ARMS, agent_prompt, build_repo, scoring_source
from bench.schema import Task, load_tasks

TEST_HINT = ("test", "assert")


class OracleUser:
    """Answers with the sentence the requirement had removed.

    The same oracle the earlier rounds used: always available, always right,
    never annoyed. A stated limitation, not a claim about real users -- it makes
    "did asking help" measurable by removing every other reason it might not.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.asked: list[str] = []

    def __call__(self, question: str, options: list[str]) -> str:
        self.asked.append(question)
        return self.task.constraints[0].text


def score(repo, task) -> dict[str, Any]:
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    try:
        got = run_candidates([scoring_source(repo, task)], probes, task.entry_point)
        ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    except Exception as exc:                       # noqa: BLE001
        return {"scorable": False, "why": f"{type(exc).__name__}: {exc}"}
    if got.invalid[0]:
        return {"scorable": False, "why": "invalid row"}
    return {"scorable": True, "correct": got.tokens[0] == ref}


def features(trace, repo) -> dict[str, Any]:
    reads: list[str] = []
    listings = runs = 0
    wrote_test = ran_test = False
    first_write = None
    for i, st in enumerate(trace.steps):
        if st.kind != "tool":
            continue
        name = st.payload.get("name")
        args = st.payload.get("args", {}) or {}
        if name == "read_file":
            reads.append(args.get("path"))
        elif name == "list_files":
            listings += 1
        elif name == "run":
            runs += 1
            if any(h in str(args.get("command", "")).lower() for h in TEST_HINT):
                ran_test = True
        elif name in ("write_file", "edit_file"):
            path = str(args.get("path") or "")
            if "test" in path.lower():
                wrote_test = True
            if first_write is None:
                first_write = i
    reads_before = sum(
        1 for i, st in enumerate(trace.steps)
        if st.kind == "tool" and st.payload.get("name") == "read_file"
        and (first_write is None or i < first_write))
    return {
        "n_reads": len(reads), "reads": reads, "listings": listings, "runs": runs,
        "reads_before_first_write": reads_before,
        "explore_before_commit": reads_before + listings,
        "opened_answer": (repo.answer_file in reads) if repo.answer_file else None,
        "wrote_test": wrote_test, "ran_test": ran_test,
        "n_steps": len(trace.steps),
    }


class Meter:
    def __init__(self) -> None:
        self.cost = 0.0
        self.errors = 0
        self.done = 0
        self.reasons: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(self, c: float) -> None:
        with self._lock:
            self.cost += c
            self.done += 1

    def fail(self, why: str = "") -> None:
        with self._lock:
            self.errors += 1
            self.reasons[why[:120]] = self.reasons.get(why[:120], 0) + 1


def one_run(model: str, task: Task, arm: str, allow_ask: bool, rep: int,
            max_steps: int, meter: Meter) -> dict[str, Any] | None:
    root = Path(tempfile.mkdtemp(prefix="aoa-traj-"))
    try:
        repo = build_repo(root, task, arm)
        oracle = OracleUser(task)
        agent = Agent(
            make_backend("openai:" + model), Workspace(root), responder=oracle,
            config=AgentConfig(max_steps=max_steps, temperature=1.0,
                               allow_ask=allow_ask, max_asks=2, seed=500 + rep))
        res = agent.run(agent_prompt(task, repo), task_id=f"{task.id}|{arm}")
        f = features(res.trace, repo)
        v = score(repo, task)
        meter.add(res.trace.cost_usd)
        return {"task": task.id, "arm": arm, "allow_ask": allow_ask, "rep": rep,
                "model": model, "outcome": res.outcome, **f,
                "n_asks": len(oracle.asked), "questions": oracle.asked,
                "scorable": v["scorable"], "correct": v.get("correct"),
                "cost_usd": round(res.trace.cost_usd, 6)}
    except Exception as exc:                       # noqa: BLE001
        # Record why. A silently dropped trajectory is worse than a loud one:
        # failures that correlate with what the agent did bias the sample.
        meter.fail(f"{type(exc).__name__}: {exc}")
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def paired_rank(rows, model, key, hi_arm, lo_arm, allow_ask) -> tuple[float, int]:
    """Within-task ranking, ties 0.5 -- immune to a global exploration bias."""
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["allow_ask"] == allow_ask:
            by[(r["task"], r["arm"])].append(r[key])
    wins = 0.0
    n = 0
    for task in {t for t, _ in by}:
        a, b = by.get((task, hi_arm)), by.get((task, lo_arm))
        if not a or not b:
            continue
        n += 1
        ma, mb = statistics.mean(a), statistics.mean(b)
        wins += 1.0 if ma > mb else (0.5 if ma == mb else 0.0)
    return (wins / n if n else float("nan")), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4-5-20251001"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=18)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--out", default="results/traj_study.json")
    args = ap.parse_args()

    tasks = [t for t in load_tasks() if not args.tasks or t.id in args.tasks]
    jobs = [(m, t, arm, ask, r)
            for m in args.models for t in tasks for arm in ARMS
            for ask in (False, True) for r in range(args.reps)]
    print(f"{len(jobs)} trajectories: {len(args.models)} models x {len(tasks)} tasks "
          f"x {len(ARMS)} arms x 2 ask-modes x {args.reps} reps")

    meter = Meter()
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one_run, m, t, arm, ask, r, args.max_steps, meter)
                for m, t, arm, ask, r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row:
                rows.append(row)
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)}  ${meter.cost:.3f}  "
                      f"{meter.errors} errors  {time.time() - t0:.0f}s", flush=True)

    print(f"\ndone: {len(rows)} usable, {meter.errors} errors, "
          f"${meter.cost:.3f}, {time.time() - t0:.0f}s")

    def agg(pred, key):
        v = [r[key] for r in rows if pred(r) and r[key] is not None]
        return statistics.mean(v) if v else float("nan")

    # ---- H1 / H2: exploration and self-confirmation, asking off ------------
    print("\n" + "=" * 88)
    print("Asking OFF — what the agent did while looking")
    print("=" * 88)
    print(f"{'model':<28}{'arm':<12}{'explore':>9}{'reads':>8}{'opened ans':>12}"
          f"{'wrote test':>12}{'correct':>9}")
    print("-" * 88)
    for m in args.models:
        for arm in ARMS:
            p = lambda r, m=m, a=arm: (r["model"] == m and r["arm"] == a
                                       and not r["allow_ask"])
            oa = agg(p, "opened_answer")
            print(f"{m if arm == ARMS[0] else '':<28}{arm:<12}"
                  f"{agg(p, 'explore_before_commit'):>9.2f}"
                  f"{agg(p, 'n_reads'):>8.2f}"
                  f"{('n/a' if oa != oa else f'{oa:.2f}'):>12}"
                  f"{agg(p, 'wrote_test'):>12.2f}"
                  f"{agg(p, 'correct'):>9.2f}")
        print("-" * 88)

    print("\nH1  within-task ranking of exploration, unfindable > findable")
    print("    (0.50 = the agent searched the same amount whether or not")
    print("     the answer was there)")
    for m in args.models:
        for ask in (False, True):
            a, n = paired_rank(rows, m, "explore_before_commit",
                               "unfindable", "findable", ask)
            print(f"    {m:<28}ask={'on ' if ask else 'off'}  {a:.2f}   (n={n})")

    # ---- H2 ----------------------------------------------------------------
    print("\n" + "=" * 88)
    print("H2  the agent's own test, against whether it was actually right")
    print("=" * 88)
    wrote = [r for r in rows if r["wrote_test"] and r["scorable"]]
    ran = [r for r in wrote if r["ran_test"]]
    wrong_after = [r for r in ran if r["correct"] is False]
    print(f"    wrote a test          {len(wrote)}/{len(rows)}")
    print(f"    wrote AND ran it      {len(ran)}")
    print(f"    ...and was wrong      {len(wrong_after)}"
          f"   ({len(wrong_after) / max(1, len(ran)):.0%} of those that ran one)")
    print("    a test written from the agent's own reading cannot catch that")
    print("    reading being wrong -- it encodes it.")

    # ---- H3: the deliverable ----------------------------------------------
    print("\n" + "=" * 88)
    print("H3  asking ON — does the ask rate track whether an answer exists?")
    print("=" * 88)
    print(f"{'model':<28}{'arm':<12}{'ask rate':>10}{'mean asks':>11}"
          f"{'correct':>9}{'correct(off)':>14}")
    print("-" * 88)
    for m in args.models:
        for arm in ARMS:
            on = lambda r, m=m, a=arm: (r["model"] == m and r["arm"] == a
                                        and r["allow_ask"])
            off = lambda r, m=m, a=arm: (r["model"] == m and r["arm"] == a
                                         and not r["allow_ask"])
            asked = [r for r in rows if on(r)]
            rate = (sum(1 for r in asked if r["n_asks"] > 0) / len(asked)
                    if asked else float("nan"))
            print(f"{m if arm == ARMS[0] else '':<28}{arm:<12}{rate:>10.2f}"
                  f"{agg(on, 'n_asks'):>11.2f}{agg(on, 'correct'):>9.2f}"
                  f"{agg(off, 'correct'):>14.2f}")
        print("-" * 88)
    print("H3 wants the ask rate HIGH on unfindable and LOW on findable.")
    print("Flat means the agent has no notion of whether an answer was there.")

    qs = [q for r in rows for q in r["questions"]][:6]
    if qs:
        print("\nsample questions the agent actually asked:")
        for q in qs:
            print("   " + q.replace("\n", " ")[:120])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "cost_usd": meter.cost,
                                          "errors": meter.errors}, indent=2,
                                         ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
