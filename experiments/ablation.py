"""What is each of the agent's components actually worth?

Pre-registered in docs/PREREG_ablation.md before this ran.

The previous rounds asked when a model should ask. This one asks something
narrower and more useful for defending the agent itself: of the four things
added to it recently, is any of them measurably load-bearing? Each carries a
written justification and three carry a number, but every one of those numbers
came from a single trajectory. An anecdote is not an effect.

    full             baseline
    no_search        search / find_files removed from the schema entirely
    garbled_errors   command output decoded as strict UTF-8, replacing failures
    no_recovery      a malformed tool call aborts the run

The last two are monkey patches that restore what the code *actually did*
before it was fixed, rather than a guess at a bad implementation. Because those
patches are module-global, conditions run sequentially and only the runs within
one condition are parallel.

THE RULE THAT DECIDES HOW TO READ A NULL

An ablation cannot produce a larger effect than its trigger rate. If the
baseline never emits a malformed tool call, `no_recovery` measuring zero says
the task set cannot reach that code -- not that error recovery is worthless.
So the baseline is instrumented to count how often each ablated component is
reached at all, and those rates are reported as the ceiling on each effect.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    py -3 experiments/ablation.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.llm as llm_mod
import agent.loop as loop_mod
import agent.workspace as ws_mod
from agent.llm import make_backend
from agent.loop import Agent, AgentConfig
from agent.tools import build_toolset
from agent.workspace import Workspace
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.harness import EVAL_SEED
from bench.scaffold import TASK_PROMPT, agent_prompt, build_repo, scoring_source
from bench.schema import Task, load_tasks
from experiments.provenance import (git_is_dirty, randomized_block_schedule,
                                    run_manifest, save_artifact, snapshot_text,
                                    unified_patch)

CONDITIONS = ("full", "no_search", "garbled_errors", "no_recovery")
# New confirmatory runs can separate the two mechanisms that the historical
# `no_recovery` bundle changed together.
RECOVERY_CONDITIONS = ("full", "no_parse_recovery",
                       "no_argument_recovery", "no_recovery")

# Round 9: the two termination fixes, crossed. `old_env` restores the broken
# sandbox (the shell's own `python`, no PYTHONPATH) so the fix can be measured
# against what was actually there rather than against a guess.
TERMINATION = ("old_env_no_budget", "path_fix", "budget", "both")
KNOWN_CONDITIONS = set(CONDITIONS) | set(RECOVERY_CONDITIONS) | set(TERMINATION)
_DATED_MODEL = re.compile(r"(?:20\d{2}-\d{2}-\d{2}|20\d{6})(?:$|[^0-9])")

# Per-thread record of what the ablated components were asked to handle during
# one trajectory. This is how trigger rates are measured rather than guessed.
_local = threading.local()


def _counters() -> dict[str, int]:
    if not hasattr(_local, "c"):
        _local.c = {"decode_fallback": 0, "bad_args": 0}
    return _local.c


# ---------------------------------------------------------------------------
# The ablations, each restoring a specific earlier behaviour.
# ---------------------------------------------------------------------------

_REAL_TOOLCHAIN = ws_mod._toolchain_env


def _broken_toolchain(root):
    """The environment before the fix: whatever the shell's PATH happened to say.

    On this machine that resolves `python` to msys2's interpreter, which has no
    pytest, and leaves the workspace un-importable.
    """
    import os
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


_REAL_DECODE = ws_mod._decode
_REAL_CHECK = loop_mod.check_arguments
_REAL_PARSE = llm_mod._parse_arguments


def _instrumented_decode(raw: bytes) -> str:
    """The real decoder, plus a count of how often the fallback was needed."""
    if raw:
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            _counters()["decode_fallback"] += 1
    return _REAL_DECODE(raw)


def _broken_decode(raw: bytes) -> str:
    """What the code did before: one codec for every writer into the pipe.

    The child is told to emit UTF-8, but a command that does not exist never
    runs and the shell writes the error in the OEM code page instead. This is
    the exact line that turned "'pytest' is not recognized" into U+FFFD.
    """
    if not raw:
        return ""
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        _counters()["decode_fallback"] += 1
    return raw.decode("utf-8", errors="replace")


class _AblatedCrash(Exception):
    """Raised where the pre-fix code raised, so the abort is attributable."""


def _instrumented_check(tool, args):
    bad = _REAL_CHECK(tool, args)
    if bad:
        _counters()["bad_args"] += 1
    return bad


def _strict_check(tool, args):
    """Pre-fix behaviour: a bad call is a crash, not a message."""
    bad = _REAL_CHECK(tool, args)
    if bad:
        _counters()["bad_args"] += 1
        raise _AblatedCrash(bad)
    return ""


def _strict_parse(blob):
    """Pre-fix behaviour: json.loads straight into the caller."""
    return json.loads(blob or "{}")


def apply_condition(cond: str) -> None:
    """Patch the modules for one condition. Conditions never overlap in time."""
    ws_mod._toolchain_env = (_broken_toolchain
                             if cond in ("old_env_no_budget", "budget")
                             else _REAL_TOOLCHAIN)
    ws_mod._decode = _broken_decode if cond == "garbled_errors" else _instrumented_decode
    loop_mod.check_arguments = (_strict_check
                                if cond in ("no_argument_recovery", "no_recovery")
                                else _instrumented_check)
    llm_mod._parse_arguments = (_strict_parse
                                if cond in ("no_parse_recovery", "no_recovery")
                                else _REAL_PARSE)


def restore() -> None:
    ws_mod._toolchain_env = _REAL_TOOLCHAIN
    ws_mod._decode = _REAL_DECODE
    loop_mod.check_arguments = _REAL_CHECK
    llm_mod._parse_arguments = _REAL_PARSE


# ---------------------------------------------------------------------------
# Scoring: behaviour against the reference, never text inspection.
# ---------------------------------------------------------------------------

def score(repo, task: Task) -> dict[str, Any]:
    discriminating = [a for c in task.constraints for a in c.discriminating_args]
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=40, seed=EVAL_SEED,
                          extra=discriminating), task.precondition)
    # Execution infrastructure failures must escape to `one_run`, which labels
    # them outcome="error". Converting them to a missing score here would make
    # the confirmatory analyser count infrastructure noise as model failure.
    got = run_candidates([scoring_source(repo, task)], probes, task.entry_point)
    ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    matches = sum(a == b for a, b in zip(got.tokens[0], ref))
    total = len(ref)
    graded = matches / total if total else 0.0
    if got.invalid[0]:
        # Invalid candidate code is a model failure, not an infrastructure
        # failure. Keep the graded evidence instead of turning it into missing
        # data that could be silently excluded from a condition.
        return {"scorable": True, "correct": False,
                "behaviour_matches": matches, "behaviour_total": total,
                "behaviour_frac": graded, "invalid_candidate": True}
    return {"scorable": True, "correct": got.tokens[0] == ref,
            "behaviour_matches": matches, "behaviour_total": total,
            "behaviour_frac": graded, "invalid_candidate": False}


def trajectory_features(trace) -> dict[str, Any]:
    used_search = runs = failed_runs = 0
    for st in trace.steps:
        if st.kind != "tool":
            continue
        name = st.payload.get("name")
        if name in ("search", "find_files"):
            used_search += 1
        elif name == "run":
            runs += 1
            if "exit code: 0" not in str(st.payload.get("result", "")):
                failed_runs += 1
    # Model calls, not trace records: a trace record is written per model call
    # AND per tool call, so len(trace.steps) roughly doubles the agent's steps
    # and cannot be compared against max_steps.
    model_calls = sum(1 for st in trace.steps if st.kind == "model")
    tool_calls = sum(1 for st in trace.steps if st.kind == "tool")
    usage = trace.usage if trace else None
    return {"n_search": used_search, "n_runs": runs, "n_failed_runs": failed_runs,
            "n_model_calls": model_calls, "n_tool_calls": tool_calls,
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
            "total_tokens": ((usage.input_tokens + usage.output_tokens)
                             if usage else 0)}


class Meter:
    def __init__(self, total: int) -> None:
        self.cost = 0.0
        self.done = 0
        self.total = total
        self.errors: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(self, c: float) -> None:
        with self._lock:
            self.cost += c
            self.done += 1

    def fail(self, why: str) -> None:
        with self._lock:
            self.errors[why[:100]] = self.errors.get(why[:100], 0) + 1
            self.done += 1


def prompt_for(task: Task, repo, unambiguous: bool) -> str:
    """The task text. `unambiguous` is the manipulation check, not a condition.

    The scaffold's default prompt has one constraint deleted on purpose, so the
    agent can only get it right by finding that constraint in the repo. That is
    the right design for the ask-or-act question and the wrong one for this
    study: if every condition floors at zero correct, a null says the task was
    unwinnable, not that the component was worthless. Running the same harness
    on the complete prompt shows whether scoring can register success at all.
    """
    if not unambiguous:
        return agent_prompt(task, repo)
    return TASK_PROMPT.format(target=repo.target,
                              requirement=task.prompt_full.strip())


def one_run(cond: str, task: Task, rep: int, model: str, max_steps: int,
            meter: Meter, unambiguous: bool = False, temperature: float = 1.0,
            max_tokens: int = 4096,
            capture_artifact: bool = False) -> dict[str, Any]:
    """One trajectory. Always returns a row -- a crash is an outcome, not a gap."""
    _local.c = {"decode_fallback": 0, "bad_args": 0}
    root = Path(tempfile.mkdtemp(prefix="nanocode-abl-"))
    row: dict[str, Any] = {"condition": cond, "task": task.id, "rep": rep,
                           "worker_pid": os.getpid(),
                           "model": model, "crashed": False, "why": ""}
    agent: Agent | None = None
    before: dict[str, str] = {}
    try:
        repo = build_repo(root, task, "findable")
        before = snapshot_text(root) if capture_artifact else {}
        ws = Workspace(root)
        tools = build_toolset(ws, allow_ask=False)
        if cond == "no_search":
            # Removed from the schema, not discouraged in the prompt. A
            # capability the model cannot see is a real ablation.
            tools = {k: v for k, v in tools.items() if k not in ("search", "find_files")}
        backend_spec = model if ":" in model else "openai:" + model
        agent = Agent(make_backend(backend_spec), ws, tools=tools,
                      config=AgentConfig(max_steps=max_steps, temperature=temperature,
                                         max_tokens=max_tokens,
                                         allow_ask=False, seed=700 + rep,
                                         show_budget=cond in ("budget", "both")))
        try:
            res = agent.run(prompt_for(task, repo, unambiguous),
                            task_id=f"{task.id}|{cond}")
            row.update(outcome=res.outcome, cost_usd=round(res.trace.cost_usd, 6),
                       **trajectory_features(res.trace))
            meter.add(res.trace.cost_usd)
        except _AblatedCrash as exc:
            # The ablation did what the old code did. Still scored below: the
            # pre-fix code also left partial work on disk, and pretending the
            # run never happened would bias the comparison.
            trace = agent.last_trace
            if trace:
                trace.outcome = "crashed"
                trace.record("error", {"why": str(exc)[:200]})
            features = trajectory_features(trace) if trace else {
                "n_search": 0, "n_runs": 0, "n_failed_runs": 0,
                "n_model_calls": 0, "n_tool_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            cost = trace.cost_usd if trace else 0.0
            row.update(outcome="crashed", crashed=True, why=str(exc)[:120],
                       cost_usd=round(cost, 6), **features)
            meter.add(cost)
        except json.JSONDecodeError as exc:
            trace = agent.last_trace
            if trace:
                trace.outcome = "crashed"
                trace.record("error", {"why": f"JSONDecodeError: {exc}"[:200]})
            features = trajectory_features(trace) if trace else {
                "n_search": 0, "n_runs": 0, "n_failed_runs": 0,
                "n_model_calls": 0, "n_tool_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            cost = trace.cost_usd if trace else 0.0
            row.update(outcome="crashed", crashed=True, why=f"JSONDecodeError: {exc}",
                       cost_usd=round(cost, 6), **features)
            meter.add(cost)
        row.update(score(repo, task))
        row.update(_counters())
        return row
    except Exception as exc:                       # noqa: BLE001
        # Infrastructure failure, not an ablation effect. Kept and labelled so
        # it cannot be silently confused with one.
        meter.fail(f"{type(exc).__name__}: {exc}")
        trace = agent.last_trace if agent else None
        if trace:
            trace.outcome = "error"
            trace.record("error", {"why": f"{type(exc).__name__}: {exc}"[:200]})
        features = trajectory_features(trace) if trace else {
            "n_model_calls": 0, "n_tool_calls": 0, "n_search": 0,
            "n_runs": 0, "n_failed_runs": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        row.update(outcome="error", why=f"{type(exc).__name__}: {exc}"[:160],
                   scorable=False, correct=None,
                   cost_usd=round(trace.cost_usd, 6) if trace else 0.0,
                   decode_fallback=0, bad_args=0)
        row.update(features)
        return row
    finally:
        if capture_artifact and agent and agent.last_trace:
            row["_artifact"] = {
                "run_key": f"{task.id}-{cond}-r{rep}",
                "row": {k: v for k, v in row.items() if k != "_artifact"},
                "trace": agent.last_trace.to_dict(),
                "patch": unified_patch(before, root),
            }
        shutil.rmtree(root, ignore_errors=True)


def isolated_one_run(job) -> dict[str, Any]:
    """One condition in one process: module patches cannot cross trajectories."""
    cond, task, rep, model, max_steps, unambiguous, temperature, max_tokens = job
    meter = Meter(1)
    apply_condition(cond)
    try:
        return one_run(cond, task, rep, model, max_steps, meter, unambiguous,
                       temperature, max_tokens,
                       capture_artifact=True)
    finally:
        restore()


def rate(rows: list[dict], cond: str, key: str, pred=None) -> float:
    v = [r for r in rows if r["condition"] == cond and r.get("outcome") != "error"]
    if not v:
        return float("nan")
    pred = pred or (lambda r: bool(r.get(key)))
    return sum(1 for r in v if pred(r)) / len(v)


def mean_of(rows: list[dict], cond: str, key: str) -> float:
    v = [r[key] for r in rows if r["condition"] == cond
         and r.get(key) is not None and r.get("outcome") != "error"]
    return statistics.mean(v) if v else float("nan")


def success(rows: list[dict], cond: str) -> float:
    """Correctness over every non-infrastructure run, crashes counted as wrong."""
    v = [r for r in rows if r["condition"] == cond and r.get("outcome") != "error"]
    if not v:
        return float("nan")
    return sum(1 for r in v if r.get("correct") is True) / len(v)


def paired_by_task(rows: list[dict], hi: str, lo: str) -> tuple[float, int]:
    """Within-task ranking, ties 0.5. Immune to a global difficulty offset."""
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("outcome") == "error":
            continue
        by[(r["task"], r["condition"])].append(1.0 if r.get("correct") is True else 0.0)
    wins, n = 0.0, 0
    for task in {t for t, _ in by}:
        a, b = by.get((task, hi)), by.get((task, lo))
        if not a or not b:
            continue
        n += 1
        ma, mb = statistics.mean(a), statistics.mean(b)
        wins += 1.0 if ma > mb else (0.5 if ma == mb else 0.0)
    return (wins / n if n else float("nan")), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=18)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Maximum output tokens per model call.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--unambiguous", action="store_true",
                    help="Manipulation check: give the complete requirement.")
    ap.add_argument("--out", default="results/ablation.json")
    ap.add_argument("--artifacts", default=None,
                    help="Per-run trace/patch directory (default: OUT stem + _artifacts).")
    ap.add_argument("--schedule-seed", type=int, default=20260901)
    ap.add_argument("--require-clean", action="store_true",
                    help="Abort unless the git worktree is clean.")
    ap.add_argument("--require-model-snapshot", action="store_true",
                    help="Abort unless MODEL contains a date-stamped snapshot id.")
    ap.add_argument("--fail-if-output-exists", action="store_true",
                    help="Refuse to overwrite the result or artifact directory.")
    args = ap.parse_args()

    unknown_conditions = sorted(set(args.conditions) - KNOWN_CONDITIONS)
    if unknown_conditions:
        ap.error(f"unknown conditions: {', '.join(unknown_conditions)}")
    if len(set(args.conditions)) != len(args.conditions):
        ap.error("conditions must not contain duplicates")
    if args.reps <= 0 or args.max_steps <= 0 or args.max_tokens <= 0 or args.workers <= 0:
        ap.error("reps, max-steps, max-tokens and workers must be positive")
    if args.temperature < 0:
        ap.error("temperature must be non-negative")
    if args.require_clean and git_is_dirty():
        ap.error("confirmatory run requires a clean git worktree")
    if args.require_model_snapshot and not _DATED_MODEL.search(args.model):
        ap.error("confirmatory run requires a date-stamped model snapshot")

    available = load_tasks()
    available_ids = {t.id for t in available}
    requested = args.tasks or sorted(available_ids)
    unknown_tasks = sorted(set(requested) - available_ids)
    if unknown_tasks:
        ap.error(f"unknown tasks: {', '.join(unknown_tasks)}")
    if len(set(requested)) != len(requested):
        ap.error("tasks must not contain duplicates")
    tasks = [t for t in available if t.id in set(requested)]
    out = Path(args.out)
    artifact_dir = Path(args.artifacts) if args.artifacts else \
        out.with_suffix("").with_name(out.stem + "_artifacts")
    if args.fail_if_output_exists and (out.exists() or artifact_dir.exists()):
        ap.error("result or artifact target already exists")

    blocks = [(t, r) for t in tasks for r in range(args.reps)]
    randomized = randomized_block_schedule(
        blocks, args.conditions, seed=args.schedule_seed)
    jobs = [(cond, task, rep, args.model, args.max_steps, args.unambiguous,
             args.temperature, args.max_tokens)
            for cond, (task, rep) in randomized]
    total = len(jobs)
    print(f"{total} trajectories: {len(args.conditions)} conditions x "
          f"{len(tasks)} tasks x {args.reps} reps, model={args.model}")

    meter = Meter(total)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    # Conditions are shuffled within task x repetition blocks. Each job gets a
    # process, so module-global ablation patches never coexist in one interpreter.
    # `max_tasks_per_child=1` is intentional: ablation conditions patch module
    # globals. `restore()` remains defence in depth, but a fresh interpreter is
    # the isolation boundary promised by the confirmatory protocol.
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        futs = [ex.submit(isolated_one_run, job) for job in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            artifact = row.pop("_artifact", None)
            if artifact:
                row["artifact"] = str(save_artifact(artifact_dir, artifact))
            rows.append(row)
            if row.get("outcome") == "error":
                meter.fail(row.get("why", "error"))
            else:
                meter.add(float(row.get("cost_usd") or 0.0))
            if meter.done % max(1, len(args.conditions) * 3) == 0:
                print(f"  done {meter.done}/{total}  ${meter.cost:.3f}  "
                      f"{time.time() - t0:.0f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    schedule = [{"condition": c, "task": t.id, "rep": r}
                for c, t, r, *_ in jobs]
    out.write_text(json.dumps({"manifest": run_manifest(args),
                               "model": args.model, "reps": args.reps,
                               "schedule": schedule, "rows": rows}, indent=1),
                   encoding="utf-8")
    print(f"\ndone: {len(rows)} rows, ${meter.cost:.3f}, {time.time() - t0:.0f}s")
    if meter.errors:
        print("infrastructure errors (excluded, not silent):")
        for why, n in sorted(meter.errors.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}  {why}")

    # ---- trigger rates: the ceiling on every effect below -------------------
    print("\n" + "=" * 78)
    print("TRIGGER RATES in the baseline -- no ablation can beat these")
    print("=" * 78)
    print(f"  used search at all        {rate(rows, 'full', 'n_search'):.2f}")
    print(f"  emitted a malformed call  {rate(rows, 'full', 'bad_args'):.2f}")
    print(f"  hit non-UTF-8 output      {rate(rows, 'full', 'decode_fallback'):.2f}")
    print(f"  ran any command           {rate(rows, 'full', 'n_runs'):.2f}")
    print(f"  had a command fail        {rate(rows, 'full', 'n_failed_runs'):.2f}")

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'condition':<18}{'correct':>9}{'finished':>10}{'crashed':>9}"
          f"{'calls':>8}{'search':>8}{'$/run':>9}")
    print("-" * 78)
    for cond in args.conditions:
        cr = [r for r in rows if r["condition"] == cond]
        spend = sum(r.get("cost_usd") or 0.0 for r in cr)
        print(f"{cond:<18}{success(rows, cond):>9.2f}"
              f"{rate(rows, cond, '', lambda r: r.get('outcome') == 'finished'):>10.2f}"
              f"{rate(rows, cond, 'crashed'):>9.2f}"
              f"{mean_of(rows, cond, 'n_model_calls'):>8.1f}"
              f"{mean_of(rows, cond, 'n_search'):>8.2f}"
              f"{(spend / max(len(cr), 1)):>9.4f}")

    print("\nwithin-task ranking, full > ablated  (0.50 = no difference)")
    for cond in args.conditions:
        if cond == "full":
            continue
        r, n = paired_by_task(rows, "full", cond)
        print(f"  full > {cond:<18}{r:.2f}   (n={n} tasks)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
