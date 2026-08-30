"""ReAct against plan-then-execute, on real tasks in a real repository.

Pre-registered in docs/PREREG_architecture.md before this ran.

The loop architecture is the largest design decision in this project and the
only one with no evidence behind it. Everything held constant except the
control flow -- same tools, same model, same orientation, same step budget,
same scoring.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    py -3 experiments/architecture.py --model claude-sonnet-5 --reps 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import make_backend
from agent.loop import Agent, AgentConfig
from agent.plan_execute import PlanExecuteAgent
from agent.workspace import Workspace
from bench.repo_tasks import TASKS, RepoTask, stage

ARMS = ("react", "plan_execute")


def score(ws: Workspace, task: RepoTask) -> dict[str, Any]:
    """Both halves must hold: no regression, and the new behaviour works.

    The verifier is written in only now, after the agent has stopped, so it was
    never available to be read and satisfied literally.
    """
    regression = ws.run("python -m pytest tests/ -q", timeout=180)
    vf = ws.root / "tests" / f"verify_{task.id}.py"
    vf.write_text(task.verifier, encoding="utf-8")
    behaviour = ws.run(f"python -m pytest {vf.relative_to(ws.root).as_posix()} -q",
                       timeout=180)
    vf.unlink(missing_ok=True)
    return {
        "regression_ok": regression.returncode == 0,
        "behaviour_ok": behaviour.returncode == 0,
        "correct": regression.returncode == 0 and behaviour.returncode == 0,
        "regression_tail": (regression.stdout.strip().splitlines() or [""])[-1][:60],
        "behaviour_tail": (behaviour.stdout.strip().splitlines() or [""])[-1][:60],
    }


def touched(ws: Workspace, task: RepoTask) -> int:
    """How many of the files a correct answer needs were actually modified."""
    n = 0
    for rel in task.touches:
        src = Path(__file__).resolve().parents[1] / rel
        dst = ws.root / rel
        try:
            if dst.read_text(encoding="utf-8", errors="replace") != \
               src.read_text(encoding="utf-8", errors="replace"):
                n += 1
        except OSError:
            pass
    return n


class Meter:
    """Counts tokens, and stops the sweep at a ceiling.

    The ceiling is in tokens rather than dollars because this relay publishes
    no price list: a dollar cap would be enforced against a number I made up.
    Tokens are what is actually measured, so tokens are what is capped.
    """

    def __init__(self, cap: float) -> None:
        self.tokens = 0
        self.cost = 0.0
        self.done = 0
        self.errors = 0
        self.cap = cap
        self.stopped = False
        self._lock = threading.Lock()

    def add(self, tokens: int) -> bool:
        with self._lock:
            self.tokens += tokens
            self.done += 1
            # One thrashing trajectory burned far more than the mean in an
            # earlier sweep, so this carries a hard ceiling rather than
            # trusting a per-run estimate that was already wrong by 10x once.
            if self.tokens > self.cap:
                self.stopped = True
            return self.stopped

    def fail(self) -> None:
        with self._lock:
            self.errors += 1


def one_run(arm: str, task: RepoTask, rep: int, model: str, max_steps: int,
            meter: Meter) -> dict[str, Any] | None:
    if meter.stopped:
        return None
    root = Path(tempfile.mkdtemp(prefix="nanocode-arch-"))
    try:
        stage(root)
        ws = Workspace(root)
        cfg = AgentConfig(max_steps=max_steps, temperature=1.0, allow_ask=False,
                          seed=1300 + rep)
        backend = make_backend("openai:" + model)
        t0 = time.monotonic()
        if arm == "react":
            res = Agent(backend, ws, config=cfg).run(task.prompt, task_id=task.id)
            plan = ""
        else:
            out = PlanExecuteAgent(backend, ws, config=cfg).run(task.prompt,
                                                                task_id=task.id)
            res, plan = out.result, out.plan
        secs = time.monotonic() - t0
        row = {
            "arm": arm, "task": task.id, "rep": rep, "model": model,
            "outcome": res.outcome, "secs": round(secs, 1),
            "n_model_calls": sum(1 for s in res.trace.steps if s.kind == "model"),
            "n_tool_calls": sum(1 for s in res.trace.steps if s.kind == "tool"),
            "n_tool_errors": sum(1 for s in res.trace.steps
                                 if s.kind == "tool" and "error" in s.payload),
            "touched": touched(ws, task), "must_touch": len(task.touches),
            "plan_len": len(plan.splitlines()) if plan else 0,
            "plan": plan[:1200],
            # Tokens are the primary cost unit here. This relay publishes no
            # price list, so a dollar figure would be a guess dressed as a
            # measurement; both arms run the same model, so tokens compare
            # cleanly anyway.
            "input_tokens": res.trace.usage.input_tokens,
            "output_tokens": res.trace.usage.output_tokens,
            "cost_usd": round(res.trace.cost_usd, 5),
        }
        row.update(score(ws, task))
        meter.add(res.trace.usage.input_tokens + res.trace.usage.output_tokens)
        return row
    except Exception as exc:                       # noqa: BLE001
        meter.fail()
        return {"arm": arm, "task": task.id, "rep": rep, "model": model,
                "outcome": "error", "why": f"{type(exc).__name__}: {exc}"[:200],
                "correct": False, "regression_ok": False, "behaviour_ok": False,
                "cost_usd": 0.0, "n_model_calls": 0, "n_tool_calls": 0}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=6_000_000,
                    help="Hard ceiling in total tokens; the sweep stops past it.")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="results/architecture.json")
    args = ap.parse_args()

    tasks = [t for t in TASKS if not args.tasks or t.id in args.tasks]
    arms = [a for a in ARMS if a in args.arms]
    jobs = [(a, t, r) for a in arms for t in tasks for r in range(args.reps)]
    print(f"{len(jobs)} runs: {len(arms)} arms x {len(tasks)} tasks x {args.reps} "
          f"reps, model={args.model}, cap {args.budget/1e6:.1f}M tokens", flush=True)

    meter = Meter(args.budget)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one_run, a, t, r, args.model, args.max_steps, meter)
                for a, t, r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row:
                rows.append(row)
            if i % 8 == 0:
                print(f"  {i}/{len(jobs)}  {meter.tokens/1e6:.2f}M tok  {meter.errors} err  "
                      f"{time.time() - t0:.0f}s", flush=True)

    if meter.stopped:
        print(f"\n!! budget cap ${args.budget} reached; {len(rows)} runs completed")
    print(f"\ndone: {len(rows)} rows, ${meter.cost:.2f}, {meter.errors} errors, "
          f"{time.time() - t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False),
                              encoding="utf-8")

    print(f"\n{'arm':<14}{'correct':>9}{'regr ok':>9}{'behav ok':>10}"
          f"{'calls':>8}{'tools':>8}{'ktok/run':>10}")
    print("-" * 69)
    for arm in arms:
        g = [r for r in rows if r["arm"] == arm]
        if not g:
            continue
        f = lambda k: sum(1 for r in g if r.get(k)) / len(g)
        print(f"{arm:<14}{f('correct'):>9.2f}{f('regression_ok'):>9.2f}"
              f"{f('behaviour_ok'):>10.2f}"
              f"{sum(r['n_model_calls'] for r in g) / len(g):>8.1f}"
              f"{sum(r['n_tool_calls'] for r in g) / len(g):>8.1f}"
              f"{sum(r.get('input_tokens', 0) + r.get('output_tokens', 0) for r in g) / len(g) / 1000:>10.1f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
