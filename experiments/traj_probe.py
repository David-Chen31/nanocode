"""Go/no-go: run the agent once per arm and find out what a trajectory costs.

Seven rounds and the agent loop had never been run. Before designing a study
around it, three things have to be measured rather than estimated:

  cost         dollars and steps for one trajectory
  scorability  can the function the agent wrote be judged by execution, in place
  legibility   does the trace say which files it opened, and in what order

Two trajectories, one task: `findable` and `unfindable`. At n=1 the comparison
proves nothing -- it is here to show whether the measurement would be visible at
all if the effect existed.

    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    py -3 experiments/traj_probe.py --task t08_round_price
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import make_backend
from agent.loop import Agent, AgentConfig
from agent.workspace import Workspace
from askoract.execute import run_candidates
from askoract.probes import filter_probes, synthesize_probes
from bench.harness import CALIB_SEED
from bench.scaffold import agent_prompt, build_repo, scoring_source
from bench.schema import load_tasks


def never_answers(question: str, options: list[str]) -> str:
    """The probe runs with asking disabled; this exists only as a guard."""
    raise AssertionError("ask_user should be unreachable in this probe")


def score(repo, task) -> dict[str, Any]:
    """Did the agent's function behave like the reference on held-out probes?"""
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=24, seed=CALIB_SEED), task.precondition)
    try:
        got = run_candidates([scoring_source(repo, task)], probes, task.entry_point)
        ref = run_candidates([task.reference], probes, task.entry_point).tokens[0]
    except Exception as exc:              # noqa: BLE001 - report, do not crash
        return {"scorable": False, "why": f"{type(exc).__name__}: {exc}"}
    if got.invalid[0]:
        return {"scorable": False, "why": "candidate row invalid (import or syntax)",
                "detail": got.shown[0][:3]}
    return {"scorable": True, "correct": got.tokens[0] == ref,
            "n_probes": len(probes)}


def summarise(trace) -> dict[str, Any]:
    """The trajectory features the real study would be built on."""
    reads, listings, runs, writes, edits = [], 0, 0, [], []
    first_write_at = None
    for i, st in enumerate(trace.steps):
        if st.kind != "tool":
            continue
        name, args = st.payload.get("name"), st.payload.get("args", {})
        if name == "read_file":
            reads.append(args.get("path"))
        elif name == "list_files":
            listings += 1
        elif name == "run":
            runs += 1
        elif name in ("write_file", "edit_file"):
            (writes if name == "write_file" else edits).append(
                args.get("path") or args.get("file_path") or "?")
            if first_write_at is None:
                first_write_at = i
    return {
        "steps": len(trace.steps),
        "model_calls": sum(1 for s in trace.steps if s.kind == "model"),
        "reads": reads,
        "n_reads": len(reads),
        "listings": listings,
        "runs": runs,
        "writes": writes + edits,
        # The measurement the whole study turns on: how much looking happened
        # before the agent committed to an answer.
        "reads_before_first_write": sum(
            1 for i, st in enumerate(trace.steps)
            if st.kind == "tool" and st.payload.get("name") == "read_file"
            and (first_write_at is None or i < first_write_at)),
        "opened_the_answer": None,   # filled in by the caller, which knows the arm
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t08_round_price")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--arms", nargs="+", default=["findable", "unfindable"])
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--keep", action="store_true", help="keep the workspaces")
    ap.add_argument("--out", default="results/traj_probe.json")
    args = ap.parse_args()

    tasks = {t.id: t for t in load_tasks()}
    if args.task not in tasks:
        print(f"no such task {args.task}; have {sorted(tasks)}")
        return 2
    task = tasks[args.task]

    report: dict[str, Any] = {"task": task.id, "model": args.model, "runs": {}}
    for arm in args.arms:
        root = Path(tempfile.mkdtemp(prefix=f"aoa-{arm}-"))
        repo = build_repo(root, task, arm)
        ws = Workspace(root)
        backend = make_backend("openai:" + args.model)
        agent = Agent(backend, ws, responder=never_answers,
                      config=AgentConfig(max_steps=args.max_steps, temperature=0.0,
                                         allow_ask=False, seed=11))

        print(f"\n{'=' * 76}\n{arm}  ({root})\n{'=' * 76}")
        t0 = time.time()
        res = agent.run(agent_prompt(task, repo), task_id=f"{task.id}|{arm}")
        wall = time.time() - t0

        feats = summarise(res.trace)
        feats["opened_the_answer"] = (
            repo.answer_file in feats["reads"] if repo.answer_file else None)
        verdict = score(repo, task)

        print(f"outcome        {res.outcome}")
        print(f"steps          {feats['steps']}  "
              f"({feats['model_calls']} model calls)")
        print(f"list/read/run  {feats['listings']} / {feats['n_reads']} / {feats['runs']}")
        print(f"files read     {feats['reads']}")
        print(f"wrote          {feats['writes']}")
        print(f"answer file    {repo.answer_file or '(none — nothing answers)'}"
              f"   opened={feats['opened_the_answer']}")
        print(f"scorable       {verdict['scorable']}"
              + ("" if verdict["scorable"] else f"  <- {verdict.get('why')}"))
        if verdict["scorable"]:
            print(f"correct        {verdict['correct']}  "
                  f"(over {verdict['n_probes']} probes)")
        print(f"cost           ${res.trace.cost_usd:.4f}   {wall:.0f}s")

        report["runs"][arm] = {
            "outcome": res.outcome, "features": feats, "verdict": verdict,
            "cost_usd": round(res.trace.cost_usd, 5),
            "wall_seconds": round(wall, 1),
            "root": str(root),
            "final_target": (root / repo.target).read_text(encoding="utf-8")[-1500:],
        }
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    total = sum(r["cost_usd"] for r in report["runs"].values())
    n = len(report["runs"])
    print(f"\n{'=' * 76}")
    print(f"{n} trajectories, ${total:.4f} total, ${total / n:.4f} each")
    scorable = sum(r["verdict"]["scorable"] for r in report["runs"].values())
    print(f"scorable: {scorable}/{n}")
    print("\nGO/NO-GO: a full study is 12 tasks x 3 arms x 5 runs x 3 models = 540")
    print(f"trajectories, so roughly ${total / n * 540:.0f} at this rate.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
