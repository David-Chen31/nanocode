"""Command line entry point for the bare agent.

    python -m agent.cli "add a docstring to utils.py" --workspace ./sandbox

With no API key configured this falls back to the fixture backend, which will
tell you plainly that it has no recorded response rather than pretending.
"""
from __future__ import annotations

import argparse
import json
import sys

from .llm import make_backend
from .loop import Agent, AgentConfig, cli_responder
from .trace import Step
from .workspace import Workspace


def _survivable_console() -> None:
    """Never let printing the result be the thing that fails.

    The console's encoding is whatever the OS says -- cp936 on a Chinese
    Windows box -- and a model summary containing a check mark or an emoji
    cannot be encoded in it. `print` then raises UnicodeEncodeError *after* the
    agent has finished the task, so a completed run ends in a traceback and the
    work looks lost when it is actually on disk.

    The encoding is left alone, because it is correct for this console and
    changing it would turn readable Chinese into mojibake. Only the error
    handler moves: an unencodable character becomes '?' instead of an
    exception.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass          # not a reconfigurable stream; nothing to do


def _short(value: object, limit: int = 100) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _print_live_step(step: Step) -> None:
    """Show a compact, truthful live view without dumping file contents."""
    tag = f"[{step.index + 1:02d}]"
    if step.kind == "model":
        calls = step.payload.get("tool_calls") or []
        print(f"{tag} model -> {', '.join(calls) if calls else 'text'}", flush=True)
        return
    if step.kind == "tool":
        name = step.payload.get("name", "?")
        args = dict(step.payload.get("args") or {})
        if "content" in args:
            args["content"] = f"<{len(str(args['content']))} chars>"
        shown = {key: _short(value) for key, value in args.items()}
        print(f"{tag} tool  {name} {json.dumps(shown, ensure_ascii=False)}", flush=True)
        first = str(step.payload.get("result", "")).splitlines()
        if first:
            print("     " + _short(first[0], 120), flush=True)
        return
    print(f"{tag} {step.kind} {_short(step.payload)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    _survivable_console()
    ap = argparse.ArgumentParser(prog="agent", description="Run the coding agent on one task.")
    ap.add_argument("task", help="What the agent should do.")
    ap.add_argument("--workspace", default=None, help="Directory to work in (default: temp dir).")
    ap.add_argument("--backend", default=None, help="e.g. anthropic:claude-sonnet-5 (default: $NANOCODE_BACKEND).")
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--max-asks", type=int, default=3)
    ap.add_argument("--max-tokens-total", type=int, default=None,
                    help="Soft cap: stop between calls near this recorded token total.")
    ap.add_argument("--max-tool-calls", type=int, default=None,
                    help="Stop after processing this many tool calls.")
    ap.add_argument("--no-ask", action="store_true", help="Remove the ask_user tool entirely.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--trace", default="results/traces/cli.jsonl")
    ap.add_argument("--verbose", action="store_true",
                    help="Print each model/tool trace step as it happens.")
    args = ap.parse_args(argv)

    backend = make_backend(args.backend)
    ws = Workspace(args.workspace)
    cfg = AgentConfig(max_steps=args.max_steps, max_asks=args.max_asks,
                      allow_ask=not args.no_ask, temperature=args.temperature,
                      max_total_tokens=args.max_tokens_total,
                      max_tool_calls=args.max_tool_calls)
    agent = Agent(backend, ws, config=cfg, responder=cli_responder,
                  trace_observer=_print_live_step if args.verbose else None)

    print(f"workspace: {ws.root}")
    print(f"backend:   {backend.name}:{backend.model}\n")

    result = agent.run(args.task)
    path = result.trace.save(args.trace)

    print("\n--- done ---")
    print("outcome: " + result.outcome)
    if result.summary:
        print("summary: " + result.summary)
    t = result.trace
    print(f"steps: {len(t.steps)}  asks: {t.n_asks}  "
          f"tokens: {t.usage.input_tokens}in/{t.usage.output_tokens}out  "
          f"cost: ${t.cost_usd:.4f}")
    print("trace appended to " + str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
