"""The agent loop.

A plain tool-calling ReAct loop, kept small on purpose: it is the control
condition for the research, so anything clever belongs in a policy module that
can be ablated, not in here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .context import ContextPolicy, Conversation, clip_tool_output
from .llm import LLMBackend, ToolCall, Usage
from .tools import AskUser, Finish, Tool, build_toolset, check_arguments
from .trace import Trace
from .workspace import Workspace

SYSTEM = """You are a coding agent working inside a sandboxed workspace.

Work in small steps: inspect what is there, make a change, run something to check
it, then move on. Prefer running code over asserting that code is correct.

When the task is done, call `finish` with a one-line summary. If you have a tool
for asking the user, use it only when an answer would actually change the code
you write -- never to confirm something you can determine yourself by reading or
running the code."""


IDLE_NUDGE = (
    "You replied without calling a tool. If the task is complete, call `finish` "
    "with a summary. If not, call the tool you need next -- describing what you "
    "intend to do does not do it."
)


def orientation(ws: Workspace) -> str:
    """Tell the agent where it is before it has to guess.

    Measured on a real repository: with no orientation the agent assumed the
    root was `/workspace`, then spent turns on `pwd && ls -la` and absolute
    paths into the temp directory before finding anything. Those turns come out
    of the step budget.
    """
    return ("\n\nYou are working in a project directory. Every path you pass to "
            "a tool is relative to that directory -- do not use absolute paths. "
            "Here is what it contains:\n\n" + ws.overview())


def budget_note(used: int, total: int) -> str:
    """Tell the agent where it is in its step budget.

    Without this the agent cannot see the budget at all, and an agent that
    cannot see a limit cannot manage it -- it works until it is cut off
    mid-thought. Measured: two thirds of runs that produced correct code never
    called `finish`, burning every remaining step.

    The warning only appears near the end. Saying it every turn would make it
    background noise, and worse, would push toward stopping early on a task
    that has plenty of budget left -- which trades correctness for a better
    termination number, the wrong direction.
    """
    left = total - used
    line = f"\n\nStep {used + 1} of {total}."
    if left <= 3:
        return line + (" You are nearly out of steps. Finish now: make sure the "
                       "file is in its best state, then call `finish` and say "
                       "plainly what is done and what is not.")
    if left <= total // 3:
        return line + (" Budget is running low -- prefer completing the change "
                       "over further exploration.")
    return line


class UserResponder(Protocol):
    """Answers an agent's question. A human at a terminal, or an oracle sim."""

    def __call__(self, question: str, options: list[str]) -> str: ...


def cli_responder(question: str, options: list[str]) -> str:
    print("\n[agent asks] " + question)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    raw = input("your answer> ").strip()
    if options and raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw


@dataclass
class AgentConfig:
    max_steps: int = 24
    temperature: float = 0.0
    max_asks: int = 3
    allow_ask: bool = True
    max_tokens: int = 4096
    seed: int | None = 0
    # How many tool-free turns to nudge through before accepting the text as
    # the final answer. Zero restores the old behaviour of stopping at the
    # first one, which mistook a preamble for a conclusion.
    max_idle_turns: int = 2
    # A ceiling on total tokens for one run, or None for no ceiling. The step
    # budget bounds how many times the agent acts, which is not the same thing:
    # twenty turns over a large repository cost far more than twenty over a
    # small one, and only this bounds that. Tokens rather than dollars because
    # the token count is measured while a price may not be known.
    max_total_tokens: int | None = None
    # Show the agent its remaining step budget. Off would mean an agent that
    # cannot see the limit it is being judged against.
    show_budget: bool = True
    context: ContextPolicy = field(default_factory=ContextPolicy)


@dataclass
class AgentResult:
    outcome: str
    summary: str
    trace: Trace
    workspace: Workspace
    asked: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        workspace: Workspace,
        *,
        config: AgentConfig | None = None,
        responder: UserResponder | None = None,
        tools: dict[str, Tool] | None = None,
    ) -> None:
        self.backend = backend
        self.ws = workspace
        self.cfg = config or AgentConfig()
        self.responder = responder or cli_responder
        self.tools = tools or build_toolset(workspace, allow_ask=self.cfg.allow_ask)

    def run(self, task: str, *, task_id: str = "adhoc", system: str | None = None) -> AgentResult:
        trace = Trace(
            run_id=uuid.uuid4().hex[:12],
            task_id=task_id,
            model=self.backend.model,
            backend=self.backend.name,
            config={"max_steps": self.cfg.max_steps, "temperature": self.cfg.temperature,
                    "max_asks": self.cfg.max_asks, "allow_ask": self.cfg.allow_ask,
                    "seed": self.cfg.seed, "show_budget": self.cfg.show_budget,
                    "max_idle_turns": self.cfg.max_idle_turns,
                    "max_total_tokens": self.cfg.max_total_tokens},
        )
        convo = Conversation(policy=self.cfg.context, backend=self.backend.name)
        convo.add({"role": "user", "content": task})
        schemas = [t.schema() for t in self.tools.values()]
        asked: list[dict[str, Any]] = []
        outcome, summary = "max_steps", ""
        idle = 0
        # Built once: the layout is stable enough over one run, and rebuilding
        # it every turn would walk the tree on every model call.
        where = orientation(self.ws)

        for step in range(self.cfg.max_steps):
            # Checked before the call rather than after, so the ceiling is a
            # ceiling. Work already written to disk is kept -- the run stops,
            # it does not roll back.
            spent = trace.usage.input_tokens + trace.usage.output_tokens
            if self.cfg.max_total_tokens and spent >= self.cfg.max_total_tokens:
                outcome = "token_budget"
                summary = (f"Stopped after {spent} tokens, at the configured "
                           f"ceiling of {self.cfg.max_total_tokens}. Any files "
                           f"already written are on disk.")
                trace.record("budget", {"spent": spent,
                                        "limit": self.cfg.max_total_tokens})
                break
            # Enforce the context budget before every call, not after the API
            # has already refused one.
            convo.compact(lambda ev: trace.record("compact", ev))
            base_system = (system or SYSTEM) + where
            resp = self.backend.complete(
                convo.render(),
                # Appended to the system prompt rather than pushed into the
                # history: the count changes every turn, and a history full of
                # stale "step 4 of 24" messages is both wasted context and
                # actively misleading.
                system=(base_system + budget_note(step, self.cfg.max_steps)
                        if self.cfg.show_budget else base_system),
                tools=schemas,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                seed=self.cfg.seed,
                # Passed per call rather than stored on the backend. A backend
                # shared by two agents has one slot for a hook, so the second
                # agent silently inherits the first agent's retries -- and a
                # retry that nothing records looks exactly like a slow call.
                on_retry=lambda n, d, why: trace.record(
                    "retry", {"attempt": n, "delay_s": round(d, 2), "why": why}),
            )
            trace.record("model", {"text": resp.text,
                                   "tool_calls": [tc.name for tc in resp.tool_calls]},
                         resp.usage)

            if not resp.tool_calls:
                # A turn with no tool call is ambiguous: the model may be done,
                # or it may be narrating before it acts ("Let me start by
                # exploring the workspace..."). Treating both as the final
                # answer ended real runs at step one with nothing done, so the
                # model is asked once, and only a repeat is taken as final.
                if idle < self.cfg.max_idle_turns:
                    idle += 1
                    trace.record("idle", {"text": resp.text[:400], "nudge": idle})
                    convo.add({"role": "assistant", "content": resp.text or "..."})
                    convo.add({"role": "user", "content": IDLE_NUDGE})
                    continue
                outcome, summary = "text_only", resp.text
                break
            idle = 0

            convo.add(_assistant_turn(resp.text, resp.tool_calls, self.backend.name))
            for tc in resp.tool_calls:
                convo.note_call(tc.id, tc.name, tc.arguments)
            results: list[tuple[ToolCall, str]] = []
            stop: tuple[str, str] | None = None

            for tc in resp.tool_calls:
                tool = self.tools.get(tc.name)
                if tool is None:
                    # Name the alternatives. A model that invents `grep` can act
                    # on "did you mean search?"; it cannot act on "no".
                    known = ", ".join(sorted(self.tools))
                    msg = f"no such tool {tc.name!r}. Available tools: {known}"
                    results.append((tc, "error: " + msg))
                    trace.record("tool", {"name": tc.name, "args": tc.arguments,
                                          "error": msg})
                    continue
                bad = check_arguments(tool, tc.arguments)
                if bad:
                    # Caught before dispatch so the message can quote the schema
                    # rather than a Python TypeError about a closure.
                    results.append((tc, "error: " + bad))
                    trace.record("tool", {"name": tc.name, "args": tc.arguments,
                                          "error": bad})
                    continue
                try:
                    out = tool.fn(**tc.arguments)
                    # Clip on the way in. Nothing unbounded ever reaches history,
                    # so no single result can blow the budget by itself.
                    out, clipped = clip_tool_output(out, self.cfg.context)
                    convo.n_clipped += clipped
                    results.append((tc, out))
                    trace.record("tool", {"name": tc.name, "args": tc.arguments,
                                          "result": out[:2000], "clipped": clipped})
                except AskUser as ask:
                    if len(asked) >= self.cfg.max_asks:
                        results.append((tc, "You have used your question budget. Proceed "
                                            "with your best judgement and state the assumption."))
                        trace.record("ask", {"question": ask.question, "refused": True})
                        continue
                    answer = self.responder(ask.question, ask.options)
                    asked.append({"question": ask.question, "options": ask.options,
                                  "answer": answer})
                    trace.n_asks += 1
                    results.append((tc, "User answered: " + answer))
                    trace.record("ask", {"question": ask.question, "options": ask.options,
                                         "answer": answer})
                except Finish as fin:
                    results.append((tc, "ok"))
                    trace.record("end", {"summary": fin.summary})
                    stop = ("finished", fin.summary)
                except Exception as exc:  # tool crashed; let the model see it
                    msg = f"{type(exc).__name__}: {exc}"
                    results.append((tc, "error: " + msg))
                    trace.record("tool", {"name": tc.name, "args": tc.arguments, "error": msg})

            for turn in _tool_result_turns(results, self.backend.name):
                convo.add(turn)
            if stop:
                outcome, summary = stop
                break

        trace.outcome = outcome
        trace.record("end", {"context": convo.stats()})
        return AgentResult(outcome=outcome, summary=summary, trace=trace,
                           workspace=self.ws, asked=asked)


def _assistant_turn(text: str, calls: list[ToolCall], backend: str) -> dict[str, Any]:
    if backend == "openai":
        return {
            "role": "assistant",
            "content": text or None,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name,
                                         "arguments": _json(c.arguments)}} for c in calls],
        }
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for c in calls:
        content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
    return {"role": "assistant", "content": content}


def _tool_result_turns(results: list[tuple[ToolCall, str]], backend: str) -> list[dict[str, Any]]:
    """OpenAI wants one message per tool result; Anthropic wants them batched."""
    if backend == "openai":
        return [{"role": "tool", "tool_call_id": tc.id, "content": out} for tc, out in results]
    return [{"role": "user",
             "content": [{"type": "tool_result", "tool_use_id": tc.id, "content": out}
                         for tc, out in results]}]


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
