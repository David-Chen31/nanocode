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
from .tools import AskUser, Finish, Tool, build_toolset
from .trace import Trace
from .workspace import Workspace

SYSTEM = """You are a coding agent working inside a sandboxed workspace.

Work in small steps: inspect what is there, make a change, run something to check
it, then move on. Prefer running code over asserting that code is correct.

When the task is done, call `finish` with a one-line summary. If you have a tool
for asking the user, use it only when an answer would actually change the code
you write -- never to confirm something you can determine yourself by reading or
running the code."""


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
                    "seed": self.cfg.seed},
        )
        convo = Conversation(policy=self.cfg.context, backend=self.backend.name)
        convo.add({"role": "user", "content": task})
        schemas = [t.schema() for t in self.tools.values()]
        asked: list[dict[str, Any]] = []
        outcome, summary = "max_steps", ""

        for _ in range(self.cfg.max_steps):
            # Enforce the context budget before every call, not after the API
            # has already refused one.
            convo.compact(lambda ev: trace.record("compact", ev))
            resp = self.backend.complete(
                convo.render(),
                system=system or SYSTEM,
                tools=schemas,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                seed=self.cfg.seed,
            )
            trace.record("model", {"text": resp.text,
                                   "tool_calls": [tc.name for tc in resp.tool_calls]},
                         resp.usage)

            if not resp.tool_calls:
                # No tool call and no finish: treat the text as the final answer.
                outcome, summary = "text_only", resp.text
                break

            convo.add(_assistant_turn(resp.text, resp.tool_calls, self.backend.name))
            for tc in resp.tool_calls:
                convo.note_call(tc.id, tc.name, tc.arguments)
            results: list[tuple[ToolCall, str]] = []
            stop: tuple[str, str] | None = None

            for tc in resp.tool_calls:
                tool = self.tools.get(tc.name)
                if tool is None:
                    results.append((tc, "error: no such tool " + tc.name))
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
