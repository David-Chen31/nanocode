"""A plan-then-execute loop, as the alternative to the ReAct loop in loop.py.

The loop architecture is the largest design decision in this project and the
only one with no evidence behind it -- "ReAct is standard" is an appeal to
convention, not a defence. This module exists so the choice can be measured
instead of asserted.

BUILDING THE STRONGEST FAIR VERSION, NOT A STRAW MAN

A comparison is worthless if the alternative is built to lose, and a weak
version is easy to build by accident. So:

- Same tools, same model, same temperature, same scoring.
- Same total budget. The planning call consumes one step of `max_steps`,
  because it costs a model call and pretending otherwise would hand this arm a
  free turn.
- Same orientation. The planner sees the repository layout, so it is not
  planning blind about structure.
- **The plan is guidance, not a script.** Execution may depart from it, and the
  prompt says so explicitly. Forcing blind execution of a plan written before
  any file was read would be the straw man: nobody builds that, and beating it
  would prove nothing.

What differs is only this: ReAct decides the next action from the current
state, every turn. Plan-then-execute commits to a sequence first, then carries
it as context while acting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMBackend
from .loop import SYSTEM, Agent, AgentConfig, AgentResult, UserResponder, orientation
from .tools import Tool
from .workspace import Workspace

PLANNER_SYSTEM = """You are planning a change to a codebase before making it.

Write a short numbered plan: the concrete steps needed, in order, naming the
files you expect to touch and how you will verify the change worked. Between
three and seven steps. No prose before or after the list.

You cannot run anything yet, so do not guess at contents you have not seen --
if a step needs information, make "read X" the step."""

PLAN_PREFACE = """You are executing a plan you wrote for this task.

{plan}

Follow it, but it is guidance rather than a script: you wrote it before reading
any file, so if what you find contradicts it, say so briefly and do the right
thing instead. Verify your work by running it before calling `finish`."""


@dataclass
class PlanResult:
    plan: str
    result: AgentResult


class PlanExecuteAgent:
    """Plan once, then execute with the plan in context."""

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
        self.responder = responder
        self.tools = tools
        self.last_trace = None

    def plan_for(self, task: str) -> tuple[str, Any]:
        """One model call, no tools, to produce the plan."""
        resp = self.backend.complete(
            [{"role": "user", "content": task}],
            # The same repository map the executing agent gets, so any
            # difference between the arms is the control flow and not what
            # either one was allowed to know.
            system=PLANNER_SYSTEM + orientation(self.ws),
            tools=None,
            temperature=self.cfg.temperature,
            max_tokens=(min(self.cfg.max_tokens, self.cfg.max_total_tokens)
                        if self.cfg.max_total_tokens is not None
                        else self.cfg.max_tokens),
            seed=self.cfg.seed,
        )
        return resp.text.strip(), resp.usage

    def run(self, task: str, *, task_id: str = "adhoc") -> PlanResult:
        plan, usage = self.plan_for(task)

        remaining_tokens = self.cfg.max_total_tokens
        if remaining_tokens is not None:
            remaining_tokens = max(
                0, remaining_tokens - usage.input_tokens - usage.output_tokens)
        executor_steps = max(0, self.cfg.max_steps - 1)
        # If planning exhausted the soft token budget, do not make a second
        # model call. Zero executor steps is valid: the planning call was the
        # arm's one allowed turn.
        if remaining_tokens == 0:
            executor_steps = 0

        agent = Agent(
            self.backend, self.ws,
            # The planning call already spent a turn; the executor gets the
            # rest, so both arms make at most max_steps model calls.
            config=_with_steps(self.cfg, executor_steps,
                               max_total_tokens=remaining_tokens),
            responder=self.responder, tools=self.tools,
        )
        try:
            res = agent.run(task, task_id=task_id,
                            system=SYSTEM + "\n\n" + PLAN_PREFACE.format(
                                plan=plan or "(no plan)"))
        except Exception:
            # Preserve the planner's spend even when execution crashes before
            # it can return an AgentResult.
            self.last_trace = agent.last_trace
            if self.last_trace:
                self.last_trace.record("plan", {"plan": plan[:4000]}, usage)
            raise

        # Fold the planning call into the trace so cost and call counts include
        # it. Reporting an arm's cost without the call that defines it would
        # flatter this arm for no reason.
        res.trace.record("plan", {"plan": plan[:4000]}, usage)
        self.last_trace = res.trace
        return PlanResult(plan=plan, result=res)


def _with_steps(cfg: AgentConfig, steps: int, *,
                max_total_tokens: int | None = None) -> AgentConfig:
    return AgentConfig(
        max_steps=steps, temperature=cfg.temperature, max_asks=cfg.max_asks,
        allow_ask=cfg.allow_ask, max_tokens=cfg.max_tokens, seed=cfg.seed,
        max_idle_turns=cfg.max_idle_turns, show_budget=cfg.show_budget,
        max_total_tokens=(cfg.max_total_tokens if max_total_tokens is None
                          else max_total_tokens),
        max_tool_calls=cfg.max_tool_calls,
        context=cfg.context,
    )
