"""Trajectory recording.

A Trace contains each model call, tool call, token and cost record. The CLI
appends it to JSONL; live experiment harnesses persist one artifact per run with
the trace and final patch. Agent.run itself does not choose an output path.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm import Usage


@dataclass
class Step:
    index: int
    kind: str                      # "model" | "tool" | "ask" | "end"
    payload: dict[str, Any]
    t_wall: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trace:
    run_id: str
    task_id: str
    model: str
    backend: str
    config: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    started: float = field(default_factory=time.time)
    outcome: str = "incomplete"
    n_asks: int = 0
    observer: Callable[[Step], None] | None = field(
        default=None, repr=False, compare=False)

    def record(self, kind: str, payload: dict[str, Any], usage: Usage | None = None) -> None:
        step = Step(
            index=len(self.steps),
            kind=kind,
            payload=payload,
            t_wall=round(time.time() - self.started, 3),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
        self.steps.append(step)
        if usage:
            self.usage.add(usage)
        if self.observer:
            try:
                self.observer(step)
            except Exception:
                # A display/progress hook must never turn a successful agent
                # action into a failed run. The durable trace above is intact.
                pass

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd(self.model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model": self.model,
            "backend": self.backend,
            "config": self.config,
            "outcome": self.outcome,
            "n_asks": self.n_asks,
            "n_steps": len(self.steps),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "model_calls": self.usage.calls,
            "cost_usd": round(self.cost_usd, 6),
            "wall_seconds": round(time.time() - self.started, 3),
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        return p
