"""Trajectory recording.

Every run writes a full JSONL trace: each model call, each tool call, tokens and
dollars. Publishing traces (not just final patches) is one of the reproducibility
requirements this project holds itself to.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

    def record(self, kind: str, payload: dict[str, Any], usage: Usage | None = None) -> None:
        self.steps.append(Step(
            index=len(self.steps),
            kind=kind,
            payload=payload,
            t_wall=round(time.time() - self.started, 3),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        ))
        if usage:
            self.usage.add(usage)

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
