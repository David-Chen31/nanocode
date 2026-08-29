"""Task schema for the counterfactual ambiguity diagnostic set.

Each task exists in two forms that differ by exactly one sentence:

    prompt_full       fully specifies the behaviour
    prompt_ambiguous  the same task with one constraint sentence removed

The removed sentence is the ground truth. That construction is what makes the
set diagnostic rather than merely hard: we know what the agent *should* ask
about, so localisation can be scored, not just final success.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Constraint:
    """A sentence removed from the full prompt, plus the input that reveals it."""

    id: str
    text: str
    # Inputs on which the reference and a plausible wrong reading diverge. Used
    # only for scoring localisation -- never shown to the probe generator.
    discriminating_args: list[list[Any]] = field(default_factory=list)


@dataclass
class Task:
    id: str
    entry_point: str
    prompt_ambiguous: str
    prompt_full: str
    reference: str
    seed_args: list[list[Any]]
    constraints: list[Constraint]
    candidates_ambiguous: list[str]
    candidates_full: list[str]
    # "easy"  -> samples visibly disagree under the ambiguous prompt
    # "hard"  -> samples mostly converge on one reading despite the ambiguity,
    #            so BSE should be low and the task should be a miss. Keeping
    #            these in the set is what stops the benchmark from being rigged.
    # A predicate over the arguments describing the domain the prompt already
    # restricts itself to ("Assume size >= 1"). It is part of the *stated* spec in
    # both variants, so filtering probes with it leaks nothing about the removed
    # constraint -- it just stops the ranking from being dominated by inputs the
    # task never claimed to handle.
    precondition: str = ""
    difficulty: str = "easy"
    notes: str = ""

    def prompt(self, variant: str) -> str:
        return self.prompt_ambiguous if variant == "ambiguous" else self.prompt_full

    def candidates(self, variant: str) -> list[str]:
        return self.candidates_ambiguous if variant == "ambiguous" else self.candidates_full


def load_tasks(package: str = "bench.tasks") -> list[Task]:
    """Import every task module in the package and collect its TASK/TASKS."""
    mod = importlib.import_module(package)
    tasks: list[Task] = []
    for info in pkgutil.iter_modules(mod.__path__):
        if info.name.startswith("_"):
            continue
        sub = importlib.import_module(f"{package}.{info.name}")
        task = getattr(sub, "TASK", None)
        if isinstance(task, Task):
            tasks.append(task)
        for t in getattr(sub, "TASKS", []) or []:
            if isinstance(t, Task):
                tasks.append(t)
    tasks.sort(key=lambda t: t.id)
    return tasks
