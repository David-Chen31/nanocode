"""Turn a disagreement into one closed question.

Deciding *whether* to ask is only half the problem. A question like "what do you
want?" costs the user the same interruption and returns almost nothing, so the
question is built from the specific probe the samples split on, and its options
are the behaviours actually observed. That makes it answerable in one click and
makes the answer directly usable: it names a behaviour token, which filters the
candidate set without any further interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bse import BSEResult, ProbeDisagreement
from .execute import BehaviourMatrix


@dataclass
class Question:
    task_id: str
    probe_index: int
    args: list[Any]
    text: str
    options: list[str] = field(default_factory=list)         # human readable
    option_tokens: list[str] = field(default_factory=list)   # canonical behaviour
    option_support: list[int] = field(default_factory=list)  # samples per option

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "probe_index": self.probe_index,
                "args": self.args, "text": self.text, "options": self.options,
                "option_tokens": self.option_tokens, "support": self.option_support}


def _call_repr(entry_point: str, args: list[Any]) -> str:
    return entry_point + "(" + ", ".join(repr(a) for a in args) + ")"


def build_question(
    task_id: str,
    entry_point: str,
    disagreement: ProbeDisagreement,
    matrix: BehaviourMatrix,
) -> Question:
    """Phrase the top disagreement as a multiple-choice question."""
    ordered = sorted(disagreement.behaviours.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    tokens = [tok for tok, _ in ordered]
    support = [len(ids) for _, ids in ordered]
    options = [matrix.display(tok, disagreement.index) for tok in tokens]

    call = _call_repr(entry_point, disagreement.args)
    text = (
        f"The task does not say what {call} should do, and my candidate "
        f"implementations split on it. Which behaviour is correct?"
    )
    return Question(task_id=task_id, probe_index=disagreement.index,
                    args=disagreement.args, text=text, options=options,
                    option_tokens=tokens, option_support=support)


def next_question(
    task_id: str,
    entry_point: str,
    result: BSEResult,
    matrix: BehaviourMatrix,
    *,
    exclude: set[int] | None = None,
    min_entropy: float = 1e-9,
) -> Question | None:
    """Highest-entropy probe not already asked about, or None if all settled."""
    exclude = exclude or set()
    for d in result.probe_ranking:
        if d.index in exclude or d.entropy <= min_entropy:
            continue
        return build_question(task_id, entry_point, d, matrix)
    return None
