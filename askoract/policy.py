"""The ask-or-act gate.

Framing. Asking costs one interruption and buys a reduction in specification
uncertainty. Under a value-of-information rule the agent asks when

    E[gain from the answer]  >  cost of the interruption

Both sides need units. We model the expected gain as monotone increasing in the
uncertainty signal, and the interruption cost as a constant per question. Under
those two assumptions the VOI rule *is* a threshold on the signal -- so this
module implements a threshold and says so plainly rather than dressing a
threshold up as decision theory. What the framing buys is the sweep: varying the
interruption cost traces out the whole success-versus-budget frontier, which is
the object the experiment actually compares.

The second decision the gate makes is what to do with the answer. Filtering the
candidate set by the answered behaviour and then taking the largest surviving
behavioural class is the cheapest possible use of it, and is deliberately not
clever: any gain has to come from the signal, not from the selection rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .bse import BSEResult, compute_bse
from .execute import BehaviourMatrix
from .question import Question, next_question
from .signals import Signal, SignalInput


# An oracle answers a question with a canonical behaviour token, or None when
# none of the offered options is what the user wanted.
Responder = Callable[[Question], "str | None"]


@dataclass
class Decision:
    task_id: str
    score: float
    n_asks: int
    chosen_id: str
    questions: list[dict[str, Any]] = field(default_factory=list)
    unmatched_answers: int = 0
    surviving: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "score": round(self.score, 6),
                "n_asks": self.n_asks, "chosen_id": self.chosen_id,
                "questions": self.questions, "surviving": self.surviving,
                "unmatched_answers": self.unmatched_answers}


def _largest_class(matrix: BehaviourMatrix, rows: list[int]) -> int:
    """Index of a representative from the largest behavioural class among rows.

    Ties are broken by the smallest candidate index so the whole pipeline stays
    deterministic under a fixed seed.
    """
    groups: dict[tuple[str, ...], list[int]] = {}
    for i in rows:
        groups.setdefault(tuple(matrix.tokens[i]), []).append(i)
    best = max(groups.values(), key=lambda g: (len(g), -g[0]))
    return best[0]


class AskOrActPolicy:
    """Score the task, ask while the score clears the bar and budget remains."""

    def __init__(
        self,
        signal: Signal,
        threshold: float,
        *,
        max_asks: int = 2,
        drop_invalid: bool = True,
    ) -> None:
        self.signal = signal
        self.threshold = threshold
        self.max_asks = max_asks
        self.drop_invalid = drop_invalid

    def decide(
        self,
        task_id: str,
        entry_point: str,
        sources: list[str],
        matrix: BehaviourMatrix,
        responder: Responder,
        *,
        signal_extra: dict[str, Any] | None = None,
        budget_left: int | None = None,
    ) -> Decision:
        rows = matrix.valid_rows() if self.drop_invalid else list(range(matrix.n_candidates))
        if not rows:
            rows = list(range(matrix.n_candidates))

        result = compute_bse(matrix, drop_invalid=self.drop_invalid)
        inp = SignalInput(task_id=task_id, sources=sources, matrix=matrix,
                          bse=result, **(signal_extra or {}))
        score = self.signal.score(inp)

        decision = Decision(task_id=task_id, score=score, n_asks=0, chosen_id="")
        allowed = self.max_asks if budget_left is None else min(self.max_asks, budget_left)
        asked_probes: set[int] = set()

        while score >= self.threshold and decision.n_asks < allowed:
            sub = matrix.subset(rows)
            sub_result = compute_bse(sub, drop_invalid=self.drop_invalid)
            q = next_question(task_id, entry_point, sub_result, sub, exclude=asked_probes)
            if q is None:
                break

            answer = responder(q)
            decision.n_asks += 1
            asked_probes.add(q.probe_index)
            record = q.to_dict()
            record["answer"] = answer
            decision.questions.append(record)

            if answer is None:
                # The user rejected every option. Nothing in the candidate set is
                # right; keep the set as it is and stop asking about this task.
                decision.unmatched_answers += 1
                break

            survivors = [i for i in rows if matrix.tokens[i][q.probe_index] == answer]
            if not survivors:
                decision.unmatched_answers += 1
                break
            rows = survivors
            if len(rows) == 1:
                break
            score = compute_bse(matrix.subset(rows), drop_invalid=self.drop_invalid).bse

        decision.surviving = len(rows)
        decision.chosen_id = matrix.candidate_ids[_largest_class(matrix, rows)]
        return decision
