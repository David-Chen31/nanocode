"""Precomputation, the oracle user, and success scoring.

Execution is the expensive part of every experiment here, so each task variant is
executed once and cached; sweeping thresholds, signals and budgets afterwards is
free. That is what makes a full Pareto sweep over five signals runnable on a
laptop in under a minute.

The user simulator is the reference implementation. Asked "what should f([]) do?",
it answers with what the reference actually does. No model is involved, so the
answers are deterministic, free, and cannot be gamed by phrasing -- at the cost
of modelling a user who is always available, always right, and never annoyed.
That limitation is stated in the results rather than papered over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from askoract.bse import BSEResult, compute_bse
from askoract.execute import BehaviourMatrix, run_candidates
from askoract.probes import filter_probes, synthesize_probes
from askoract.question import Question

from .schema import Task

CALIB_SEED = 0     # probes the agent is allowed to see
EVAL_SEED = 7      # held-out probes that decide success


def _norm(args: Any) -> str:
    return json.dumps(args, sort_keys=True, default=str)


@dataclass
class TaskArtifacts:
    task: Task
    variant: str
    probes: list[list[Any]]
    eval_probes: list[list[Any]]
    matrix: BehaviourMatrix
    eval_matrix: BehaviourMatrix
    ref_probe_row: list[str]
    ref_eval_row: list[str]
    # Present only when candidates came from a live model. `mean_logprobs` powers
    # the token-entropy baseline and `self_report` the self-reported-confidence
    # baseline; both are None in fixture mode, where the signals are skipped
    # rather than faked.
    live_sources: list[str] | None = None
    mean_logprobs: list[float] | None = None
    self_report: float | None = None
    direct_ask: float | None = None
    bse: BSEResult = field(init=False)

    def __post_init__(self) -> None:
        self.bse = compute_bse(self.matrix)

    @property
    def sources(self) -> list[str]:
        return self.live_sources if self.live_sources is not None \
            else self.task.candidates(self.variant)

    def is_correct(self, candidate_id: str) -> bool:
        """A candidate succeeds when it is behaviourally identical to the
        reference on every held-out probe. This is differential testing against
        the reference, not a hand-written suite, so it cannot under-test the
        constraint the task removed."""
        i = self.eval_matrix.candidate_ids.index(candidate_id)
        return self.eval_matrix.tokens[i] == self.ref_eval_row

    def constraint_probe_indices(self) -> set[int]:
        """Probe positions that reveal one of the removed constraints.

        Used only for scoring localisation. The probe generator never sees this.
        """
        wanted = {_norm(a) for c in self.task.constraints for a in c.discriminating_args}
        return {j for j, p in enumerate(self.probes) if _norm(p) in wanted}

    def majority_id(self) -> str:
        return self.bse.majority_class[0] if self.bse.majority_class else \
            self.matrix.candidate_ids[0]


def build_artifacts(task: Task, variant: str, *, n_probes: int = 24,
                    n_eval: int = 40,
                    sources: list[str] | None = None,
                    mean_logprobs: list[float] | None = None,
                    self_report: float | None = None,
                    direct_ask: float | None = None) -> TaskArtifacts:
    probes = filter_probes(
        synthesize_probes(task.seed_args, n=n_probes, seed=CALIB_SEED), task.precondition)

    # The held-out set always contains the inputs that discriminate the removed
    # constraint, so a candidate that guessed wrong cannot pass by luck.
    discriminating = [a for c in task.constraints for a in c.discriminating_args]
    eval_probes = filter_probes(
        synthesize_probes(task.seed_args, n=n_eval, seed=EVAL_SEED, extra=discriminating),
        task.precondition)

    live = sources
    sources = sources if sources is not None else task.candidates(variant)
    ids = [f"c{i}" for i in range(len(sources))]

    matrix = run_candidates(sources, probes, task.entry_point, candidate_ids=ids)
    eval_matrix = run_candidates(sources, eval_probes, task.entry_point, candidate_ids=ids)
    ref_probe = run_candidates([task.reference], probes, task.entry_point,
                               candidate_ids=["ref"])
    ref_eval = run_candidates([task.reference], eval_probes, task.entry_point,
                              candidate_ids=["ref"])

    return TaskArtifacts(
        task=task, variant=variant, probes=probes, eval_probes=eval_probes,
        matrix=matrix, eval_matrix=eval_matrix,
        ref_probe_row=ref_probe.tokens[0], ref_eval_row=ref_eval.tokens[0],
        live_sources=live, mean_logprobs=mean_logprobs, self_report=self_report,
        direct_ask=direct_ask,
    )


class OracleUser:
    """Answers a question by consulting the reference implementation."""

    def __init__(self, artifacts: TaskArtifacts) -> None:
        self.art = artifacts
        self.asked: list[Question] = []
        self.on_constraint = 0
        self.off_constraint = 0
        self.rejected = 0

    def __call__(self, q: Question) -> str | None:
        self.asked.append(q)
        if q.probe_index in self.art.constraint_probe_indices():
            self.on_constraint += 1
        else:
            self.off_constraint += 1
        want = self.art.ref_probe_row[q.probe_index]
        if want in q.option_tokens:
            return want
        # None of the sampled behaviours is what the user wanted. A real user
        # would say so; the agent learns the answer is outside its candidate set.
        self.rejected += 1
        return None
