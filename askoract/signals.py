"""Uncertainty signals, behind one interface so the gate can be ablated.

Every signal maps a task's candidate set to a score in [0, 1] meaning "how
unsure am I about what was asked for". The point of the interface is that the
gating policy never learns which signal it is holding, so swapping BSE for a
lexical or random signal changes exactly one thing in the experiment.

Grouping:
  behavioural   BSE, MaxProbeEntropy, DistinctClasses  -- need execution
  lexical       TextDiversity                          -- source text only
  model-report  TokenEntropy, SelfReport               -- need a live backend
  control       Random, Constant
"""
from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .bse import BSEResult, compute_bse, normalised_entropy
from .execute import BehaviourMatrix


@dataclass
class SignalInput:
    """Everything a signal is allowed to see. Notably not the hidden constraint."""

    task_id: str
    sources: list[str]
    matrix: BehaviourMatrix | None = None
    bse: BSEResult | None = None
    mean_logprobs: list[float] | None = None
    self_report: float | None = None
    direct_ask: float | None = None


class Signal(Protocol):
    name: str
    requires_live: bool

    def score(self, inp: SignalInput) -> float: ...


class BSESignal:
    """Normalised entropy over behavioural equivalence classes. The proposal."""

    name = "bse"
    requires_live = False

    def score(self, inp: SignalInput) -> float:
        if inp.bse is None:
            raise ValueError("BSESignal needs a precomputed BSEResult")
        return inp.bse.bse


class MaxProbeEntropySignal:
    """Ablation: max per-probe entropy instead of vector-level class entropy.

    Answers "is the aggregator doing any work?". Vector-level entropy counts two
    candidates as different if they differ *anywhere*; the max-probe version only
    looks at the single most contested input.
    """

    name = "max_probe_entropy"
    requires_live = False

    def score(self, inp: SignalInput) -> float:
        if inp.bse is None or not inp.bse.probe_ranking:
            return 0.0
        return inp.bse.probe_ranking[0].entropy


class DistinctClassesSignal:
    """Ablation: just count behavioural classes, ignoring their sizes.

    If this matches BSE, the entropy weighting adds nothing and the simpler
    statistic should be preferred.
    """

    name = "distinct_classes"
    requires_live = False

    def score(self, inp: SignalInput) -> float:
        if inp.bse is None or inp.bse.n_candidates <= 1:
            return 0.0
        return (inp.bse.n_classes - 1) / (inp.bse.n_candidates - 1)


_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|\S")


def _normalise_source(src: str) -> list[str]:
    lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    return _TOKEN.findall("\n".join(lines))


class TextDiversitySignal:
    """The control that decides whether execution matters.

    Mean pairwise lexical distance between candidate sources. If this scores as
    well as BSE, nothing is gained by running the code and the whole premise of
    the method collapses -- so this is the baseline that matters most.
    """

    name = "text_diversity"
    requires_live = False

    def score(self, inp: SignalInput) -> float:
        toks = [_normalise_source(s) for s in inp.sources]
        n = len(toks)
        if n < 2:
            return 0.0
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                ratio = difflib.SequenceMatcher(None, toks[i], toks[j]).ratio()
                dists.append(1.0 - ratio)
        return sum(dists) / len(dists)


class TokenEntropySignal:
    """Baseline: mean per-token logprob of the sampled completions.

    The standard confidence proxy in NLP. On code it cannot separate harmless
    syntactic variation from a genuine behavioural fork, which is the hypothesis
    this baseline exists to test.
    """

    name = "token_entropy"
    requires_live = True

    def score(self, inp: SignalInput) -> float:
        lps = [lp for lp in (inp.mean_logprobs or []) if lp is not None]
        if not lps:
            raise ValueError("token_entropy requires logprobs from a live backend")
        # Map mean logprob (<= 0) into [0, 1]; -2.0 nats/token is treated as
        # maximal uncertainty, which covers the usual range for code completions.
        avg = sum(lps) / len(lps)
        return max(0.0, min(1.0, -avg / 2.0))


class SelfReportSignal:
    """Baseline: ask the model how confident it is, use 1 - confidence."""

    name = "self_report"
    requires_live = True

    def score(self, inp: SignalInput) -> float:
        if inp.self_report is None:
            raise ValueError("self_report requires a live backend")
        return max(0.0, min(1.0, 1.0 - inp.self_report))


class DirectAskSignal:
    """Baseline: ask the model outright how under-specified the task is.

    This is the baseline the execution machinery has to beat, and the one most
    easily skipped: it needs no sampling, no probes and no execution, so if it
    matches BSE then the entire behavioural pipeline is unjustified overhead.
    """

    name = "direct_ask"
    requires_live = True

    def score(self, inp: SignalInput) -> float:
        if inp.direct_ask is None:
            raise ValueError("direct_ask requires a live backend")
        return max(0.0, min(1.0, inp.direct_ask))


class RandomSignal:
    """Control: uniform noise. Any gate built on this defines the floor."""

    name = "random"
    requires_live = False

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def score(self, inp: SignalInput) -> float:
        return random.Random(f"{self._seed}:{inp.task_id}").random()


class ConstantSignal:
    """Control: realises always-ask (1.0) and never-ask (0.0)."""

    requires_live = False

    def __init__(self, value: float, name: str) -> None:
        self.value = value
        self.name = name

    def score(self, inp: SignalInput) -> float:
        return self.value


def offline_signals(seed: int = 0) -> list[Signal]:
    """Every signal that runs without an API key."""
    return [BSESignal(), MaxProbeEntropySignal(), DistinctClassesSignal(),
            TextDiversitySignal(), RandomSignal(seed)]


def build_signal_input(task_id: str, sources: list[str], matrix: BehaviourMatrix,
                       *, drop_invalid: bool = True, **extra: Any) -> SignalInput:
    return SignalInput(
        task_id=task_id,
        sources=sources,
        matrix=matrix,
        bse=compute_bse(matrix, drop_invalid=drop_invalid),
        **extra,
    )
