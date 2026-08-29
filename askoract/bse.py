"""Behavioural Spec Entropy (BSE) and disagreement localisation.

BSE answers "how ambiguous is this task?" without knowing the right answer. It is
the normalised Shannon entropy over *behavioural equivalence classes* of sampled
candidate implementations:

    classes  = candidates grouped by identical behaviour vectors over the probes
    BSE      = H(class size distribution) / log(n_candidates)   in [0, 1]

0 means every sample behaves identically -- the task pinned the behaviour down.
1 means every sample behaves differently -- the prompt constrained nothing.

Localisation answers "ambiguous about *what*". Per-probe entropy over behaviour
tokens ranks the probes on which the samples disagree most; the top probe is the
one worth asking about.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .execute import BehaviourMatrix


@dataclass
class ProbeDisagreement:
    index: int
    args: list[Any]
    entropy: float                       # normalised, in [0, 1]
    behaviours: dict[str, list[str]]     # behaviour token -> candidate ids


@dataclass
class BSEResult:
    bse: float
    n_classes: int
    n_candidates: int
    classes: list[list[str]]             # candidate ids per equivalence class
    class_tokens: list[tuple[str, ...]]  # the behaviour vector of each class
    probe_ranking: list[ProbeDisagreement]
    n_invalid: int = 0

    @property
    def majority_class(self) -> list[str]:
        return max(self.classes, key=len) if self.classes else []

    @property
    def top_probe(self) -> ProbeDisagreement | None:
        return self.probe_ranking[0] if self.probe_ranking else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bse": round(self.bse, 6),
            "n_classes": self.n_classes,
            "n_candidates": self.n_candidates,
            "n_invalid": self.n_invalid,
            "class_sizes": [len(c) for c in self.classes],
            "top_probe": None if not self.top_probe else {
                "index": self.top_probe.index,
                "args": self.top_probe.args,
                "entropy": round(self.top_probe.entropy, 6),
                "behaviours": self.top_probe.behaviours,
            },
        }


def normalised_entropy(counts: list[int]) -> float:
    """Shannon entropy of a size distribution, normalised by log(total).

    Normalising by log(total) rather than log(len(counts)) keeps the scale
    comparable across tasks with the same sample budget: the maximum is reached
    only when every sample lands in its own class.
    """
    total = sum(counts)
    if total <= 1:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h / math.log(total)


def compute_bse(matrix: BehaviourMatrix, *, drop_invalid: bool = True) -> BSEResult:
    """Cluster candidates by behaviour vector and score the disagreement.

    `drop_invalid` removes candidates that never ran (syntax error, missing entry
    point). A candidate that does not parse is a generation failure, not evidence
    that the specification was ambiguous -- keeping it would inflate BSE. The flag
    exists so the choice can be ablated rather than assumed.
    """
    n_invalid = sum(matrix.invalid)
    work = matrix.subset(matrix.valid_rows()) if drop_invalid else matrix

    if work.n_candidates == 0:
        return BSEResult(0.0, 0, 0, [], [], [], n_invalid=n_invalid)

    groups: dict[tuple[str, ...], list[str]] = {}
    for cid, row in zip(work.candidate_ids, work.tokens):
        groups.setdefault(tuple(row), []).append(cid)

    classes = sorted(groups.values(), key=len, reverse=True)
    class_tokens = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    bse = normalised_entropy([len(c) for c in classes])

    ranking: list[ProbeDisagreement] = []
    for j in range(work.n_probes):
        col = [work.tokens[i][j] for i in range(work.n_candidates)]
        counts = Counter(col)
        by_behaviour: dict[str, list[str]] = {}
        for cid, tok in zip(work.candidate_ids, col):
            by_behaviour.setdefault(tok, []).append(cid)
        ranking.append(ProbeDisagreement(
            index=j,
            args=work.probes[j],
            entropy=normalised_entropy(list(counts.values())),
            behaviours=by_behaviour,
        ))

    # Ties broken by probe index so the ranking is deterministic across runs.
    ranking.sort(key=lambda p: (-p.entropy, p.index))

    return BSEResult(
        bse=bse,
        n_classes=len(classes),
        n_candidates=work.n_candidates,
        classes=classes,
        class_tokens=[tuple(t) for t in class_tokens],
        probe_ranking=ranking,
        n_invalid=n_invalid,
    )
