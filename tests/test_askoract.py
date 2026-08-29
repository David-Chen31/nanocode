"""Unit checks for the measurement machinery.

These guard the properties the experiments depend on. If BSE stopped
distinguishing behaviour from syntax, every number in results/ would still look
plausible -- so the invariants are asserted directly.
"""
from __future__ import annotations

import math

import pytest

from askoract.bse import compute_bse, normalised_entropy
from askoract.execute import run_candidates
from askoract.policy import AskOrActPolicy
from askoract.probes import filter_probes, synthesize_probes
from askoract.question import build_question
from askoract.signals import (BSESignal, DistinctClassesSignal, SignalInput,
                              TextDiversitySignal)

IDENTITY_A = "def f(xs):\n    return sorted(xs)\n"
IDENTITY_B = "def f(items):\n    out = list(items)\n    out.sort()\n    return out\n"
DIFFERENT = "def f(xs):\n    return sorted(xs, reverse=True)\n"
RAISES = "def f(xs):\n    if not xs:\n        raise ValueError('empty')\n    return sorted(xs)\n"
BROKEN = "def f(xs)\n    return xs\n"          # syntax error
NO_ENTRY = "def g(xs):\n    return xs\n"       # entry point missing


def _matrix(sources, probes=None):
    return run_candidates(sources, probes or [[[3, 1, 2]], [[]], [[1]]], "f")


# ---------------------------------------------------------------- entropy

def test_normalised_entropy_bounds():
    assert normalised_entropy([6]) == 0.0
    assert normalised_entropy([1, 1, 1, 1, 1, 1]) == pytest.approx(1.0)
    assert 0.0 < normalised_entropy([5, 1]) < 1.0
    assert normalised_entropy([]) == 0.0


# ---------------------------------------------------------------- execution

def test_lexically_different_but_behaviourally_identical_collapses():
    """The premise of the whole method: syntax must not create entropy."""
    res = compute_bse(_matrix([IDENTITY_A, IDENTITY_B]))
    assert res.n_classes == 1
    assert res.bse == 0.0


def test_behavioural_difference_creates_entropy():
    res = compute_bse(_matrix([IDENTITY_A, DIFFERENT]))
    assert res.n_classes == 2
    assert res.bse == pytest.approx(1.0)


def test_exception_type_is_an_observable_behaviour():
    res = compute_bse(_matrix([IDENTITY_A, RAISES]))
    # They differ only on the empty probe, which is enough to split them.
    assert res.n_classes == 2
    top = res.top_probe
    assert top is not None and top.args == [[]]   # one argument: the empty list


def test_in_place_mutation_is_observed():
    in_place = "def f(xs):\n    xs.sort()\n    return None\n"
    copies = "def f(xs):\n    return sorted(xs)\n"
    res = compute_bse(_matrix([in_place, copies]))
    assert res.n_classes == 2


def test_invalid_candidates_are_dropped_not_counted_as_disagreement():
    """A syntax error is a generation failure, not evidence of ambiguity."""
    m = _matrix([IDENTITY_A, IDENTITY_B, BROKEN, NO_ENTRY])
    kept = compute_bse(m, drop_invalid=True)
    assert kept.n_invalid == 2
    assert kept.n_candidates == 2
    assert kept.bse == 0.0

    counted = compute_bse(m, drop_invalid=False)
    assert counted.bse > 0.0  # the ablation inflates, as expected


# ---------------------------------------------------------------- probes

def test_probes_keep_seeds_and_add_edge_cases():
    probes = synthesize_probes([[[1, 2, 3], 2]], n=20, seed=0)
    assert [[1, 2, 3], 2] in probes
    assert any(p[0] == [] for p in probes), "empty list edge case should appear"
    assert len(probes) <= 20
    assert len(probes) == len({repr(p) for p in probes}), "probes must be unique"


def test_probes_are_deterministic():
    a = synthesize_probes([[[1, 2], 3]], n=15, seed=4)
    b = synthesize_probes([[[1, 2], 3]], n=15, seed=4)
    assert a == b


def test_precondition_filters_out_of_contract_inputs():
    probes = synthesize_probes([[[1, 2, 3], 2]], n=24, seed=0)
    src = "def precondition(xs, size):\n    return isinstance(size, int) and size >= 1\n"
    kept = filter_probes(probes, src)
    assert all(p[1] >= 1 for p in kept)
    assert len(kept) < len(probes)


def test_filter_never_empties_the_probe_set():
    probes = [[0], [-1]]
    src = "def precondition(x):\n    return x > 100\n"
    assert filter_probes(probes, src) == probes


# ---------------------------------------------------------------- signals

def test_text_diversity_is_blind_to_behaviour():
    """The control baseline must behave as advertised, or H1 proves nothing."""
    same_behaviour = SignalInput("t", [IDENTITY_A, IDENTITY_B],
                                 bse=compute_bse(_matrix([IDENTITY_A, IDENTITY_B])))
    assert TextDiversitySignal().score(same_behaviour) > 0.3   # lexically far apart
    assert BSESignal().score(same_behaviour) == 0.0            # behaviourally identical


def test_distinct_classes_signal_range():
    m = _matrix([IDENTITY_A, IDENTITY_B, DIFFERENT])
    inp = SignalInput("t", [IDENTITY_A, IDENTITY_B, DIFFERENT], bse=compute_bse(m))
    assert DistinctClassesSignal().score(inp) == pytest.approx(0.5)  # 2 classes of 3


# ---------------------------------------------------------------- question + policy

def test_question_offers_the_observed_behaviours():
    m = _matrix([IDENTITY_A, RAISES])
    res = compute_bse(m)
    q = build_question("t", "f", res.top_probe, m)
    assert len(q.option_tokens) == 2
    assert any("raises ValueError" in o for o in q.options)
    assert sum(q.option_support) == 2


def test_policy_never_asks_below_threshold():
    m = _matrix([IDENTITY_A, IDENTITY_B])          # bse == 0
    calls = []

    def responder(q):
        calls.append(q)
        return q.option_tokens[0]

    d = AskOrActPolicy(BSESignal(), 0.5).decide("t", "f", [IDENTITY_A, IDENTITY_B],
                                                m, responder)
    assert d.n_asks == 0 and not calls


def test_policy_asks_and_narrows_the_candidate_set():
    sources = [IDENTITY_A, IDENTITY_B, RAISES]
    m = _matrix(sources)
    # Answer with whatever the majority does, which should eliminate RAISES.
    def responder(q):
        return q.option_tokens[0]

    d = AskOrActPolicy(BSESignal(), 0.1, max_asks=1).decide("t", "f", sources, m, responder)
    assert d.n_asks == 1
    assert d.surviving == 2
    assert d.chosen_id in {"c0", "c1"}


def test_policy_handles_a_user_who_rejects_every_option():
    sources = [IDENTITY_A, RAISES]
    m = _matrix(sources)
    d = AskOrActPolicy(BSESignal(), 0.1, max_asks=2).decide(
        "t", "f", sources, m, lambda q: None)
    assert d.n_asks == 1                # stops asking once nothing matched
    assert d.unmatched_answers == 1
    assert d.chosen_id in {"c0", "c1"}  # still returns something


def test_policy_is_deterministic():
    sources = [IDENTITY_A, IDENTITY_B, DIFFERENT, RAISES]
    m = _matrix(sources)
    runs = [AskOrActPolicy(BSESignal(), 0.1, max_asks=2).decide(
        "t", "f", sources, m, lambda q: q.option_tokens[0]).chosen_id for _ in range(5)]
    assert len(set(runs)) == 1
