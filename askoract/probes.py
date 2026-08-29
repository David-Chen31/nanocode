"""Probe input synthesis.

Probes are the inputs on which candidate implementations are compared. They must
be generated *without* consulting the hidden constraint that the diagnostic set
removed -- otherwise the experiment measures nothing but the leak. So the
generator only ever sees the entry point's seed inputs (the kind of example that
appears in any task statement) and mutates them by type.
"""
from __future__ import annotations

import random
from typing import Any

# Edge-case values by type. These are the usual suspects that specifications
# forget to mention, which is exactly why they discriminate.
EDGE_VALUES: dict[type, list[Any]] = {
    list: [[], [0], [1], [1, 1], [-1, 0, 1], [2, 1], [1, 2, 3, 4, 5], [1000000, 1]],
    str: ["", " ", "a", "aa", "Ab", "  a  ", "\n", "abc"],
    int: [0, 1, -1, 2, -2, 10, 1000000],
    float: [0.0, 1.0, -1.0, 0.5, 2.0, 1e-9],
    dict: [{}, {"a": 1}, {"a": 1, "b": 2}],
    tuple: [(), (1,), (1, 2)],
    bool: [True, False],
    type(None): [None],
}


def _variants_for(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return EDGE_VALUES[bool]
    for typ, vals in EDGE_VALUES.items():
        if typ is not bool and isinstance(value, typ):
            return vals
    return [value]


def _key(args: list[Any]) -> str:
    return repr(args)


def synthesize_probes(
    seed_args: list[list[Any]],
    *,
    n: int = 24,
    seed: int = 0,
    extra: list[list[Any]] | None = None,
) -> list[list[Any]]:
    """Build a probe set from seed argument tuples by per-argument mutation.

    The seeds themselves are always kept -- a probe set that only holds edge cases
    would miss disagreements in the ordinary path.
    """
    rng = random.Random(seed)
    out: list[list[Any]] = []
    seen: set[str] = set()

    def add(args: list[Any]) -> None:
        k = _key(args)
        if k not in seen:
            seen.add(k)
            out.append(list(args))

    for s in seed_args:
        add(s)
    for e in extra or []:
        add(e)

    # One positional argument replaced at a time keeps each probe interpretable:
    # when candidates split on a probe, the differing argument is obvious.
    pool: list[list[Any]] = []
    for s in seed_args:
        for pos in range(len(s)):
            for v in _variants_for(s[pos]):
                cand = list(s)
                cand[pos] = v
                if _key(cand) not in seen:
                    pool.append(cand)

    rng.shuffle(pool)
    for cand in pool:
        if len(out) >= n:
            break
        add(cand)

    return out[:n]


def compile_precondition(source: str):
    """Compile a task precondition into a callable, or None when absent."""
    if not source or not source.strip():
        return None
    ns: dict[str, Any] = {}
    exec(compile(source, "<precondition>", "exec"), ns)
    fn = ns.get("precondition")
    if not callable(fn):
        raise ValueError("precondition source must define precondition(*args)")
    return fn


def filter_probes(probes: list[list[Any]], source: str) -> list[list[Any]]:
    """Drop probes that fall outside the domain the prompt already restricted.

    Asking the user what `chunk(xs, 0)` should do is a wasted question when both
    prompt variants say "assume size >= 1". Without this filter, out-of-contract
    inputs dominate the disagreement ranking -- see the `no_precondition` ablation.
    """
    fn = compile_precondition(source)
    if fn is None:
        return probes
    kept = []
    for args in probes:
        try:
            if fn(*args):
                kept.append(args)
        except Exception:
            continue
    return kept or probes
