"""Diagnostic tasks 05-08. Includes two `hard` tasks.

A `hard` task is one where the deleted sentence contradicts the strong prior:
almost every sample converges on the common reading, so BSE stays low and the
gate will not ask -- and the majority candidate is wrong. These are the honest
negatives. A diagnostic set without them would flatter any disagreement-based
signal, because it would only ever contain ambiguities that happen to be visible.
"""
from __future__ import annotations

from bench.schema import Constraint, Task

# --------------------------------------------------------------------------
# 05 merge_configs -- who wins a key conflict
# --------------------------------------------------------------------------

_MG_FULL = (
    "Write merge_configs(base, override) that returns a new dict containing all "
    "keys from both. Neither input may be modified. When a key is present in "
    "both, the value from override wins."
)
_MG_AMBIG = (
    "Write merge_configs(base, override) that returns a new dict containing all "
    "keys from both. Neither input may be modified."
)
_MG_REF = '''
def merge_configs(base, override):
    out = dict(base)
    out.update(override)
    return out
'''

T05 = Task(
    id="t05_merge_configs",
    entry_point="merge_configs",
    prompt_ambiguous=_MG_AMBIG,
    prompt_full=_MG_FULL,
    reference=_MG_REF,
    seed_args=[[{"a": 1, "b": 2}, {"b": 9, "c": 3}], [{"a": 1}, {"a": 2}]],
    constraints=[Constraint(
        "conflict_winner",
        "When a key is present in both, the value from override wins.",
        [[{"a": 1, "b": 2}, {"b": 9, "c": 3}]])],
    candidates_ambiguous=[
        _MG_REF,
        '''
def merge_configs(base, override):
    return {**base, **override}
''',
        # base wins
        '''
def merge_configs(base, override):
    out = dict(override)
    out.update(base)
    return out
''',
        # base wins, written the other way
        '''
def merge_configs(base, override):
    return {**override, **base}
''',
        # refuses to choose
        '''
def merge_configs(base, override):
    overlap = set(base) & set(override)
    if overlap:
        raise ValueError("conflicting keys: " + ",".join(sorted(overlap)))
    return {**base, **override}
''',
        '''
def merge_configs(base, override):
    merged = {}
    for k, v in base.items():
        merged[k] = v
    for k, v in override.items():
        merged[k] = v
    return merged
''',
    ],
    candidates_full=[
        _MG_REF,
        '''
def merge_configs(base, override):
    return {**base, **override}
''',
        '''
def merge_configs(base, override):
    merged = dict(base)
    for k, v in override.items():
        merged[k] = v
    return merged
''',
        '''
def merge_configs(base, override):
    merged = {}
    merged.update(base)
    merged.update(override)
    return merged
''',
        '''
def merge_configs(base, override):
    out = {k: v for k, v in base.items()}
    for k in override:
        out[k] = override[k]
    return out
''',
        '''
import copy

def merge_configs(base, override):
    out = copy.copy(base)
    out.update(override)
    return out
''',
    ],
    precondition='''
def precondition(base, override):
    return isinstance(base, dict) and isinstance(override, dict)
''',
    difficulty="easy",
    notes="Three readings: override wins, base wins, refuse.",
)


# --------------------------------------------------------------------------
# 06 parse_range -- HARD: the prior says inclusive, the spec says exclusive
# --------------------------------------------------------------------------

_PR_FULL = (
    "Write parse_range(spec) where spec is a string like '3-7'. Return the list of "
    "integers it denotes. The end of the range is exclusive, so '3-7' gives "
    "[3, 4, 5, 6]. Assume spec is well formed, with a non-negative start no larger "
    "than the end."
)
_PR_AMBIG = (
    "Write parse_range(spec) where spec is a string like '3-7'. Return the list of "
    "integers it denotes. Assume spec is well formed, with a non-negative start no "
    "larger than the end."
)
_PR_REF = '''
def parse_range(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b)))
'''

_PR_INCLUSIVE = '''
def parse_range(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))
'''

T06 = Task(
    id="t06_parse_range",
    entry_point="parse_range",
    prompt_ambiguous=_PR_AMBIG,
    prompt_full=_PR_FULL,
    reference=_PR_REF,
    seed_args=[["3-7"], ["0-1"]],
    constraints=[Constraint(
        "exclusive_end",
        "The end of the range is exclusive, so '3-7' gives [3, 4, 5, 6].",
        [["3-7"]])],
    candidates_ambiguous=[
        _PR_INCLUSIVE,
        '''
def parse_range(spec):
    start, end = spec.split("-")
    return [i for i in range(int(start), int(end) + 1)]
''',
        '''
def parse_range(spec):
    parts = spec.split("-")
    lo = int(parts[0])
    hi = int(parts[1])
    out = []
    for i in range(lo, hi + 1):
        out.append(i)
    return out
''',
        '''
def parse_range(spec):
    a, b = spec.split("-", 1)
    return list(range(int(a.strip()), int(b.strip()) + 1))
''',
        '''
def parse_range(spec):
    lo, hi = map(int, spec.split("-"))
    return list(range(lo, hi + 1))
''',
        # the one sample that reads it as exclusive
        _PR_REF,
    ],
    candidates_full=[
        _PR_REF,
        '''
def parse_range(spec):
    lo, hi = map(int, spec.split("-"))
    return list(range(lo, hi))
''',
        '''
def parse_range(spec):
    start, end = spec.split("-")
    return [i for i in range(int(start), int(end))]
''',
        '''
def parse_range(spec):
    a, b = spec.split("-", 1)
    return list(range(int(a.strip()), int(b.strip())))
''',
        '''
def parse_range(spec):
    parts = spec.split("-")
    out = []
    for i in range(int(parts[0]), int(parts[1])):
        out.append(i)
    return out
''',
        '''
def parse_range(spec):
    lo, hi = [int(p) for p in spec.split("-")]
    return list(range(lo, hi))
''',
    ],
    precondition='''
def precondition(spec):
    if not isinstance(spec, str) or spec.count("-") != 1:
        return False
    a, b = spec.split("-")
    return a.isdigit() and b.isdigit() and int(a) <= int(b)
''',
    difficulty="hard",
    notes="5 of 6 samples read the dash as inclusive. BSE stays low; the gate "
          "should not ask, and the majority answer is wrong. An honest miss.",
)


# --------------------------------------------------------------------------
# 07 dedupe -- keep the first or the last occurrence
# --------------------------------------------------------------------------

_DD_FULL = (
    "Write dedupe(xs) that returns a new list with duplicates removed. The "
    "surviving elements keep the relative order of their last occurrence in xs, "
    "so [1, 2, 1, 3] gives [2, 1, 3]. Do not modify xs."
)
_DD_AMBIG = (
    "Write dedupe(xs) that returns a new list with duplicates removed, keeping "
    "the elements in order. Do not modify xs."
)
_DD_REF = '''
def dedupe(xs):
    out = []
    for i, x in enumerate(xs):
        if x not in xs[i + 1:]:
            out.append(x)
    return out
'''

T07 = Task(
    id="t07_dedupe",
    entry_point="dedupe",
    prompt_ambiguous=_DD_AMBIG,
    prompt_full=_DD_FULL,
    reference=_DD_REF,
    seed_args=[[[1, 2, 1, 3]], [[1, 2, 3]]],
    constraints=[Constraint(
        "which_occurrence",
        "The surviving elements keep the relative order of their last occurrence.",
        [[[1, 2, 1, 3]]])],
    candidates_ambiguous=[
        # keep first (the common reading)
        '''
def dedupe(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''',
        '''
def dedupe(xs):
    return list(dict.fromkeys(xs))
''',
        '''
def dedupe(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out
''',
        # keep last
        _DD_REF,
        '''
def dedupe(xs):
    rev = []
    seen = set()
    for x in reversed(xs):
        if x not in seen:
            seen.add(x)
            rev.append(x)
    return list(reversed(rev))
''',
        # sorted set: loses order entirely
        '''
def dedupe(xs):
    return sorted(set(xs))
''',
    ],
    candidates_full=[
        _DD_REF,
        '''
def dedupe(xs):
    rev = []
    seen = set()
    for x in reversed(xs):
        if x not in seen:
            seen.add(x)
            rev.append(x)
    return list(reversed(rev))
''',
        '''
def dedupe(xs):
    seen = set()
    out = []
    for x in reversed(xs):
        if x not in seen:
            seen.add(x)
            out.insert(0, x)
    return out
''',
        '''
def dedupe(xs):
    last = {}
    for i, x in enumerate(xs):
        last[x] = i
    keep = sorted(last.values())
    return [xs[i] for i in keep]
''',
        '''
def dedupe(xs):
    out = []
    n = len(xs)
    for i in range(n):
        if xs[i] not in xs[i + 1:]:
            out.append(xs[i])
    return out
''',
        '''
def dedupe(xs):
    result = list(dict.fromkeys(reversed(xs)))
    result.reverse()
    return result
''',
    ],
    precondition='''
def precondition(xs):
    return isinstance(xs, list) and all(isinstance(x, (int, str, bool)) for x in xs)
''',
    difficulty="easy",
    notes="First-vs-last is a genuine three-way split, including a sorted-set answer.",
)


# --------------------------------------------------------------------------
# 08 round_price -- HARD: Python's round() is banker's, the spec wants half-up
# --------------------------------------------------------------------------

_RP_FULL = (
    "Write round_price(x) that rounds the float x to two decimal places and "
    "returns a float. A value exactly halfway between two candidates rounds away "
    "from zero, so 0.125 becomes 0.13."
)
_RP_AMBIG = (
    "Write round_price(x) that rounds the float x to two decimal places and "
    "returns a float."
)
_RP_REF = '''
from decimal import Decimal, ROUND_HALF_UP

def round_price(x):
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
'''

T08 = Task(
    id="t08_round_price",
    entry_point="round_price",
    prompt_ambiguous=_RP_AMBIG,
    prompt_full=_RP_FULL,
    reference=_RP_REF,
    seed_args=[[0.125], [0.625], [1.234]],
    constraints=[Constraint(
        "halfway_direction",
        "A value exactly halfway rounds away from zero, so 0.125 becomes 0.13.",
        [[0.125], [0.625]])],
    candidates_ambiguous=[
        # everyone reaches for round(), which is banker's rounding
        '''
def round_price(x):
    return round(x, 2)
''',
        '''
def round_price(x):
    return float(round(x, 2))
''',
        '''
def round_price(x):
    return round(float(x), 2)
''',
        '''
def round_price(x):
    return float(f"{x:.2f}")
''',
        '''
def round_price(x):
    return round(x * 100) / 100
''',
        # the one sample that thinks about money
        _RP_REF,
    ],
    candidates_full=[
        _RP_REF,
        '''
from decimal import Decimal, ROUND_HALF_UP

def round_price(x):
    d = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(d)
''',
        '''
import math

def round_price(x):
    scaled = x * 100
    if scaled >= 0:
        return math.floor(scaled + 0.5) / 100
    return math.ceil(scaled - 0.5) / 100
''',
        '''
from decimal import Decimal, ROUND_HALF_UP

def round_price(x):
    return float(Decimal(str(x)).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP))
''',
        '''
import math

def round_price(x):
    sign = 1 if x >= 0 else -1
    return sign * math.floor(abs(x) * 100 + 0.5) / 100
''',
        '''
from decimal import Decimal, ROUND_HALF_UP

def round_price(x):
    q = Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(q)
''',
    ],
    precondition='''
def precondition(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)
''',
    difficulty="hard",
    notes="round() is the overwhelming prior. BSE stays low and the majority is "
          "wrong -- a miss no disagreement-based signal can catch.",
)


TASKS = [T05, T06, T07, T08]
