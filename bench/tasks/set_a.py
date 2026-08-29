"""Diagnostic tasks 01-04.

Construction rule for every task in this set:

  * `prompt_full` pins down every behaviour the reference exhibits.
  * `prompt_ambiguous` is `prompt_full` with exactly one sentence deleted.
  * the candidate lists are what a model plausibly samples from each prompt.

Two deliberate choices keep the set from being rigged in the method's favour:
the `full` candidate lists are not always behaviourally identical (real sampling
has incidental variation), and the set contains `hard` tasks where samples
converge on the common reading even though the prompt left it open -- those are
misses the signal is supposed to be scored on.
"""
from __future__ import annotations

from bench.schema import Constraint, Task

# --------------------------------------------------------------------------
# 01 remove_outliers -- what happens on empty input
# --------------------------------------------------------------------------

_RO_FULL = (
    "Write a function remove_outliers(xs, k) that returns a new list holding the "
    "values of xs whose absolute distance from the mean is at most k population "
    "standard deviations (ddof=0). Preserve the original order and do not modify "
    "xs. Assume k >= 0. If xs is empty, return an empty list."
)

_RO_AMBIG = (
    "Write a function remove_outliers(xs, k) that returns a new list holding the "
    "values of xs whose absolute distance from the mean is at most k population "
    "standard deviations (ddof=0). Preserve the original order and do not modify xs. "
    "Assume k >= 0."
)

_RO_REF = '''
def remove_outliers(xs, k):
    if not xs:
        return []
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return [x for x in xs if abs(x - m) <= k * sd]
'''

_RO_AMBIG_CANDS = [
    # no guard: division by zero surfaces from the mean
    '''
def remove_outliers(xs, k):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return [x for x in xs if abs(x - m) <= k * sd]
''',
    # no guard, statistics module: a different exception type
    '''
import statistics

def remove_outliers(xs, k):
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs)
    return [x for x in xs if abs(x - m) <= k * sd]
''',
    # explicit refusal
    '''
def remove_outliers(xs, k):
    if not xs:
        raise ValueError("xs must not be empty")
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return [x for x in xs if abs(x - m) <= k * sd]
''',
    # guard returning empty
    '''
def remove_outliers(xs, k):
    if len(xs) == 0:
        return []
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    sd = var ** 0.5
    out = []
    for x in xs:
        if abs(x - mean) <= k * sd:
            out.append(x)
    return out
''',
    # no guard again (the common reading)
    '''
def remove_outliers(xs, k):
    n = len(xs)
    mean = sum(xs) / n
    sd = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
    return [x for x in xs if abs(x - mean) <= k * sd]
''',
    # guard returning empty, list() copy style
    '''
def remove_outliers(xs, k):
    data = list(xs)
    if not data:
        return []
    mean = sum(data) / len(data)
    sd = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return [x for x in data if abs(x - mean) <= k * sd]
''',
]

_RO_FULL_CANDS = [
    _RO_REF,
    '''
def remove_outliers(xs, k):
    if len(xs) == 0:
        return []
    mean = sum(xs) / len(xs)
    sd = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
    return [x for x in xs if abs(x - mean) <= k * sd]
''',
    '''
import statistics

def remove_outliers(xs, k):
    if not xs:
        return []
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs)
    return [x for x in xs if abs(x - m) <= k * sd]
''',
    '''
def remove_outliers(xs, k):
    if not xs:
        return []
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = var ** 0.5
    out = []
    for x in xs:
        if abs(x - mean) <= k * sd:
            out.append(x)
    return out
''',
    '''
def remove_outliers(xs, k):
    if not xs:
        return []
    data = list(xs)
    mean = sum(data) / len(data)
    sd = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return [x for x in data if abs(x - mean) <= k * sd]
''',
    # incidental sampling variation: strict inequality at the boundary. Not the
    # removed constraint, but real sampling noise the signal has to live with.
    '''
def remove_outliers(xs, k):
    if not xs:
        return []
    mean = sum(xs) / len(xs)
    sd = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
    return [x for x in xs if abs(x - mean) < k * sd or sd == 0]
''',
]

T01 = Task(
    id="t01_remove_outliers",
    entry_point="remove_outliers",
    prompt_ambiguous=_RO_AMBIG,
    prompt_full=_RO_FULL,
    reference=_RO_REF,
    seed_args=[[[1, 2, 3, 10], 2.0], [[5, 5, 5], 1.0]],
    constraints=[Constraint("empty_input", "If xs is empty, return an empty list.",
                            [[[], 2.0]])],
    candidates_ambiguous=_RO_AMBIG_CANDS,
    candidates_full=_RO_FULL_CANDS,
    precondition='''
def precondition(xs, k):
    return isinstance(xs, list) and isinstance(k, (int, float)) and k >= 0
''',
    difficulty="easy",
    notes="Classic forgotten edge case; samples split three ways on empty input.",
)


# --------------------------------------------------------------------------
# 02 top_k -- how ties are broken
# --------------------------------------------------------------------------

_TK_FULL = (
    "Write top_k(items, k) where items is a list of [name, score] pairs (each a "
    "2-element list). Return a list of the k pairs with the highest scores, "
    "ordered by score descending. When two items have the same score, the one "
    "that appeared earlier in items comes first. Assume k >= 0. If k is larger than "
    "len(items), return all of them."
)

_TK_AMBIG = (
    "Write top_k(items, k) where items is a list of [name, score] pairs (each a "
    "2-element list). Return a list of the k pairs with the highest scores, "
    "ordered by score descending. Assume k >= 0. If k is larger than len(items), "
    "return all of them."
)

_TK_REF = '''
def top_k(items, k):
    ranked = sorted(items, key=lambda p: -p[1])
    return ranked[:k]
'''

_TK_AMBIG_CANDS = [
    _TK_REF,
    # reverse=True is also stable on ties -> same behaviour as the reference
    '''
def top_k(items, k):
    return sorted(items, key=lambda p: p[1], reverse=True)[:k]
''',
    # alphabetical tie-break: a different reading
    '''
def top_k(items, k):
    return sorted(items, key=lambda p: (-p[1], p[0]))[:k]
''',
    # reverse the whole comparison, which flips ties too
    '''
def top_k(items, k):
    return sorted(items, key=lambda p: (p[1], p[0]), reverse=True)[:k]
''',
    # heapq.nlargest is stable in the same way as sorted
    '''
import heapq

def top_k(items, k):
    return heapq.nlargest(k, items, key=lambda p: p[1])
''',
    # alphabetical tie-break again (a common instinct)
    '''
def top_k(items, k):
    ranked = sorted(items, key=lambda p: p[0])
    ranked.sort(key=lambda p: p[1], reverse=True)
    return ranked[:k]
''',
]

_TK_FULL_CANDS = [
    _TK_REF,
    '''
def top_k(items, k):
    return sorted(items, key=lambda p: p[1], reverse=True)[:k]
''',
    '''
import heapq

def top_k(items, k):
    return heapq.nlargest(k, items, key=lambda p: p[1])
''',
    '''
def top_k(items, k):
    indexed = [(-p[1], i, p) for i, p in enumerate(items)]
    indexed.sort()
    return [p for _, _, p in indexed[:k]]
''',
    '''
def top_k(items, k):
    out = sorted(items, key=lambda p: -p[1])
    return out[:k] if k < len(out) else out
''',
    '''
def top_k(items, k):
    ranked = list(items)
    ranked.sort(key=lambda p: p[1], reverse=True)
    return ranked[:k]
''',
]

T02 = Task(
    id="t02_top_k",
    entry_point="top_k",
    prompt_ambiguous=_TK_AMBIG,
    prompt_full=_TK_FULL,
    reference=_TK_REF,
    seed_args=[[[["a", 3], ["b", 5], ["c", 3]], 2], [[["x", 1], ["y", 1]], 1]],
    constraints=[Constraint(
        "tie_order",
        "When two items have the same score, the one that appeared earlier in items comes first.",
        [[[["a", 3], ["b", 5], ["c", 3]], 2]])],
    candidates_ambiguous=_TK_AMBIG_CANDS,
    candidates_full=_TK_FULL_CANDS,
    precondition='''
def precondition(items, k):
    return (isinstance(items, list) and isinstance(k, int) and k >= 0
            and all(isinstance(p, list) and len(p) == 2 for p in items))
''',
    difficulty="easy",
    notes="Tie-breaking splits between 'stable' and 'alphabetical' readings.",
)


# --------------------------------------------------------------------------
# 03 chunk -- what happens to the trailing partial chunk
# --------------------------------------------------------------------------

_CH_FULL = (
    "Write chunk(xs, size) that splits xs into consecutive sublists of length "
    "size, in order. Assume size >= 1. If the last sublist would be shorter than "
    "size, include it anyway."
)

_CH_AMBIG = (
    "Write chunk(xs, size) that splits xs into consecutive sublists of length "
    "size, in order. Assume size >= 1."
)

_CH_REF = '''
def chunk(xs, size):
    return [xs[i:i + size] for i in range(0, len(xs), size)]
'''

_CH_AMBIG_CANDS = [
    _CH_REF,
    '''
def chunk(xs, size):
    out = []
    for i in range(0, len(xs), size):
        out.append(xs[i:i + size])
    return out
''',
    # drops the short tail
    '''
def chunk(xs, size):
    return [xs[i:i + size] for i in range(0, len(xs), size) if i + size <= len(xs)]
''',
    # drops the short tail via integer division
    '''
def chunk(xs, size):
    n = len(xs) // size
    return [xs[i * size:(i + 1) * size] for i in range(n)]
''',
    '''
def chunk(xs, size):
    result = []
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == size:
            result.append(buf)
            buf = []
    if buf:
        result.append(buf)
    return result
''',
    # buffers but discards the remainder
    '''
def chunk(xs, size):
    result = []
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == size:
            result.append(buf)
            buf = []
    return result
''',
]

_CH_FULL_CANDS = [
    _CH_REF,
    '''
def chunk(xs, size):
    out = []
    for i in range(0, len(xs), size):
        out.append(xs[i:i + size])
    return out
''',
    '''
def chunk(xs, size):
    result = []
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == size:
            result.append(buf)
            buf = []
    if buf:
        result.append(buf)
    return result
''',
    '''
def chunk(xs, size):
    return [list(xs[i:i + size]) for i in range(0, len(xs), size)]
''',
    '''
def chunk(xs, size):
    data = list(xs)
    return [data[i:i + size] for i in range(0, len(data), size)]
''',
    '''
import math

def chunk(xs, size):
    n = math.ceil(len(xs) / size) if xs else 0
    return [xs[i * size:(i + 1) * size] for i in range(n)]
''',
]

T03 = Task(
    id="t03_chunk",
    entry_point="chunk",
    prompt_ambiguous=_CH_AMBIG,
    prompt_full=_CH_FULL,
    reference=_CH_REF,
    seed_args=[[[1, 2, 3, 4, 5], 2], [[1, 2, 3, 4], 2]],
    constraints=[Constraint(
        "partial_tail",
        "If the last sublist would be shorter than size, include it anyway.",
        [[[1, 2, 3, 4, 5], 2]])],
    candidates_ambiguous=_CH_AMBIG_CANDS,
    candidates_full=_CH_FULL_CANDS,
    precondition='''
def precondition(xs, size):
    return isinstance(xs, list) and isinstance(size, int) and size >= 1
''',
    difficulty="easy",
    notes="Half the samples silently drop the remainder.",
)


# --------------------------------------------------------------------------
# 04 normalize -- degenerate range
# --------------------------------------------------------------------------

_NM_FULL = (
    "Write normalize(xs) that min-max scales a list of numbers into [0.0, 1.0] "
    "and returns a new list of floats, preserving order. Assume xs is non-empty. "
    "If every value in xs is the same, return a list of 0.0 of the same length."
)

_NM_AMBIG = (
    "Write normalize(xs) that min-max scales a list of numbers into [0.0, 1.0] "
    "and returns a new list of floats, preserving order. Assume xs is non-empty."
)

_NM_REF = '''
def normalize(xs):
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]
'''

_NM_AMBIG_CANDS = [
    _NM_REF,
    # unguarded: division by zero
    '''
def normalize(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) for x in xs]
''',
    # all ones
    '''
def normalize(xs):
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [1.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]
''',
    # midpoint
    '''
def normalize(xs):
    lo = min(xs)
    hi = max(xs)
    span = hi - lo
    if span == 0:
        return [0.5 for _ in xs]
    return [(x - lo) / span for x in xs]
''',
    # zeros, written differently
    '''
def normalize(xs):
    lo = min(xs)
    hi = max(xs)
    if hi - lo == 0:
        return [0.0 for _ in xs]
    return [float(x - lo) / float(hi - lo) for x in xs]
''',
    # unguarded again
    '''
def normalize(xs):
    mn = min(xs)
    rng = max(xs) - mn
    return [(v - mn) / rng for v in xs]
''',
]

_NM_FULL_CANDS = [
    _NM_REF,
    '''
def normalize(xs):
    lo = min(xs)
    hi = max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]
''',
    '''
def normalize(xs):
    lo, hi = min(xs), max(xs)
    span = hi - lo
    if span == 0:
        return [0.0] * len(xs)
    return [float(x - lo) / span for x in xs]
''',
    '''
def normalize(xs):
    lo, hi = min(xs), max(xs)
    if lo == hi:
        return [0.0] * len(xs)
    out = []
    for x in xs:
        out.append((x - lo) / (hi - lo))
    return out
''',
    '''
def normalize(xs):
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]
''',
    '''
def normalize(xs):
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0] * len(xs)
    return [round((x - lo) / (hi - lo), 6) for x in xs]
''',
]

T04 = Task(
    id="t04_normalize",
    entry_point="normalize",
    prompt_ambiguous=_NM_AMBIG,
    prompt_full=_NM_FULL,
    reference=_NM_REF,
    seed_args=[[[1, 2, 3]], [[5, 5, 5]]],
    constraints=[Constraint(
        "degenerate_range",
        "If every value in xs is the same, return a list of 0.0 of the same length.",
        [[[5, 5, 5]]])],
    candidates_ambiguous=_NM_AMBIG_CANDS,
    candidates_full=_NM_FULL_CANDS,
    precondition='''
def precondition(xs):
    return isinstance(xs, list) and len(xs) > 0 and all(isinstance(x, (int, float)) for x in xs)
''',
    difficulty="easy",
    notes="Four distinct readings of the zero-range case.",
)


TASKS = [T01, T02, T03, T04]
