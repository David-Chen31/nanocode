"""Diagnostic tasks 09-12. Includes one `hard` task."""
from __future__ import annotations

from bench.schema import Constraint, Task

# --------------------------------------------------------------------------
# 09 split_name -- what a single-token name means
# --------------------------------------------------------------------------

_SN_FULL = (
    "Write split_name(full) that takes a name string and returns a [first, last] "
    "pair of strings, splitting on the first space. Leading and trailing spaces "
    "are stripped first. If the name has no space in it, the whole name is the "
    "last name and the first name is the empty string."
)
_SN_AMBIG = (
    "Write split_name(full) that takes a name string and returns a [first, last] "
    "pair of strings, splitting on the first space. Leading and trailing spaces "
    "are stripped first."
)
_SN_REF = '''
def split_name(full):
    s = full.strip()
    if " " not in s:
        return ["", s]
    first, last = s.split(" ", 1)
    return [first, last]
'''

T09 = Task(
    id="t09_split_name",
    entry_point="split_name",
    prompt_ambiguous=_SN_AMBIG,
    prompt_full=_SN_FULL,
    reference=_SN_REF,
    seed_args=[["Ada Lovelace"], ["Prince"]],
    constraints=[Constraint(
        "single_token",
        "If the name has no space, the whole name is the last name.",
        [["Prince"]])],
    candidates_ambiguous=[
        # whole thing becomes the first name
        '''
def split_name(full):
    s = full.strip()
    parts = s.split(" ", 1)
    if len(parts) == 1:
        return [parts[0], ""]
    return [parts[0], parts[1]]
''',
        '''
def split_name(full):
    s = full.strip()
    if " " in s:
        a, b = s.split(" ", 1)
        return [a, b]
    return [s, ""]
''',
        # whole thing becomes the last name
        _SN_REF,
        # refuses
        '''
def split_name(full):
    s = full.strip()
    if " " not in s:
        raise ValueError("name must contain a space")
    a, b = s.split(" ", 1)
    return [a, b]
''',
        '''
def split_name(full):
    parts = full.strip().split(" ", 1)
    return [parts[0], parts[1] if len(parts) > 1 else ""]
''',
        '''
def split_name(full):
    s = full.strip()
    idx = s.find(" ")
    if idx < 0:
        return ["", s]
    return [s[:idx], s[idx + 1:]]
''',
    ],
    candidates_full=[
        _SN_REF,
        '''
def split_name(full):
    s = full.strip()
    idx = s.find(" ")
    if idx == -1:
        return ["", s]
    return [s[:idx], s[idx + 1:]]
''',
        '''
def split_name(full):
    s = full.strip()
    parts = s.split(" ", 1)
    if len(parts) == 1:
        return ["", parts[0]]
    return parts
''',
        '''
def split_name(full):
    s = full.strip()
    if " " not in s:
        return ["", s]
    return list(s.split(" ", 1))
''',
        '''
def split_name(full):
    s = full.strip()
    bits = s.split(" ", 1)
    return ["", bits[0]] if len(bits) < 2 else [bits[0], bits[1]]
''',
        '''
def split_name(full):
    s = full.strip()
    if not s or " " not in s:
        return ["", s]
    a, b = s.split(" ", 1)
    return [a, b]
''',
    ],
    precondition='''
def precondition(full):
    return isinstance(full, str)
''',
    difficulty="easy",
    notes="Three readings for a name with no space.",
)


# --------------------------------------------------------------------------
# 10 moving_average -- a window wider than the data
# --------------------------------------------------------------------------

_MA_FULL = (
    "Write moving_average(xs, w) returning the list of averages of every "
    "consecutive window of length w, in order. Assume w >= 1 and xs is a list of "
    "numbers. If w is larger than len(xs), return an empty list."
)
_MA_AMBIG = (
    "Write moving_average(xs, w) returning the list of averages of every "
    "consecutive window of length w, in order. Assume w >= 1 and xs is a list of "
    "numbers."
)
_MA_REF = '''
def moving_average(xs, w):
    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]
'''

T10 = Task(
    id="t10_moving_average",
    entry_point="moving_average",
    prompt_ambiguous=_MA_AMBIG,
    prompt_full=_MA_FULL,
    reference=_MA_REF,
    seed_args=[[[1, 2, 3, 4], 2], [[1, 2], 3]],
    constraints=[Constraint(
        "window_too_wide",
        "If w is larger than len(xs), return an empty list.",
        [[[1, 2], 3]])],
    candidates_ambiguous=[
        _MA_REF,
        '''
def moving_average(xs, w):
    out = []
    for i in range(len(xs) - w + 1):
        out.append(sum(xs[i:i + w]) / w)
    return out
''',
        # averages the short window instead
        '''
def moving_average(xs, w):
    if w > len(xs):
        return [sum(xs) / len(xs)]
    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]
''',
        # refuses
        '''
def moving_average(xs, w):
    if w > len(xs):
        raise ValueError("window larger than input")
    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]
''',
        '''
def moving_average(xs, w):
    n = len(xs)
    return [sum(xs[i:i + w]) / w for i in range(max(0, n - w + 1))]
''',
        # partial windows at the tail
        '''
def moving_average(xs, w):
    out = []
    for i in range(len(xs)):
        window = xs[i:i + w]
        out.append(sum(window) / len(window))
    return out
''',
    ],
    candidates_full=[
        _MA_REF,
        '''
def moving_average(xs, w):
    if w > len(xs):
        return []
    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]
''',
        '''
def moving_average(xs, w):
    n = len(xs)
    return [sum(xs[i:i + w]) / w for i in range(max(0, n - w + 1))]
''',
        '''
def moving_average(xs, w):
    out = []
    for i in range(len(xs) - w + 1):
        out.append(sum(xs[i:i + w]) / w)
    return out
''',
        '''
def moving_average(xs, w):
    if len(xs) < w:
        return []
    result = []
    for i in range(0, len(xs) - w + 1):
        chunk = xs[i:i + w]
        result.append(sum(chunk) / len(chunk))
    return result
''',
        '''
def moving_average(xs, w):
    return [sum(xs[i:i + w]) / w for i in range(len(xs)) if i + w <= len(xs)]
''',
    ],
    precondition='''
def precondition(xs, w):
    return (isinstance(xs, list) and isinstance(w, int) and not isinstance(w, bool)
            and w >= 1 and all(isinstance(x, (int, float)) for x in xs))
''',
    difficulty="easy",
    notes="Four readings when the window does not fit.",
)


# --------------------------------------------------------------------------
# 11 truncate -- HARD: does the ellipsis count toward the limit
# --------------------------------------------------------------------------

_TR_FULL = (
    "Write truncate(s, n) that returns s unchanged when len(s) <= n. Otherwise "
    "return a shortened string of length exactly n that ends in '...'. Assume "
    "n >= 3."
)
_TR_AMBIG = (
    "Write truncate(s, n) that returns s unchanged when len(s) <= n. Otherwise "
    "return a shortened version of s that ends in '...'. Assume n >= 3."
)
_TR_REF = '''
def truncate(s, n):
    if len(s) <= n:
        return s
    return s[:n - 3] + "..."
'''

T11 = Task(
    id="t11_truncate",
    entry_point="truncate",
    prompt_ambiguous=_TR_AMBIG,
    prompt_full=_TR_FULL,
    reference=_TR_REF,
    seed_args=[["abcdefgh", 5], ["abc", 5]],
    constraints=[Constraint(
        "ellipsis_in_budget",
        "The result must be exactly n characters, so the ellipsis eats into the budget.",
        [["abcdefgh", 5]])],
    candidates_ambiguous=[
        # the overwhelming prior: n characters plus the ellipsis
        '''
def truncate(s, n):
    if len(s) <= n:
        return s
    return s[:n] + "..."
''',
        '''
def truncate(s, n):
    return s if len(s) <= n else s[:n] + "..."
''',
        '''
def truncate(s, n):
    if len(s) <= n:
        return s
    out = s[:n]
    return out + "..."
''',
        '''
def truncate(s, n):
    if len(s) > n:
        return "".join(s[:n]) + "..."
    return s
''',
        '''
def truncate(s, n):
    return s[:n] + "..." if len(s) > n else s
''',
        # the one sample that budgets for the ellipsis
        _TR_REF,
    ],
    candidates_full=[
        _TR_REF,
        '''
def truncate(s, n):
    return s if len(s) <= n else s[:n - 3] + "..."
''',
        '''
def truncate(s, n):
    if len(s) <= n:
        return s
    keep = n - 3
    return s[:keep] + "..."
''',
        '''
def truncate(s, n):
    if len(s) <= n:
        return s
    out = s[:max(0, n - 3)]
    return out + "..."
''',
        '''
def truncate(s, n):
    if len(s) <= n:
        return s
    result = s[:n - 3]
    result += "..."
    return result
''',
        '''
def truncate(s, n):
    return s if len(s) <= n else "".join(list(s)[:n - 3]) + "..."
''',
    ],
    precondition='''
def precondition(s, n):
    return (isinstance(s, str) and isinstance(n, int) and not isinstance(n, bool)
            and n >= 3)
''',
    difficulty="hard",
    notes="'first n chars plus ellipsis' is the near-universal reading. BSE is "
          "low and the majority is wrong.",
)


# --------------------------------------------------------------------------
# 12 sort_scores -- in place or a copy
# --------------------------------------------------------------------------

_SS_FULL = (
    "Write sort_scores(xs) that orders a list of numbers from largest to "
    "smallest. Sort xs in place and return None."
)
_SS_AMBIG = (
    "Write sort_scores(xs) that orders a list of numbers from largest to smallest."
)
_SS_REF = '''
def sort_scores(xs):
    xs.sort(reverse=True)
    return None
'''

T12 = Task(
    id="t12_sort_scores",
    entry_point="sort_scores",
    prompt_ambiguous=_SS_AMBIG,
    prompt_full=_SS_FULL,
    reference=_SS_REF,
    seed_args=[[[3, 1, 2]], [[1]]],
    constraints=[Constraint(
        "in_place_or_copy",
        "Sort xs in place and return None.",
        [[[3, 1, 2]]])],
    candidates_ambiguous=[
        # returns a new list, leaves the input alone
        '''
def sort_scores(xs):
    return sorted(xs, reverse=True)
''',
        '''
def sort_scores(xs):
    return sorted(xs)[::-1]
''',
        # in place, returns None
        _SS_REF,
        # in place, but also returns the list
        '''
def sort_scores(xs):
    xs.sort(reverse=True)
    return xs
''',
        '''
def sort_scores(xs):
    out = list(xs)
    out.sort(reverse=True)
    return out
''',
        '''
def sort_scores(xs):
    xs.sort(key=lambda v: -v)
    return None
''',
    ],
    candidates_full=[
        _SS_REF,
        '''
def sort_scores(xs):
    xs.sort(key=lambda v: -v)
    return None
''',
        '''
def sort_scores(xs):
    xs.sort(reverse=True)
''',
        '''
def sort_scores(xs):
    xs[:] = sorted(xs, reverse=True)
    return None
''',
        '''
def sort_scores(xs):
    xs.sort()
    xs.reverse()
    return None
''',
        '''
def sort_scores(xs):
    n = len(xs)
    ordered = sorted(xs, reverse=True)
    for i in range(n):
        xs[i] = ordered[i]
    return None
''',
    ],
    precondition='''
def precondition(xs):
    return isinstance(xs, list) and all(isinstance(x, (int, float)) for x in xs)
''',
    difficulty="easy",
    notes="Mutation is only visible because the runner records argument state "
          "after the call, not just the return value.",
)


TASKS = [T09, T10, T11, T12]
