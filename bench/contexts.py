"""Repository context for each diagnostic task, in two matched flavours.

The question this fixture exists to answer: when a requirement omits a
behavioural decision, can the surrounding codebase supply it?

Two arms, deliberately matched in length and style so the only difference is
information content:

  convention  sibling functions that resolve the analogous decision the way the
              reference does -- demonstrated in code, never stated in prose. The
              docstrings are written to describe *what* each helper returns and
              never *how* it treats the case the task omitted.

  distractor  sibling functions of the same size and register that say nothing
              about the omitted decision. This is the control that separates
              "the convention was readable" from "more context made the model
              more careful".

The convention arm is deliberately the easy case -- a sibling three lines above
that literally guards the empty list is about as legible as a convention can
get. That makes it an upper bound: if the model will not pick it up here, it
will not pick it up from a real repository.
"""
from __future__ import annotations

# A pool of module-mates that are plausible neighbours for any of the tasks and
# resolve none of the omitted decisions. Used to pad both arms to equal length.
_NEUTRAL = '''
def is_within(value, lo, hi):
    """Return True when value lies between lo and hi inclusive."""
    return lo <= value <= hi


def describe(value):
    """Return a short human label for a numeric value."""
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    return "positive"
'''

CONTEXTS: dict[str, dict[str, str]] = {}


def _add(task_id: str, convention: str, distractor: str) -> None:
    CONTEXTS[task_id] = {
        "convention": convention.strip() + "\n\n" + _NEUTRAL.strip() + "\n",
        "distractor": distractor.strip() + "\n\n" + _NEUTRAL.strip() + "\n",
    }


# --------------------------------------------------------------- t01 empty -> []
_add(
    "t01_remove_outliers",
    '''
def drop_negatives(xs):
    """Return the values of xs that are not negative."""
    if not xs:
        return []
    return [x for x in xs if x >= 0]


def clip_to_range(xs, lo, hi):
    """Return the values of xs that fall inside the given bounds."""
    if not xs:
        return []
    return [x for x in xs if lo <= x <= hi]
''',
    '''
def mean_absolute_error(a, b):
    """Return the mean absolute difference between two equal-length series."""
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def percent_change(old, new):
    """Return the change from old to new as a percentage of old."""
    return (new - old) / old * 100.0
''',
)

# ---------------------------------------------------- t02 ties keep input order
_add(
    "t02_top_k",
    '''
def order_by_priority(items):
    """Return items ordered from highest priority to lowest."""
    decorated = [(-p[1], i, p) for i, p in enumerate(items)]
    decorated.sort()
    return [p for _, _, p in decorated]


def order_by_weight(items):
    """Return items ordered from heaviest to lightest."""
    decorated = [(-p[1], i, p) for i, p in enumerate(items)]
    decorated.sort()
    return [p for _, _, p in decorated]
''',
    '''
def total_score(items):
    """Return the sum of the scores in a list of pairs."""
    return sum(p[1] for p in items)


def names_of(items):
    """Return just the names from a list of pairs."""
    return [p[0] for p in items]
''',
)

# ------------------------------------------------- t03 keep the short last tail
_add(
    "t03_chunk",
    '''
def pages(xs, per_page):
    """Return xs grouped into pages of at most per_page entries."""
    return [xs[i:i + per_page] for i in range(0, len(xs), per_page)]


def batches(xs, size):
    """Return xs grouped into batches for sequential processing."""
    out = []
    for i in range(0, len(xs), size):
        out.append(xs[i:i + size])
    return out
''',
    '''
def flatten(groups):
    """Return one list holding every element of every group."""
    return [x for g in groups for x in g]


def longest(groups):
    """Return the length of the largest group."""
    return max((len(g) for g in groups), default=0)
''',
)

# ------------------------------------------- t04 all values equal -> zeros
_add(
    "t04_normalize",
    '''
def scale_to_unit(xs):
    """Return xs rescaled into the unit interval."""
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def rescale_positive(xs):
    """Return xs rescaled so the largest value becomes one."""
    hi = max(xs)
    if hi == 0:
        return [0.0] * len(xs)
    return [x / hi for x in xs]
''',
    '''
def total(xs):
    """Return the sum of xs."""
    return sum(xs)


def spread(xs):
    """Return the difference between the largest and smallest value."""
    return max(xs) - min(xs)
''',
)

# ----------------------------------------------------- t05 the override wins
_add(
    "t05_merge_configs",
    '''
def merge_headers(base, extra):
    """Return one header mapping built from both inputs."""
    out = dict(base)
    out.update(extra)
    return out


def merge_defaults(defaults, supplied):
    """Return one settings mapping built from both inputs."""
    out = dict(defaults)
    out.update(supplied)
    return out
''',
    '''
def key_count(mapping):
    """Return how many keys a mapping holds."""
    return len(mapping)


def sorted_keys(mapping):
    """Return the keys of a mapping in alphabetical order."""
    return sorted(mapping)
''',
)

# ------------------------------------------------------- t06 end is exclusive
_add(
    "t06_parse_range",
    '''
def parse_slice(spec):
    """Return the indices denoted by a slice written as 'a:b'."""
    a, b = spec.split(":")
    return list(range(int(a), int(b)))


def parse_span(spec):
    """Return the indices denoted by a span written as 'a..b'."""
    a, b = spec.split("..")
    return list(range(int(a), int(b)))
''',
    '''
def is_numeric(spec):
    """Return True when every character of spec is a digit."""
    return spec.isdigit()


def normalise(spec):
    """Return spec with surrounding whitespace removed."""
    return spec.strip()
''',
)

# -------------------------------------------------- t07 keep the last occurrence
_add(
    "t07_dedupe",
    '''
def latest_events(xs):
    """Return the events of xs with earlier repeats removed."""
    out = []
    for i, x in enumerate(xs):
        if x not in xs[i + 1:]:
            out.append(x)
    return out


def latest_tags(xs):
    """Return the tags of xs with earlier repeats removed."""
    seen = set()
    rev = []
    for x in reversed(xs):
        if x not in seen:
            seen.add(x)
            rev.append(x)
    return list(reversed(rev))
''',
    '''
def count_unique(xs):
    """Return how many distinct values xs holds."""
    return len(set(xs))


def most_common(xs):
    """Return the value that appears most often in xs."""
    return max(set(xs), key=xs.count)
''',
)

# ------------------------------------------------------- t08 halfway rounds up
_add(
    "t08_round_price",
    '''
from decimal import Decimal, ROUND_HALF_UP


def round_tax(x):
    """Return x as a tax amount with two decimal places."""
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def round_fee(x):
    """Return x as a fee with two decimal places."""
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
''',
    '''
def to_cents(x):
    """Return an amount expressed in whole cents."""
    return int(x * 100)


def format_amount(x):
    """Return an amount rendered with a currency prefix."""
    return "$" + str(x)
''',
)

# ------------------------------------------- t09 a lone token is the last name
_add(
    "t09_split_name",
    '''
def split_label(s):
    """Return a [prefix, body] pair split on the first space."""
    s = s.strip()
    if " " not in s:
        return ["", s]
    a, b = s.split(" ", 1)
    return [a, b]


def split_title(s):
    """Return a [qualifier, subject] pair split on the first space."""
    s = s.strip()
    if " " not in s:
        return ["", s]
    a, b = s.split(" ", 1)
    return [a, b]
''',
    '''
def initials(s):
    """Return the first letter of each whitespace-separated word."""
    return "".join(w[0] for w in s.split())


def word_count(s):
    """Return how many whitespace-separated words s holds."""
    return len(s.split())
''',
)

# ------------------------------------------------ t10 window wider than data -> []
_add(
    "t10_moving_average",
    '''
def rolling_sum(xs, w):
    """Return the sum of every consecutive window of length w."""
    return [sum(xs[i:i + w]) for i in range(len(xs) - w + 1)]


def rolling_max(xs, w):
    """Return the largest value in every consecutive window of length w."""
    return [max(xs[i:i + w]) for i in range(len(xs) - w + 1)]
''',
    '''
def series_total(xs):
    """Return the sum of the whole series."""
    return sum(xs)


def series_range(xs):
    """Return the difference between the largest and smallest value."""
    return max(xs) - min(xs)
''',
)

# ------------------------------------------- t11 the result is exactly n chars
_add(
    "t11_truncate",
    '''
def shorten(s, n):
    """Return s shortened to fit a field of width n."""
    if len(s) <= n:
        return s
    return s[:n - 3] + "..."


def shorten_title(s, n):
    """Return a title shortened to fit a column of width n."""
    if len(s) <= n:
        return s
    return s[:n - 3] + "..."
''',
    '''
def pad_right(s, n):
    """Return s padded with spaces to a width of n."""
    return s + " " * max(0, n - len(s))


def strip_quotes(s):
    """Return s with any surrounding double quotes removed."""
    return s.strip('"')
''',
)

# ------------------------------------------------- t12 sort in place, return None
_add(
    "t12_sort_scores",
    '''
def sort_names(xs):
    """Order a list of names alphabetically."""
    xs.sort()
    return None


def sort_timestamps(xs):
    """Order a list of timestamps from oldest to newest."""
    xs.sort()
    return None
''',
    '''
def median(xs):
    """Return the middle value of a sorted copy of xs."""
    s = sorted(xs)
    return s[len(s) // 2]


def top_value(xs):
    """Return the largest value in xs."""
    return max(xs)
''',
)


def context_for(task_id: str, arm: str) -> str | None:
    """Return the module text for one task and arm, or None for the bare arm."""
    if arm == "bare":
        return None
    return CONTEXTS[task_id][arm]


ARMS = ("bare", "distractor", "convention")


# ---------------------------------------------------------------------------
# The decisive control: siblings that demonstrate the OPPOSITE of the reference.
#
# The convention arm shows that a legible sibling can move the model onto the
# right behaviour. That alone does not prove the repository beat the prior --
# it could be that the sibling merely reinforced something the model half
# believed. Inverting the sibling settles it. If a task the model got RIGHT on
# its own turns wrong when the module demonstrates the other convention, then
# the repository is genuinely overriding the prior, in both directions.
#
# Practically this is the warning half of the finding: a bad convention already
# in your codebase will be copied into new code, silently.
# ---------------------------------------------------------------------------

ANTI: dict[str, str] = {
    "t01_remove_outliers": '''
def drop_negatives(xs):
    """Return the values of xs that are not negative."""
    if not xs:
        raise ValueError("xs must not be empty")
    return [x for x in xs if x >= 0]


def clip_to_range(xs, lo, hi):
    """Return the values of xs that fall inside the given bounds."""
    if not xs:
        raise ValueError("xs must not be empty")
    return [x for x in xs if lo <= x <= hi]
''',
    "t02_top_k": '''
def order_by_priority(items):
    """Return items ordered from highest priority to lowest."""
    return sorted(items, key=lambda p: (-p[1], p[0]))


def order_by_weight(items):
    """Return items ordered from heaviest to lightest."""
    return sorted(items, key=lambda p: (-p[1], p[0]))
''',
    "t03_chunk": '''
def pages(xs, per_page):
    """Return xs grouped into pages of exactly per_page entries."""
    n = len(xs) // per_page
    return [xs[i * per_page:(i + 1) * per_page] for i in range(n)]


def batches(xs, size):
    """Return xs grouped into batches for sequential processing."""
    n = len(xs) // size
    return [xs[i * size:(i + 1) * size] for i in range(n)]
''',
    "t04_normalize": '''
def scale_to_unit(xs):
    """Return xs rescaled into the unit interval."""
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [1.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def rescale_positive(xs):
    """Return xs rescaled so the largest value becomes one."""
    hi = max(xs)
    if hi == 0:
        return [1.0] * len(xs)
    return [x / hi for x in xs]
''',
    "t05_merge_configs": '''
def merge_headers(base, extra):
    """Return one header mapping built from both inputs."""
    out = dict(extra)
    out.update(base)
    return out


def merge_defaults(defaults, supplied):
    """Return one settings mapping built from both inputs."""
    out = dict(supplied)
    out.update(defaults)
    return out
''',
    "t06_parse_range": '''
def parse_slice(spec):
    """Return the indices denoted by a slice written as 'a:b'."""
    a, b = spec.split(":")
    return list(range(int(a), int(b) + 1))


def parse_span(spec):
    """Return the indices denoted by a span written as 'a..b'."""
    a, b = spec.split("..")
    return list(range(int(a), int(b) + 1))
''',
    "t07_dedupe": '''
def latest_events(xs):
    """Return the events of xs with repeats removed."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def latest_tags(xs):
    """Return the tags of xs with repeats removed."""
    return list(dict.fromkeys(xs))
''',
    "t08_round_price": '''
def round_tax(x):
    """Return x as a tax amount with two decimal places."""
    return round(x, 2)


def round_fee(x):
    """Return x as a fee with two decimal places."""
    return round(x, 2)
''',
    "t09_split_name": '''
def split_label(s):
    """Return a [prefix, body] pair split on the first space."""
    s = s.strip()
    if " " not in s:
        return [s, ""]
    a, b = s.split(" ", 1)
    return [a, b]


def split_title(s):
    """Return a [qualifier, subject] pair split on the first space."""
    s = s.strip()
    if " " not in s:
        return [s, ""]
    a, b = s.split(" ", 1)
    return [a, b]
''',
    "t10_moving_average": '''
def rolling_sum(xs, w):
    """Return the sum of every consecutive window of length w."""
    if w > len(xs):
        return [sum(xs)]
    return [sum(xs[i:i + w]) for i in range(len(xs) - w + 1)]


def rolling_max(xs, w):
    """Return the largest value in every consecutive window of length w."""
    if w > len(xs):
        return [max(xs)]
    return [max(xs[i:i + w]) for i in range(len(xs) - w + 1)]
''',
    "t11_truncate": '''
def shorten(s, n):
    """Return s shortened to fit a field of width n."""
    if len(s) <= n:
        return s
    return s[:n] + "..."


def shorten_title(s, n):
    """Return a title shortened to fit a column of width n."""
    if len(s) <= n:
        return s
    return s[:n] + "..."
''',
    "t12_sort_scores": '''
def sort_names(xs):
    """Order a list of names alphabetically."""
    return sorted(xs)


def sort_timestamps(xs):
    """Order a list of timestamps from oldest to newest."""
    return sorted(xs)
''',
}

for _tid, _src in ANTI.items():
    CONTEXTS[_tid]["anti"] = _src.strip() + "\n\n" + _NEUTRAL.strip() + "\n"

ARMS = ("bare", "distractor", "convention", "anti")


# ---------------------------------------------------------------------------
# Dilution arm: the same convention, pushed away from the point of use.
#
# The obvious objection to the convention result is that a near-identical helper
# sitting three lines above is really few-shot prompting, not a repository
# convention. This arm keeps the convention-bearing siblings byte-for-byte and
# only changes where they sit: six unrelated helpers are inserted after them, so
# the convention is the *oldest* thing in the module rather than the nearest.
# If the effect survives distance it is reading a convention; if it evaporates
# it was proximity all along.
# ---------------------------------------------------------------------------

_FILLER = '''
def clamp(value, lo, hi):
    """Return value confined to the range between lo and hi."""
    return max(lo, min(hi, value))


def as_bool(text):
    """Return the boolean a configuration string denotes."""
    return text.strip().lower() in {"1", "true", "yes", "on"}


def slugify(text):
    """Return text lowercased with spaces replaced by hyphens."""
    return "-".join(text.lower().split())


def repeat(text, times):
    """Return text concatenated with itself the given number of times."""
    return text * times


def first_line(text):
    """Return everything before the first newline."""
    return text.split("\n", 1)[0]


def initial_of(word):
    """Return the first character of a word, uppercased."""
    return word[:1].upper()
'''

for _tid, _bundle in CONTEXTS.items():
    _conv = _bundle["convention"].split(_NEUTRAL.strip())[0].rstrip()
    _bundle["diluted"] = (_conv + "\n\n" + _FILLER.strip() + "\n\n"
                          + _NEUTRAL.strip() + "\n")

# ---------------------------------------------------------------------------
# Prose arm: the same convention, stated in words instead of shown in code.
#
# arXiv 2607.27250 ran a controlled ablation of natural-language context files
# (AGENTS.md / CLAUDE.md) on real repositories and found a null: correctness did
# not move, and the authors ruled out low file quality as the explanation. The
# convention arm here is the same information carried by a different channel --
# demonstrated in a sibling function rather than asserted in a guideline -- and
# it moves correctness by +3.1/12. Either the channel matters or my fixtures are
# an artefact, and those two possibilities are distinguishable in one arm.
#
# Matching the indirectness is the whole design problem. The convention siblings
# resolve the *analogous* decision on a *different* function and never mention
# the task at hand; if the prose named the task it would simply be the deleted
# sentence handed back, and the comparison would be rigged. So each rule below
# is stated as a general property of the module, entailing the answer without
# ever addressing the function being written.
#
# The arm is otherwise built on `distractor`, not on `convention`: the siblings
# carry nothing, so the prose is the only channel with the information in it.
# That leaves this arm slightly *longer* than the ones it is compared against,
# which biases in favour of prose -- the conservative direction for a null.
# ---------------------------------------------------------------------------

_PROSE = {
    "t01_remove_outliers":
        "Helpers that filter a sequence return an empty sequence when given an "
        "empty one. They never raise on empty input.",
    "t02_top_k":
        "Helpers that order or rank entries are stable: when two entries compare "
        "equal, the one that appeared earlier in the input stays earlier in the "
        "output.",
    "t03_chunk":
        "Helpers that group a sequence keep a final short group rather than "
        "dropping it or padding it out to full size.",
    "t04_normalize":
        "Helpers that rescale a series return all zeros when the series has no "
        "spread, rather than raising or producing NaN.",
    "t05_merge_configs":
        "Helpers that combine two mappings let the second argument win wherever "
        "the two disagree on a key.",
    "t06_parse_range":
        "Helpers that parse or produce a span treat its end as exclusive, "
        "following Python's own slicing convention.",
    "t07_dedupe":
        "Helpers that remove duplicates keep the last occurrence of each item, "
        "so the surviving order follows where each item last appeared.",
    "t08_round_price":
        "Helpers that round monetary amounts round a value sitting exactly "
        "halfway away from zero, not to the nearest even digit.",
    "t09_split_name":
        "Helpers that split a person's name into parts treat a single bare "
        "token as the trailing part, leaving the leading part empty.",
    "t10_moving_average":
        "Helpers that slide a window over a series return an empty list when "
        "the window is wider than the data.",
    "t11_truncate":
        "Helpers that shorten text produce a result of exactly the requested "
        "length; any ellipsis they add counts against that budget.",
    "t12_sort_scores":
        "Helpers that sort reorder the caller's own list in place and return "
        "None, rather than building a new list.",
}

# Framed the way a real project guideline is framed, with the planted rule in
# first position -- the most prominent slot, so a null here is not a null about
# burying it.
_PROSE_HEADER = (
    '"""Shared helpers for this package.\n'
    '\n'
    'Conventions that hold for every helper in this module:\n'
    '\n'
    '  * {rule}\n'
    '  * Helpers are pure unless their name says otherwise.\n'
    '  * Docstrings say what a helper returns, not how it is implemented.\n'
    '"""'
)

for _tid, _bundle in CONTEXTS.items():
    _bundle["prose"] = (_PROSE_HEADER.format(rule=_PROSE[_tid]) + "\n\n"
                        + _bundle["distractor"])

# ---------------------------------------------------------------------------
# Conflict arm: the module contradicts itself.
#
# Every arm so far presents one convention, consistently. `anti` showed that a
# wrong one is followed silently, and the T2 experiment showed the detector is
# *correct* to call it settled -- the code really does settle it. That made the
# blind spot a proposition rather than a finding: the information needed to
# catch it is provably absent from the page.
#
# This arm is the exit from that argument. Two siblings resolve the SAME
# analogous decision in OPPOSITE ways, so the module no longer determines the
# answer -- and, unlike `anti`, the fact that it does not is fully visible. A
# purely structural signal is available. The question is whether it is used.
#
# Built by byte-for-byte reuse: one sibling from `convention`, one from `anti`.
# Nothing new is written, so the arm inherits no fresh authoring bias and lines
# up exactly against both parents.
#
# Two orderings, because otherwise "which side wins" is confounded with "which
# side came first":
#   conflict    the reference-correct variant appears FIRST
#   conflict_r  the wrong variant appears first
#
# Ground truth for T2 is OPEN: a module that contradicts itself has not settled
# the question.
# ---------------------------------------------------------------------------


def _split_siblings(block: str) -> tuple[str, list[str]]:
    """Return (leading imports, [one text per top-level def]) for a block."""
    lines = block.strip().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("def ")]
    if not starts:
        return "", []
    header = "\n".join(lines[: starts[0]]).strip()
    funcs = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        funcs.append("\n".join(lines[s:e]).rstrip())
    return header, funcs


def _assemble(header: str, first: str, second: str) -> str:
    parts = ([header] if header else []) + [first, second, _NEUTRAL.strip()]
    return "\n\n\n".join(parts) + "\n"


for _tid, _bundle in CONTEXTS.items():
    _h, _cf = _split_siblings(_bundle["convention"].split(_NEUTRAL.strip())[0])
    _, _af = _split_siblings(_bundle["anti"].split(_NEUTRAL.strip())[0])
    if len(_cf) < 2 or len(_af) < 2:
        raise RuntimeError(f"{_tid}: expected two siblings in each parent arm")
    # The import line lives in the convention block and is needed by whichever
    # variant uses Decimal, so it rides along in both orderings identically.
    _bundle["conflict"] = _assemble(_h, _cf[0], _af[1])
    _bundle["conflict_r"] = _assemble(_h, _af[0], _cf[1])

ARMS = ("bare", "distractor", "prose", "convention", "diluted", "anti",
        "conflict", "conflict_r")
