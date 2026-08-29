"""Turn one diagnostic task into a small repository the agent has to search.

Seven rounds of this project measured what a model writes when context is handed
to it. An agent is not handed context; it goes looking. That changes the
question, because looking can fail, and a failed search is information the
single-turn setting cannot produce.

So the arms move out of the prompt and into a file tree. The decisive design
rule is that the answer never sits in the file the agent was told to edit --
otherwise this is the round-4 setup again with extra steps. It lives one import
away, in a module the agent must decide to open.

    findable    pkg/helpers.py demonstrates the convention the reference follows
    unfindable  pkg/helpers.py holds unrelated helpers; nothing anywhere answers
    decoy       pkg/helpers.py demonstrates the OPPOSITE convention

The three are byte-identical apart from the body of that one module, and each
reuses the sibling functions already written for the corresponding single-turn
arm (`convention`, `distractor`, `anti`), so the trajectory study inherits the
controls rather than re-deriving them.

Filler modules exist so that listing the tree is not the same as reading it: an
agent that opens everything is behaving differently from one that opens the
right thing, and the trace has to be able to tell those apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench.contexts import CONTEXTS
from bench.schema import Task

# The arm in the file tree, and the single-turn arm whose siblings it reuses.
ARM_SOURCE = {"findable": "convention", "unfindable": "distractor", "decoy": "anti"}
ARMS = tuple(ARM_SOURCE)

_README = '''# {pkg}

Small internal utility package.

Layout:

- `{pkg}/core.py` — the operations callers use directly
- `{pkg}/helpers.py` — shared building blocks the core operations are built on
- `{pkg}/text.py`, `{pkg}/seq.py` — odds and ends
- `tests/` — unit tests

House style: helpers stay small and are covered by a test each.
'''

_CORE_HEAD = '''"""Operations this package exposes to callers."""


def describe_result(value):
    """Return a short human label for a result value."""
    if value is None:
        return "no result"
    return "ok: " + repr(value)


def is_ready(state):
    """Return True when a state mapping is complete enough to act on."""
    return bool(state) and state.get("status") == "ready"
'''

_TEXT = '''"""String odds and ends."""


def titlecase_words(s):
    """Return s with each whitespace-separated word capitalised."""
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def strip_prefix(s, prefix):
    """Return s without a leading prefix, if it has one."""
    return s[len(prefix):] if s.startswith(prefix) else s


def collapse_spaces(s):
    """Return s with runs of whitespace reduced to single spaces."""
    return " ".join(s.split())
'''

_SEQ = '''"""Sequence odds and ends."""


def pairwise(xs):
    """Return consecutive pairs of xs."""
    return list(zip(xs, xs[1:]))


def interleave(a, b):
    """Return the elements of a and b, alternating."""
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return out


def index_of(xs, value):
    """Return the position of value in xs, or -1 when it is absent."""
    try:
        return xs.index(value)
    except ValueError:
        return -1
'''

_TEST_HELPERS = '''"""Unit tests for the shared helpers."""
from {pkg} import helpers


def test_helpers_import():
    assert helpers is not None
'''

_TEST_CORE = '''"""Unit tests for the core operations."""
from {pkg} import core


def test_describe_result():
    assert core.describe_result(None) == "no result"


def test_is_ready():
    assert core.is_ready({{"status": "ready"}})
    assert not core.is_ready({{}})
'''


@dataclass
class Repo:
    """Where everything ended up, so the trace can be scored against it."""

    root: Path
    pkg: str
    target: str          # the file the agent was told to edit
    answer_file: str | None   # the file that resolves the omission, if any
    files: list[str]


def build_repo(root: str | Path, task: Task, arm: str, *, pkg: str = "toolkit") -> Repo:
    """Materialise the repository for one task under one arm."""
    if arm not in ARM_SOURCE:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    root = Path(root)
    siblings = CONTEXTS[task.id][ARM_SOURCE[arm]]

    files = {
        "README.md": _README.format(pkg=pkg),
        f"{pkg}/__init__.py": f'"""{pkg} package."""\n',
        f"{pkg}/core.py": _CORE_HEAD,
        f"{pkg}/helpers.py": '"""Shared building blocks."""\n\n\n' + siblings,
        f"{pkg}/text.py": _TEXT,
        f"{pkg}/seq.py": _SEQ,
        "tests/test_helpers.py": _TEST_HELPERS.format(pkg=pkg),
        "tests/test_core.py": _TEST_CORE.format(pkg=pkg),
    }
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    return Repo(
        root=root, pkg=pkg, target=f"{pkg}/core.py",
        # `unfindable` genuinely has no answer anywhere -- that is the whole
        # point of the arm, and the scorer must not pretend otherwise.
        answer_file=(f"{pkg}/helpers.py" if arm in ("findable", "decoy") else None),
        files=sorted(files),
    )


TASK_PROMPT = """Add one function to `{target}` in this package.

{requirement}

Keep the existing contents of the file. When you are done, call `finish`."""


def agent_prompt(task: Task, repo: Repo) -> str:
    """What the agent is told. It is never told that helpers.py is relevant."""
    return TASK_PROMPT.format(target=repo.target,
                              requirement=task.prompt_ambiguous.strip())


IMPORT_SHIM = """import sys
sys.path.insert(0, {root!r})
from {pkg}.core import {entry}
"""


def scoring_source(repo: Repo, task: Task) -> str:
    """A stand-in 'candidate source' that imports what the agent actually wrote.

    Lets the existing differential-execution machinery score a function living
    in a real package, imports and all, without loosening what it checks.
    """
    return IMPORT_SHIM.format(root=str(repo.root), pkg=repo.pkg,
                              entry=task.entry_point)
