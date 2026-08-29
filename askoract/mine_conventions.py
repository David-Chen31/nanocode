"""How often does a real module already decide the thing a requirement omitted?

Every number in this project so far comes from twelve tasks I wrote, with
conventions I wrote, against reference implementations I wrote. The base rate
that would tell us whether any of it matters in practice -- how often real
sibling code actually resolves an omitted behavioural decision, and how often
the siblings disagree with each other -- cannot come from that fixture. It can
come from reading real code, statically, for nothing.

Scope is deliberately narrow and honest: this only looks at functions that
*explicitly take a position*. A function that would crash on empty input but
never says so is not counted as deciding anything. So the question answered is

    given a module containing two or more functions that explicitly decide
    dimension D, do they decide it the same way?

which is exactly the precondition my `convention` and `conflict` arms assume.

Three dimensions, chosen because they are the ones my tasks turn on and because
each is detectable from the AST without guessing:

    D1  empty sequence input   -> EMPTY / RAISE / NONE / OTHER
    D3  mutate or copy         -> IN_PLACE / COPY
    D5  merge precedence       -> SECOND_WINS / FIRST_WINS

    py -3 askoract/mine_conventions.py --limit 4000
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sysconfig
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


# --------------------------------------------------------------------------- D1
def _is_empty_test(node: ast.expr, params: set[str]) -> str | None:
    """Return the parameter name this expression tests for emptiness, if any."""
    # if not xs
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        if isinstance(node.operand, ast.Name) and node.operand.id in params:
            return node.operand.id
        # if not len(xs)
        if (isinstance(node.operand, ast.Call)
                and isinstance(node.operand.func, ast.Name)
                and node.operand.func.id == "len"
                and node.operand.args
                and isinstance(node.operand.args[0], ast.Name)
                and node.operand.args[0].id in params):
            return node.operand.args[0].id
    # if len(xs) == 0  /  if xs == []  /  if len(xs) < 1
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, op, right = node.left, node.ops[0], node.comparators[0]
        zero = (isinstance(right, ast.Constant) and right.value == 0)
        emptylit = isinstance(right, (ast.List, ast.Tuple, ast.Dict)) and not getattr(right, "elts", getattr(right, "keys", [1]))
        if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "len" and left.args
                and isinstance(left.args[0], ast.Name) and left.args[0].id in params
                and isinstance(op, (ast.Eq, ast.Lt, ast.LtE)) and zero):
            return left.args[0].id
        if (isinstance(left, ast.Name) and left.id in params
                and isinstance(op, ast.Eq) and emptylit):
            return left.id
    return None


_EMPTY_LITERALS = (ast.List, ast.Tuple, ast.Dict, ast.Set)


def _classify_guard(body: list[ast.stmt], param: str) -> str | None:
    """What does the guarded branch do?"""
    if not body:
        return None
    st = body[0]
    if isinstance(st, ast.Raise):
        return "RAISE"
    if isinstance(st, ast.Return):
        v = st.value
        if v is None:
            return "NONE"
        if isinstance(v, ast.Constant) and v.value is None:
            return "NONE"
        # return []  /  return ()  /  return {}  /  return ""
        if isinstance(v, _EMPTY_LITERALS):
            elts = getattr(v, "elts", None)
            keys = getattr(v, "keys", None)
            if not elts and not keys:
                return "EMPTY"
        if isinstance(v, ast.Constant) and v.value in ("", b""):
            return "EMPTY"
        # return xs -- handing back the empty container it was given
        if isinstance(v, ast.Name) and v.id == param:
            return "EMPTY"
        # return list(xs) / list() / dict() ...
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id in {"list", "tuple", "dict", "set", "str"}):
            if not v.args or (isinstance(v.args[0], ast.Name) and v.args[0].id == param):
                return "EMPTY"
        return "OTHER"
    return None


def d1(fn: ast.FunctionDef, params: set[str]) -> str | None:
    # Only an *early* guard counts: a decision the function announces up front.
    for st in fn.body[:3]:
        if isinstance(st, ast.If):
            p = _is_empty_test(st.test, params)
            if p:
                verdict = _classify_guard(st.body, p)
                if verdict:
                    return verdict + "|" + p
    return None


# --------------------------------------------------------------------------- D3
def d3(fn: ast.FunctionDef, params: set[str]) -> str | None:
    """Does the function mutate a parameter in place, or build a new object?"""
    mutates = False
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"sort", "reverse"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in params):
            mutates = True
    if mutates:
        # In place only counts if it does not then hand back a new object.
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None:
                if not (isinstance(node.value, ast.Constant)
                        and node.value.value is None):
                    return None  # mixed; do not force a label
        return "IN_PLACE"
    for node in ast.walk(fn):
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"sorted", "reversed"}
                and node.value.args
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id in params):
            return "COPY"
    return None


# --------------------------------------------------------------------------- D5
def d5(fn: ast.FunctionDef, params: set[str]) -> str | None:
    """On a two-mapping merge, which argument wins a key collision?"""
    seeded: dict[str, str] = {}          # local name -> which param seeded it
    for st in ast.walk(fn):
        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Call)
                and isinstance(st.value.func, ast.Name)
                and st.value.func.id == "dict"
                and st.value.args
                and isinstance(st.value.args[0], ast.Name)
                and st.value.args[0].id in params):
            seeded[st.targets[0].id] = st.value.args[0].id
        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Dict) and st.value.keys == [None]
                and isinstance(st.value.values[0], ast.Name)
                and st.value.values[0].id in params):
            seeded[st.targets[0].id] = st.value.values[0].id
    if not seeded:
        return None
    order = list(params)
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in seeded
                and node.args and isinstance(node.args[0], ast.Name)
                and node.args[0].id in params):
            base, over = seeded[node.func.value.id], node.args[0].id
            if base == over or base not in order or over not in order:
                return None
            return "SECOND_WINS" if order.index(over) > order.index(base) else "FIRST_WINS"
    return None


DIMENSIONS = {"D1_empty": d1, "D3_mutate": d3, "D5_merge": d5}


def iter_sources(limit: int) -> Iterator[tuple[str, str, str]]:
    """Yield (project, path, source) for real, human-written Python."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    site = Path(sysconfig.get_paths()["purelib"])
    seen = 0
    buckets: list[tuple[str, list[Path]]] = [
        ("stdlib", sorted(stdlib.glob("*.py")) + sorted(stdlib.glob("*/*.py")))]
    for pkg in ("numpy", "scipy", "sklearn", "matplotlib", "transformers",
                "fontTools", "setuptools", "openai", "anthropic", "PyInstaller"):
        d = site / pkg
        if d.is_dir():
            buckets.append((pkg, sorted(d.rglob("*.py"))))
    for project, paths in buckets:
        for p in paths:
            low = str(p).lower()
            if "test" in low or "vendor" in low or "third_party" in low:
                continue
            if seen >= limit:
                return
            try:
                src = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            seen += 1
            yield project, str(p), src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--out", default="results/convention_baserate.json")
    args = ap.parse_args()

    # module -> dimension -> [(function name, verdict)]
    found: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list))
    owner: dict[str, str] = {}
    # A module is a topic, not a convention scope. Reading the disagreeing
    # modules by hand showed the "disagreements" were nearly all between
    # functions with nothing to do with each other -- ntpath.commonpath raising
    # on an empty sequence while normcase returns an empty string is not a
    # convention conflict, it is two unrelated jobs. The narrower scope that
    # actually models sibling-hood is a family: functions in one module that
    # guard the SAME parameter.
    family: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    n_files = n_funcs = n_parsed = 0

    for project, path, src in iter_sources(args.limit):
        n_files += 1
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        n_parsed += 1
        owner[path] = project
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            n_funcs += 1
            a = node.args
            params = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
            params.discard("self")
            params.discard("cls")
            if not params:
                continue
            for dim, fnc in DIMENSIONS.items():
                try:
                    v = fnc(node, params)
                except Exception:
                    v = None
                if v:
                    # d1 tags its verdict with the parameter it guarded, so the
                    # family scope can be built; the module-level tally must not
                    # see that tag or two functions guarding differently-named
                    # parameters would count as disagreeing on sight.
                    guarded = None
                    if "|" in v:
                        v, guarded = v.split("|", 1)
                    found[path][dim].append((node.name, v))
                    if guarded is not None:
                        family[(path, guarded)].append((node.name, v))

    print(f"scanned {n_files} files ({n_parsed} parsed), {n_funcs} functions\n")

    report: dict[str, Any] = {"files": n_files, "functions": n_funcs, "dims": {}}
    for dim in DIMENSIONS:
        mods_any = [m for m in found if found[m].get(dim)]
        mods_two = [m for m in mods_any if len(found[m][dim]) >= 2]
        agree, disagree = [], []
        verdicts: Counter[str] = Counter()
        for m in mods_two:
            vs = [v for _, v in found[m][dim]]
            verdicts.update(vs)
            (agree if len(set(vs)) == 1 else disagree).append(m)
        for m in mods_any:
            if len(found[m][dim]) == 1:
                verdicts.update(v for _, v in found[m][dim])

        n_two = len(mods_two)
        print("=" * 74)
        print(f"{dim}")
        print("=" * 74)
        print(f"  modules with >=1 function taking a position   {len(mods_any):>6}"
              f"   ({len(mods_any) / max(1, n_parsed):.1%} of parsed)")
        print(f"  modules with >=2  -- a local convention could exist"
              f"  {n_two:>6}   ({n_two / max(1, n_parsed):.1%})")
        if n_two:
            print(f"  of those, the siblings AGREE                  {len(agree):>6}"
                  f"   ({len(agree) / n_two:.1%})")
            print(f"                       they DISAGREE            {len(disagree):>6}"
                  f"   ({len(disagree) / n_two:.1%})")
        print(f"  verdict mix: {dict(verdicts.most_common())}")

        report["dims"][dim] = {
            "modules_any": len(mods_any), "modules_two_plus": n_two,
            "parsed": n_parsed,
            "agree": len(agree), "disagree": len(disagree),
            "availability": round(n_two / max(1, n_parsed), 4),
            "agreement": round(len(agree) / n_two, 4) if n_two else None,
            "verdicts": dict(verdicts),
            # Keep the disagreeing modules: P3 in the pre-registration requires
            # reading ten of them by hand, and a number alone will not settle it.
            "disagreeing_examples": [
                {"project": owner.get(m, "?"), "module": m,
                 "functions": found[m][dim]}
                for m in disagree[:40]],
            "agreeing_examples": [
                {"project": owner.get(m, "?"), "module": m,
                 "functions": found[m][dim]}
                for m in agree[:20]],
        }
        print()

    fam2 = {k: v for k, v in family.items() if len(v) >= 2}
    fam_agree = [k for k, v in fam2.items() if len({x for _, x in v}) == 1]
    print("=" * 74)
    print("D1_empty, scoped to a FUNCTION FAMILY (same module, same guarded param)")
    print("=" * 74)
    print(f"  families with >=2 functions taking a position  {len(fam2):>6}"
          f"   ({len(fam2) / max(1, n_parsed):.1%} of parsed modules)")
    if fam2:
        print(f"  of those, the siblings AGREE                  {len(fam_agree):>6}"
              f"   ({len(fam_agree) / len(fam2):.1%})")
        print(f"                       they DISAGREE            "
              f"{len(fam2) - len(fam_agree):>6}"
              f"   ({1 - len(fam_agree) / len(fam2):.1%})")
    print()
    print("  disagreeing families:")
    for k, v in fam2.items():
        if len({x for _, x in v}) > 1:
            print(f"    {owner.get(k[0], '?')}/{os.path.basename(k[0])}  param={k[1]}")
            for fn, ver in v:
                print(f"        {ver:<7} {fn}")
    print()
    print("  a sample of agreeing families:")
    for k in fam_agree[:8]:
        vs = family[k]
        print(f"    {owner.get(k[0], '?')}/{os.path.basename(k[0])}  param={k[1]}"
              f"  -> {vs[0][1]}   ({', '.join(f for f, _ in vs)})")
    report["family_scope"] = {
        "families_two_plus": len(fam2), "agree": len(fam_agree),
        "disagree": len(fam2) - len(fam_agree),
        "availability": round(len(fam2) / max(1, n_parsed), 4),
        "agreement": round(len(fam_agree) / len(fam2), 4) if fam2 else None,
        "families": {f"{owner.get(k[0], '?')}|{os.path.basename(k[0])}|{k[1]}": v
                     for k, v in fam2.items()},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
