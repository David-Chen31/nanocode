"""Select and validate a temporal holdout of real public GitHub PR tasks.

Selection rules are frozen in docs/PREREG_open_source_data.md. This script
never runs an agent and never looks at model outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import platform
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.open_source_schema import (ALLOWED_LICENSES, is_test_path,
                                      load_open_source_tasks)

VERSION = 1
WINDOW_START = "2024-07-19"
WINDOW_END = "2025-07-18"
REPOS = (
    "pallets/click",
    "encode/httpx",
    "pytest-dev/pytest",
    "pydantic/pydantic",
    "psf/requests",
    "Textualize/rich",
)
EXCLUDED_TITLE = re.compile(
    r"\b(release|bump|dependenc(?:y|ies)|pre-commit|changelog|typo|docs?|documentation)\b",
    re.I,
)
BOT = re.compile(r"(?:\[bot\]$|dependabot|pre-commit-ci)", re.I)
DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.M)


def _request(url: str, *, accept: str = "application/vnd.github+json") -> bytes:
    headers = {"Accept": accept, "User-Agent": "nanocode-open-data-audit",
               "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"GitHub request failed {exc.code}: {url}: {detail}") from exc


def _json(url: str) -> Any:
    return json.loads(_request(url))


def _search(repo: str) -> tuple[str, list[dict[str, Any]]]:
    query = (f"repo:{repo} is:pr is:merged "
             f"created:{WINDOW_START}..{WINDOW_END} "
             f"merged:{WINDOW_START}..{WINDOW_END}")
    params = urllib.parse.urlencode({"q": query, "sort": "created", "order": "asc",
                                    "per_page": 100})
    data = _json("https://api.github.com/search/issues?" + params)
    return query, data.get("items", [])


def _paths(patch: bytes) -> list[str]:
    text = patch.decode("utf-8", "replace")
    return list(dict.fromkeys(b for _, b in DIFF_HEADER.findall(text)))


def _patch_url(repo: str, number: int) -> str:
    return f"https://github.com/{repo}/pull/{number}.patch"


def _patch_chunks(patch: bytes) -> list[tuple[str, bytes]]:
    matches = list(DIFF_HEADER.finditer(patch.decode("utf-8", "replace")))
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(patch)
        chunks.append((match.group(2), patch[start:end]))
    return chunks


def _safe_extract_zip(blob: bytes, destination: Path) -> str:
    """Extract a GitHub source archive without trusting member paths."""
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="nanocode-oss-archive-") as td:
        archive = Path(td) / "source.zip"
        archive.write_bytes(blob)
        unpacked = Path(td) / "unpacked"
        unpacked.mkdir()
        with ZipFile(archive) as zf:
            for info in zf.infolist():
                target = (unpacked / info.filename).resolve()
                if unpacked.resolve() not in target.parents and target != unpacked.resolve():
                    raise ValueError(f"unsafe archive member: {info.filename}")
            zf.extractall(unpacked)
        roots = [p for p in unpacked.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise ValueError("GitHub archive did not contain exactly one root directory")
        archive_root = roots[0]
        top_name = archive_root.name
        for child in archive_root.iterdir():
            shutil.move(str(child), destination / child.name)
        return top_name


def materialize(manifest_path: str | Path, task_id: str,
                workspace: str | Path, grader: str | Path) -> dict[str, Any]:
    """Fetch immutable base source and keep gold/hidden patches outside it."""
    _, tasks = load_open_source_tasks(manifest_path)
    matches = [task for task in tasks if task.id == task_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate task id: {task_id}")
    task = matches[0]
    workspace = Path(workspace).resolve()
    grader = Path(grader).resolve()
    if workspace.exists() or grader.exists():
        raise FileExistsError("workspace and grader targets must not already exist")
    if workspace in grader.parents or grader in workspace.parents:
        raise ValueError("grader and agent workspace must not contain each other")

    owner, repo_name = task.repo.split("/", 1)
    archive_url = f"https://codeload.github.com/{owner}/{repo_name}/zip/{task.base_sha}"
    archive = _request(archive_url, accept="application/zip")
    patch = _request(task.patch_url, accept="application/vnd.github.patch")
    got_patch_hash = hashlib.sha256(patch).hexdigest()
    if got_patch_hash != task.patch_sha256:
        raise ValueError("upstream PR patch no longer matches the frozen SHA-256")

    try:
        archive_root = _safe_extract_zip(archive, workspace)
        grader.mkdir(parents=True, exist_ok=False)
        chunks = _patch_chunks(patch)
        test_chunks = [chunk for path, chunk in chunks if path in task.test_files]
        if len(test_chunks) != len(task.test_files):
            raise ValueError("could not recover every frozen test-file patch")
        hidden_patch = b"".join(test_chunks)
        (grader / "gold.patch").write_bytes(patch)
        (grader / "hidden_tests.patch").write_bytes(hidden_patch)
        (grader / "prompt.md").write_text(
            f"# {task.title}\n\n{task.body.strip()}\n", encoding="utf-8")
        record = {"task": asdict(task), "archive_url": archive_url,
                  "archive_sha256": hashlib.sha256(archive).hexdigest(),
                  "archive_root": archive_root,
                  "hidden_tests_sha256": hashlib.sha256(hidden_patch).hexdigest()}
        (grader / "materialization.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return record
    except Exception:
        # A half-staged workspace could otherwise be mistaken for valid data.
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(grader, ignore_errors=True)
        raise


def prepare_grader(workspace: str | Path, grader: str | Path,
                   output: str | Path) -> list[str]:
    """Copy an agent result and apply only the hidden test-file changes."""
    workspace, grader, output = map(lambda p: Path(p).resolve(),
                                    (workspace, grader, output))
    if output.exists():
        raise FileExistsError(f"grading target already exists: {output}")
    patch = grader / "hidden_tests.patch"
    meta = json.loads((grader / "materialization.json").read_text(encoding="utf-8"))
    expected = meta["hidden_tests_sha256"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
        raise ValueError("hidden test patch failed its materialization checksum")
    shutil.copytree(workspace, output)
    try:
        check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=output,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=60)
        if check.returncode:
            raise RuntimeError("hidden tests do not apply cleanly: " + check.stderr[:500])
        subprocess.run(["git", "apply", str(patch)], cwd=output, check=True,
                       capture_output=True, timeout=60)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return list(meta["task"]["test_files"])


def _apply_patch_copy(workspace: Path, patch: Path, output: Path) -> None:
    shutil.copytree(workspace, output)
    try:
        check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=output,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=60)
        if check.returncode:
            raise RuntimeError("patch does not apply cleanly: " + check.stderr[:500])
        subprocess.run(["git", "apply", str(patch)], cwd=output, check=True,
                       capture_output=True, timeout=60)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def validate_red_green(manifest_path: str | Path, task_id: str,
                       scratch: str | Path, *, timeout: int = 300) -> dict[str, Any]:
    """Check that hidden tests fail on base and pass with the public gold patch."""
    scratch = Path(scratch).resolve()
    if scratch.exists():
        raise FileExistsError(f"scratch target already exists: {scratch}")
    workspace = scratch / "workspace"
    grader = scratch / "grader"
    baseline = scratch / "baseline"
    gold = scratch / "gold"
    scratch.mkdir(parents=True)
    try:
        record = materialize(manifest_path, task_id, workspace, grader)
        test_files = prepare_grader(workspace, grader, baseline)
        _apply_patch_copy(workspace, grader / "gold.patch", gold)

        def run(root: Path) -> dict[str, Any]:
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_files],
                                  cwd=root, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
            text = (proc.stdout + "\n" + proc.stderr).strip()
            return {"returncode": proc.returncode, "output_tail": text[-4000:]}

        before = run(baseline)
        after = run(gold)
        assertion_red = before["returncode"] == 1 and bool(
            re.search(r"\b\d+ failed\b", before["output_tail"]))
        if before["returncode"] == 0:
            status = "BASELINE_ALREADY_PASSES"
        elif not assertion_red:
            status = "INFRASTRUCTURE_ERROR"
        elif after["returncode"] != 0:
            status = "GOLD_DOES_NOT_PASS"
        else:
            status = "VALIDATED"
        return {"task_id": task_id, "repo": record["task"]["repo"],
                "status": status, "test_files": test_files,
                "baseline": before, "gold": after}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def validate_all(manifest_path: str | Path, out: str | Path, *,
                 timeout: int = 300) -> dict[str, Any]:
    document, tasks = load_open_source_tasks(manifest_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite validation record: {out}")
    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"validating {i}/{len(tasks)} {task.id}...", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="nanocode-oss-validate-parent-") as td:
                row = validate_red_green(manifest_path, task.id, Path(td) / "run",
                                         timeout=timeout)
        except Exception as exc:  # infrastructure is data, never a silent deletion
            row = {"task_id": task.id, "repo": task.repo,
                   "status": "INFRASTRUCTURE_ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:1000]}
        rows.append(row)
        print(f"  {row['status']}", flush=True)
    counts = Counter(row["status"] for row in rows)
    result = {"schema_version": 1,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_tasks_sha256": document["tasks_sha256"],
              "python": sys.version, "platform": platform.platform(),
              "timeout_seconds": timeout, "counts": dict(counts), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return result


def _eligible_preview(item: dict[str, Any], patch: bytes) -> tuple[bool, str, list[str]]:
    if BOT.search((item.get("user") or {}).get("login", "")):
        return False, "bot author", []
    if EXCLUDED_TITLE.search(item.get("title") or ""):
        return False, "excluded title", []
    paths = _paths(patch)
    if not 2 <= len(paths) <= 8:
        return False, "file count", paths
    tests = [p for p in paths if p.endswith(".py") and is_test_path(p)]
    code = [p for p in paths if p.endswith(".py") and not is_test_path(p)]
    if not tests:
        return False, "no test file", paths
    if not code:
        return False, "no python code file", paths
    return True, "eligible preview", paths


def _repo_license(repo: str) -> tuple[str, dict[str, Any]]:
    data = _json(f"https://api.github.com/repos/{repo}")
    license_id = (data.get("license") or {}).get("spdx_id")
    return license_id, {"html_url": data.get("html_url"),
                        "default_branch": data.get("default_branch"),
                        "license_spdx": license_id,
                        "stars_at_selection": data.get("stargazers_count")}


def select(per_repo: int = 5) -> dict[str, Any]:
    tasks = []
    audit: dict[str, Any] = {}
    repositories = {}
    for repo in REPOS:
        print(f"checking {repo}...", flush=True)
        license_id, repo_meta = _repo_license(repo)
        repositories[repo] = repo_meta
        if license_id not in ALLOWED_LICENSES:
            audit[repo] = {"selected": 0, "error": f"license {license_id} not allowed"}
            continue
        query, candidates = _search(repo)
        excluded: Counter[str] = Counter()
        selected = []
        for rank, item in enumerate(candidates, 1):
            if len(selected) >= per_repo:
                break
            number = int(item["number"])
            patch_url = _patch_url(repo, number)
            if rank == 1 or rank % 10 == 0:
                print(f"  candidate {rank}/{len(candidates)}, "
                      f"selected {len(selected)}/{per_repo}", flush=True)
            patch = _request(patch_url, accept="application/vnd.github.patch")
            ok, reason, paths = _eligible_preview(item, patch)
            if not ok:
                excluded[reason] += 1
                continue
            detail = _json(f"https://api.github.com/repos/{repo}/pulls/{number}")
            if not detail.get("merged_at"):
                excluded["not merged on detail"] += 1
                continue
            if detail.get("additions", 10**9) > 300 or detail.get("deletions", 10**9) > 200:
                excluded["patch size"] += 1
                continue
            if detail.get("changed_files") != len(paths):
                excluded["patch file metadata mismatch"] += 1
                continue
            code = [p for p in paths if p.endswith(".py") and not is_test_path(p)]
            tests = [p for p in paths if p.endswith(".py") and is_test_path(p)]
            owner_repo = repo.replace("/", "_").lower()
            task = {
                "id": f"oss_{owner_repo}_pr{number}",
                "repo": repo,
                "pr_number": number,
                "pr_url": detail["html_url"],
                "title": detail.get("title") or "",
                "body": detail.get("body") or "",
                "created_at": detail["created_at"],
                "merged_at": detail["merged_at"],
                "base_sha": detail["base"]["sha"],
                "head_sha": detail["head"]["sha"],
                "merge_sha": detail["merge_commit_sha"],
                "license_spdx": license_id,
                "additions": detail["additions"],
                "deletions": detail["deletions"],
                "changed_files": detail["changed_files"],
                "code_files": code,
                "test_files": tests,
                "patch_url": patch_url,
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "selection_rank": rank,
            }
            selected.append(task)
            tasks.append(task)
            print(f"    selected PR #{number} ({len(selected)}/{per_repo})", flush=True)
        audit[repo] = {"query": query, "candidate_pool": len(candidates),
                       "selected": len(selected), "excluded_before_cutoff": dict(excluded)}
    canonical = json.dumps(tasks, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode()
    return {
        "schema_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code": "experiments/open_source_data.py",
        "selection_preregistration": "docs/PREREG_open_source_data.md",
        "snapshot_model": "gpt-4o-mini-2024-07-18",
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "repositories": repositories,
        "audit": audit,
        "tasks_sha256": hashlib.sha256(canonical).hexdigest(),
        "tasks": tasks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--out", default="bench/open_source_tasks.json")
    choose.add_argument("--per-repo", type=int, default=5)
    check = sub.add_parser("validate")
    check.add_argument("path", nargs="?", default="bench/open_source_tasks.json")
    fetch = sub.add_parser("materialize")
    fetch.add_argument("task_id")
    fetch.add_argument("--manifest", default="bench/open_source_tasks.json")
    fetch.add_argument("--workspace", required=True)
    fetch.add_argument("--grader", required=True)
    grade = sub.add_parser("prepare-grader")
    grade.add_argument("--workspace", required=True)
    grade.add_argument("--grader", required=True)
    grade.add_argument("--out", required=True)
    one = sub.add_parser("validate-task")
    one.add_argument("task_id")
    one.add_argument("--manifest", default="bench/open_source_tasks.json")
    one.add_argument("--scratch", required=True)
    one.add_argument("--timeout", type=int, default=300)
    all_tasks = sub.add_parser("validate-all")
    all_tasks.add_argument("--manifest", default="bench/open_source_tasks.json")
    all_tasks.add_argument("--out", default="bench/open_source_validation.json")
    all_tasks.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    if args.command == "select":
        out = Path(args.out)
        if out.exists():
            ap.error(f"refusing to overwrite frozen manifest: {out}")
        if not 1 <= args.per_repo <= 5:
            ap.error("--per-repo must be between 1 and the pre-registered maximum 5")
        document = select(args.per_repo)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"wrote {len(document['tasks'])} tasks to {out}")
        for repo, audit in document["audit"].items():
            print(f"  {repo:<24} {audit.get('selected', 0)} selected")
        return 0 if len(document["tasks"]) == len(REPOS) * args.per_repo else 2
    if args.command == "validate":
        document, tasks = load_open_source_tasks(args.path)
        print(f"valid: {len(tasks)} tasks, sha256={document['tasks_sha256']}")
        return 0
    if args.command == "materialize":
        record = materialize(args.manifest, args.task_id, args.workspace, args.grader)
        print(f"materialized {args.task_id} at {record['task']['base_sha']}")
        return 0
    if args.command == "prepare-grader":
        tests = prepare_grader(args.workspace, args.grader, args.out)
        print("hidden tests applied; suggested command:")
        print("  py -3 -m pytest " + " ".join(tests))
        return 0
    if args.command == "validate-task":
        result = validate_red_green(args.manifest, args.task_id, args.scratch,
                                    timeout=args.timeout)
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 0 if result["status"] == "VALIDATED" else 2
    result = validate_all(args.manifest, args.out, timeout=args.timeout)
    print("validation counts: " + json.dumps(result["counts"], sort_keys=True))
    return 0 if result["counts"] == {"VALIDATED": len(result["rows"])} else 2


if __name__ == "__main__":
    raise SystemExit(main())
