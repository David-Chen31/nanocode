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
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    document, tasks = load_open_source_tasks(args.path)
    print(f"valid: {len(tasks)} tasks, sha256={document['tasks_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
