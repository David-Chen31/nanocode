"""Schema and fail-closed validation for the external GitHub PR task layer."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SNAPSHOT_AT = datetime(2024, 7, 18, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 7, 18, 23, 59, 59, tzinfo=timezone.utc)
ALLOWED_LICENSES = {"MIT", "BSD-3-Clause", "Apache-2.0", "ISC"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _time(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _safe_path(raw: str) -> bool:
    p = PurePosixPath(raw)
    return bool(raw) and not p.is_absolute() and ".." not in p.parts


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path.lower())
    return ("tests" in p.parts or "test" in p.parts
            or p.name.startswith("test_") or p.name.endswith("_test.py"))


@dataclass(frozen=True)
class OpenSourceTask:
    id: str
    repo: str
    pr_number: int
    pr_url: str
    title: str
    body: str
    created_at: str
    merged_at: str
    base_sha: str
    head_sha: str
    merge_sha: str
    license_spdx: str
    additions: int
    deletions: int
    changed_files: int
    code_files: tuple[str, ...]
    test_files: tuple[str, ...]
    patch_url: str
    patch_sha256: str
    selection_rank: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "OpenSourceTask":
        return cls(**{**row, "code_files": tuple(row["code_files"]),
                      "test_files": tuple(row["test_files"])})

    def validate(self) -> list[str]:
        errors = []
        if not self.id.startswith("oss_"):
            errors.append("id must start with oss_")
        if self.license_spdx not in ALLOWED_LICENSES:
            errors.append("license is not in the pre-registered allowlist")
        if not SNAPSHOT_AT < _time(self.merged_at) <= WINDOW_END:
            errors.append("merged_at is outside the temporal holdout")
        if not SNAPSHOT_AT < _time(self.created_at) <= WINDOW_END:
            errors.append("created_at is outside the temporal holdout")
        if _time(self.created_at) > _time(self.merged_at):
            errors.append("created_at is after merged_at")
        for name, value in (("base_sha", self.base_sha), ("head_sha", self.head_sha),
                            ("merge_sha", self.merge_sha)):
            if not SHA_RE.fullmatch(value or ""):
                errors.append(f"{name} is not a full git SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", self.patch_sha256 or ""):
            errors.append("patch_sha256 is invalid")
        if not self.code_files or not self.test_files:
            errors.append("task needs both code and test files")
        if any(not _safe_path(p) for p in self.code_files + self.test_files):
            errors.append("unsafe repository path")
        if any(is_test_path(p) for p in self.code_files):
            errors.append("code_files contains a test path")
        if any(not is_test_path(p) for p in self.test_files):
            errors.append("test_files contains a non-test path")
        if not (2 <= self.changed_files <= 8):
            errors.append("changed_files is outside the selection rule")
        if self.additions > 300 or self.deletions > 200:
            errors.append("patch size is outside the selection rule")
        if (self.pr_number <= 0 or self.selection_rank <= 0
                or self.additions < 0 or self.deletions < 0):
            errors.append("numeric provenance fields must be non-negative")
        expected_pr = f"https://github.com/{self.repo}/pull/{self.pr_number}"
        if self.pr_url != expected_pr or self.patch_url != expected_pr + ".patch":
            errors.append("PR URLs do not match repository and number")
        return errors


def load_open_source_tasks(path: str | Path | None = None) -> tuple[dict, list[OpenSourceTask]]:
    path = Path(path) if path else Path(__file__).with_name("open_source_tasks.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    tasks = [OpenSourceTask.from_dict(row) for row in document.get("tasks", [])]
    errors = []
    if document.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    if document.get("snapshot_model") != "gpt-4o-mini-2024-07-18":
        errors.append("snapshot_model does not match the temporal design")
    if document.get("window") != {"start": "2024-07-19", "end": "2025-07-18"}:
        errors.append("selection window does not match the temporal design")
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task ids")
    sources = [(t.repo, t.pr_number) for t in tasks]
    if len(sources) != len(set(sources)):
        errors.append("duplicate repository PRs")
    for task in tasks:
        errors.extend(f"{task.id}: {msg}" for msg in task.validate())
    repositories = set((document.get("repositories") or {}).keys())
    if {t.repo for t in tasks} - repositories:
        errors.append("task refers to a repository absent from repository metadata")
    audit = document.get("audit") or {}
    for repo, record in audit.items():
        actual = sum(t.repo == repo for t in tasks)
        if record.get("selected") != actual:
            errors.append(f"{repo}: audit selected count does not match tasks")
    canonical = json.dumps(document.get("tasks", []), sort_keys=True,
                           ensure_ascii=False, separators=(",", ":")).encode()
    expected = document.get("tasks_sha256")
    if expected != hashlib.sha256(canonical).hexdigest():
        errors.append("tasks_sha256 does not match manifest content")
    if errors:
        raise ValueError("invalid open-source task manifest:\n- " + "\n- ".join(errors))
    return document, tasks
