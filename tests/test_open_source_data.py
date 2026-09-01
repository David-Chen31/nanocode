import hashlib
import json

import pytest

from bench.open_source_schema import OpenSourceTask, load_open_source_tasks
from experiments.open_source_data import _eligible_preview, _paths


def _row(**updates):
    row = {
        "id": "oss_owner_repo_pr1", "repo": "owner/repo", "pr_number": 1,
        "pr_url": "https://github.com/owner/repo/pull/1", "title": "Fix edge case",
        "body": "Handle empty input.", "created_at": "2024-08-01T00:00:00Z",
        "merged_at": "2024-08-02T00:00:00Z", "base_sha": "a" * 40,
        "head_sha": "b" * 40, "merge_sha": "c" * 40, "license_spdx": "MIT",
        "additions": 20, "deletions": 3, "changed_files": 2,
        "code_files": ["src/pkg/core.py"], "test_files": ["tests/test_core.py"],
        "patch_url": "https://github.com/owner/repo/pull/1.patch",
        "patch_sha256": "d" * 64, "selection_rank": 1,
    }
    row.update(updates)
    return row


def _document(row):
    tasks = [row]
    canonical = json.dumps(tasks, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode()
    return {"schema_version": 1, "tasks": tasks,
            "tasks_sha256": hashlib.sha256(canonical).hexdigest()}


def test_manifest_loader_accepts_a_valid_temporal_task(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(_document(_row())), encoding="utf-8")
    _, tasks = load_open_source_tasks(p)
    assert tasks == [OpenSourceTask.from_dict(_row())]


@pytest.mark.parametrize("updates", [
    {"merged_at": "2024-07-18T00:00:00Z"},
    {"test_files": ["../hidden_test.py"]},
    {"license_spdx": "NOASSERTION"},
    {"base_sha": "short"},
])
def test_manifest_loader_rejects_boundary_and_provenance_violations(tmp_path, updates):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(_document(_row(**updates))), encoding="utf-8")
    with pytest.raises(ValueError):
        load_open_source_tasks(p)


def test_patch_preview_requires_code_and_tests_without_reading_model_outcomes():
    patch = (b"diff --git a/src/pkg/core.py b/src/pkg/core.py\n"
             b"diff --git a/tests/test_core.py b/tests/test_core.py\n")
    item = {"user": {"login": "human"}, "title": "Fix empty input"}
    assert _paths(patch) == ["src/pkg/core.py", "tests/test_core.py"]
    assert _eligible_preview(item, patch)[0] is True


def test_patch_preview_excludes_bots_and_docs_only_changes():
    patch = b"diff --git a/docs/index.md b/docs/index.md\n"
    bot = {"user": {"login": "dependabot[bot]"}, "title": "Update package"}
    assert _eligible_preview(bot, patch)[1] == "bot author"
