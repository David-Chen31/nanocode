import hashlib
import io
import json
from zipfile import ZipFile

import pytest

from bench.open_source_schema import (OpenSourceTask, is_pytest_entrypoint,
                                      load_open_source_tasks,
                                      load_validated_open_source_tasks)
from experiments.open_source_data import (_eligible_preview, _patch_chunks, _paths,
                                          _safe_extract_zip, prepare_grader)


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
    return {"schema_version": 1,
            "snapshot_model": "gpt-4o-mini-2024-07-18",
            "window": {"start": "2024-07-19", "end": "2025-07-18"},
            "repositories": {"owner/repo": {"license_spdx": "MIT"}},
            "audit": {"owner/repo": {"selected": 1}}, "tasks": tasks,
            "tasks_sha256": hashlib.sha256(canonical).hexdigest()}


def test_manifest_loader_accepts_a_valid_temporal_task(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(_document(_row())), encoding="utf-8")
    _, tasks = load_open_source_tasks(p)
    assert tasks == [OpenSourceTask.from_dict(_row())]


@pytest.mark.parametrize("updates", [
    {"merged_at": "2024-07-18T00:00:00Z"},
    {"created_at": "2024-07-18T00:00:00Z"},
    {"test_files": ["../hidden_test.py"]},
    {"license_spdx": "NOASSERTION"},
    {"base_sha": "short"},
])
def test_manifest_loader_rejects_boundary_and_provenance_violations(tmp_path, updates):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(_document(_row(**updates))), encoding="utf-8")
    with pytest.raises(ValueError):
        load_open_source_tasks(p)


def _validation(manifest, *, status="VALIDATED"):
    row = {"task_id": manifest["tasks"][0]["id"], "status": status,
           "baseline": {"returncode": 1}, "gold": {"returncode": 0}}
    return {"schema_version": 1,
            "manifest_tasks_sha256": manifest["tasks_sha256"],
            "counts": {status: 1}, "rows": [row]}


def test_validated_loader_returns_only_audited_red_green_tasks(tmp_path):
    manifest = _document(_row())
    tasks_path = tmp_path / "tasks.json"
    validation_path = tmp_path / "validation.json"
    tasks_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation_path.write_text(json.dumps(_validation(manifest)), encoding="utf-8")

    _, tasks = load_validated_open_source_tasks(tasks_path, validation_path)
    assert [task.id for task in tasks] == ["oss_owner_repo_pr1"]


def test_committed_validation_exposes_nineteen_tasks_from_five_repositories():
    _, tasks = load_validated_open_source_tasks()
    assert len(tasks) == 19
    assert {task.repo for task in tasks} == {
        "pallets/click", "encode/httpx", "pytest-dev/pytest",
        "pydantic/pydantic", "Textualize/rich",
    }


def test_validated_loader_accepts_a_red_timeout_when_gold_passes(tmp_path):
    manifest = _document(_row())
    validation = _validation(manifest)
    validation["rows"][0]["baseline"] = {"returncode": 124, "timed_out": True}
    tasks_path = tmp_path / "tasks.json"
    validation_path = tmp_path / "validation.json"
    tasks_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    _, tasks = load_validated_open_source_tasks(tasks_path, validation_path)
    assert len(tasks) == 1


@pytest.mark.parametrize("change", ["manifest_hash", "missing_row", "false_green"])
def test_validated_loader_fails_closed_on_audit_tampering(tmp_path, change):
    manifest = _document(_row())
    validation = _validation(manifest)
    if change == "manifest_hash":
        validation["manifest_tasks_sha256"] = "0" * 64
    elif change == "missing_row":
        validation["rows"] = []
        validation["counts"] = {}
    else:
        validation["rows"][0]["gold"]["returncode"] = 1
    tasks_path = tmp_path / "tasks.json"
    validation_path = tmp_path / "validation.json"
    tasks_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid open-source validation"):
        load_validated_open_source_tasks(tasks_path, validation_path)


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


def test_pytest_entrypoint_excludes_fixture_and_expected_output_modules():
    paths = ["tests/mypy/modules/frozen_field.py",
             "tests/mypy/outputs/1.0.1/frozen_field.py",
             "tests/mypy/test_mypy.py"]
    assert [path for path in paths if is_pytest_entrypoint(path)] == [
        "tests/mypy/test_mypy.py"]


def test_patch_chunks_can_hide_test_changes_from_the_agent():
    patch = (b"mail header\n"
             b"diff --git a/src/core.py b/src/core.py\ncode\n"
             b"diff --git a/tests/test_core.py b/tests/test_core.py\ntest\n")
    chunks = _patch_chunks(patch)
    assert [path for path, _ in chunks] == ["src/core.py", "tests/test_core.py"]
    assert chunks[1][1].startswith(b"diff --git a/tests/")
    assert b"src/core.py" not in chunks[1][1]


def test_source_archive_drops_githubs_top_directory(tmp_path):
    buf = io.BytesIO()
    with ZipFile(buf, "w") as zf:
        zf.writestr("repo-sha/src/core.py", "value = 1\n")
    dest = tmp_path / "workspace"
    root = _safe_extract_zip(buf.getvalue(), dest)
    assert root == "repo-sha"
    assert (dest / "src/core.py").read_text() == "value = 1\n"


def test_prepare_grader_applies_hidden_patch_to_a_copy(tmp_path):
    workspace = tmp_path / "workspace"
    grader = tmp_path / "grader"
    workspace.mkdir()
    grader.mkdir()
    (workspace / "tests").mkdir()
    (workspace / "tests/test_core.py").write_text("old\n", encoding="utf-8")
    patch = (b"diff --git a/tests/test_core.py b/tests/test_core.py\n"
             b"index 3367afd..3e75765 100644\n"
             b"--- a/tests/test_core.py\n"
             b"+++ b/tests/test_core.py\n"
             b"@@ -1 +1 @@\n-old\n+new\n")
    (grader / "hidden_tests.patch").write_bytes(patch)
    meta = {"hidden_tests_sha256": hashlib.sha256(patch).hexdigest(),
            "task": {"test_files": ["tests/test_core.py"]}}
    (grader / "materialization.json").write_text(json.dumps(meta), encoding="utf-8")
    output = tmp_path / "scoring"
    assert prepare_grader(workspace, grader, output) == ["tests/test_core.py"]
    assert (output / "tests/test_core.py").read_text() == "new\n"
    assert (workspace / "tests/test_core.py").read_text() == "old\n"


def test_prepare_grader_overlays_sha_frozen_hidden_files(tmp_path):
    workspace = tmp_path / "workspace"
    grader = tmp_path / "grader"
    hidden = grader / "hidden_tests/tests"
    workspace.mkdir()
    hidden.mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "tests/test_core.py").write_text("old\n", encoding="utf-8")
    (workspace / "tests/test_removed.py").write_text("remove me\n", encoding="utf-8")
    payload = b"new\n"
    (hidden / "test_core.py").write_bytes(payload)
    meta = {
        "materialization_version": 2,
        "task": {"test_files": ["tests/test_core.py", "tests/test_removed.py"]},
        "hidden_tests": [
            {"path": "tests/test_core.py", "action": "copy",
             "sha256": hashlib.sha256(payload).hexdigest()},
            {"path": "tests/test_removed.py", "action": "delete"},
        ],
    }
    (grader / "materialization.json").write_text(json.dumps(meta), encoding="utf-8")

    output = tmp_path / "scoring"
    assert prepare_grader(workspace, grader, output) == meta["task"]["test_files"]
    assert (output / "tests/test_core.py").read_bytes() == payload
    assert not (output / "tests/test_removed.py").exists()
    assert (workspace / "tests/test_core.py").read_text() == "old\n"


def test_prepare_grader_rejects_tampered_hidden_file(tmp_path):
    workspace = tmp_path / "workspace"
    grader = tmp_path / "grader"
    hidden = grader / "hidden_tests/tests"
    workspace.mkdir()
    hidden.mkdir(parents=True)
    (hidden / "test_core.py").write_bytes(b"tampered\n")
    meta = {
        "materialization_version": 2,
        "task": {"test_files": ["tests/test_core.py"]},
        "hidden_tests": [{"path": "tests/test_core.py", "action": "copy",
                          "sha256": hashlib.sha256(b"expected\n").hexdigest()}],
    }
    (grader / "materialization.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum failed"):
        prepare_grader(workspace, grader, tmp_path / "scoring")
    assert not (tmp_path / "scoring").exists()


def test_prepare_grader_rejects_escaping_hidden_path(tmp_path):
    workspace = tmp_path / "workspace"
    grader = tmp_path / "grader"
    workspace.mkdir()
    grader.mkdir()
    meta = {
        "materialization_version": 2,
        "task": {"test_files": ["../escape.py"]},
        "hidden_tests": [{"path": "../escape.py", "action": "delete"}],
    }
    (grader / "materialization.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe hidden test path"):
        prepare_grader(workspace, grader, tmp_path / "scoring")
    assert not (tmp_path / "scoring").exists()
