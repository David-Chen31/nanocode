"""Stage and grade validated public tasks without coupling to one agent.

Only the staged ``workspace`` and ``task.md`` should be handed to an agent.
The sibling ``grader`` directory contains hidden tests and the gold tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.open_source_schema import (is_pytest_entrypoint,
                                      load_validated_open_source_tasks)
from experiments.open_source_container import (_check_runner_requirements,
                                                _container_test,
                                                _load_environments)
from experiments.open_source_data import materialize, prepare_grader
from experiments.provenance import snapshot_text, unified_patch

DEFAULT_MANIFEST = "bench/open_source_tasks.json"
DEFAULT_VALIDATION = "bench/open_source_validation_linux_v6.json"
DEFAULT_ENVIRONMENTS = "bench/open_source_environments_v4.json"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _grade_status(result: dict[str, Any]) -> str:
    output = result.get("output_tail", "")
    if result.get("returncode") == 0 and re.search(r"\b\d+ passed\b", output):
        return "PASS"
    if result.get("returncode") == 1 and re.search(r"\b\d+ failed\b", output):
        return "FAIL"
    return "INFRASTRUCTURE_ERROR"


def stage(task_id: str, out: str | Path, *, manifest_path: str | Path,
          validation_path: str | Path, environment_path: str | Path) -> dict[str, Any]:
    out = Path(out).resolve()
    manifest_path = Path(manifest_path).resolve()
    validation_path = Path(validation_path).resolve()
    environment_path = Path(environment_path).resolve()
    if out.exists():
        raise FileExistsError(f"staging target already exists: {out}")
    validation, tasks = load_validated_open_source_tasks(
        manifest_path, validation_path)
    selected = [task for task in tasks if task.id == task_id]
    if len(selected) != 1:
        raise ValueError(f"task is not in the validated external subset: {task_id}")
    task = selected[0]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    environments = _load_environments(manifest, environment_path)
    environment = environments["repositories"][task.repo]
    try:
        materialize(manifest_path, task_id, out / "workspace", out / "grader")
        base_snapshot = snapshot_text(out / "workspace")
        (out / "grader" / "base_snapshot.json").write_text(
            json.dumps(base_snapshot, ensure_ascii=False) + "\n", encoding="utf-8")
        prompt = f"# {task.title}\n\n{task.body.strip()}\n"
        (out / "task.md").write_text(prompt, encoding="utf-8")
        record = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": task.id, "repo": task.repo,
            "manifest_tasks_sha256": manifest["tasks_sha256"],
            "manifest_file": str(manifest_path),
            "manifest_file_sha256": _sha256(manifest_path),
            "validation_file": str(validation_path),
            "validation_file_sha256": _sha256(validation_path),
            "environment_file": str(environment_path),
            "environment_file_sha256": _sha256(environment_path),
            "environment_image_id": environment["image_id"],
            "workspace": "workspace", "grader": "grader", "prompt": "task.md",
            "validation_counts": validation["counts"],
        }
        (out / "adapter.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return record
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise


def grade(run_dir: str | Path, out: str | Path, *, timeout: int = 180) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite grade result: {out}")
    record = json.loads((run_dir / "adapter.json").read_text(encoding="utf-8"))
    manifest_path = record["manifest_file"]
    validation_path = record["validation_file"]
    environment_path = record["environment_file"]
    if _sha256(validation_path) != record["validation_file_sha256"]:
        raise ValueError("validation file changed since staging")
    if _sha256(environment_path) != record["environment_file_sha256"]:
        raise ValueError("environment file changed since staging")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if _sha256(manifest_path) != record["manifest_file_sha256"]:
        raise ValueError("manifest file changed since staging")
    _, tasks = load_validated_open_source_tasks(manifest_path, validation_path)
    task = next((item for item in tasks if item.id == record["task_id"]), None)
    if task is None:
        raise ValueError("staged task is no longer in the validated subset")
    environments = _load_environments(manifest, environment_path)
    environment = environments["repositories"][task.repo]
    if environment["image_id"] != record["environment_image_id"]:
        raise ValueError("environment image differs from the staged image")

    scoring = run_dir / "scoring"
    test_files = prepare_grader(run_dir / "workspace", run_dir / "grader", scoring)
    pytest_files = [path for path in test_files if is_pytest_entrypoint(path)]
    if not pytest_files:
        raise ValueError("task has no directly collectable pytest file")
    _check_runner_requirements(pytest_files, environment)
    pytest_args = ["--test-mypy"] if task.repo == "pydantic/pydantic" else []
    test_result = _container_test(environment["tag"], scoring, pytest_files,
                                  timeout, pytest_args)
    before = json.loads((run_dir / "grader" / "base_snapshot.json").read_text(
        encoding="utf-8"))
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task.id, "repo": task.repo,
        "status": _grade_status(test_result),
        "test_files": test_files, "pytest_files": pytest_files,
        "test": test_result,
        "agent_patch": unified_patch(before, run_dir / "workspace"),
        "environment_image_id": environment["image_id"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("stage")
    prepare.add_argument("task_id")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--manifest", default=DEFAULT_MANIFEST)
    prepare.add_argument("--validation", default=DEFAULT_VALIDATION)
    prepare.add_argument("--environments", default=DEFAULT_ENVIRONMENTS)
    score = sub.add_parser("grade")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.command == "stage":
        record = stage(args.task_id, args.out, manifest_path=args.manifest,
                       validation_path=args.validation,
                       environment_path=args.environments)
        print(f"staged {record['task_id']} in {args.out}")
        print(f"give the agent only: {args.out}/task.md and {args.out}/workspace")
        return 0
    result = grade(args.run_dir, args.out, timeout=args.timeout)
    print(json.dumps({"task_id": result["task_id"], "status": result["status"]}))
    return {"PASS": 0, "FAIL": 2, "INFRASTRUCTURE_ERROR": 3}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
