"""Build frozen repository environments and audit public PR tasks in Linux.

Candidate selection remains in open_source_data.py.  This module only resolves
infrastructure errors; it never changes or substitutes the frozen task list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.open_source_schema import is_pytest_entrypoint, load_open_source_tasks
from experiments.open_source_data import materialize, prepare_grader

BASE_IMAGE = (
    "python:3.11.9-slim-bookworm@"
    "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)
DOCKERFILE = Path(__file__).with_name("open_source_env.Dockerfile")
PROFILES = {
    "pallets/click": ("click", "oss_pallets_click_pr2788"),
    "encode/httpx": ("httpx", "oss_encode_httpx_pr3250"),
    "pytest-dev/pytest": ("pytest", "oss_pytest-dev_pytest_pr12656"),
    "pydantic/pydantic": ("pydantic", "oss_pydantic_pydantic_pr9932"),
    "psf/requests": ("requests", "oss_psf_requests_pr6963"),
    "Textualize/rich": ("rich", "oss_textualize_rich_pr3454"),
}
RUNNER_REQUIREMENTS = {
    "tests/mypy/test_mypy.py": "mypy==",
}


def _run(args: list[str], *, cwd: Path | None = None, timeout: int = 1800,
         check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if check and proc.returncode:
        output = (proc.stdout + "\n" + proc.stderr).strip()
        raise RuntimeError(f"command failed ({proc.returncode}): {output[-4000:]}")
    return proc


def _image_tag(profile: str) -> str:
    return f"nanocode-oss-{profile}:py311-v4"


def build_images(manifest_path: str | Path, out: str | Path) -> dict[str, Any]:
    """Build one dependency image per repository and record resolved packages."""
    manifest, tasks = load_open_source_tasks(manifest_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite environment record: {out}")
    task_by_id = {task.id: task for task in tasks}
    if set(PROFILES) != {task.repo for task in tasks}:
        raise ValueError("container profiles do not exactly cover manifest repositories")

    rows = {}
    dockerfile_sha = hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()
    for repo, (profile, bootstrap_id) in PROFILES.items():
        task = task_by_id[bootstrap_id]
        print(f"building {repo} from {bootstrap_id}...", flush=True)
        with tempfile.TemporaryDirectory(prefix="nanocode-oss-image-") as td:
            root = Path(td)
            workspace, grader = root / "workspace", root / "grader"
            materialize(manifest_path, bootstrap_id, workspace, grader)
            tag = _image_tag(profile)
            proc = _run([
                "docker", "build", "--pull=false", "--file", str(DOCKERFILE),
                "--build-arg", f"BASE_IMAGE={BASE_IMAGE}",
                "--build-arg", f"PROFILE={profile}", "--tag", tag, ".",
            ], cwd=workspace)
            print(proc.stdout[-1000:], flush=True)
            image_id = _run([
                "docker", "image", "inspect", tag, "--format", "{{.Id}}"
            ]).stdout.strip()
            freeze = _run([
                "docker", "run", "--rm", "--network", "none", tag,
                "cat", "/opt/pip-freeze.txt",
            ]).stdout.splitlines()
            rows[repo] = {
                "profile": profile, "tag": tag, "image_id": image_id,
                "bootstrap_task_id": bootstrap_id,
                "bootstrap_base_sha": task.base_sha,
                "pip_freeze": freeze,
            }
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_tasks_sha256": manifest["tasks_sha256"],
        "base_image": BASE_IMAGE,
        "dockerfile": str(DOCKERFILE.relative_to(DOCKERFILE.parents[1])).replace("\\", "/"),
        "dockerfile_sha256": dockerfile_sha,
        "host_platform": platform.platform(),
        "repositories": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return record


def _load_environments(manifest: dict[str, Any], path: str | Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = []
    if record.get("schema_version") != 1:
        errors.append("unsupported environment schema")
    if record.get("manifest_tasks_sha256") != manifest["tasks_sha256"]:
        errors.append("environment record refers to another manifest")
    if record.get("base_image") != BASE_IMAGE:
        errors.append("base image differs from the frozen image")
    if record.get("dockerfile_sha256") != hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest():
        errors.append("Dockerfile checksum differs from the frozen environment")
    repositories = record.get("repositories") or {}
    if set(repositories) != set(PROFILES):
        errors.append("environment record does not cover every repository exactly")
    for repo, row in repositories.items():
        expected_profile, expected_task = PROFILES.get(repo, (None, None))
        if (row.get("profile"), row.get("bootstrap_task_id")) != (
                expected_profile, expected_task):
            errors.append(f"{repo}: environment profile provenance differs")
        current = _run([
            "docker", "image", "inspect", row.get("tag", ""),
            "--format", "{{.Id}}",
        ], check=False).stdout.strip()
        if current != row.get("image_id"):
            errors.append(f"{repo}: local Docker image does not match recorded image id")
    if errors:
        raise ValueError("invalid open-source environment record:\n- " +
                         "\n- ".join(errors))
    return record


def _container_test(image: str, root: Path, test_files: list[str],
                    timeout: int, pytest_args: list[str] | None = None) -> dict[str, Any]:
    tests = " ".join(shlex.quote(path) for path in test_files)
    options = " ".join(shlex.quote(arg) for arg in (pytest_args or []))
    command = (
        "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST=8.4.0.dev0 "
        "python -m pip install --no-deps --no-build-isolation -e /task "
        f"&& timeout --signal=TERM --kill-after=5s {timeout}s "
        "python -m pytest -q " + options + " " + tests
    )
    proc = _run([
        "docker", "run", "--rm", "--network", "none",
        "--mount", f"type=bind,source={root},target=/task",
        image, "sh", "-lc", command,
    ], timeout=timeout + 120, check=False)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return {"returncode": proc.returncode,
            "timed_out": proc.returncode in {124, 137},
            "output_tail": output[-4000:]}


def _classify(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_passed = bool(re.search(r"\b\d+ passed\b", before["output_tail"]))
    after_passed = bool(re.search(r"\b\d+ passed\b", after["output_tail"]))
    assertion_red = before["returncode"] == 1 and bool(
        re.search(r"\b\d+ failed\b", before["output_tail"]))
    if before["returncode"] == 0:
        return "BASELINE_ALREADY_PASSES" if before_passed else "INFRASTRUCTURE_ERROR"
    if not assertion_red and not before.get("timed_out", False):
        return "INFRASTRUCTURE_ERROR"
    if after["returncode"] != 0:
        return "GOLD_DOES_NOT_PASS"
    if not after_passed:
        return "INFRASTRUCTURE_ERROR"
    return "VALIDATED"


def _check_runner_requirements(test_files: list[str], environment: dict[str, Any]) -> None:
    frozen = environment.get("pip_freeze") or []
    for path in test_files:
        prefix = RUNNER_REQUIREMENTS.get(path)
        if prefix and not any(line.startswith(prefix) for line in frozen):
            raise RuntimeError(
                f"test runner dependency absent from frozen image: {prefix[:-2]}")


def validate_all(manifest_path: str | Path, environment_path: str | Path,
                 out: str | Path, *, timeout: int = 180) -> dict[str, Any]:
    manifest, tasks = load_open_source_tasks(manifest_path)
    environments = _load_environments(manifest, environment_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite validation record: {out}")
    rows = []
    for index, task in enumerate(tasks, 1):
        print(f"validating {index}/{len(tasks)} {task.id}...", flush=True)
        try:
            with tempfile.TemporaryDirectory(
                    prefix="nanocode-oss-container-validate-") as td:
                root = Path(td)
                workspace, grader = root / "workspace", root / "grader"
                baseline = root / "baseline"
                materialize(manifest_path, task.id, workspace, grader)
                test_files = prepare_grader(workspace, grader, baseline)
                pytest_files = [path for path in test_files
                                if is_pytest_entrypoint(path)]
                if not pytest_files:
                    raise ValueError("task has no directly collectable pytest file")
                environment = environments["repositories"][task.repo]
                _check_runner_requirements(pytest_files, environment)
                image = environment["tag"]
                pytest_args = ["--test-mypy"] if task.repo == "pydantic/pydantic" else []
                before = _container_test(image, baseline, pytest_files, timeout,
                                         pytest_args)
                after = _container_test(image, grader / "gold_source", pytest_files,
                                        timeout, pytest_args)
                row = {"task_id": task.id, "repo": task.repo,
                       "status": _classify(before, after),
                       "test_files": test_files, "pytest_files": pytest_files,
                       "baseline": before, "gold": after}
        except Exception as exc:
            row = {"task_id": task.id, "repo": task.repo,
                   "status": "INFRASTRUCTURE_ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:4000]}
        rows.append(row)
        print(f"  {row['status']}", flush=True)
    counts = Counter(row["status"] for row in rows)
    environment_bytes = Path(environment_path).read_bytes()
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_tasks_sha256": manifest["tasks_sha256"],
        "environment_record": str(environment_path),
        "environment_record_sha256": hashlib.sha256(environment_bytes).hexdigest(),
        "runner": "docker-linux", "network_during_tests": "none",
        "timeout_seconds": timeout, "counts": dict(counts), "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-images")
    build.add_argument("--manifest", default="bench/open_source_tasks.json")
    build.add_argument("--out", default="bench/open_source_environments_v4.json")
    validate = sub.add_parser("validate-all")
    validate.add_argument("--manifest", default="bench/open_source_tasks.json")
    validate.add_argument("--environments", default="bench/open_source_environments_v4.json")
    validate.add_argument("--out", default="bench/open_source_validation_linux_v6.json")
    validate.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.command == "build-images":
        record = build_images(args.manifest, args.out)
        print(f"wrote {len(record['repositories'])} environments to {args.out}")
        return 0
    result = validate_all(args.manifest, args.environments, args.out,
                          timeout=args.timeout)
    print("validation counts: " + json.dumps(result["counts"], sort_keys=True))
    return 0 if result["counts"] == {"VALIDATED": len(result["rows"])} else 2


if __name__ == "__main__":
    raise SystemExit(main())
