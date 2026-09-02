import hashlib
import json

import pytest

from experiments.open_source_container import (_check_runner_requirements, _classify,
                                                _load_environments, BASE_IMAGE,
                                                DOCKERFILE, PROFILES)


@pytest.mark.parametrize(("before", "after", "expected"), [
    ((0, "1 passed"), (0, "1 passed"), "BASELINE_ALREADY_PASSES"),
    ((2, "collection error"), (0, "1 passed"), "INFRASTRUCTURE_ERROR"),
    ((1, "1 failed"), (1, "1 failed"), "GOLD_DOES_NOT_PASS"),
    ((1, "1 failed"), (0, "1 passed"), "VALIDATED"),
])
def test_container_status_classifier(before, after, expected):
    red = {"returncode": before[0], "output_tail": before[1]}
    green = {"returncode": after[0], "output_tail": after[1]}
    assert _classify(red, green) == expected


def test_base_timeout_is_red_only_when_gold_passes():
    timed_out = {"returncode": 124, "timed_out": True, "output_tail": ""}
    green = {"returncode": 0, "timed_out": False, "output_tail": "1 passed"}
    still_red = {"returncode": 124, "timed_out": True, "output_tail": ""}
    assert _classify(timed_out, green) == "VALIDATED"
    assert _classify(timed_out, still_red) == "GOLD_DOES_NOT_PASS"


def test_all_skipped_is_not_misreported_as_baseline_pass():
    skipped = {"returncode": 0, "timed_out": False,
               "output_tail": "60 skipped in 1.0s"}
    assert _classify(skipped, skipped) == "INFRASTRUCTURE_ERROR"


def test_runner_requirement_is_checked_against_frozen_packages():
    files = ["tests/mypy/test_mypy.py"]
    with pytest.raises(RuntimeError, match="dependency absent"):
        _check_runner_requirements(files, {"pip_freeze": ["pytest==8.2.2"]})
    _check_runner_requirements(files, {"pip_freeze": ["mypy==1.1.1"]})


def test_environment_record_fails_closed_before_image_inspection(tmp_path):
    manifest = {"tasks_sha256": "a" * 64}
    record = {
        "schema_version": 1,
        "manifest_tasks_sha256": "wrong",
        "base_image": BASE_IMAGE,
        "dockerfile_sha256": hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest(),
        "repositories": {},
    }
    path = tmp_path / "environments.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid open-source environment"):
        _load_environments(manifest, path)


def test_profiles_cover_the_six_preregistered_repositories():
    assert len(PROFILES) == 6
    assert set(PROFILES) == {
        "pallets/click", "encode/httpx", "pytest-dev/pytest",
        "pydantic/pydantic", "psf/requests", "Textualize/rich",
    }
