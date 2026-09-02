import pytest

from experiments.open_source_adapter import (_grade_status, DEFAULT_ENVIRONMENTS,
                                              DEFAULT_MANIFEST, DEFAULT_VALIDATION,
                                              stage)


@pytest.mark.parametrize(("returncode", "output", "expected"), [
    (0, "12 passed in 1.0s", "PASS"),
    (1, "1 failed, 11 passed", "FAIL"),
    (0, "12 skipped in 1.0s", "INFRASTRUCTURE_ERROR"),
    (2, "collection error", "INFRASTRUCTURE_ERROR"),
    (124, "", "INFRASTRUCTURE_ERROR"),
])
def test_grade_status_never_calls_skips_or_infrastructure_a_pass(
        returncode, output, expected):
    assert _grade_status({"returncode": returncode, "output_tail": output}) == expected


def test_stage_rejects_candidate_that_is_not_in_validated_subset(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="not in the validated external subset"):
        stage("oss_psf_requests_pr6963", out,
              manifest_path=DEFAULT_MANIFEST,
              validation_path=DEFAULT_VALIDATION,
              environment_path=DEFAULT_ENVIRONMENTS)
    assert not out.exists()
