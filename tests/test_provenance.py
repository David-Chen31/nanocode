from experiments.provenance import (randomized_block_schedule, run_manifest,
                                    snapshot_text, unified_patch)


def test_unified_patch_records_changed_added_and_deleted_files(tmp_path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "gone.txt").write_text("gone\n", encoding="utf-8")
    before = snapshot_text(tmp_path)

    (tmp_path / "a.txt").write_text("new\n", encoding="utf-8")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "added.txt").write_text("added\n", encoding="utf-8")
    patch = unified_patch(before, tmp_path)

    assert "-old" in patch and "+new" in patch
    assert "gone.txt" in patch and "added.txt" in patch


def test_schedule_randomizes_treatments_inside_complete_blocks():
    a = randomized_block_schedule([("t1", 0), ("t2", 0)],
                                  ["full", "ablated"], seed=7)
    b = randomized_block_schedule([("t1", 0), ("t2", 0)],
                                  ["full", "ablated"], seed=7)
    assert a == b
    for block in (("t1", 0), ("t2", 0)):
        assert {t for t, got_block in a if got_block == block} == {"full", "ablated"}


def test_manifest_records_execution_capacity_without_credentials():
    got = run_manifest({"model": "fixture"})
    assert got["python_executable"]
    assert got["cpu_count"] >= 1
    assert "api_key" not in str(got).lower()
