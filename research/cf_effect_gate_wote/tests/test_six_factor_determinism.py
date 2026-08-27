from __future__ import annotations

from pathlib import Path

from research.cf_effect_gate_wote.src.independent_label_store import (
    SixFactorIndependentCandidateLabelWriter,
)
from research.cf_effect_gate_wote.src.six_factor_verdict import (
    compare_six_factor_runs,
)
from research.cf_effect_gate_wote.tests.test_six_factor_store import six_factor_scene


def test_v2_logical_hash_is_path_independent(tmp_path: Path) -> None:
    scenes = [six_factor_scene("scene-a"), six_factor_scene("scene-b")]
    first = tmp_path / "first-path"
    second = tmp_path / "different-path"
    for root in (first, second):
        SixFactorIndependentCandidateLabelWriter(
            root, "a" * 64, "d" * 64, shard_scenes=1
        ).write(scenes)
    rows, summary = compare_six_factor_runs(
        first, second, expected_scenes=2, pass_status="TEST_PASS"
    )
    assert summary["pass"]
    assert summary["status"] == "TEST_PASS"
    assert summary["run1_logical_sha256"] == summary["run2_logical_sha256"]
    assert summary["max_run_to_run_error"] == 0
    assert all(row["raw_progress_array_equal"] for row in rows)
