from __future__ import annotations

from pathlib import Path

import numpy as np

from research.cf_effect_gate_wote.src.independent_label_store import (
    IndependentCandidateLabelWriter,
    IndependentLabelRecord,
    IndependentLabelScene,
)
from research.cf_effect_gate_wote.src.metrics import pdms_from_factors
from research.cf_effect_gate_wote.src.relabel_consistency import compare_independent_runs


def _scenes() -> list[IndependentLabelScene]:
    scenes = []
    for index in range(2):
        factors = np.full((256, 5), 0.5 + index * 0.1, dtype=np.float32)
        score = pdms_from_factors(factors).astype(np.float32)
        scenes.append(
            IndependentLabelScene(
                IndependentLabelRecord(
                    f"scene-{index}", "a" * 64, "b" * 64, chr(99 + index) * 64
                ),
                factors,
                score,
                int(np.argmax(score)),
                np.arange(256, dtype=np.int64),
            )
        )
    return scenes


def test_two_independent_writes_have_exact_logical_hash(tmp_path: Path) -> None:
    for name in ("run1", "run2"):
        IndependentCandidateLabelWriter(
            tmp_path / name, "a" * 64, "e" * 64, shard_scenes=1
        ).write(_scenes())
    rows, summary = compare_independent_runs(tmp_path / "run1", tmp_path / "run2")
    assert summary["status"] == "PASS"
    assert summary["logical_sha256_equal"]
    assert summary["manifest_bytes_equal"]
    assert summary["max_factor_absolute_error"] == 0
    assert all(row["oracle_index_equal"] for row in rows)

