from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.independent_label_store import (
    IndependentCandidateLabelStore,
    IndependentCandidateLabelWriter,
    IndependentLabelRecord,
    IndependentLabelScene,
    ScoreReconstructionError,
    validate_label_arrays,
)
from research.cf_effect_gate_wote.src.metrics import pdms_from_factors


def _scene(token: str = "scene-a") -> IndependentLabelScene:
    rng = np.random.default_rng(4)
    factors = rng.uniform(0.1, 1.0, size=(256, 5)).astype(np.float32)
    score = pdms_from_factors(factors).astype(np.float32)
    return IndependentLabelScene(
        record=IndependentLabelRecord(token, "a" * 64, "b" * 64, "c" * 64),
        factors=factors,
        score=score,
        oracle_index=int(np.argmax(score)),
        candidate_indices=np.arange(256, dtype=np.int64),
    )


def test_independent_label_store_round_trip_and_explicit_join(tmp_path: Path) -> None:
    scene = _scene()
    root = tmp_path / "labels"
    IndependentCandidateLabelWriter(root, "a" * 64, "d" * 64).write([scene])
    store = IndependentCandidateLabelStore(root)
    loaded = store.join_scene("scene-a", "a" * 64, "b" * 64)
    np.testing.assert_array_equal(loaded.factors, scene.factors)
    np.testing.assert_array_equal(loaded.score, scene.score)
    assert loaded.oracle_index == scene.oracle_index
    with pytest.raises(RuntimeError, match="trajectory join mismatch"):
        store.join_scene("scene-a", "a" * 64, "e" * 64)


def test_score_reassembles_within_one_e_minus_six() -> None:
    scene = _scene()
    assert scene.validate() <= 1e-6


def test_driving_direction_multiplier_fails_five_factor_contract() -> None:
    scene = _scene()
    evaluator_score = scene.score.copy()
    evaluator_score[7] *= 0.5
    with pytest.raises(ScoreReconstructionError) as captured:
        validate_label_arrays(
            scene.factors,
            evaluator_score,
            int(np.argmax(evaluator_score)),
            scene.candidate_indices,
        )
    assert captured.value.mismatched_indices.tolist() == [7]


def test_label_store_sidecar_contains_metric_cache_hash(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    IndependentCandidateLabelWriter(root, "a" * 64, "d" * 64).write([_scene()])
    sidecar = (root / "shard-00000.json").read_text(encoding="utf-8")
    assert '"metric_cache_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"' in sidecar

