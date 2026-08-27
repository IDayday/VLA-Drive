from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.independent_label_store import (
    IndependentCandidateLabelStore,
    IndependentLabelRecord,
    IndependentLabelStoreError,
    SIX_FACTOR_LABEL_SCHEMA_VERSION,
    SixFactorIndependentCandidateLabelStore,
    SixFactorIndependentCandidateLabelWriter,
    SixFactorIndependentLabelScene,
)
from research.cf_effect_gate_wote.src.six_factor_metrics import (
    SIX_FACTOR_ORDER,
    pdms_from_six_factors,
)


def six_factor_scene(token: str = "scene-v2") -> SixFactorIndependentLabelScene:
    rng = np.random.default_rng(20260827)
    factors = rng.uniform(0.1, 1.0, size=(256, 6)).astype(np.float32)
    score = pdms_from_six_factors(factors).astype(np.float32)
    return SixFactorIndependentLabelScene(
        record=IndependentLabelRecord(token, "a" * 64, "a" * 64, "c" * 64),
        factors=factors,
        score=score,
        raw_progress=np.linspace(0, 10, 256, dtype=np.float32),
        oracle_index=int(np.argmax(score)),
        candidate_indices=np.arange(256, dtype=np.int64),
    )


def test_v2_store_round_trip_schema_and_raw_progress(tmp_path: Path) -> None:
    scene = six_factor_scene()
    root = tmp_path / "labels"
    SixFactorIndependentCandidateLabelWriter(root, "a" * 64, "d" * 64).write([scene])
    store = SixFactorIndependentCandidateLabelStore(root)
    loaded = store.join_scene(
        "scene-v2", "a" * 64, "a" * 64, np.arange(256, dtype=np.int64)
    )
    np.testing.assert_array_equal(loaded.factors, scene.factors)
    np.testing.assert_array_equal(loaded.score, scene.score)
    np.testing.assert_array_equal(loaded.raw_progress, scene.raw_progress)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SIX_FACTOR_LABEL_SCHEMA_VERSION
    assert manifest["factor_order"] == list(SIX_FACTOR_ORDER)
    assert manifest["progress_normalization_scope"] == "all_256_candidates_within_scene"
    assert manifest["candidate_set_dependent_ep"] is True
    with np.load(root / "shard-00000.npz", allow_pickle=False) as archive:
        assert archive["factors"].shape == (1, 256, 6)
        assert archive["raw_progress"].shape == (1, 256)


def test_v2_sidecar_contains_metric_cache_hash(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    SixFactorIndependentCandidateLabelWriter(root, "a" * 64, "d" * 64).write(
        [six_factor_scene()]
    )
    sidecar = json.loads((root / "shard-00000.json").read_text(encoding="utf-8"))
    assert sidecar["records"][0]["metric_cache_sha256"] == "c" * 64


def test_old_reader_never_misparses_v2_store(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    SixFactorIndependentCandidateLabelWriter(root, "a" * 64, "d" * 64).write(
        [six_factor_scene()]
    )
    with pytest.raises(IndependentLabelStoreError, match="unsupported independent"):
        IndependentCandidateLabelStore(root)
