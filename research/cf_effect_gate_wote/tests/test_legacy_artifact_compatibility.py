from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.feature_store import sha256_file
from research.cf_effect_gate_wote.src.independent_label_store import (
    IndependentCandidateLabelStore,
    IndependentCandidateLabelWriter,
    IndependentLabelRecord,
    IndependentLabelScene,
    IndependentLabelStoreError,
    SixFactorIndependentCandidateLabelStore,
)
from research.cf_effect_gate_wote.src.metrics import pdms_from_factors


LEGACY_REPORT_TREE_SHA256 = (
    "dbc800d748be19c5245b6cd33a4460b4212f9abbb2a1d334e3766609e582dfd2"
)


def test_v1_store_remains_readable_and_v2_reader_rejects_it(tmp_path: Path) -> None:
    factors = np.full((256, 5), 0.75, dtype=np.float32)
    score = pdms_from_factors(factors).astype(np.float32)
    scene = IndependentLabelScene(
        IndependentLabelRecord("legacy", "a" * 64, "b" * 64, "c" * 64),
        factors,
        score,
        int(np.argmax(score)),
        np.arange(256, dtype=np.int64),
    )
    root = tmp_path / "v1"
    IndependentCandidateLabelWriter(root, "a" * 64, "d" * 64).write([scene])
    loaded = IndependentCandidateLabelStore(root).join_scene(
        "legacy", "a" * 64, "b" * 64
    )
    np.testing.assert_array_equal(loaded.factors, factors)
    with pytest.raises(IndependentLabelStoreError, match="six-factor"):
        SixFactorIndependentCandidateLabelStore(root)


def test_historical_report_tree_hash_is_unchanged() -> None:
    project_root = Path(__file__).resolve().parents[3]
    files = sorted(
        path
        for directory in (
            project_root / "reports/cf_effect_gate_wote",
            project_root / "reports/cf_effect_wote_relabel",
        )
        for path in directory.rglob("*")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(project_root)
        digest.update(f"{sha256_file(path)}  {relative}\n".encode("utf-8"))
    assert digest.hexdigest() == LEGACY_REPORT_TREE_SHA256
