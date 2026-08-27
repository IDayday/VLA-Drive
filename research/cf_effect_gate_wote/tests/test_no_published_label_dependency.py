from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from research.cf_effect_gate_wote.src import independent_relabel
from research.cf_effect_gate_wote.src.cache_wote_features import (
    _score_dictionary_for_label_source,
)
from research.cf_effect_gate_wote.src.feature_store import (
    BASE_ANCHOR_FEATURE_SCHEMA_VERSION,
    CacheIdentity,
    FeatureShardWriter,
    SceneCacheRecord,
)


def test_independent_relabel_has_no_published_score_reader_dependency() -> None:
    source = Path(independent_relabel.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "CandidateScoreTable" not in source
    assert "formatted_pdm_score_256.npy" not in source
    assert not any("candidate_alignment" in name for name in imports)


def test_label_source_none_never_opens_published_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("published score file was opened")

    monkeypatch.setattr(
        "research.cf_effect_gate_wote.src.cache_wote_features._load_score_dictionary",
        forbidden,
    )
    assert _score_dictionary_for_label_source(tmp_path, "none") is None


def test_label_free_sidecar_omits_factor_labels_and_label_hash(tmp_path: Path) -> None:
    identity = CacheIdentity(
        run_id="label-free",
        split="train",
        checkpoint_sha256="a" * 64,
        wote_commit_sha="b" * 40,
        feature_schema_version=BASE_ANCHOR_FEATURE_SCHEMA_VERSION,
        label_source="none",
        candidate_bank_hash="c" * 64,
    )
    writer = FeatureShardWriter(tmp_path / "cache", identity)
    writer.write_shard(
        0,
        {"trajectory": np.zeros((1, 256, 8, 3), dtype=np.float32)},
        (
            SceneCacheRecord(
                "scene",
                tuple(range(256)),
                trajectory_hash="d" * 64,
                candidate_bank_hash="c" * 64,
            ),
        ),
    )
    writer.finalize()
    sidecar = json.loads(
        (tmp_path / "cache" / "shard-00000.json").read_text(encoding="utf-8")
    )
    assert "label_hash" not in sidecar["records"][0]
    assert "factor_labels" not in sidecar["arrays"]
    assert sidecar["schema_version"] == BASE_ANCHOR_FEATURE_SCHEMA_VERSION
