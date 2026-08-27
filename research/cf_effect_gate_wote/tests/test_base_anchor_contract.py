from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.cache_wote_features import (
    assert_base_anchor_contract,
)
from research.cf_effect_gate_wote.src.feature_store import FeatureShardReader


def test_base_anchor_tensors_are_bitwise_equal() -> None:
    anchors = np.arange(256 * 8 * 3, dtype=np.float32).reshape(256, 8, 3)
    output = {
        "base_trajectory_anchors": anchors.copy(),
        "trajectory_anchor_raw": anchors[None].copy(),
        "all_trajectory": anchors.copy(),
    }
    assert_base_anchor_contract(output, anchors)
    original = output["all_trajectory"][3, 4, 1]
    output["all_trajectory"][3, 4, 1] = np.nextafter(
        original, np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ValueError, match="all_trajectory"):
        assert_base_anchor_contract(output, anchors)


def test_old_feature_cache_schema_remains_readable(tmp_path: Path) -> None:
    root = tmp_path / "old"
    root.mkdir()
    np.savez(root / "shard-00000.npz", value=np.ones((1, 2), dtype=np.float16))
    import hashlib

    shard_sha = hashlib.sha256((root / "shard-00000.npz").read_bytes()).hexdigest()
    sidecar = {
        "schema_version": "wote_debug.v1",
        "records": [
            {
                "scene_token": "old-scene",
                "candidate_indices": list(range(256)),
                "trajectory_hash": "a" * 64,
                "label_hash": "b" * 64,
            }
        ],
    }
    (root / "shard-00000.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    sidecar_sha = hashlib.sha256(
        (root / "shard-00000.json").read_bytes()
    ).hexdigest()
    logical = {
        "identity": {
            "run_id": "old",
            "split": "smoke",
            "checkpoint_sha256": "c" * 64,
            "wote_commit_sha": "d" * 40,
            "feature_schema_version": "wote_debug.v1",
            "candidate_count": 256,
            "horizon": 8,
        },
        "scene_count": 1,
        "shards": [
            {
                "path": "shard-00000.npz",
                "sha256": shard_sha,
                "sidecar": "shard-00000.json",
                "sidecar_sha256": sidecar_sha,
                "scene_count": 1,
            }
        ],
    }
    from research.cf_effect_gate_wote.src.feature_store import stable_json_hash

    manifest = {**logical, "logical_content_sha256": stable_json_hash(logical)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert len(list(FeatureShardReader(root).iter_shards())) == 1
