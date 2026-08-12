import json

import pytest
import torch

from starVLA.cache.navsim_feature_cache import (
    NavsimFeatureCacheReader,
    RankCacheWriter,
    write_manifest,
)


def test_vggt_query_cache_requires_completed_manifest_and_exact_payload(tmp_path):
    with RankCacheWriter(tmp_path, "vggt_query", rank=0, map_size_bytes=8 * 1024 * 1024) as writer:
        writer.put(
            "scene-token",
            {
                "features": torch.ones(27, 6, dtype=torch.bfloat16),
                "valid_mask": torch.ones(27, dtype=torch.bool),
            },
        )
    write_manifest(
        tmp_path,
        "vggt_query",
        {
            "world_size": 1,
            "query_count": 27,
            "feature_dim": 6,
            "active_slot_mask": [True] * 27,
        },
    )

    reader = NavsimFeatureCacheReader(tmp_path, components=("vggt_query",), strict=True)
    payload = reader.get("vggt_query", 0, "scene-token")
    assert payload["features"].shape == (27, 6)
    assert payload["valid_mask"].all()


def test_vggt_query_cache_fails_on_incomplete_manifest(tmp_path):
    component = tmp_path / "vggt_query"
    component.mkdir(parents=True)
    (component / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "complete": False}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not marked complete"):
        NavsimFeatureCacheReader(tmp_path, components=("vggt_query",), strict=True)
