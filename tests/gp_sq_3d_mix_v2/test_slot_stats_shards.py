import json
import inspect
from pathlib import Path

import torch

from starVLA.gp_sq3dmix_v2 import (
    descriptor_projection,
    pooled_scene_descriptor,
    tensor_sha256,
)
from tools.compute_gp_sq3dmix_slot_stats import (
    _compute_shard,
    _contract_sha,
    _merge,
)
import tools.compute_gp_sq3dmix_slot_stats as slot_stats_tool


def _partial(root: Path, contract: dict, shard_id: int, tokens, indices, mean_value):
    descriptors = torch.nn.functional.normalize(
        torch.arange(len(tokens) * 128, dtype=torch.float32).reshape(len(tokens), 128)
        + 1
        + shard_id,
        dim=-1,
    ).half()
    payload = {
        "schema_version": 2,
        "shard_id": shard_id,
        "contract_sha256": _contract_sha(contract),
        "count": len(tokens),
        "pooled_feature_slot_mean": torch.full(
            (180, 2048), mean_value, dtype=torch.float64
        ),
        "pooled_scene_descriptor_sum": descriptors.double().sum(dim=0),
        "token_descriptors": descriptors,
        "tokens": tokens,
        "source_indices": indices,
    }
    torch.save(payload, root / f"partial_{shard_id}.pt")


def test_slot_stat_shard_merge_matches_weighted_single_process_mean(tmp_path):
    tokens = ["a", "b", "c"]
    contract = {
        "schema_version": 2,
        "shard_count": 2,
        "sample_count": 3,
        "view_order": ["cam_f0", "cam_l0", "cam_r0"],
        "pooling_layout": [3, 6, 10],
        "feature_dimension": 2048,
        "descriptor_dimension": 128,
        "descriptor_projection_seed": 20260824,
        "descriptor_projection_shape": [2048, 128],
        "descriptor_projection_sha256": "a" * 64,
        "token_order_sha256": "b" * 64,
    }
    _partial(tmp_path, contract, 0, ["a", "c"], [0, 2], 2.0)
    _partial(tmp_path, contract, 1, ["b"], [1], 5.0)
    manifest = _merge(tmp_path, tokens, contract)
    result = torch.load(
        tmp_path / "gp_sq3dmix_pooled_stats.pt",
        map_location="cpu",
        weights_only=True,
    )["pooled_feature_slot_mean"]
    torch.testing.assert_close(result, torch.full_like(result, 3.0))
    assert manifest["completed_shards"] == [0, 1]
    descriptor_asset = torch.load(
        tmp_path / "pooled_scene_descriptors.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert descriptor_asset["tokens"] == tokens


def test_slot_stat_resume_does_not_recount_completed_shard(tmp_path):
    tokens = ["a", "b"]
    contract = {"schema_version": 2, "shard_count": 1, "sample_count": 2}
    _partial(tmp_path, contract, 0, tokens, [0, 1], 1.0)
    before = (tmp_path / "partial_0.pt").stat().st_mtime_ns
    result = _compute_shard(
        str(tmp_path / "missing-cache"),
        str(tmp_path),
        tokens,
        1,
        0,
        contract,
    )
    after = (tmp_path / "partial_0.pt").stat().st_mtime_ns
    assert result["resumed"] is True
    assert before == after


def test_descriptor_projection_and_descriptor_are_deterministic():
    first = descriptor_projection()
    second = descriptor_projection()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert tensor_sha256(first) == tensor_sha256(second)
    features = torch.randn(180, 2048, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(
        pooled_scene_descriptor(features, first),
        pooled_scene_descriptor(features, second),
        rtol=0,
        atol=0,
    )


def test_parallel_slot_stats_use_spawn_not_fork():
    source = inspect.getsource(slot_stats_tool.main)
    assert 'multiprocessing.get_context("spawn")' in source
