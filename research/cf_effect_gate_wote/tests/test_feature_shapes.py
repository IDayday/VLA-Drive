from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.cf_effect_gate_wote.src.cache_wote_features import (
    assert_official_equivalence,
    environment_only_future,
    validate_debug_shapes,
)
from research.cf_effect_gate_wote.src.feature_store import (
    CacheIdentity,
    FeatureShardReader,
    FeatureShardWriter,
    SceneCacheRecord,
    fixed_random_projection,
    stable_array_hash,
)
from research.cf_effect_gate_wote.src.models.probe_heads import (
    MatchedCapacityFactorProbe,
    MatchedInputComposer,
    audit_parameters,
)
from research.cf_effect_gate_wote.src.train_probe import ProbeScene, raw_scene_inputs


def _debug_output(batch: int = 2) -> dict[str, torch.Tensor]:
    candidates, horizon, steps, cells, channels = 256, 8, 1, 64, 256
    return {
        "trajectory": torch.zeros(batch, horizon, 3),
        "current_bev_tokens": torch.zeros(batch, cells, channels),
        "current_bev_pool": torch.zeros(batch, channels),
        "ego_status_feature": torch.zeros(batch, 8),
        "trajectory_anchor_raw": torch.zeros(batch, candidates, horizon, 3),
        "trajectory_anchor_feature": torch.zeros(batch, candidates, channels),
        "candidate_current_feature": torch.zeros(batch, candidates, channels),
        "future_ego_features_by_step": torch.zeros(
            batch, candidates, steps, channels
        ),
        "future_bev_tokens_by_step": torch.zeros(
            batch, candidates, steps, cells, channels
        ),
        "future_bev_pool_by_step": torch.zeros(
            batch, candidates, steps, channels
        ),
        "reward_feature": torch.zeros(batch, candidates, channels),
        "all_trajectory": torch.zeros(candidates, horizon, 3),
        "base_trajectory_anchors": torch.zeros(candidates, horizon, 3),
        "trajectory_offsets": torch.zeros(batch, candidates, horizon, 3),
        "im_rewards": torch.zeros(batch, candidates),
        "sim_rewards": torch.zeros(batch, candidates, 5),
        "final_rewards": torch.zeros(batch, candidates),
        "selected_index": torch.zeros(batch, dtype=torch.long),
    }


def test_debug_feature_shapes_and_finiteness() -> None:
    output = _debug_output()
    validate_debug_shapes(output)
    output["reward_feature"][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_debug_shapes(output)


def test_official_equivalence_contract() -> None:
    reference = _debug_output(batch=1)
    instrumented = {key: value.clone() for key, value in reference.items()}
    assert_official_equivalence(reference, instrumented)
    instrumented["final_rewards"][0, 0] = 1e-4
    with pytest.raises(AssertionError):
        assert_official_equivalence(reference, instrumented)


def test_environment_only_mask_is_candidate_specific() -> None:
    current = np.zeros((1, 64, 256), dtype=np.float32)
    spatial = np.arange(64, dtype=np.float32)[None, None, None, :, None]
    future = np.broadcast_to(spatial, (1, 2, 1, 64, 256)).copy()
    trajectories = np.zeros((1, 2, 8, 3), dtype=np.float32)
    trajectories[0, 0, :, :2] = (4.0, -20.0)
    trajectories[0, 1, :, :2] = (24.0, 20.0)
    projection = fixed_random_projection(256, 64)
    first = environment_only_future(current, future, trajectories, projection)
    second = environment_only_future(current, future, trajectories, projection)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 2, 1, 64)
    assert not np.array_equal(first[:, 0], first[:, 1])


def test_feature_shards_are_byte_reproducible(tmp_path: Path) -> None:
    identity = CacheIdentity(
        run_id="shape-test",
        split="smoke",
        checkpoint_sha256="a" * 64,
        wote_commit_sha="b" * 40,
    )
    trajectory = np.arange(256 * 8 * 3, dtype=np.float32).reshape(256, 8, 3)
    labels = np.linspace(0, 1, 256 * 5, dtype=np.float32).reshape(256, 5)
    record = SceneCacheRecord(
        scene_token="token-1",
        candidate_indices=tuple(range(256)),
        trajectory_hash=stable_array_hash(trajectory),
        label_hash=stable_array_hash(labels),
    )
    shard_hashes: list[str] = []
    for name in ("first", "second"):
        writer = FeatureShardWriter(tmp_path / name, identity)
        writer.write_shard(
            0,
            {
                "trajectory": trajectory[None],
                "factor_labels": labels[None],
            },
            (record,),
        )
        manifest_path = writer.finalize()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_hashes.append(manifest["shards"][0]["sha256"])
        reader = FeatureShardReader(tmp_path / name, identity)
        loaded = list(reader.iter_shards())
        assert len(loaded) == 1
        assert loaded[0][1]["trajectory"].dtype == np.float16
    assert shard_hashes[0] == shard_hashes[1]


def test_feature_store_refuses_overwrite(tmp_path: Path) -> None:
    identity = CacheIdentity(
        run_id="no-overwrite",
        split="smoke",
        checkpoint_sha256="a" * 64,
        wote_commit_sha="b" * 40,
    )
    FeatureShardWriter(tmp_path / "cache", identity)
    with pytest.raises(RuntimeError, match="existing cache root"):
        FeatureShardWriter(tmp_path / "cache", identity)


def test_matched_probe_has_fixed_capacity_and_shapes() -> None:
    composer = MatchedInputComposer()
    probe = MatchedCapacityFactorProbe(hidden_dim=256)
    audit = audit_parameters(composer, probe)
    common = composer(
        torch.zeros(2, 16, 64),
        torch.zeros(2, 16, 256),
        torch.zeros(2, 16, 768),
    )
    output = probe(common)
    assert audit.trainable_parameters == 332_037
    assert common.shape == (2, 16, 1024)
    assert output["factors"].shape == (2, 16, 5)
    assert output["score"].shape == (2, 16)


def test_effect_swap_changes_only_auxiliary_slot() -> None:
    candidates = 256
    frozen = {
        "trajectory": np.arange(candidates * 8 * 3, dtype=np.float32).reshape(
            candidates, 8, 3
        ),
        "factor_labels": np.zeros((candidates, 5), dtype=np.float32),
        "ego_status_feature": np.zeros(8, dtype=np.float32),
        "current_bev_pool": np.arange(256, dtype=np.float32),
        "selected_index": np.asarray(0, dtype=np.int64),
    }
    effects = {
        "ego_effect": np.arange(candidates * 8 * 16, dtype=np.float32).reshape(
            candidates, 8, 16
        ),
        "map_effect": np.zeros((candidates, 8, 8), dtype=np.float32),
        "actor_effect": np.zeros((candidates, 8, 16, 13), dtype=np.float32),
        "actor_mask": np.ones((candidates, 8, 16), dtype=bool),
        "interaction_mask": np.zeros((candidates, 8, 16), dtype=bool),
        "shared_logged_future": np.zeros((8, 16, 8), dtype=np.float32),
        "shared_actor_mask": np.zeros((8, 16), dtype=bool),
    }
    scene = ProbeScene("swap", frozen, effects)
    ordinary = raw_scene_inputs(scene, "oracle_replay_effect")
    permutation = np.roll(np.arange(candidates), 1)
    swapped = raw_scene_inputs(
        scene, "oracle_replay_effect", effect_permutation=permutation
    )
    np.testing.assert_array_equal(ordinary[0], swapped[0])
    np.testing.assert_array_equal(ordinary[1], swapped[1])
    np.testing.assert_array_equal(ordinary[3], swapped[3])
    assert not np.array_equal(ordinary[2], swapped[2])
