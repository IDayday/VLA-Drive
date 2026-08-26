from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from research.cf_effect_gate_wote.src.evaluate_probe import evaluate_checkpoint
from research.cf_effect_gate_wote.src.effect_prediction import (
    _train_predictor_trial,
    compare_effect_caches,
    write_predicted_effect_cache,
)
from research.cf_effect_gate_wote.src.feature_store import (
    CacheIdentity,
    FeatureShardReader,
    FeatureShardWriter,
    SceneCacheRecord,
    stable_array_hash,
)
from research.cf_effect_gate_wote.src.train_probe import _train_trial


def _write_frozen_cache(root: Path, split: str, token: str) -> None:
    rng = np.random.default_rng(17)
    trajectory = rng.normal(size=(256, 8, 3)).astype(np.float32)
    factors = rng.uniform(size=(256, 5)).astype(np.float32)
    factors[:, [0, 1, 3]] = (factors[:, [0, 1, 3]] > 0.25).astype(np.float32)
    identity = CacheIdentity(
        run_id=f"probe-{split}",
        split=split,
        checkpoint_sha256="a" * 64,
        wote_commit_sha="b" * 40,
    )
    record = SceneCacheRecord(
        scene_token=token,
        candidate_indices=tuple(range(256)),
        trajectory_hash=stable_array_hash(trajectory),
        label_hash=stable_array_hash(factors),
    )
    writer = FeatureShardWriter(root, identity)
    writer.write_shard(
        0,
        {
            "current_bev_tokens": rng.normal(size=(1, 64, 256)).astype(np.float32),
            "current_bev_pool": rng.normal(size=(1, 256)).astype(np.float32),
            "ego_status_feature": rng.normal(size=(1, 8)).astype(np.float32),
            "trajectory": trajectory[None],
            "factor_labels": factors[None],
            "selected_index": np.asarray([0], dtype=np.int64),
        },
        (record,),
    )
    writer.finalize()


def test_probe_train_eval_contract(tmp_path: Path) -> None:
    train = tmp_path / "train"
    val = tmp_path / "val"
    test = tmp_path / "test"
    _write_frozen_cache(train, "train", "train-token")
    _write_frozen_cache(val, "val", "val-token")
    _write_frozen_cache(test, "test", "test-token")
    checkpoint = tmp_path / "probe.pt"
    metadata = _train_trial(
        train_cache=train,
        val_cache=val,
        train_effects=None,
        val_effects=None,
        model_type="direct_current",
        seed=0,
        learning_rate=3.0e-4,
        pairwise_weight=0.5,
        hidden_dim=32,
        max_epochs=1,
        patience=1,
        batch_scenes=1,
        train_candidates=16,
        device=torch.device("cpu"),
        output=checkpoint,
    )
    result = evaluate_checkpoint(
        checkpoint,
        test,
        effect_cache=None,
        batch_scenes=1,
        device=torch.device("cpu"),
    )
    assert metadata["best_epoch"] == 0
    assert result.tokens == ("test-token",)
    assert result.predicted_scores.shape == (1, 256)
    assert result.predicted_factors.shape == (1, 256, 5)
    assert np.isfinite(result.predicted_scores).all()


def _write_effect_cache(root: Path, frozen_root: Path) -> None:
    reader = FeatureShardReader(frozen_root)
    sidecar, frozen = next(reader.iter_shards())
    record_data = sidecar["records"][0]
    rng = np.random.default_rng(29)
    actor_mask = rng.uniform(size=(1, 256, 8, 16)) > 0.2
    interaction = (rng.uniform(size=(1, 256, 8, 16)) > 0.8) & actor_mask
    record = SceneCacheRecord(
        scene_token=record_data["scene_token"],
        candidate_indices=tuple(record_data["candidate_indices"]),
        trajectory_hash=record_data["trajectory_hash"],
        label_hash=record_data["label_hash"],
    )
    identity_data = reader.manifest["identity"]
    identity = CacheIdentity(
        run_id=f"{identity_data['run_id']}-effects",
        split=identity_data["split"],
        checkpoint_sha256=identity_data["checkpoint_sha256"],
        wote_commit_sha=identity_data["wote_commit_sha"],
        feature_schema_version="replay_effect.v1",
    )
    writer = FeatureShardWriter(root, identity)
    writer.write_shard(
        0,
        {
            "ego_effect": rng.normal(size=(1, 256, 8, 16)).astype(np.float32),
            "map_effect": rng.normal(size=(1, 256, 8, 8)).astype(np.float32),
            "actor_effect": rng.normal(size=(1, 256, 8, 16, 13)).astype(np.float32),
            "actor_mask": actor_mask,
            "interaction_mask": interaction,
            "shared_logged_future": rng.normal(size=(1, 8, 16, 8)).astype(np.float32),
            "shared_actor_mask": np.ones((1, 8, 16), dtype=bool),
        },
        (record,),
    )
    writer.finalize()


def test_forward_effect_train_cache_contract(tmp_path: Path) -> None:
    train = tmp_path / "forward-train"
    val = tmp_path / "forward-val"
    test = tmp_path / "forward-test"
    _write_frozen_cache(train, "train", "forward-train-token")
    _write_frozen_cache(val, "val", "forward-val-token")
    _write_frozen_cache(test, "test", "forward-test-token")
    train_effects = tmp_path / "forward-train-effects"
    val_effects = tmp_path / "forward-val-effects"
    test_effects = tmp_path / "forward-test-effects"
    _write_effect_cache(train_effects, train)
    _write_effect_cache(val_effects, val)
    _write_effect_cache(test_effects, test)
    checkpoint = tmp_path / "effect-predictor.pt"
    config = {
        "hidden_dim": 64,
        "decoder_layers": 2,
        "attention_heads": 8,
        "actor_slots": 16,
        "max_parameters": 10_000_000,
        "max_epochs": 1,
        "patience": 1,
        "batch_scenes": 1,
        "candidate_chunk": 16,
        "train_candidates_per_scene": 16,
        "temporal_consistency_weight": 0.1,
    }
    trial = _train_predictor_trial(
        train_cache=train,
        val_cache=val,
        train_effects=train_effects,
        val_effects=val_effects,
        seed=0,
        learning_rate=3.0e-4,
        config=config,
        device=torch.device("cpu"),
        output=checkpoint,
    )
    predicted = tmp_path / "predicted-test-effects"
    write_predicted_effect_cache(
        checkpoint=checkpoint,
        frozen_root=test,
        oracle_effect_root=test_effects,
        output=predicted,
        device=torch.device("cpu"),
        candidate_chunk=16,
    )
    metrics = compare_effect_caches(test_effects, predicted, seed=0, split="test")
    assert trial["trainable_parameters"] < 10_000_000
    assert metrics["scene_count"] == 1
    assert np.isfinite(metrics["ego_effect_mae"])
