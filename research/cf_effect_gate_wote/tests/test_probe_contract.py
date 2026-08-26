from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from research.cf_effect_gate_wote.src.evaluate_probe import evaluate_checkpoint
from research.cf_effect_gate_wote.src.feature_store import (
    CacheIdentity,
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
