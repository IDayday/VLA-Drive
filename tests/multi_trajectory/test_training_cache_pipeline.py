import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.multi_trajectory.cache_schema import (
    CandidateCacheManifest,
    ProposalCacheRecord,
    SUPRIM_METRICS,
    TRAINING_METRIC_SCHEMA,
    TrainingCacheRecord,
    load_training_record,
    mark_training_cache_complete,
    save_proposal_record,
    save_training_record,
    write_manifest,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_resampler import (
    trajectory_8_to_40,
)
from starVLA.model.modules.action_model.multi_trajectory.candidate_types import (
    DynamicScorerOutput,
)
from starVLA.model.modules.action_model.multi_trajectory.planner import (
    _resolve_coarse_score_targets,
)
from starVLA.training.train_starvla import _save_ddp_drs_component_checkpoints
from tools.generate_ddp_drs_training_cache import (
    _normalize_precomputed_static_scores,
)


def _cache(tmp_path):
    manifest = CandidateCacheManifest(
        split="train",
        ddp_checkpoint_sha="d" * 64,
        repository_commit_sha="r" * 40,
        generator_config_hash="g" * 64,
        seed=3047,
        metric_schema=tuple(TRAINING_METRIC_SCHEMA),
        label_source_split="train",
    )
    write_manifest(tmp_path, manifest)
    trajectory_8 = torch.zeros(64, 8, 3)
    trajectory_8[..., 0] = torch.arange(1, 9, dtype=torch.float32)
    trajectory_40 = trajectory_8_to_40(trajectory_8).numpy()
    proposal = ProposalCacheRecord(
        token="token-a",
        split="train",
        ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
        repository_commit_sha=manifest.repository_commit_sha,
        generator_config_hash=manifest.generator_config_hash,
        seed=manifest.seed,
        candidate_ids=np.arange(64, dtype=np.int64),
        trajectory_8=trajectory_8.numpy(),
        trajectory_40=trajectory_40,
        target_trajectory_8=np.zeros((8, 3), dtype=np.float32),
    )
    save_proposal_record(tmp_path, proposal, manifest)
    dynamic = {
        name: np.full((64,), index / 10.0, dtype=np.float16)
        for index, name in enumerate(TRAINING_METRIC_SCHEMA)
    }
    static = {
        name: np.full((8192,), index / 10.0, dtype=np.float16)
        for index, name in enumerate(TRAINING_METRIC_SCHEMA)
    }
    record = TrainingCacheRecord(
        proposal=proposal,
        dynamic_metrics=dynamic,
        static_metrics=static,
        dynamic_final_score=np.ones(64, dtype=np.float16),
        static_final_score=np.ones(8192, dtype=np.float16),
    )
    save_training_record(tmp_path, record, manifest)
    return manifest, record


def test_training_cache_requires_completion_marker(tmp_path):
    manifest, _ = _cache(tmp_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        load_training_record(
            tmp_path,
            "token-a",
            expected_split="train",
            expected_ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
        )


def test_training_cache_roundtrip_and_stage_targets(tmp_path):
    manifest, expected = _cache(tmp_path)
    mark_training_cache_complete(tmp_path, tokens=["token-a"])
    actual = load_training_record(
        tmp_path,
        "token-a",
        expected_split="train",
        expected_ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
        expected_generator_config_hash=manifest.generator_config_hash,
    )
    np.testing.assert_array_equal(
        actual.proposal.trajectory_8, expected.proposal.trajectory_8
    )
    drivor = actual.training_targets("train_drivor")
    assert set(drivor["drivor_scores"]) == {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    }
    assert drivor["drivor_scores"]["comfort"].shape == (64,)
    static = actual.training_targets("train_suprim_static")
    assert set(static["coarse_scores"]) == set(SUPRIM_METRICS)
    assert static["coarse_scores"]["ego_progress"].shape == (8192,)
    joint = actual.training_targets("train_suprim_joint")
    assert set(joint["coarse_scores"]) == {"static", "dynamic"}
    assert joint["coarse_scores"]["dynamic"]["ego_progress"].shape == (64,)


def test_training_cache_rejects_generator_mismatch(tmp_path):
    manifest, _ = _cache(tmp_path)
    mark_training_cache_complete(tmp_path, tokens=["token-a"])
    with pytest.raises(ValueError, match="generator config hash mismatch"):
        load_training_record(
            tmp_path,
            "token-a",
            expected_split="train",
            expected_ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
            expected_generator_config_hash="wrong",
        )


def test_joint_cache_labels_follow_dynamic_topk_order():
    topk_indices = torch.tensor([[3, 1], [0, 2]])
    output = DynamicScorerOutput(
        sub_scores={},
        aggregate_score=torch.zeros(2, 4),
        topk_indices=topk_indices,
        topk_trajectories=torch.zeros(2, 2, 8, 3),
        score_states=torch.zeros(2, 4, 256),
    )
    targets = {
        "static": {"metric": torch.tensor([[10.0, 11.0], [20.0, 21.0]])},
        "dynamic": {
            "metric": torch.tensor(
                [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]]
            )
        },
    }
    resolved = _resolve_coarse_score_targets(targets, output)
    torch.testing.assert_close(
        resolved["metric"],
        torch.tensor([[10.0, 11.0, 103.0, 101.0], [20.0, 21.0, 200.0, 202.0]]),
    )


def test_training_stage_exports_strict_component_checkpoints(tmp_path):
    cfg = OmegaConf.create(
        {
            "multi_trajectory": {
                "enabled": True,
                "training_stage": "train_drivor",
                "scene_compressor": {"scene_dim": 2048},
                "planning": {"planning_dim": 256},
            }
        }
    )
    state = {
        "multi_trajectory_planner.scene_compressor.scene_queries": torch.ones(
            1, 16, 2048
        ),
        "multi_trajectory_planner.dynamic_scorer.metric_heads.weight": torch.ones(
            2, 2
        ),
        "action_model.unchanged": torch.ones(1),
    }
    _save_ddp_drs_component_checkpoints(cfg, state, tmp_path)
    scene = torch.load(
        tmp_path / "scene_compressor.pt", map_location="cpu", weights_only=True
    )
    scorer = torch.load(
        tmp_path / "dynamic_scorer.pt", map_location="cpu", weights_only=True
    )
    assert set(scene["state_dict"]) == {"scene_queries"}
    assert set(scorer["state_dict"]) == {"metric_heads.weight"}
    assert scene["ddp_drs_checkpoint"]["scene_dim"] == 2048
    assert scene["ddp_drs_checkpoint"]["planning_dim"] == 256
    assert scene["ddp_drs_checkpoint"]["inference_ready"] is True


def test_official_suprim_static_score_mapping():
    payload = {
        "token-a": {
            name: np.full((8192,), index, dtype=np.float16)
            for index, name in enumerate(SUPRIM_METRICS)
        }
    }
    payload["token-a"]["pdm_score"] = np.ones(8192, dtype=np.float16)
    metrics, final_score = _normalize_precomputed_static_scores(
        "token-a", payload
    )
    assert tuple(metrics) == tuple(TRAINING_METRIC_SCHEMA)
    np.testing.assert_array_equal(metrics["comfort"], metrics["history_comfort"])
    np.testing.assert_array_equal(final_score, np.ones(8192, dtype=np.float16))
