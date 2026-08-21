from dataclasses import replace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.modules.action_model.multi_trajectory.cache_schema import (
    REQUIRED_METRICS,
    CandidateCacheManifest,
    CandidateCacheRecord,
    load_record,
    save_record,
    write_manifest,
)
from starVLA.model.modules.action_model.multi_trajectory.checkpointing import (
    load_base_checkpoint_strict,
)
from starVLA.model.modules.action_model.multi_trajectory.planner import (
    _load_checkpoint_file,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_resampler import (
    trajectory_8_to_40,
)


class _ExtendedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 2)
        self.multi_trajectory_planner = nn.Sequential(nn.Linear(2, 1))


def _base_state(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("multi_trajectory_planner.")
    }


def test_original_checkpoint_uses_local_strict_compatibility(capsys):
    source = _ExtendedModel()
    target = _ExtendedModel()
    original_state = _base_state(source)
    planner_before = {
        key: value.detach().clone()
        for key, value in target.state_dict().items()
        if key.startswith("multi_trajectory_planner.")
    }

    load_base_checkpoint_strict(target, original_state)

    for key, expected in original_state.items():
        torch.testing.assert_close(target.state_dict()[key], expected)
    for key, expected in planner_before.items():
        torch.testing.assert_close(target.state_dict()[key], expected)
    output = capsys.readouterr().out
    assert "missing base keys: []" in output
    assert "unexpected keys: []" in output
    assert "locally allowed planner keys" in output


def test_original_checkpoint_rejects_non_planner_mismatch():
    model = _ExtendedModel()
    state = _base_state(model)
    state.pop("base.bias")
    state["unrelated.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="does not strictly match"):
        load_base_checkpoint_strict(model, state)


def test_full_ddp_drs_checkpoint_load_is_strict(capsys):
    source = _ExtendedModel()
    target = _ExtendedModel()
    full_state = {
        key: value.detach().clone() for key, value in source.state_dict().items()
    }
    load_base_checkpoint_strict(target, full_state)
    for key, expected in full_state.items():
        torch.testing.assert_close(target.state_dict()[key], expected)
    assert "full checkpoint missing keys: []" in capsys.readouterr().out


def test_strict_inference_rejects_untrained_donor_initializer(tmp_path):
    path = tmp_path / "warmstart.pth"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1)},
            "ddp_drs_checkpoint": {
                "component": "test",
                "scene_dim": 2048,
                "planning_dim": 256,
                "inference_ready": False,
                "requires_training": ["adapter.weight"],
            },
        },
        path,
    )
    with pytest.raises(RuntimeError, match="rejects a donor warm-start"):
        _load_checkpoint_file(str(path), reject_training_initializer=True)
    loaded = _load_checkpoint_file(str(path), reject_training_initializer=False)
    torch.testing.assert_close(loaded["weight"], torch.ones(1))


def test_checkpoint_rejects_256_scene_dim(tmp_path):
    path = tmp_path / "old-scene.pth"
    torch.save(
        {
            "state_dict": {"scene_queries": torch.zeros(1, 16, 256)},
            "ddp_drs_checkpoint": {
                "component": "scene_compressor",
                "scene_dim": 256,
                "planning_dim": 256,
                "inference_ready": False,
                "requires_training": [],
            },
        },
        path,
    )
    with pytest.raises(RuntimeError, match="Old 256-wide scene checkpoints"):
        _load_checkpoint_file(str(path))


def _cache_fixture():
    seed = 17
    generator = torch.Generator().manual_seed(seed)
    trajectory_8 = torch.randn(4, 8, 3, generator=generator)
    trajectory_40 = trajectory_8_to_40(trajectory_8)
    metrics = {
        name: np.linspace(0.1, 0.9, 4, dtype=np.float32)
        for name in REQUIRED_METRICS
    }
    manifest = CandidateCacheManifest(
        split="trainval",
        ddp_checkpoint_sha="ddp-sha",
        repository_commit_sha="repo-sha",
        generator_config_hash="config-sha",
        seed=seed,
        metric_schema=REQUIRED_METRICS,
        label_source_split="trainval",
    )
    record = CandidateCacheRecord(
        token="sample-token",
        split=manifest.split,
        ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
        repository_commit_sha=manifest.repository_commit_sha,
        generator_config_hash=manifest.generator_config_hash,
        seed=manifest.seed,
        candidate_ids=np.arange(4, dtype=np.int64),
        trajectory_8=trajectory_8.numpy(),
        trajectory_40=trajectory_40.numpy(),
        metrics=metrics,
        final_score=np.linspace(0.2, 0.8, 4, dtype=np.float32),
        metric_schema=REQUIRED_METRICS,
    )
    return manifest, record


def test_candidate_cache_round_trip_validates_training_contract(tmp_path):
    manifest, record = _cache_fixture()
    write_manifest(tmp_path, manifest)
    save_record(tmp_path, record, manifest)

    loaded = load_record(
        tmp_path,
        record.token,
        expected_split=manifest.split,
        expected_ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
    )
    assert loaded.token == record.token
    np.testing.assert_array_equal(loaded.candidate_ids, record.candidate_ids)
    np.testing.assert_allclose(loaded.trajectory_8, record.trajectory_8)
    np.testing.assert_allclose(loaded.trajectory_40, record.trajectory_40)
    assert loaded.metric_schema == REQUIRED_METRICS


def test_candidate_cache_rejects_hash_and_split_mismatch(tmp_path):
    manifest, record = _cache_fixture()
    write_manifest(tmp_path, manifest)
    save_record(tmp_path, record, manifest)
    with pytest.raises(ValueError, match="split"):
        load_record(
            tmp_path,
            record.token,
            expected_split="other",
            expected_ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
        )
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_record(
            tmp_path,
            record.token,
            expected_split=manifest.split,
            expected_ddp_checkpoint_sha="wrong",
        )


def test_candidate_cache_rejects_test_label_leakage():
    manifest, _ = _cache_fixture()
    with pytest.raises(ValueError, match="test-label"):
        replace(manifest, label_source_split="navsim/test/metric_cache").validate()


def test_candidate_cache_rejects_time_convention_mismatch():
    manifest, record = _cache_fixture()
    record.trajectory_40 = record.trajectory_40.copy()
    record.trajectory_40[:, 4, 0] += 1.0
    with pytest.raises(ValueError, match="8/40 time convention"):
        record.validate(manifest)


def test_inference_cli_path_override_beats_environment_and_yaml(
    monkeypatch,
):
    import infer

    monkeypatch.setenv("DDP_DRS_SUPRIM_VOCAB", "/environment/vocab.npy")
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer.py",
            "--ckpt_dir",
            "/checkpoint",
            "--datalist_path",
            "/data/list.json",
            "--out_dir",
            "/output",
            "--suprim_vocab_path",
            "/cli path/vocab.npy",
        ],
    )
    args = infer.parse_args()
    assert args.suprim_vocab_path == "/cli path/vocab.npy"

    config = OmegaConf.create(
        {"multi_trajectory": {"suprim": {"vocab_path": "/yaml/vocab.npy"}}}
    )
    infer.apply_ddp_drs_path_overrides(
        config,
        {"multi_trajectory.suprim.vocab_path": args.suprim_vocab_path},
    )
    assert config.multi_trajectory.suprim.vocab_path == "/cli path/vocab.npy"
