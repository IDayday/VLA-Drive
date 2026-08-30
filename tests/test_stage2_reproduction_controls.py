import json
import pickle
import pickletools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.planning.training.stage2_reproduction_sampler import (
    ReferenceGlobalBatchDistributedSampler,
)
from local_stage2.audit_stage2_sampler import audit as audit_sampler_order
from local_stage2.audit_stage2_checkpoint_history import (
    _unpickle_metadata,
    _value_opcode_after_key,
)
from local_stage2.audit_stage2_long_target_integrity import _heading_deltas
from local_stage2.audit_stage2_lr_schedule_signature import _relative_lr
from local_stage2.audit_stage2_scheduler_presence import _loop_progress
from local_stage2.compare_stage2_proposal_artifacts import (
    _grouped_bootstrap_ci,
)
from local_stage2.audit_stage2_public_runtime import (
    _compose_config,
    _stratified_samples,
)
from local_stage2.snapshot_validation_milestone import _snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bare_agent(frozen_backbone_mode="eval"):
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    torch.nn.Module.__init__(agent)
    agent.backbone = torch.nn.Sequential(torch.nn.Dropout(p=0.5))
    agent.action_head = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )
    agent.vlm_config = SimpleNamespace(
        freeze_backbone=True,
        frozen_backbone_mode=frozen_backbone_mode,
    )
    agent.batch_size = 2
    agent.num_gpus = 8
    agent.scheduler_args = None
    return agent


def test_frozen_backbone_eval_mode_is_explicit():
    agent = _bare_agent("eval")
    agent.train(True)
    assert agent.action_head.training
    assert not agent.backbone.training
    assert all(parameter.requires_grad for parameter in agent.action_head.parameters())


def test_frozen_backbone_train_mode_reproduces_module_semantics():
    agent = _bare_agent("train")
    for parameter in agent.backbone.parameters():
        parameter.requires_grad = False
    agent.train(True)
    assert agent.action_head.training
    assert agent.backbone.training
    assert not any(parameter.requires_grad for parameter in agent.backbone.parameters())


def test_invalid_frozen_backbone_mode_fails_closed():
    agent = _bare_agent("ambiguous")
    with pytest.raises(ValueError, match="frozen_backbone_mode"):
        agent.train(True)


def test_multinode_launcher_locks_transformers_on_both_nodes():
    launcher = (
        REPO_ROOT / "local_stage2/launch_stage2_multinode_reproduction.sh"
    ).read_text()
    trainer = (REPO_ROOT / "local_stage2/train_stage2_full.sh").read_text()
    assert "STAGE2_TRANSFORMERS_OVERLAY" in launcher
    assert "STAGE2_REQUIRE_TRANSFORMERS_VERSION" in launcher
    assert '"transformers":transformers.__version__' in launcher
    assert "Expected the locked Transformers" in launcher
    assert "STAGE2_REQUIRE_TRANSFORMERS_VERSION" in trainer
    assert "import transformers" in trainer
    assert '"STAGE2_WORLD_SIZE=16"' in launcher
    assert '"agent.num_gpus=${STAGE2_WORLD_SIZE}"' in trainer


def test_long_target_interpolation_reaches_extra_logged_horizon():
    from local_stage2.build_stage2_long_target_cache import build_long_trajectory

    logged = np.zeros((10, 3), dtype=np.float64)
    logged[:, 0] = np.arange(1, 11, dtype=np.float64)
    long_target = build_long_trajectory(logged, num_poses=8, additional_poses=2)

    assert long_target.shape == (8, 3)
    assert long_target[-1, 0] == pytest.approx(10.0)
    assert np.all(np.diff(long_target[:, 0]) > 0)


def test_long_target_integrity_audit_wraps_heading_deltas():
    trajectory = np.zeros((8, 3), dtype=np.float64)
    trajectory[:, 2] = [3.0, 3.1, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6]
    raw, wrapped = _heading_deltas(trajectory)
    assert np.max(np.abs(raw)) > np.pi
    assert np.max(np.abs(wrapped)) < 0.2


def test_checkpoint_audit_distinguishes_stripped_training_state():
    payload = pickle.dumps(
        {"optimizer_states": [], "lr_schedulers": []}, protocol=2
    )
    operations = list(pickletools.genops(payload))
    assert _value_opcode_after_key(operations, "optimizer_states") == "EMPTY_LIST"
    assert _value_opcode_after_key(operations, "lr_schedulers") == "EMPTY_LIST"


def test_checkpoint_audit_decodes_loop_progress_without_tensor_storage():
    checkpoint = {
        "loops": {
            "fit_loop": {
                "epoch_loop.scheduler_progress": {
                    "total": {"ready": 12, "completed": 12},
                    "current": {"ready": 4, "completed": 4},
                },
                "epoch_loop.automatic_optimization.optim_progress": {
                    "optimizer": {
                        "step": {
                            "total": {"ready": 12, "completed": 12},
                            "current": {"ready": 4, "completed": 4},
                        }
                    }
                },
            }
        }
    }
    decoded = _unpickle_metadata(pickle.dumps(checkpoint, protocol=2))
    progress = decoded["loops"]["fit_loop"]["epoch_loop.scheduler_progress"]
    assert progress["total"]["completed"] == 12


def test_scheduler_presence_audit_reads_lightning_loop_state(tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            "global_step": 12,
            "lr_schedulers": [{"last_epoch": 12}],
            "loops": {
                "fit_loop": {
                    "epoch_loop.scheduler_progress": {
                        "total": {"ready": 12, "completed": 12},
                        "current": {"ready": 4, "completed": 4},
                    },
                    "epoch_loop.automatic_optimization.optim_progress": {
                        "optimizer": {
                            "step": {
                                "total": {"ready": 12, "completed": 12},
                                "current": {"ready": 4, "completed": 4},
                            }
                        }
                    },
                }
            },
        },
        path,
    )
    report = _loop_progress(path)
    assert report["global_step"] == 12
    assert report["scheduler_progress"]["total"]["completed"] == 12
    assert report["saved_scheduler_state_count"] == 1


def test_generic_stage2_launcher_retains_compatibility_default():
    launcher = (REPO_ROOT / "local_stage2/train_stage2_full.sh").read_text()
    assert (
        'STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES="${STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES:--1}"'
        in launcher
    )
    assert (
        '"agent.action_head_config.long_trajectory_additional_poses=${STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES}"'
        in launcher
    )


def test_reproduction_entry_defaults_to_recovered_long2_recipe():
    common = (REPO_ROOT / "local_stage2/common.sh").read_text()
    reproduction = (
        REPO_ROOT / "local_stage2/train_stage2_reproduction.sh"
    ).read_text()
    multinode = (
        REPO_ROOT / "local_stage2/launch_stage2_multinode_reproduction.sh"
    ).read_text()

    assert "DRIVEVLA_NAVTRAIN_LONG2_FEATURE_CACHE" in common
    assert (
        'STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES="${STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES:-2}"'
        in reproduction
    )
    assert 'STAGE2_SCHEDULER="${STAGE2_SCHEDULER:-source_cosine}"' in reproduction
    assert (
        'STAGE2_REQUIRE_LIGHTNING_VERSION="${STAGE2_REQUIRE_LIGHTNING_VERSION:-2.2.1}"'
        in reproduction
    )
    assert (
        'STAGE2_REQUIRE_TRANSFORMERS_VERSION="${STAGE2_REQUIRE_TRANSFORMERS_VERSION:-4.48.3}"'
        in reproduction
    )
    assert '"STAGE2_SCHEDULER=${STAGE2_SCHEDULER:-source_cosine}"' in multinode
    assert (
        '"STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES=${STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES:-2}"'
        in multinode
    )


def test_paired_artifact_bootstrap_is_log_grouped_and_deterministic():
    differences = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    logs = ["log_a", "log_a", "log_b", "log_b"]
    first = _grouped_bootstrap_ci(
        differences, logs, seed=7, samples=5_000
    )
    second = _grouped_bootstrap_ci(
        differences, logs, seed=7, samples=5_000
    )
    assert first == second
    assert first[0] == pytest.approx(-1.0)
    assert first[1] == pytest.approx(1.0)


def test_rl_zt3_priority_controls_are_bounded_matched_and_use_authorized_gpus():
    launcher = (
        REPO_ROOT
        / "local_stage2/watch_rl_zt3_and_launch_tf437_control.sh"
    ).read_text()
    assert 'gpu_list="3,5,6,7"' in launcher
    assert launcher.count("STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES=-1") == 2
    assert launcher.count("feature_cache_navtrain_full") == 2
    assert "STAGE2_NUM_GPUS=4" in launcher
    assert "STAGE2_BATCH_SIZE=1" in launcher
    assert "STAGE2_ACCUMULATE_GRAD_BATCHES=4" in launcher
    assert "STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE=16" in launcher
    assert "STAGE2_REQUIRE_TRANSFORMERS_VERSION=4.48.3" in launcher
    assert "4.37.2)" in launcher
    assert (
        '"STAGE2_REQUIRE_TRANSFORMERS_VERSION=${transformers_version}"'
        in launcher
    )
    assert '"peft": "0.10.0"' in launcher
    assert "stage2_source_cosine_seed2_tf448_peft010_4x1_acc4_step1000_rlzt3" in launcher
    assert "stage2_source_cosine_seed2_tf448_peft010_clip1_4x1_acc4_step1000_rlzt3" in launcher
    assert '"trainer.params.gradient_clip_val=${gradient_clip_val}"' in launcher
    assert (
        "launch_followup_control \\\n"
        "  stage2_source_cosine_seed2_tf448_peft010_clip1_4x1_acc4_step1000_rlzt3 \\\n"
        "  1.0 4.48.3"
        in launcher
    )
    assert (
        "launch_followup_control \\\n"
        "  stage2_source_cosine_seed2_tf437_peft010_4x1_acc4_step1000_rlzt3 \\\n"
        "  0.0 4.37.2"
        in launcher
    )
    first = launcher.index(
        'experiment="stage2_source_cosine_seed2_tf448_peft010'
    )
    clip = launcher.index("clip1_4x1")
    tf437 = launcher.rindex("tf437_peft010_4x1")
    assert first < clip < tf437
    assert "trainer.params.limit_train_batches=4000" in launcher
    assert "trainer.params.max_epochs=1" in launcher


def test_public_runtime_subset_round_robins_logs(tmp_path):
    for log_name, tokens in {
        "log_a": ("a0", "a1", "a2"),
        "log_b": ("b0", "b1"),
        "log_c": ("c0",),
    }.items():
        for token in tokens:
            sample_dir = tmp_path / log_name / token
            sample_dir.mkdir(parents=True)
            (sample_dir / "internvl_feature.gz").touch()
            (sample_dir / "trajectory_target.gz").touch()

    selected = _stratified_samples(
        tmp_path, ("log_a", "log_b", "log_c"), count=5
    )
    assert [(path.parent.name, path.name) for path in selected] == [
        ("log_a", "a0"),
        ("log_b", "b0"),
        ("log_c", "c0"),
        ("log_a", "a1"),
        ("log_b", "b1"),
    ]


def test_public_runtime_quotes_checkpoint_paths_for_hydra():
    args = SimpleNamespace(
        checkpoint=Path("/tmp/best-epoch=25-step=167856.ckpt"),
        vlm_path=Path("/tmp/InternVL model"),
        flash_attention=False,
        batch_size=2,
    )
    config = _compose_config(args)
    assert config.agent.checkpoint_path == str(args.checkpoint)
    assert config.agent.vlm_config.vlm_path == str(args.vlm_path)


@pytest.mark.parametrize(
    ("decay_norm_and_bias", "expected_groups", "expected_zero_decay_groups"),
    [(False, 2, 1), (True, 1, 0)],
)
def test_optimizer_decay_semantics_are_selectable(
    decay_norm_and_bias, expected_groups, expected_zero_decay_groups
):
    agent = _bare_agent()
    for parameter in agent.backbone.parameters():
        parameter.requires_grad = False
    agent._lr_args = {
        "name": "AdamW",
        "base_lr": 1e-4,
        "base_batch_size": 16,
        "decay_norm_and_bias": decay_norm_and_bias,
    }
    optimizer = agent.get_optimizers()[0]
    assert len(optimizer.param_groups) == expected_groups
    assert sum(group["weight_decay"] == 0 for group in optimizer.param_groups) == (
        expected_zero_decay_groups
    )
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter in agent.action_head.parameters()
        if parameter.requires_grad
    }
    assert optimized == expected


@pytest.mark.parametrize(
    ("base_lr", "base_batch_size", "effective_batch_size", "expected_lr"),
    [
        (1e-4, 16, 16, 1e-4),
        (1e-4, 64, 16, 5e-5),
        (5e-4, 64, 16, 2.5e-4),
        (5e-4, 64, 2, 5e-4 * (2 / 64) ** 0.5),
    ],
)
def test_optimizer_effective_learning_rate_is_explicit(
    base_lr, base_batch_size, effective_batch_size, expected_lr
):
    agent = _bare_agent()
    for parameter in agent.backbone.parameters():
        parameter.requires_grad = False
    agent._lr_args = {
        "name": "AdamW",
        "base_lr": base_lr,
        "base_batch_size": base_batch_size,
        "effective_global_batch_size": effective_batch_size,
        "decay_norm_and_bias": True,
    }
    optimizer = agent.get_optimizers()[0]
    assert all(
        group["lr"] == pytest.approx(expected_lr)
        for group in optimizer.param_groups
    )


def test_source_cosine_schedule_reaches_peak_then_zero():
    agent = _bare_agent()
    for parameter in agent.backbone.parameters():
        parameter.requires_grad = False
    agent._lr_args = {
        "name": "AdamW",
        "base_lr": 1e-4,
        "base_batch_size": 16,
        "effective_global_batch_size": 16,
        "decay_norm_and_bias": True,
    }
    agent.scheduler_args = OmegaConf.create(
        {
            "dataset_size": 160,
            "num_epochs": 2,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.0,
            "action_head_min_lr_ratio": 0.0,
            "vlm_min_lr_ratio": 0.0,
            "start_lr_ratio": 1e-6,
        }
    )
    optimizers, scheduler_configs = agent.get_optimizers()
    optimizer = optimizers[0]
    scheduler = scheduler_configs[0]["scheduler"]
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-10)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4)
    for _ in range(18):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-12)


def test_lr_signature_uses_the_released_warmup_cosine_shape():
    assert _relative_lr(0, total_steps=20, warmup_steps=2) == pytest.approx(1e-6)
    assert _relative_lr(2, total_steps=20, warmup_steps=2) == pytest.approx(1.0)
    assert _relative_lr(20, total_steps=20, warmup_steps=2) == pytest.approx(0.0)


def test_source_cosine_matches_released_sequential_lr_inside_horizon():
    """The LambdaLR rewrite must preserve the released dormant branch."""
    agent = _bare_agent()
    for parameter in agent.backbone.parameters():
        parameter.requires_grad = False
    agent._lr_args = {
        "name": "AdamW",
        "base_lr": 1e-4,
        "base_batch_size": 16,
        "effective_global_batch_size": 16,
        "decay_norm_and_bias": True,
    }
    agent.scheduler_args = OmegaConf.create(
        {
            "dataset_size": 160,
            "num_epochs": 2,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.0,
            "action_head_min_lr_ratio": 0.0,
            "vlm_min_lr_ratio": 0.0,
            "start_lr_ratio": 1e-6,
        }
    )
    optimizers, scheduler_configs = agent.get_optimizers()
    optimizer = optimizers[0]
    scheduler = scheduler_configs[0]["scheduler"]

    released_parameter = torch.nn.Parameter(torch.ones(()))
    released_optimizer = torch.optim.AdamW(
        [released_parameter], lr=1e-4
    )
    released_warmup = torch.optim.lr_scheduler.LinearLR(
        released_optimizer, start_factor=1e-6, total_iters=2
    )
    released_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        released_optimizer, T_max=18, eta_min=0.0, last_epoch=-1
    )
    released_scheduler = torch.optim.lr_scheduler.SequentialLR(
        released_optimizer,
        schedulers=[released_warmup, released_cosine],
        milestones=[2],
    )

    assert scheduler.get_last_lr()[0] == pytest.approx(
        released_scheduler.get_last_lr()[0]
    )
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        released_optimizer.step()
        released_scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(
            released_scheduler.get_last_lr()[0], abs=1e-12
        )


def _global_batches(
    dataset_size,
    world_size,
    local_batch,
    epoch,
    shuffle=True,
    gradient_accumulation_steps=1,
):
    dataset = TensorDataset(torch.arange(dataset_size))
    rank_sequences = []
    for rank in range(world_size):
        sampler = ReferenceGlobalBatchDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            per_rank_batch_size=local_batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            reference_global_batch_size=16,
            shuffle=shuffle,
            seed=0,
        )
        sampler.set_epoch(epoch)
        sequence = list(sampler)
        samples_per_optimizer_step = local_batch * gradient_accumulation_steps
        rank_sequences.append([
            sequence[offset : offset + samples_per_optimizer_step]
            for offset in range(0, len(sequence), samples_per_optimizer_step)
        ])
    return [
        [item for rank in rank_sequences for item in rank[step]]
        for step in range(len(rank_sequences[0]))
    ]


@pytest.mark.parametrize(("world_size", "local_batch"), [(16, 1), (8, 2), (4, 4)])
def test_reference_sampler_has_expected_shape(world_size, local_batch):
    batches = _global_batches(103_288, world_size, local_batch, epoch=0)
    assert len(batches) == 6_456
    assert all(len(batch) == 16 for batch in batches)


def test_eight_by_two_replays_sixteen_by_one_exactly():
    for epoch in (0, 1, 26):
        reference = _global_batches(103_288, 16, 1, epoch)
        reproduced = _global_batches(103_288, 8, 2, epoch)
        assert [sorted(batch) for batch in reproduced] == [
            sorted(batch) for batch in reference
        ]


def test_eight_by_one_accumulate_two_replays_sixteen_by_one_exactly():
    for epoch in (0, 1, 26):
        reference = _global_batches(103_288, 16, 1, epoch)
        reproduced = _global_batches(
            103_288,
            8,
            1,
            epoch,
            gradient_accumulation_steps=2,
        )
        assert [sorted(batch) for batch in reproduced] == [
            sorted(batch) for batch in reference
        ]


def test_four_by_one_accumulate_four_replays_sixteen_by_one_exactly():
    for epoch in (0, 1, 26):
        reference = _global_batches(103_288, 16, 1, epoch)
        reproduced = _global_batches(
            103_288,
            4,
            1,
            epoch,
            gradient_accumulation_steps=4,
        )
        assert [sorted(batch) for batch in reproduced] == [
            sorted(batch) for batch in reference
        ]


def test_reference_sampler_pads_after_shuffle():
    batches = _global_batches(103_288, 16, 1, epoch=0)
    flat = [item for batch in batches for item in batch]
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(103_288, generator=generator).tolist()
    assert flat == permutation + permutation[:8]


def test_reference_sampler_validation_order_is_stable():
    batches = _global_batches(18_179, 8, 2, epoch=7, shuffle=False)
    flat = [item for batch in batches for item in batch]
    assert sorted(flat) == sorted(list(range(18_179)) + list(range(13)))
    assert [sorted(batch) for batch in batches] == [
        sorted(
            list(range(offset, min(offset + 16, 18_179)))
            + list(range(max(0, offset + 16 - 18_179)))
        )
        for offset in range(0, 18_192, 16)
    ]


def test_reference_sampler_rejects_global_batch_mismatch():
    dataset = TensorDataset(torch.arange(32))
    with pytest.raises(ValueError, match="Current effective global batch"):
        ReferenceGlobalBatchDistributedSampler(
            dataset,
            num_replicas=8,
            rank=0,
            per_rank_batch_size=1,
            reference_global_batch_size=16,
        )


def test_legacy_prepad_changes_the_complete_training_order():
    report = audit_sampler_order(103_288, 16, 0)
    assert report["padding_samples"] == 8
    assert report["optimizer_steps"] == 6_456
    assert report["exact_same_global_batch_count"] == 0
    assert report["same_position_fraction"] < 0.001
    assert report["mean_batch_member_overlap_fraction"] < 0.001


def test_corrected_long2_epoch0_closes_public_proposal_ceiling_gap():
    result = json.loads(
        (
            REPO_ROOT
            / "reports/stage2_reproduction_diagnosis/corrected_long2_epoch0_result.json"
        ).read_text()
    )
    current = result["validation"]["best_of_64_pdms"]
    no_long = result["matched_schedule_no_long_epoch0"]["best_of_64_pdms"]
    public = result["public_final_reference"]["best_of_64_pdms"]
    recovery = (current - no_long) / (public - no_long)
    assert current > no_long
    assert recovery == pytest.approx(
        result["proposal_ceiling_gap_to_public"][
            "fraction_of_no_long_gap_recovered"
        ]
    )
    assert recovery > 0.90


def test_official_stage2_target_is_no_memory_base_not_retrieval_score():
    result = json.loads(
        (
            REPO_ROOT
            / "reports/stage2_reproduction_diagnosis/official_benchmark_disambiguation.json"
        ).read_text()
    )

    paper = result["paper"]
    base = paper["navsim_v1_table_3"][
        "base_model_without_memory_pdms_percent"
    ]
    retrieval = paper["navsim_v1_table_3"][
        "map_and_agent_retrieval_pdms_percent"
    ]
    scale = paper["navsim_v1_table_1"][
        "drivevla_m0_scale_with_10k_synthetic_memory_pdms_percent"
    ]
    local_public = result["locally_evaluated_modelscope_checkpoint"]

    assert base == 91.0
    assert retrieval == 92.3
    assert scale == 94.1
    assert base < retrieval < scale
    assert abs(local_public["navtest_pdms"] - base / 100.0) < 5e-4
    assert "approximately 0.910" in result["conclusion"]


def test_reproduction_scheduler_matches_released_schedule_numerically():
    result = json.loads(
        (
            REPO_ROOT
            / "reports/stage2_reproduction_diagnosis/scheduler_implementation_equivalence.json"
        ).read_text()
    )
    assert result["total_steps"] == 174_312
    assert result["warmup_steps"] == 17_431
    assert result["max_absolute_lr_difference"] < 1e-16


def test_milestone_snapshot_preserves_retained_best_without_copy(tmp_path):
    source = tmp_path / "best-epoch=9-step=64560.ckpt"
    source.write_bytes(b"checkpoint")
    last = tmp_path / "last.ckpt"
    last.symlink_to(source.name)
    destination = tmp_path / "milestones" / "epoch9.ckpt"

    resolved, method = _snapshot(last, destination)

    assert resolved == source.resolve()
    assert method == "hardlink_retained_best"
    assert destination.read_bytes() == b"checkpoint"
    assert destination.stat().st_ino == source.stat().st_ino


def test_milestone_snapshot_copies_mutable_last(tmp_path):
    last = tmp_path / "last.ckpt"
    last.write_bytes(b"checkpoint")
    destination = tmp_path / "milestones" / "epoch9.ckpt"

    resolved, method = _snapshot(last, destination)

    assert resolved == last.resolve()
    assert method == "reflink_or_copy_mutable_last"
    assert destination.read_bytes() == b"checkpoint"
    assert destination.stat().st_ino != last.stat().st_ino
