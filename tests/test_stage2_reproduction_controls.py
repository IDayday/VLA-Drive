from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.planning.training.stage2_reproduction_sampler import (
    ReferenceGlobalBatchDistributedSampler,
)
from local_stage2.audit_stage2_sampler import audit as audit_sampler_order
from local_stage2.audit_stage2_lr_schedule_signature import _relative_lr
from local_stage2.audit_stage2_public_runtime import _stratified_samples


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


def test_rl_zt3_priority_controls_are_bounded_matched_and_use_authorized_gpus():
    launcher = (
        REPO_ROOT
        / "local_stage2/watch_rl_zt3_and_launch_tf437_control.sh"
    ).read_text()
    assert 'gpu_list="3,5,6,7"' in launcher
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
