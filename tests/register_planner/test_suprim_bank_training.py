import inspect

import torch
from torch import nn

from starVLA.model.modules.register_planner.selectors import (
    DynamicDriveSuprimSelector,
    HybridDriveSuprimSelector,
    dynamic_coarse_output,
)
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
    DynamicScorerOutput,
)
from starVLA.model.modules.trajectory_scorer.drivesuprim_joint_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
)
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS
from starVLA.training import train_register_suprim
from starVLA.training.train_register_suprim import RegisterSuprimStage


def _dynamic_output(batch=2, candidates=64, topm=32, dim=32):
    indices = torch.arange(topm)[None].expand(batch, -1)
    proposals = torch.randn(batch, candidates, 8, 3)
    return DynamicScorerOutput(
        metric_logits={"metric": torch.randn(batch, candidates)},
        aggregate_score=torch.randn(batch, candidates),
        topm_indices=indices,
        topm_trajectories_8=proposals[:, :topm],
        topm_candidate_states=torch.randn(batch, topm, dim),
    )


def test_dynamic_suprim_uses_top32_only():
    output = dynamic_coarse_output(_dynamic_output())
    assert output.joint_candidates_40.shape == (2, 32, 40, 3)
    assert output.topk_candidate_states.shape == (2, 32, 32)


def test_suprim_training_does_not_instantiate_upstream_or_evaluator():
    source = inspect.getsource(train_register_suprim)
    assert "QwenRegister" not in source
    assert "build_framework" not in source
    assert "RegisterTrajectoryGenerator(" not in source
    assert "DynamicMetricSupervisor" not in source


def _hybrid(vocab_size=16, dynamic=4, topk=4):
    coarse = DriveSuprimCoarseScorer(
        static_vocab=torch.zeros(vocab_size, 40, 3),
        vocab_size=vocab_size,
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_heads=1,
        num_layers=1,
        coarse_topk=topk,
    )
    fine = DriveSuprimFineRefiner(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_heads=1,
        num_layers=2,
    )
    return coarse, fine


def test_hybrid_candidate_count_8224():
    coarse = DriveSuprimCoarseScorer(
        static_vocab=torch.zeros(8192, 40, 3),
        vocab_size=8192,
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_heads=1,
        num_layers=1,
        coarse_topk=256,
    )
    joint, metadata = coarse.build_joint_pool(
        1, torch.zeros(1, 4, 32), torch.zeros(1, 32, 40, 3)
    )
    assert joint.shape[1] == 8224
    assert metadata.source.shape == (1, 8224)


class PassDecoder(nn.Module):
    def forward(self, query, memory):
        return query


def test_hybrid_global_top256():
    coarse = DriveSuprimCoarseScorer(
        static_vocab=torch.zeros(8192, 40, 3),
        vocab_size=8192,
        scene_dim=8,
        model_dim=8,
        ffn_dim=16,
        num_heads=1,
        num_layers=1,
        coarse_topk=256,
    )
    coarse.coarse_decoder = PassDecoder()
    output = coarse(
        torch.zeros(1, 4, 8),
        torch.zeros(1, 4),
        dynamic_trajectories_40=torch.zeros(1, 32, 40, 3),
    )
    assert output.joint_candidates_40.shape[1] == 8224
    assert output.topk_indices.shape == (1, 256)


def test_hybrid_static_dynamic_share_heads():
    coarse, fine = _hybrid()
    assert set(coarse.metric_heads) == {*SUPRIM_METRICS, "imi"}
    assert set(fine.metric_heads) == {*SUPRIM_METRICS, "imi"}
    assert len(fine.metric_heads) == 9


def test_no_source_embedding():
    coarse, fine = _hybrid()
    selector = HybridDriveSuprimSelector(coarse, fine)
    assert not any("source" in name.lower() for name, _ in selector.named_modules())


def _stage(memory_source="global_scene_tokens"):
    drivor = DrivoRDynamicScorer(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
        decoder_style="donor_register",
        proj_drop=0.0,
        drop_path=0.0,
    )
    fine = DriveSuprimFineRefiner(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_heads=1,
        num_layers=2,
    )
    return RegisterSuprimStage(
        mode="dynamic",
        drivor=drivor,
        selector=DynamicDriveSuprimSelector(fine),
        dynamic_topm=4,
        memory_source=memory_source,
    )


def _batch():
    metrics = {name: torch.rand(2, 4) for name in SUPRIM_METRICS}
    metrics["comfort"] = torch.rand(2, 4)
    metrics["aggregate_score"] = torch.rand(2, 4)
    return {
        "token": ["a", "b"],
        "proposals": torch.randn(2, 4, 8, 3),
        "scene_global_tokens": torch.randn(2, 4, 32),
        "ego_state": torch.randn(2, 4),
        "gt_trajectory": torch.randn(2, 8, 3),
        "metrics": metrics,
    }


def test_dense_memory_requires_bank_component():
    stage = _stage(memory_source="dense_scene_memory")
    try:
        stage(_batch())
    except RuntimeError as error:
        assert "include_dense_memory" in str(error)
    else:
        raise AssertionError("missing dense bank component was accepted")


def test_suprim_bank_forward_backward():
    stage = _stage()
    output = stage(_batch())
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert all(parameter.grad is None for parameter in stage.drivor.parameters())
    assert any(parameter.grad is not None for parameter in stage.selector.parameters())
