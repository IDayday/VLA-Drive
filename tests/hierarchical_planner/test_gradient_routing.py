import torch

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.scene_encoder import GlobalSceneQFormer
from starVLA.model.modules.trajectory_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
    DrivoRDynamicScorer,
    HierarchicalDrivoRSuprimScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS


def _modules(detach_scene=False):
    scene = GlobalSceneQFormer(
        input_dim=12,
        hidden_dim=64,
        output_dim=64,
        num_queries=4,
        num_layers=1,
        num_heads=4,
        ffn_dim=128,
    )
    dynamic = DrivoRDynamicScorer(
        scene_dim=64,
        ego_state_dim=4,
        model_dim=32,
        ffn_dim=64,
        num_layers=1,
        num_heads=4,
    )
    coarse = DriveSuprimCoarseScorer(
        static_vocab=torch.randn(16, 40, 3),
        vocab_size=16,
        scene_dim=64,
        ego_state_dim=4,
        model_dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=1,
        coarse_topk=4,
    )
    fine = DriveSuprimFineRefiner(
        scene_dim=64,
        model_dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=2,
    )
    hierarchy = HierarchicalDrivoRSuprimScorer(
        dynamic,
        coarse,
        fine,
        detach_scene_for_scorer=detach_scene,
    )
    return scene, hierarchy


def _targets(batch, candidates):
    result = {name: torch.rand(batch, candidates) for name in SUPRIM_METRICS}
    result["comfort"] = torch.rand(batch, candidates)
    result["aggregate_score"] = torch.rand(batch, candidates)
    return result


def test_full_hierarchy_detaches_proposals_and_routes_scene_gradients():
    scene_encoder, hierarchy = _modules()
    context = scene_encoder(
        torch.randn(2, 6, 12), torch.ones(2, 6, dtype=torch.bool)
    )
    proposals = torch.randn(2, 4, 8, 3, requires_grad=True)
    result = hierarchy.forward_full(
        dynamic_proposals_8=proposals,
        dynamic_targets=_targets(2, 4),
        global_scene_tokens=context.global_tokens,
        dense_scene_memory=context.dense_memory,
        memory_key_padding_mask=context.memory_key_padding_mask,
        ego_state=torch.randn(2, 1, 4),
        gt_trajectory_8=torch.randn(2, 8, 3),
        static_targets=_targets(2, 16),
        dynamic_topm=2,
    )
    total = sum(result["losses"].values())
    total.backward()
    assert proposals.grad is None
    assert scene_encoder.input_proj.weight.grad is not None
    assert hierarchy.dynamic_prescorer.trajectory_embedding[0].weight.grad is not None
    assert hierarchy.joint_coarse_scorer.candidate_embedding[0].weight.grad is not None
    assert hierarchy.joint_fine_refiner.fine_decoder.layers[0].cross_attn.k_proj_weight.grad is not None


def test_detach_scene_for_scorer_blocks_only_scene_path():
    scene_encoder, hierarchy = _modules(detach_scene=True)
    context = scene_encoder(
        torch.randn(1, 5, 12), torch.ones(1, 5, dtype=torch.bool)
    )
    proposals = torch.randn(1, 4, 8, 3, requires_grad=True)
    result = hierarchy.forward_full(
        dynamic_proposals_8=proposals,
        dynamic_targets=_targets(1, 4),
        global_scene_tokens=context.global_tokens,
        dense_scene_memory=context.dense_memory,
        memory_key_padding_mask=context.memory_key_padding_mask,
        ego_state=torch.randn(1, 1, 4),
        gt_trajectory_8=torch.randn(1, 8, 3),
        static_targets=_targets(1, 16),
        dynamic_topm=2,
    )
    sum(result["losses"].values()).backward()
    assert proposals.grad is None
    assert scene_encoder.input_proj.weight.grad is None
    assert hierarchy.dynamic_prescorer.trajectory_embedding[0].weight.grad is not None


def test_global_scene_tokens_fine_memory_ablation():
    scene_encoder, hierarchy = _modules()
    hierarchy.fine_memory_source = "global_scene_tokens"
    context = scene_encoder(
        torch.randn(1, 6, 12), torch.ones(1, 6, dtype=torch.bool)
    )
    result = hierarchy.forward_static_only(
        global_scene_tokens=context.global_tokens,
        dense_scene_memory=context.dense_memory,
        memory_key_padding_mask=context.memory_key_padding_mask,
        ego_state=torch.randn(1, 1, 4),
        gt_trajectory_8=torch.randn(1, 8, 3),
        static_targets=_targets(1, 16),
    )
    assert result["outputs"]["selected_trajectory_40"].shape == (1, 40, 3)
