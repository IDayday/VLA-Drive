import torch
from torch import nn

from starVLA.model.modules.vggt_query.sq_3d_mix import (
    SceneConditionedGatedFusion,
)


def test_gated_fusion_matches_manual_equations():
    torch.manual_seed(3)
    module = SceneConditionedGatedFusion(scene_dim=4, vggt_dim=3)
    scene = torch.randn(2, 5, 4)
    vggt = torch.randn(2, 7, 3)

    actual, diagnostics = module(scene, vggt)
    scene_summary = scene.mean(dim=1, keepdim=True)
    geometry = module.vggt_projection(vggt)
    semantic = scene_summary.expand(-1, geometry.shape[1], -1)
    gate = torch.sigmoid(module.gate_projection(torch.cat([semantic, geometry], -1)))
    expected = gate * module.semantic_projection(semantic) + (
        1.0 - gate
    ) * module.geometry_projection(geometry)

    torch.testing.assert_close(actual, expected)
    assert all(not value.requires_grad for value in diagnostics.values())


def test_gate_is_position_and_channel_wise():
    module = SceneConditionedGatedFusion(scene_dim=6, vggt_dim=4)
    scene = torch.randn(2, 16, 6)
    vggt = torch.randn(2, 180, 4)
    scene_summary = scene.mean(dim=1, keepdim=True)
    geometry = module.vggt_projection(vggt)
    semantic = scene_summary.expand(-1, geometry.shape[1], -1)
    gate = torch.sigmoid(module.gate_projection(torch.cat([semantic, geometry], -1)))

    assert gate.shape == (2, 180, 6)


def test_no_forbidden_gated_fusion_modules():
    module = SceneConditionedGatedFusion(scene_dim=8, vggt_dim=5)
    descendants = list(module.modules())[1:]

    assert len(descendants) == 4
    assert all(isinstance(child, nn.Linear) for child in descendants)
    assert not any(
        isinstance(child, (nn.LayerNorm, nn.MultiheadAttention, nn.Embedding))
        for child in descendants
    )


def test_all_four_fusion_layers_receive_gradient():
    torch.manual_seed(11)
    module = SceneConditionedGatedFusion(scene_dim=8, vggt_dim=5)
    scene = torch.randn(2, 16, 8)
    vggt = torch.randn(2, 180, 5)

    fused, _ = module(scene, vggt)
    fused.square().mean().backward()

    for layer in (
        module.vggt_projection,
        module.gate_projection,
        module.semantic_projection,
        module.geometry_projection,
    ):
        assert layer.weight.grad is not None
        assert layer.weight.grad.detach().abs().sum() > 0
