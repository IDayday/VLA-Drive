import torch

from starVLA.model.modules.vggt_query.centered_action_geometry_reader import (
    CenteredActionGeometryReader,
)
from starVLA.model.modules.vggt_query.gp_geometry_adapter import GeometryMemoryAdapter
from starVLA.model.modules.vggt_query.gp_geometry_gate import SceneConditionedGeometryGate
from starVLA.model.modules.vggt_query.scene_summary import MaskedSceneSummary
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_geometry_per_view,
)


def _payload():
    features = torch.arange(180 * 2048, dtype=torch.float32).reshape(180, 2048)
    view_ids = torch.arange(3).repeat_interleave(60)
    uv = torch.stack((torch.linspace(-1, 1, 180), torch.linspace(1, -1, 180)), dim=-1)
    directions = torch.tensor([1.0, 2.0, 3.0]).expand(180, -1)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    rays = torch.cat((torch.arange(180).float().unsqueeze(-1).expand(-1, 3), directions), dim=-1)
    return {
        "features": features,
        "valid_mask": torch.ones(180, dtype=torch.bool),
        "view_ids": view_ids,
        "uv_coords": uv,
        "ray_features": rays,
        "patch_grid_hw": torch.tensor([[6, 10], [6, 10], [6, 10]], dtype=torch.int16),
    }


def test_metadata_pooling_shape_order_and_unit_directions():
    pooled = pool_dense_vggt_geometry_per_view([_payload()], dtype=torch.float32)
    assert pooled["features"].shape == (1, 180, 2048)
    assert pooled["view_ids"].shape == (1, 180)
    assert pooled["uv_coords"].shape == (1, 180, 2)
    assert pooled["ray_features"].shape == (1, 180, 6)
    assert pooled["view_ids"][0].tolist() == [0] * 60 + [1] * 60 + [2] * 60
    torch.testing.assert_close(
        pooled["ray_features"][..., 3:].norm(dim=-1), torch.ones(1, 180)
    )


def test_scene_summary_excludes_padding_and_action_tokens():
    module = MaskedSceneSummary(hidden_dim=4, action_query_count=2)
    hidden = torch.tensor([[[1.0, 2, 3, 4], [8.0, 7, 6, 5], [3.0, 1, 4, 2], [99.0] * 4]])
    mask = torch.tensor([[1, 1, 1, 0]])
    summary, metrics = module(hidden, mask, torch.tensor([[1, 2]]))
    expected = torch.nn.functional.layer_norm(hidden[:, :1].float(), (4,))
    torch.testing.assert_close(summary, expected)
    assert metrics["gp_sq3dmix/scene_memory_token_count"] == 1


def test_adapter_output_norm_is_bounded():
    pooled = pool_dense_vggt_geometry_per_view([_payload()], dtype=torch.float32)
    adapter = GeometryMemoryAdapter()
    output = adapter(**pooled)
    assert output.shape == (1, 180, 512)
    norms = output.norm(dim=-1)
    assert torch.isfinite(norms).all()
    assert 20.0 < norms.mean() < 25.0


def test_gate_shape_initialization_and_bounds():
    gate = SceneConditionedGeometryGate()
    output, metrics = gate(torch.randn(2, 1, 2048), torch.randn(2, 180, 512))
    assert output.shape == (2, 180, 512)
    torch.testing.assert_close(metrics["gp_sq3dmix/retention_mean"], torch.tensor(0.10), atol=1e-7, rtol=0)
    assert metrics["gp_sq3dmix/retention_min"] >= 0.05
    assert metrics["gp_sq3dmix/retention_max"] <= 0.50


def test_reader_slot_mean_identity_and_zero_init_baseline():
    reader = CenteredActionGeometryReader()
    actions = torch.randn(1, 8, 2048)
    memory = torch.randn(1, 180, 512)
    identical, _ = reader(actions, memory, memory)
    torch.testing.assert_close(identical, actions, rtol=0, atol=0)
    changed, _ = reader(actions, memory + 1, memory)
    torch.testing.assert_close(changed, actions, rtol=0, atol=0)

    # Identity is permanent, not only a property of zero initialization.
    with torch.no_grad():
        reader.readout_norm.bias.fill_(0.7)
        reader.up_projection.weight.normal_()
        reader.up_projection.bias.fill_(0.3)
    updated, _ = reader(actions, memory, memory)
    torch.testing.assert_close(updated, actions, rtol=0, atol=0)


def test_real_feature_change_affects_enhanced_query_after_reader_is_active():
    reader = CenteredActionGeometryReader()
    torch.nn.init.normal_(reader.up_projection.weight, std=1e-3)
    actions = torch.randn(1, 8, 2048)
    reference = torch.randn(1, 180, 512)
    first, _ = reader(actions, reference + 0.1 * torch.randn_like(reference), reference)
    second, _ = reader(actions, reference + 0.2 * torch.randn_like(reference), reference)
    assert not torch.equal(first, second)


def test_all_new_trainable_modules_receive_gradients_when_reader_active():
    adapter = GeometryMemoryAdapter()
    gate = SceneConditionedGeometryGate()
    reader = CenteredActionGeometryReader()
    torch.nn.init.normal_(reader.up_projection.weight, std=1e-3)
    pooled = pool_dense_vggt_geometry_per_view([_payload()], dtype=torch.float32)
    real = adapter(**pooled)
    reference_payload = dict(pooled)
    reference_payload["features"] = torch.zeros_like(pooled["features"])
    reference = adapter(**reference_payload)
    scene = torch.randn(1, 1, 2048)
    real, _ = gate(scene, real)
    reference, _ = gate(scene, reference)
    enhanced, _ = reader(torch.randn(1, 8, 2048), real, reference)
    enhanced.square().mean().backward()
    assert adapter.feature_projection.weight.grad.norm() > 0
    assert gate.gate_projection.weight.grad.norm() > 0
    assert reader.cross_attention.in_proj_weight.grad.norm() > 0
    assert reader.up_projection.weight.grad.norm() > 0
