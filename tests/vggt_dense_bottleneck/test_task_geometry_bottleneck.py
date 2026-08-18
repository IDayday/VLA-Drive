import torch

from starVLA.model.modules.vggt_query.planning_heads import AuxiliaryTrajectoryHead
from starVLA.model.modules.vggt_query.task_geometry_bottleneck import (
    PlanningConditionedDenseVGGTBottleneck,
)


def _inputs(batch=2, tokens=17, qwen_dim=32, source_dim=24):
    torch.manual_seed(20260818)
    planning = torch.randn(batch, 8, qwen_dim)
    source = torch.randn(batch, tokens, source_dim)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    view_ids = torch.arange(tokens).remainder(3).repeat(batch, 1)
    uv = torch.rand(batch, tokens, 2).mul(2).sub(1)
    rays = torch.randn(batch, tokens, 6)
    rays[..., 3:] = torch.nn.functional.normalize(rays[..., 3:], dim=-1)
    return planning, source, mask, view_ids, uv, rays


def _module():
    return PlanningConditionedDenseVGGTBottleneck(
        planning_dim=32,
        source_dim=24,
        bottleneck_dim=16,
        expected_horizons=8,
        slots_per_horizon=4,
        num_heads=4,
        ffn_expansion=2,
        detach_planning_queries=True,
        attention_dropout=0.0,
    )


def test_bottleneck_shapes_and_strict_identity_initialization():
    module = _module().eval()
    inputs = _inputs()
    enhanced, readout, task_tokens, diagnostics = module(*inputs)

    assert enhanced.shape == (2, 8, 32)
    assert readout.shape == (2, 8, 16)
    assert task_tokens.shape == (2, 32, 16)
    assert diagnostics["source_token_count_mean"].ndim == 0
    torch.testing.assert_close(enhanced, inputs[0], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(module.up_projection.weight) == 0
    assert torch.count_nonzero(module.up_projection.bias) == 0


def test_masked_source_tokens_do_not_affect_attention_output():
    module = _module().eval()
    planning, source, mask, view_ids, uv, rays = _inputs(tokens=19)
    mask[:, -3:] = False
    changed = source.clone()
    changed[:, -3:] = 100000.0
    first = module(planning, source, mask, view_ids, uv, rays)[1:3]
    second = module(planning, changed, mask, view_ids, uv, rays)[1:3]
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_attention_output_has_no_planning_query_residual():
    module = _module().eval()
    planning, source, mask, view_ids, uv, rays = _inputs()
    with torch.no_grad():
        module.source_projection.weight.zero_()
        module.view_embedding.weight.zero_()
        for mlp in (module.uv_mlp, module.ray_mlp):
            for parameter in mlp.parameters():
                parameter.zero_()
        module.cross_attention.in_proj_bias.zero_()
        module.cross_attention.out_proj.bias.zero_()
        for parameter in module.ffn.parameters():
            parameter.zero_()

    task_a = module(planning, source, mask, view_ids, uv, rays)[2]
    task_b = module(planning + 5.0, source, mask, view_ids, uv, rays)[2]
    torch.testing.assert_close(task_a, torch.zeros_like(task_a), atol=0.0, rtol=0.0)
    torch.testing.assert_close(task_b, torch.zeros_like(task_b), atol=0.0, rtol=0.0)


def test_first_backward_flow_and_auxiliary_gradient_contracts():
    planning, source, mask, view_ids, uv, rays = _inputs()

    flow_module = _module()
    flow_output = flow_module(planning, source, mask, view_ids, uv, rays)[0]
    flow_output.square().mean().backward()
    assert flow_module.up_projection.weight.grad is not None
    assert flow_module.up_projection.weight.grad.abs().sum() > 0

    aux_module = _module()
    auxiliary = AuxiliaryTrajectoryHead(input_dim=16, hidden_dim=16, action_dim=4)
    readout = aux_module(planning, source, mask, view_ids, uv, rays)[1]
    target = torch.randn(2, 8, 4)
    auxiliary(readout, target).loss.backward()
    expected = {
        "source_projection.weight": aux_module.source_projection.weight,
        "view_embedding.weight": aux_module.view_embedding.weight,
        "uv_mlp.0.weight": aux_module.uv_mlp[0].weight,
        "ray_mlp.0.weight": aux_module.ray_mlp[0].weight,
        "planning_projection.weight": aux_module.planning_projection.weight,
        "slot_embeddings": aux_module.slot_embeddings,
        "cross_attention.in_proj_weight": aux_module.cross_attention.in_proj_weight,
        "ffn.0.weight": aux_module.ffn[0].weight,
        "readout_projection.weight": aux_module.readout_projection.weight,
        "auxiliary.head.weight": auxiliary.head[-1].weight,
    }
    for name, parameter in expected.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_debug_attention_diagnostics_are_finite_with_padding():
    module = PlanningConditionedDenseVGGTBottleneck(
        planning_dim=32,
        source_dim=24,
        bottleneck_dim=16,
        expected_horizons=8,
        slots_per_horizon=4,
        num_heads=4,
        attention_dropout=0.0,
        return_attention_diagnostics=True,
    ).eval()
    planning, source, mask, view_ids, uv, rays = _inputs(tokens=19)
    mask[0, -5:] = False
    diagnostics = module(planning, source, mask, view_ids, uv, rays)[3]
    for key in (
        "attention_entropy",
        "attention_max",
        "attention_view_0_mass",
        "attention_view_1_mass",
        "attention_view_2_mass",
        "attention_entropy_horizon_0",
        "attention_entropy_horizon_7",
    ):
        assert torch.isfinite(diagnostics[key]), key
    assert diagnostics["attention_weights"].shape == (2, 4, 32, 19)
