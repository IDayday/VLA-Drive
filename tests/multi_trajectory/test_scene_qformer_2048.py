import inspect

import pytest
import torch
from torch import nn

from starVLA.model.modules.action_model.multi_trajectory.global_scene_qformer import (
    GlobalSceneQFormer,
)
from starVLA.model.modules.action_model.multi_trajectory.scene_context import (
    SceneContext,
)


@pytest.fixture(scope="module")
def full_qformer_fixture():
    torch.manual_seed(5)
    model = GlobalSceneQFormer(input_dim=64).eval()
    hidden = torch.randn(1, 3, 64)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        context = model(hidden, mask)
    return model, hidden, mask, context


def test_scene_query_parameter_shape_2048(full_qformer_fixture):
    model, _, _, _ = full_qformer_fixture
    assert model.scene_queries.shape == (1, 16, 2048)
    assert model.scene_queries.ndim == 3


def test_scene_context_shape_2048(full_qformer_fixture):
    _, hidden, _, context = full_qformer_fixture
    assert context.global_scene_tokens.shape == (1, 16, 2048)
    assert context.dense_scene_memory.shape == (1, hidden.shape[1], 2048)
    assert context.memory_key_padding_mask.shape == (1, hidden.shape[1])
    assert context.memory_key_padding_mask.dtype is torch.bool


def test_qformer_internal_width_2048(full_qformer_fixture):
    model, _, _, _ = full_qformer_fixture
    block = model.blocks[0]
    assert model.input_proj.out_features == 2048
    assert block.self_attn.embed_dim == 2048
    assert block.cross_attn.embed_dim == 2048
    linears = [module for module in block.ffn if isinstance(module, nn.Linear)]
    assert (linears[0].in_features, linears[0].out_features) == (2048, 8192)
    assert (linears[1].in_features, linears[1].out_features) == (8192, 2048)
    assert model.output_norm.normalized_shape == (2048,)


def test_scene_qformer_padding_invariance(full_qformer_fixture):
    model, hidden, mask, expected = full_qformer_fixture
    changed = hidden.clone()
    changed[:, 2] = torch.randn_like(changed[:, 2]) * 1000.0
    with torch.no_grad():
        actual = model(changed, mask)
    torch.testing.assert_close(
        actual.global_scene_tokens,
        expected.global_scene_tokens,
        rtol=1e-5,
        atol=1e-5,
    )


def test_qformer_uses_full_hidden_sequence():
    model = GlobalSceneQFormer(
        input_dim=12,
        scene_dim=24,
        num_queries=3,
        num_layers=1,
        num_heads=3,
        ffn_dim=48,
    ).eval()
    seen = []
    hook = model.input_proj.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].shape[1])
    )
    try:
        with torch.no_grad():
            model(torch.randn(2, 37, 12), torch.ones(2, 37, dtype=torch.bool))
    finally:
        hook.remove()
    assert seen == [37]


def test_scene_context_preserves_projected_memory_dtype():
    model = GlobalSceneQFormer(
        input_dim=12,
        scene_dim=24,
        num_queries=3,
        num_layers=1,
        num_heads=3,
        ffn_dim=48,
    ).to(dtype=torch.float64).eval()
    with torch.no_grad():
        context = model(
            torch.randn(1, 5, 12, dtype=torch.float64),
            torch.ones(1, 5, dtype=torch.bool),
        )
    assert context.global_scene_tokens.dtype == torch.float64
    assert context.dense_scene_memory.dtype == torch.float64


def test_scene_context_validation_rejects_all_padding():
    context = SceneContext(
        global_scene_tokens=torch.zeros(1, 16, 2048),
        dense_scene_memory=torch.zeros(1, 2, 2048),
        memory_key_padding_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="at least one valid"):
        context.validate()


def test_no_pseudo_spatial_reshape():
    source = inspect.getsource(GlobalSceneQFormer)
    assert ".reshape" not in source
    assert "4, 4" not in source
