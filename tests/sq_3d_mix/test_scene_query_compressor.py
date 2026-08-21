import torch

from starVLA.model.modules.vggt_query.scene_query_compressor import (
    SceneQueryCompressor,
)


def _compressor(input_dim: int = 32) -> SceneQueryCompressor:
    torch.manual_seed(7)
    return SceneQueryCompressor(
        input_dim=input_dim,
        hidden_dim=16,
        num_queries=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        query_init_std=1e-6,
    ).eval()


def test_scene_query_shape():
    model = _compressor()
    hidden = torch.randn(2, 13, 32)
    valid = torch.ones(2, 13, dtype=torch.bool)

    scene_tokens, diagnostics = model(hidden, valid)

    assert scene_tokens.shape == (2, 16, 32)
    assert set(diagnostics) == {
        "sq3dmix/scene_token_norm",
        "sq3dmix/scene_token_std",
        "sq3dmix/scene_token_pairwise_cosine",
        "sq3dmix/scene_memory_valid_tokens",
    }
    assert all(not value.requires_grad for value in diagnostics.values())


def test_scene_query_accepts_bfloat16_qwen_hidden_with_float32_weights():
    model = _compressor()
    hidden = torch.randn(2, 13, 32, dtype=torch.bfloat16)
    valid = torch.ones(2, 13, dtype=torch.bool)

    scene_tokens, _ = model(hidden, valid)

    assert scene_tokens.shape == (2, 16, 32)
    assert scene_tokens.dtype == torch.float32


def test_scene_query_ignores_padding():
    model = _compressor()
    hidden = torch.randn(2, 12, 32)
    valid = torch.tensor(
        [[True] * 8 + [False] * 4, [True] * 10 + [False] * 2]
    )
    changed = hidden.clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 1000

    expected, _ = model(hidden, valid)
    actual, _ = model(changed, valid)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_scene_query_excludes_action_tokens():
    model = _compressor()
    hidden = torch.randn(2, 18, 32)
    valid = torch.ones(2, 18, dtype=torch.bool)
    action_positions = torch.tensor(
        [[2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 5, 7, 9, 11, 13, 15]]
    )
    valid.scatter_(1, action_positions, False)
    changed = hidden.clone()
    changed.scatter_(
        1,
        action_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        torch.randn(2, 8, 32) * 1000,
    )

    expected, _ = model(hidden, valid)
    actual, _ = model(changed, valid)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_scene_query_uses_valid_scene_tokens():
    model = _compressor()
    hidden = torch.randn(2, 12, 32)
    valid = torch.ones(2, 12, dtype=torch.bool)
    changed = hidden.clone()
    changed[:, 4] += 20

    expected, _ = model(hidden, valid)
    actual, _ = model(changed, valid)

    assert not torch.allclose(actual, expected)


def test_scene_queries_receive_gradient():
    model = _compressor().train()
    hidden = torch.randn(2, 12, 32)
    valid = torch.ones(2, 12, dtype=torch.bool)

    scene_tokens, _ = model(hidden, valid)
    scene_tokens.square().mean().backward()

    for parameter in (
        model.scene_queries,
        model.input_projection.weight,
        model.output_projection.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.detach().abs().sum() > 0
