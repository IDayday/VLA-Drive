import torch

from starVLA.model.modules.scene_encoder import GlobalSceneQFormer


def test_global_qformer_shapes_mask_and_padding_invariance():
    torch.manual_seed(1)
    model = GlobalSceneQFormer(
        input_dim=12,
        hidden_dim=32,
        output_dim=32,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    ).eval()
    hidden = torch.randn(2, 7, 12)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.long
    )
    changed = hidden.clone()
    changed[0, 4:] = torch.randn_like(changed[0, 4:]) * 1000
    changed[1, 5:] = torch.randn_like(changed[1, 5:]) * 1000
    with torch.no_grad():
        first = model(hidden, mask)
        second = model(changed, mask)
    assert first.global_tokens.shape == (2, 4, 32)
    assert first.dense_memory.shape == (2, 7, 32)
    assert torch.equal(first.memory_key_padding_mask, ~mask.bool())
    torch.testing.assert_close(first.global_tokens, second.global_tokens)


def test_qformer_detaches_backbone_but_trains_projection():
    model = GlobalSceneQFormer(
        input_dim=8,
        hidden_dim=16,
        output_dim=16,
        num_queries=2,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    output = model(hidden, torch.ones(2, 5, dtype=torch.bool))
    output.global_tokens.square().mean().backward()
    assert hidden.grad is None
    assert model.input_proj.weight.grad is not None
    assert model.scene_queries.grad is not None


def test_production_scene_query_definition_is_2048_wide():
    # One layer keeps this structural test inexpensive while checking the
    # production width/query/head/FFN contract directly.
    model = GlobalSceneQFormer(
        input_dim=8,
        hidden_dim=2048,
        output_dim=2048,
        num_queries=16,
        num_layers=1,
        num_heads=32,
        ffn_dim=8192,
    )
    block = model.blocks[0]
    assert model.scene_queries.shape == (1, 16, 2048)
    assert model.input_proj.out_features == 2048
    assert block.self_attn.embed_dim == 2048
    assert block.cross_attn.embed_dim == 2048
    assert block.ffn[0].in_features == 2048
    assert block.ffn[0].out_features == 8192
    assert block.ffn[3].in_features == 8192
    assert block.ffn[3].out_features == 2048
