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


def test_scene_branch_can_allow_qwen_gradient_for_ablation():
    model = GlobalSceneQFormer(
        input_dim=8,
        hidden_dim=16,
        output_dim=16,
        num_queries=2,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        detach_qwen_input=False,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    model(
        hidden, torch.ones(2, 5, dtype=torch.bool)
    ).global_tokens.square().mean().backward()
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad)


def test_production_scene_query_definition_is_16_by_256_and_lightweight():
    model = GlobalSceneQFormer(
        input_dim=2048,
        hidden_dim=256,
        output_dim=256,
        num_queries=16,
        num_layers=4,
        num_heads=8,
        ffn_dim=1024,
    )
    block = model.blocks[0]
    assert model.scene_queries.shape == (1, 16, 256)
    assert model.qwen_to_scene_proj.out_features == 256
    assert block.self_attn.embed_dim == 256
    assert block.cross_attn.embed_dim == 256
    assert block.ffn[0].in_features == 256
    assert block.ffn[0].out_features == 1024
    assert block.ffn[3].in_features == 1024
    assert block.ffn[3].out_features == 256
    assert model.parameter_count < 10_000_000


def test_qformer_checkpointing_equivalence():
    torch.manual_seed(11)
    plain = GlobalSceneQFormer(
        input_dim=12,
        hidden_dim=32,
        output_dim=32,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        use_gradient_checkpointing=False,
    ).train()
    checkpointed = GlobalSceneQFormer(
        input_dim=12,
        hidden_dim=32,
        output_dim=32,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        use_gradient_checkpointing=True,
    ).train()
    checkpointed.load_state_dict(plain.state_dict())
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    plain_input = torch.randn(2, 5, 12, requires_grad=True)
    checkpoint_input = plain_input.detach().clone().requires_grad_(True)
    plain_output = plain(plain_input, mask).global_tokens
    checkpoint_output = checkpointed(checkpoint_input, mask).global_tokens
    torch.testing.assert_close(plain_output, checkpoint_output)
    plain_output.square().mean().backward()
    checkpoint_output.square().mean().backward()
    for first, second in zip(plain.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(first.grad, second.grad)
