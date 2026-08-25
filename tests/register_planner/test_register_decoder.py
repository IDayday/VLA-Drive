import torch
from torch import nn

from starVLA.model.modules.register_planner.decoder import (
    DropPath,
    RegisterTrajectoryDecoder,
)


def test_register_decoder_shapes():
    decoder = RegisterTrajectoryDecoder()
    output = decoder(torch.randn(2, 64, 256), torch.randn(2, 16, 256))
    assert len(output) == 4
    assert all(value.shape == (2, 64, 256) for value in output)


def test_register_decoder_uses_one_head_in_main_config():
    decoder = RegisterTrajectoryDecoder()
    assert all(layer.self_attn.attn.num_heads == 1 for layer in decoder.layers)
    assert all(layer.cross_attn.attn.num_heads == 1 for layer in decoder.layers)


def test_register_decoder_has_query_and_memory_layernorm():
    block = RegisterTrajectoryDecoder().layers[0]
    assert isinstance(block.cross_attn_norm_query, nn.LayerNorm)
    assert isinstance(block.cross_attn_norm_memory, nn.LayerNorm)
    assert block.cross_attn_norm_query is not block.cross_attn_norm_memory


def test_register_decoder_has_drop_path_and_proj_drop():
    block = RegisterTrajectoryDecoder().layers[0]
    assert isinstance(block.self_attn_drop_path, DropPath)
    assert isinstance(block.cross_attn_drop_path, DropPath)
    assert isinstance(block.ffn_drop_path, DropPath)
    assert isinstance(block.self_attn.proj_drop, nn.Dropout)
    assert block.self_attn.proj_drop.p == 0.1
