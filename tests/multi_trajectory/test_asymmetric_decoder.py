import inspect

import torch
from torch import nn

from starVLA.model.modules.action_model.multi_trajectory.asymmetric_decoder import (
    AsymmetricTransformerDecoder,
    AsymmetricTransformerDecoderLayer,
)


def test_asymmetric_cross_attention_shape():
    layer = AsymmetricTransformerDecoderLayer().eval()
    with torch.no_grad():
        output = layer(
            torch.randn(2, 64, 256),
            torch.randn(2, 16, 2048),
        )
    assert output.shape == (2, 64, 256)


def test_asymmetric_cross_attention_kv_dim():
    layer = AsymmetricTransformerDecoderLayer()
    assert layer.cross_attn.embed_dim == 256
    assert layer.cross_attn.kdim == 2048
    assert layer.cross_attn.vdim == 2048
    assert layer.cross_attn.q_proj_weight.shape == (256, 256)
    assert layer.cross_attn.k_proj_weight.shape == (256, 2048)
    assert layer.cross_attn.v_proj_weight.shape == (256, 2048)


def test_each_decoder_layer_has_independent_kv_projection():
    decoder = AsymmetricTransformerDecoder(num_layers=3)
    first = decoder.layers[0].cross_attn
    second = decoder.layers[1].cross_attn
    assert first is not second
    assert first.k_proj_weight is not second.k_proj_weight
    assert first.v_proj_weight is not second.v_proj_weight
    assert first.k_proj_weight.data_ptr() != second.k_proj_weight.data_ptr()


def test_asymmetric_decoder_intermediate_outputs_and_mask():
    decoder = AsymmetricTransformerDecoder(
        num_layers=3,
        planning_dim=16,
        memory_dim=32,
        num_heads=4,
        ffn_dim=64,
        return_intermediate=True,
    ).eval()
    mask = torch.tensor([[False, False, True, True]])
    seen_masks = []
    hooks = []
    for layer in decoder.layers:
        original = layer.cross_attn.forward

        def wrapped(*args, _original=original, **kwargs):
            seen_masks.append(kwargs.get("key_padding_mask"))
            return _original(*args, **kwargs)

        layer.cross_attn.forward = wrapped
    with torch.no_grad():
        outputs = decoder(
            torch.randn(1, 5, 16),
            torch.randn(1, 4, 32),
            memory_key_padding_mask=mask,
        )
    assert len(outputs) == 3
    assert all(value.shape == (1, 5, 16) for value in outputs)
    assert len(seen_masks) == 3
    assert all(value is mask for value in seen_masks)


def test_no_shared_scene_projection_to_256():
    source = inspect.getsource(AsymmetricTransformerDecoder)
    assert "memory_proj" not in source
    decoder = AsymmetricTransformerDecoder(num_layers=2)
    assert not any(
        isinstance(module, nn.Linear)
        and module.in_features == 2048
        and module.out_features == 256
        for name, module in decoder.named_modules()
        if "cross_attn" not in name
    )
