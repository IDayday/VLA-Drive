from pathlib import Path

import torch
from torch import nn

from starVLA.model.modules.vggt_query.native_codec import (
    VGGTNativeCodecConfig,
    VGGTNativeFeatureCodec,
    load_native_codec_checkpoint,
    save_native_codec_checkpoint,
)
from starVLA.model.modules.vggt_query.native_tail import (
    resume_frozen_vggt_tail_from_global,
)


def _config() -> VGGTNativeCodecConfig:
    return VGGTNativeCodecConfig(
        view_count=3,
        special_per_view=5,
        compact_rows=6,
        compact_cols=10,
        source_rows=8,
        source_cols=12,
        branch_dim=16,
        latent_dim=32,
        encoder_hidden_dim=48,
        decoder_channels=24,
    )


def test_native_codec_encodes_only_layer11_global_into_195_tokens():
    torch.manual_seed(43)
    config = _config()
    codec = VGGTNativeFeatureCodec(config)
    source_tokens = config.special_per_view + config.source_rows * config.source_cols
    layer11_global = torch.randn(
        2, config.view_count, source_tokens, config.branch_dim
    )

    latent = codec.encode(layer11_global)
    reconstructed = codec.decode(latent)
    _, cached_reconstruction = codec(layer11_global)
    direct_cached = codec.decode(latent.to(torch.bfloat16).float())

    assert latent.shape == (2, 195, config.latent_dim)
    assert codec.encoder[1].in_features == config.branch_dim
    assert codec.special_decoder[-1].out_features == config.branch_dim
    assert codec.spatial_decoder_refine[-1].out_channels == config.branch_dim
    assert reconstructed.shape == layer11_global.shape
    assert codec.decode_layer11_global(latent).shape == layer11_global.shape
    expected_ratio = (
        config.view_count
        * source_tokens
        * config.branch_dim
        / (config.compact_query_count * config.latent_dim)
    )
    assert codec.native_scalar_compression_ratio == expected_ratio
    torch.testing.assert_close(cached_reconstruction, direct_cached)


def test_student_latent_can_backprop_through_frozen_native_decoder():
    torch.manual_seed(47)
    config = _config()
    codec = VGGTNativeFeatureCodec(config).freeze_pretrained()
    student = torch.randn(2, 195, config.latent_dim, requires_grad=True)
    target = torch.randn(
        2,
        config.view_count,
        config.special_per_view + config.source_rows * config.source_cols,
        config.branch_dim,
    )

    output = codec.reconstruction_loss(student, target)
    output.loss.backward()

    assert student.grad is not None
    assert all(parameter.grad is None for parameter in codec.parameters())
    assert all(not parameter.requires_grad for parameter in codec.parameters())


def test_native_codec_checkpoint_carries_gate_and_source_identity(tmp_path: Path):
    torch.manual_seed(53)
    config = _config()
    codec = VGGTNativeFeatureCodec(config)
    mean = torch.randn(config.compact_query_count, config.latent_dim)
    scale = torch.rand(config.compact_query_count).clamp_min(0.1)
    path = tmp_path / "native codec.pt"

    save_native_codec_checkpoint(
        codec,
        path,
        latent_slot_mean=mean,
        latent_slot_scale=scale,
        source={"vggt_checkpoint_sha256": "abc"},
        gates={"teacher_codec_downstream": True},
        metrics={"depth_abs_rel": 0.1},
    )
    loaded, metadata = load_native_codec_checkpoint(path)

    assert loaded.config == config
    assert metadata["gates"]["teacher_codec_downstream"] is True
    assert metadata["encoder_source"] == {
        "layer_index": 11,
        "attention_branch": "global",
        "feature": "layer11_global",
    }
    assert metadata["source"]["vggt_checkpoint_sha256"] == "abc"
    torch.testing.assert_close(metadata["latent_slot_mean"], mean)
    torch.testing.assert_close(metadata["latent_slot_scale"], scale)


class _PositionGetter:
    def __call__(self, batch, rows, cols, device):
        return torch.zeros(batch, rows * cols, 2, device=device, dtype=torch.long)


class _Block(nn.Module):
    def __init__(self, width: int, amount: float):
        super().__init__()
        self.weight = nn.Parameter(torch.full((width,), amount))

    def forward(self, value, pos=None):
        del pos
        return value + self.weight


class _Aggregator(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.rope = object()
        self.position_getter = _PositionGetter()
        self.frame_blocks = nn.ModuleList([_Block(width, 0.01) for _ in range(24)])
        self.global_blocks = nn.ModuleList([_Block(width, 0.02) for _ in range(24)])


def test_v3_native_tail_accepts_layer11_global_without_frame_or_layer4():
    width = 8
    aggregator = _Aggregator(width).requires_grad_(False)
    layer11_global = torch.randn(1, 3, 5 + 2 * 3, width, requires_grad=True)

    tail = resume_frozen_vggt_tail_from_global(
        aggregator,
        layer11_global,
        branch_dim=width,
        source_rows=2,
        source_cols=3,
    )
    tail[23].mean().backward()

    assert tail[17].shape[-1] == 2 * width
    assert tail[23].shape[-1] == 2 * width
    assert layer11_global.grad is not None
