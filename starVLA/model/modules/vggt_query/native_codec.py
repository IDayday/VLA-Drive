"""Task-preserving 195-token codec for VGGT layer-11 global features.

The encoder has exactly one source: the native layer-11 global branch.  It
compresses that ``[B,3,1374,1024]`` tensor into 195 structured tokens, and the
decoder reconstructs exactly that same tensor.  Frozen native VGGT blocks then
continue normally from layer 12.  No earlier tap or frame branch is encoded or
synthesized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


NATIVE_CODEC_SCHEMA_VERSION = 3
NATIVE_CODEC_SOURCE_CONTRACT = {
    "layer_index": 11,
    "attention_branch": "global",
    "feature": "layer11_global",
}


@dataclass(frozen=True)
class VGGTNativeCodecConfig:
    view_count: int = 3
    special_per_view: int = 5
    compact_rows: int = 6
    compact_cols: int = 10
    source_rows: int = 37
    source_cols: int = 37
    branch_dim: int = 1024
    latent_dim: int = 1024
    encoder_hidden_dim: int = 2048
    decoder_channels: int = 256

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def compact_query_count(self) -> int:
        return self.view_count * (
            self.special_per_view + self.compact_rows * self.compact_cols
        )

    @property
    def source_query_count_per_view(self) -> int:
        return self.special_per_view + self.source_rows * self.source_cols

@dataclass
class NativeCodecLossOutput:
    loss: torch.Tensor
    reconstructed_layer11_global: torch.Tensor
    metrics: dict[str, torch.Tensor]


class VGGTNativeFeatureCodec(nn.Module):
    """Compress and reconstruct only VGGT layer-11 global."""

    def __init__(self, config: VGGTNativeCodecConfig) -> None:
        super().__init__()
        if config.compact_query_count != 195:
            raise ValueError(
                "V3 production contract requires exactly 195 compact tokens, "
                f"found {config.compact_query_count}"
            )
        self.config = config
        # Hard V3 contract: no layer-4 or frame-branch feature can enter this
        # encoder.  The input width therefore equals one native VGGT branch.
        encoder_input_dim = config.branch_dim
        decoder_output_dim = config.branch_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(encoder_input_dim),
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.encoder_hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.special_decoder = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.encoder_hidden_dim, decoder_output_dim),
        )
        self.spatial_decoder_in = nn.Sequential(
            nn.Conv2d(config.latent_dim, config.decoder_channels, kernel_size=1),
            nn.GELU(),
        )
        self.spatial_decoder_refine = nn.Sequential(
            nn.Conv2d(
                config.decoder_channels,
                config.decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(config.decoder_channels, decoder_output_dim, kernel_size=1),
        )

    @property
    def native_scalar_compression_ratio(self) -> float:
        config = self.config
        native = (
            config.view_count
            * config.source_query_count_per_view
            * config.branch_dim
        )
        compact = config.compact_query_count * config.latent_dim
        return native / compact

    def _validate_layer11_global(self, value: torch.Tensor) -> None:
        config = self.config
        expected = (
            config.view_count,
            config.source_query_count_per_view,
            config.branch_dim,
        )
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "layer-11 global must be "
                f"[B,{expected[0]},{expected[1]},{expected[2]}], "
                f"found {tuple(value.shape)}"
            )

    def encode(self, layer11_global: torch.Tensor) -> torch.Tensor:
        """Return ``[B,195,D]`` from only native layer-11 global tokens."""

        self._validate_layer11_global(layer11_global)
        config = self.config
        source = layer11_global.float()
        special = source[:, :, : config.special_per_view]
        spatial = source[:, :, config.special_per_view :]
        spatial = spatial.reshape(
            source.shape[0],
            config.view_count,
            config.source_rows,
            config.source_cols,
            -1,
        )
        pooled = F.adaptive_avg_pool2d(
            spatial.permute(0, 1, 4, 2, 3).flatten(0, 1),
            (config.compact_rows, config.compact_cols),
        )
        pooled = pooled.reshape(
            source.shape[0],
            config.view_count,
            -1,
            config.compact_rows,
            config.compact_cols,
        ).permute(0, 1, 3, 4, 2)
        compute_dtype = self.encoder[0].weight.dtype
        encoded_special = self.encoder(special.to(dtype=compute_dtype))
        encoded_spatial = self.encoder(pooled.to(dtype=compute_dtype))
        latent = torch.cat(
            (
                encoded_special.flatten(1, 2),
                encoded_spatial.flatten(1, 3),
            ),
            dim=1,
        )
        expected = (source.shape[0], config.compact_query_count, config.latent_dim)
        if tuple(latent.shape) != expected:
            raise AssertionError(
                f"native codec latent contract changed: {tuple(latent.shape)} != {expected}"
            )
        # This canonical coordinate system is the exact cache/alignment/
        # decoder contract. VGGTQueryAligner also layer-normalizes teacher
        # targets, so student output can be sent to this decoder directly
        # without learning an extra inverse affine transform.
        return F.layer_norm(latent.float(), (config.latent_dim,))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode exactly the native layer-11 global tensor."""

        config = self.config
        expected = (config.compact_query_count, config.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected:
            raise ValueError(
                f"native codec latent must be [B,{expected[0]},{expected[1]}], "
                f"found {tuple(latent.shape)}"
            )
        batch = latent.shape[0]
        special_count = config.view_count * config.special_per_view
        special = latent[:, :special_count].reshape(
            batch, config.view_count, config.special_per_view, config.latent_dim
        )
        spatial = latent[:, special_count:].reshape(
            batch,
            config.view_count,
            config.compact_rows,
            config.compact_cols,
            config.latent_dim,
        )
        compute_dtype = self.special_decoder[0].weight.dtype
        decoded_special = self.special_decoder(special.to(dtype=compute_dtype)).float()
        spatial_map = spatial.permute(0, 1, 4, 2, 3).flatten(0, 1)
        spatial_map = self.spatial_decoder_in(
            spatial_map.to(dtype=self.spatial_decoder_in[0].weight.dtype)
        )
        spatial_map = F.interpolate(
            spatial_map.float(),
            size=(config.source_rows, config.source_cols),
            mode="bilinear",
            align_corners=True,
        ).to(dtype=self.spatial_decoder_refine[0].weight.dtype)
        decoded_spatial = self.spatial_decoder_refine(spatial_map).float()
        decoded_spatial = decoded_spatial.reshape(
            batch,
            config.view_count,
            -1,
            config.source_rows,
            config.source_cols,
        ).permute(0, 1, 3, 4, 2).flatten(2, 3)
        decoded = torch.cat((decoded_special, decoded_spatial), dim=2)
        self._validate_layer11_global(decoded)
        return decoded

    def decode_layer11_global(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the sole V3 source feature."""

        return self.decode(latent)

    def forward(
        self, layer11_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDP-friendly joint encode/decode with the exact BF16 cache bottleneck."""

        latent = self.encode(layer11_global)
        cached_latent = latent.to(torch.bfloat16).to(latent.dtype)
        return latent, self.decode(cached_latent)

    def reconstruction_loss(
        self,
        latent: torch.Tensor,
        target_layer11_global: torch.Tensor,
        *,
        cosine_weight: float = 1.0,
        smooth_l1_weight: float = 1.0,
    ) -> NativeCodecLossOutput:
        reconstructed = self.decode(latent)
        self._validate_layer11_global(target_layer11_global)
        target = target_layer11_global.float()
        prediction = reconstructed.float()
        cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-6)
        smooth = F.smooth_l1_loss(prediction, target)
        loss = float(cosine_weight) * (1.0 - cosine.mean()) + float(
            smooth_l1_weight
        ) * smooth
        metrics = {
            "layer11_global_cosine": cosine.mean().detach(),
            "layer11_global_smooth_l1": smooth.detach(),
        }
        return NativeCodecLossOutput(
            loss=loss,
            reconstructed_layer11_global=reconstructed,
            metrics=metrics,
        )

    def freeze_pretrained(self) -> "VGGTNativeFeatureCodec":
        self.requires_grad_(False)
        self.eval()
        return self


def save_native_codec_checkpoint(
    codec: VGGTNativeFeatureCodec,
    path: Path | str,
    *,
    latent_slot_mean: torch.Tensor,
    latent_slot_scale: torch.Tensor,
    source: Mapping[str, Any],
    gates: Mapping[str, Any],
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    config = codec.config
    if tuple(latent_slot_mean.shape) != (
        config.compact_query_count,
        config.latent_dim,
    ):
        raise ValueError("native codec latent slot mean shape mismatch")
    if tuple(latent_slot_scale.shape) != (config.compact_query_count,):
        raise ValueError("native codec latent slot scale shape mismatch")
    payload = {
        "schema_version": NATIVE_CODEC_SCHEMA_VERSION,
        "encoder_source": dict(NATIVE_CODEC_SOURCE_CONTRACT),
        "config": asdict(config),
        "state_dict": {
            key: value.detach().cpu() for key, value in codec.state_dict().items()
        },
        "latent_slot_mean": latent_slot_mean.detach().float().cpu(),
        "latent_slot_scale": latent_slot_scale.detach().float().cpu(),
        "source": dict(source),
        "gates": dict(gates),
        "metrics": dict(metrics),
        "thresholds": dict(thresholds or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_native_codec_checkpoint(
    path: Path | str,
) -> tuple[VGGTNativeFeatureCodec, dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing VGGT native codec: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != NATIVE_CODEC_SCHEMA_VERSION:
        raise RuntimeError(
            "VGGT native codec schema mismatch: "
            f"expected {NATIVE_CODEC_SCHEMA_VERSION}, found {payload.get('schema_version')}"
        )
    if payload.get("encoder_source") != NATIVE_CODEC_SOURCE_CONTRACT:
        raise RuntimeError("VGGT native codec is not layer-11-global-only")
    config = VGGTNativeCodecConfig(**payload["config"])
    codec = VGGTNativeFeatureCodec(config)
    codec.load_state_dict(payload["state_dict"], strict=True)
    mean = payload.get("latent_slot_mean", torch.empty(0)).float()
    scale = payload.get("latent_slot_scale", torch.empty(0)).float()
    if tuple(mean.shape) != (config.compact_query_count, config.latent_dim):
        raise RuntimeError("VGGT native codec checkpoint has invalid slot mean")
    if tuple(scale.shape) != (config.compact_query_count,):
        raise RuntimeError("VGGT native codec checkpoint has invalid slot scale")
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["path"] = str(path.resolve())
    return codec, metadata
