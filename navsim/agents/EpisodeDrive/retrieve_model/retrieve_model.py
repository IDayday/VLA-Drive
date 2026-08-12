"""Dual-branch DINO Retrieve Model V1.

This is an explicit local implementation inspired by DriveVLA-M0. It is not a
claim about the paper's unpublished implementation details.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from navsim.agents.EpisodeDrive.layers.image_encoder.dinov2_lora import (
    LoRA_ViT_timm,
    timm_ViT,
)


class NativeForwardLoRAViT(LoRA_ViT_timm):
    """Apply EpisodeDrive Q/V LoRA surgery while preserving the native ViT forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit.forward_features(x)


def build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """Return a deterministic [height * width, dim] 2D sine-cosine encoding."""
    if dim % 4 != 0:
        raise ValueError(f"2D sine-cosine embedding requires dim divisible by 4, got {dim}")

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    frequency = torch.arange(dim // 4, dtype=torch.float32)
    frequency = temperature ** (-frequency / max(dim // 4 - 1, 1))

    x_phase = x.reshape(-1, 1) * frequency.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * frequency.reshape(1, -1)
    return torch.cat(
        [x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()],
        dim=-1,
    )


class DinoPatchBranch(nn.Module):
    """A frozen DINO base with branch-specific Q/V LoRA and patch projection."""

    def __init__(
        self,
        model_name: str,
        model_weights: str,
        image_size: Sequence[int],
        d_model: int = 256,
        lora_rank: int = 32,
        pooled_grid: Tuple[int, int] = (12, 20),
        max_cameras: int = 4,
    ) -> None:
        super().__init__()
        if len(image_size) != 2:
            raise ValueError(f"image_size must be [width, height], got {image_size}")

        self.image_width = int(image_size[0])
        self.image_height = int(image_size[1])
        self.pooled_grid = tuple(int(value) for value in pooled_grid)
        self.d_model = int(d_model)

        pretrained_cfg_overlay = {"file": str(model_weights)}
        vit = timm.create_model(
            model_name,
            pretrained=True,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            img_size=(self.image_height, self.image_width),
            num_classes=0,
            in_chans=3,
        )
        self.patch_size = int(vit.patch_embed.patch_size[0])
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError(
                f"Input {(self.image_height, self.image_width)} must be divisible by "
                f"DINO patch size {self.patch_size}"
            )
        self.patch_grid = (
            self.image_height // self.patch_size,
            self.image_width // self.patch_size,
        )
        self.num_prefix_tokens = int(vit.num_prefix_tokens)
        self.backbone_width = int(vit.num_features)

        if "dinov3" in model_name.lower():
            # timm implements DINOv3 as EVA with rotary position embeddings.
            # Replacing its class with the older register-based VisionTransformer would
            # silently drop the native RoPE path, so only reuse the Q/V surgery.
            self.backbone = NativeForwardLoRAViT(
                vit,
                r=int(lora_rank),
                use_qkv=False,
            )
        else:
            vit.__class__ = timm_ViT
            self.backbone = LoRA_ViT_timm(vit, r=int(lora_rank), use_qkv=False)
        self.projection = nn.Linear(self.backbone_width, self.d_model)
        self.camera_embedding = nn.Embedding(int(max_cameras), self.d_model)
        self.register_buffer(
            "image_position",
            build_2d_sincos_position_embedding(
                self.pooled_grid[0],
                self.pooled_grid[1],
                self.d_model,
            ),
            persistent=False,
        )

    def forward(
        self,
        images: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode [B, N, 3, H, W] images into [B, N*240, 256] spatial tokens."""
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,N,3,H,W], got {tuple(images.shape)}")
        batch_size, num_cameras, channels, height, width = images.shape
        expected = (3, self.image_height, self.image_width)
        if (channels, height, width) != expected:
            raise ValueError(
                f"Expected image tail {expected}, got {(channels, height, width)}"
            )

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        all_tokens = self.backbone(flat_images)
        patch_tokens = all_tokens[:, self.num_prefix_tokens :, :]

        expected_patch_count = self.patch_grid[0] * self.patch_grid[1]
        if patch_tokens.shape[1] != expected_patch_count:
            raise RuntimeError(
                f"DINO returned {patch_tokens.shape[1]} patch tokens, expected "
                f"{self.patch_grid[0]}x{self.patch_grid[1]}={expected_patch_count}"
            )
        if patch_tokens.shape[2] != self.backbone_width:
            raise RuntimeError(
                f"DINO width is {patch_tokens.shape[2]}, expected {self.backbone_width}"
            )

        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch_size * num_cameras,
            self.backbone_width,
            self.patch_grid[0],
            self.patch_grid[1],
        )
        patch_map = F.adaptive_avg_pool2d(patch_map, self.pooled_grid)
        spatial_tokens = patch_map.flatten(2).transpose(1, 2)
        spatial_tokens = self.projection(spatial_tokens)

        tokens_per_camera = self.pooled_grid[0] * self.pooled_grid[1]
        spatial_tokens = spatial_tokens.reshape(
            batch_size,
            num_cameras,
            tokens_per_camera,
            self.d_model,
        )

        if camera_ids is None:
            camera_ids = torch.arange(num_cameras, device=images.device)
            camera_ids = camera_ids.unsqueeze(0).expand(batch_size, -1)
        elif camera_ids.ndim == 1:
            camera_ids = camera_ids.unsqueeze(0).expand(batch_size, -1)
        if camera_ids.shape != (batch_size, num_cameras):
            raise ValueError(
                f"camera_ids must have shape {(batch_size, num_cameras)}, "
                f"got {tuple(camera_ids.shape)}"
            )

        image_position = self.image_position.to(
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        camera_position = self.camera_embedding(camera_ids.long())
        spatial_tokens = (
            spatial_tokens
            + image_position.reshape(1, 1, tokens_per_camera, self.d_model)
            + camera_position.unsqueeze(2)
        )
        return spatial_tokens.reshape(
            batch_size,
            num_cameras * tokens_per_camera,
            self.d_model,
        )


class ConvDecoder(nn.Module):
    """Configurable 2x upsampling decoder ending at a fixed BEV resolution."""

    def __init__(
        self,
        in_channels: int = 256,
        channels: Sequence[int] = (128, 64, 32, 16),
        out_channels: int = 1,
        input_grid: Sequence[int] = (8, 16),
        output_grid: Sequence[int] = (128, 256),
    ) -> None:
        super().__init__()
        self.input_grid = tuple(int(value) for value in input_grid)
        self.output_grid = tuple(int(value) for value in output_grid)
        if len(self.input_grid) != 2 or len(self.output_grid) != 2:
            raise ValueError("input_grid and output_grid must contain two values")
        scale = 2 ** len(channels)
        expected_output = (
            self.input_grid[0] * scale,
            self.input_grid[1] * scale,
        )
        if expected_output != self.output_grid:
            raise ValueError(
                f"{len(channels)} upsampling stages transform {self.input_grid} "
                f"into {expected_output}, not requested {self.output_grid}"
            )

        blocks = []
        current_channels = int(in_channels)
        for output_channels in channels:
            output_channels = int(output_channels)
            num_groups = min(32, output_channels)
            while output_channels % num_groups != 0:
                num_groups -= 1
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(
                        current_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups=num_groups, num_channels=output_channels),
                    nn.GELU(),
                )
            )
            current_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv2d(current_channels, int(out_channels), kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if tuple(feature_map.shape[-2:]) != self.input_grid:
            raise ValueError(
                f"ConvDecoder expects {self.input_grid} input, "
                f"got {tuple(feature_map.shape[-2:])}"
            )
        logits = self.output(self.blocks(feature_map))
        if tuple(logits.shape[-2:]) != self.output_grid:
            raise RuntimeError(
                f"Decoder produced {tuple(logits.shape)}, "
                f"expected spatial size {self.output_grid}"
            )
        return logits


class RetrieveBranch(nn.Module):
    """One independent map/agent branch."""

    def __init__(
        self,
        model_name: str,
        model_weights: str,
        image_size: Sequence[int],
        d_model: int,
        lora_rank: int,
        pooled_grid: Tuple[int, int],
        max_cameras: int,
        bev_query_grid: Tuple[int, int],
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        conv_channels: Sequence[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.bev_query_grid = tuple(int(value) for value in bev_query_grid)
        num_queries = self.bev_query_grid[0] * self.bev_query_grid[1]

        self.image_encoder = DinoPatchBranch(
            model_name=model_name,
            model_weights=model_weights,
            image_size=image_size,
            d_model=d_model,
            lora_rank=lora_rank,
            pooled_grid=pooled_grid,
            max_cameras=max_cameras,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.bev_queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.bev_queries, std=0.02)
        self.register_buffer(
            "bev_position",
            build_2d_sincos_position_embedding(
                self.bev_query_grid[0],
                self.bev_query_grid[1],
                d_model,
            ),
            persistent=False,
        )
        self.conv_decoder = ConvDecoder(
            d_model,
            conv_channels,
            out_channels=num_classes,
            input_grid=self.bev_query_grid,
        )

    def decode_tokens(
        self,
        images: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.image_encoder(images, camera_ids)
        batch_size = images.shape[0]
        bev_position = self.bev_position.to(
            device=images.device,
            dtype=memory.dtype,
        )
        queries = self.bev_queries.to(dtype=memory.dtype).expand(batch_size, -1, -1)
        queries = queries + bev_position.unsqueeze(0)
        return self.transformer(tgt=queries, memory=memory)

    def forward(
        self,
        images: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bev_tokens = self.decode_tokens(images, camera_ids)
        batch_size = images.shape[0]
        feature_map = bev_tokens.transpose(1, 2).reshape(
            batch_size,
            bev_tokens.shape[-1],
            self.bev_query_grid[0],
            self.bev_query_grid[1],
        )
        logits = self.conv_decoder(feature_map)
        key = F.normalize(bev_tokens.mean(dim=1), p=2, dim=-1)
        return logits, key, bev_tokens


class RetrieveModelV1(nn.Module):
    """Independent map and agent branches with optional branch-specific DINO backbones."""

    def __init__(
        self,
        model_name: str,
        model_weights: str,
        image_size: Sequence[int] = (1148, 672),
        d_model: int = 256,
        lora_rank: int = 32,
        pooled_grid: Sequence[int] = (12, 20),
        max_cameras: int = 4,
        bev_query_grid: Sequence[int] = (8, 16),
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        conv_channels: Sequence[int] = (128, 64, 32, 16),
        map_num_classes: int = 4,
        agent_num_classes: int = 3,
        map_model_name: Optional[str] = None,
        map_model_weights: Optional[str] = None,
        map_image_size: Optional[Sequence[int]] = None,
        agent_model_name: Optional[str] = None,
        agent_model_weights: Optional[str] = None,
        agent_image_size: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        shared_branch_kwargs = dict(
            d_model=d_model,
            lora_rank=lora_rank,
            pooled_grid=tuple(pooled_grid),
            max_cameras=max_cameras,
            bev_query_grid=tuple(bev_query_grid),
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conv_channels=tuple(conv_channels),
        )
        self.map_branch = RetrieveBranch(
            **shared_branch_kwargs,
            model_name=map_model_name or model_name,
            model_weights=map_model_weights or model_weights,
            image_size=tuple(map_image_size or image_size),
            num_classes=int(map_num_classes),
        )
        self.agent_branch = RetrieveBranch(
            **shared_branch_kwargs,
            model_name=agent_model_name or model_name,
            model_weights=agent_model_weights or model_weights,
            image_size=tuple(agent_image_size or image_size),
            num_classes=int(agent_num_classes),
        )

    @staticmethod
    def _resize_for_branch(
        images: torch.Tensor,
        branch: RetrieveBranch,
    ) -> torch.Tensor:
        """Resize a shared camera tensor only when a branch expects another grid."""
        target_size = (
            int(branch.image_encoder.image_height),
            int(branch.image_encoder.image_width),
        )
        if tuple(images.shape[-2:]) == target_size:
            return images
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,N,3,H,W], got {tuple(images.shape)}")
        batch_size, num_cameras, channels, height, width = images.shape
        resized = F.interpolate(
            images.reshape(batch_size * num_cameras, channels, height, width),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch_size, num_cameras, channels, *target_size)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        images = features["image"]
        camera_ids = features.get("camera_ids")
        map_images = self._resize_for_branch(images, self.map_branch)
        agent_images = self._resize_for_branch(images, self.agent_branch)
        map_logits, map_key, map_tokens = self.map_branch(map_images, camera_ids)
        agent_logits, agent_key, agent_tokens = self.agent_branch(agent_images, camera_ids)

        output = {
            "map_bev_logits": map_logits,
            "agent_bev_logits": agent_logits,
            "map_key": map_key,
            "agent_key": agent_key,
        }
        if return_tokens:
            output["map_bev_tokens"] = map_tokens
            output["agent_bev_tokens"] = agent_tokens
        return output

    def encode_keys(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Export normalized retrieval keys without running convolutional heads."""
        images = features["image"]
        camera_ids = features.get("camera_ids")
        map_images = self._resize_for_branch(images, self.map_branch)
        agent_images = self._resize_for_branch(images, self.agent_branch)
        map_tokens = self.map_branch.decode_tokens(map_images, camera_ids)
        agent_tokens = self.agent_branch.decode_tokens(agent_images, camera_ids)
        return {
            "map_key": F.normalize(map_tokens.mean(dim=1), p=2, dim=-1),
            "agent_key": F.normalize(agent_tokens.mean(dim=1), p=2, dim=-1),
        }
