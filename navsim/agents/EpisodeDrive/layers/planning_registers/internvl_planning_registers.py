"""InternViT-internal planning registers and patch/register separation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import math
from typing import List

import torch
from torch import nn


@dataclass
class InternVLPlanningOutput:
    patch_features: torch.Tensor
    per_tile_registers: torch.Tensor
    scene_registers: torch.Tensor
    encoded_patches: torch.Tensor


class PlanningRegisterAdapter(nn.Module, ABC):
    """Backbone-specific contract consumed by DriveVLA scene fusion."""

    @abstractmethod
    def forward(
        self,
        vlm_model: nn.Module,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
    ) -> InternVLPlanningOutput:
        raise NotImplementedError


class InternVLPlanningRegisters(PlanningRegisterAdapter):
    """Insert registers after CLS and run them through every InternViT block."""

    def __init__(
        self,
        vision_hidden_dim: int,
        num_registers: int = 16,
        register_dim: int = 256,
        tile_aggregation: str = "mean",
        init_std: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_registers <= 0:
            raise ValueError("num_registers must be positive")
        if register_dim <= 0:
            raise ValueError("register_dim must be positive")
        if tile_aggregation != "mean":
            raise ValueError(
                "InternVL PlanReg-WM-V1 supports only tile_register_aggregation='mean'; "
                f"got {tile_aggregation!r}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.vision_hidden_dim = int(vision_hidden_dim)
        self.num_registers = int(num_registers)
        self.register_dim = int(register_dim)
        self.tile_aggregation = tile_aggregation
        self.planning_registers = nn.Parameter(
            torch.empty(
                1,
                self.num_registers,
                self.vision_hidden_dim,
                **factory_kwargs,
            )
        )
        nn.init.normal_(self.planning_registers, mean=0.0, std=float(init_std))
        self.register_norm = nn.LayerNorm(
            self.vision_hidden_dim, **factory_kwargs
        )
        self.register_projection = nn.Linear(
            self.vision_hidden_dim,
            self.register_dim,
            **factory_kwargs,
        )

    @staticmethod
    def validate_vision_structure(vision_model: nn.Module) -> None:
        embeddings = getattr(vision_model, "embeddings", None)
        encoder = getattr(vision_model, "encoder", None)
        if embeddings is None:
            raise RuntimeError(
                "InternVL planning registers require vision_model.embeddings; "
                "the loaded trust_remote_code structure is unsupported"
            )
        if encoder is None:
            raise RuntimeError(
                "InternVL planning registers require vision_model.encoder; "
                "the loaded trust_remote_code structure is unsupported"
            )
        try:
            signature = inspect.signature(encoder.forward)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Cannot inspect InternVL vision_model.encoder.forward"
            ) from exc
        if "inputs_embeds" not in signature.parameters:
            raise RuntimeError(
                "InternVL planning registers require "
                "vision_model.encoder(inputs_embeds=...); got signature "
                f"{signature}"
            )
        layers = getattr(encoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError(
                "InternVL planning registers require a non-empty "
                "vision_model.encoder.layers"
            )

    @classmethod
    def validate_runtime_structure(cls, vlm_model: nn.Module) -> None:
        vision_model = getattr(vlm_model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError(
                "InternVL planning registers require vlm_model.vision_model"
            )
        cls.validate_vision_structure(vision_model)
        if getattr(vlm_model, "select_layer", -1) != -1:
            raise RuntimeError(
                "PlanReg-WM-V1 runs and consumes all InternViT encoder layers, "
                "so InternVL select_layer must be -1"
            )
        if not callable(getattr(vlm_model, "pixel_shuffle", None)):
            raise RuntimeError("InternVL model does not expose callable pixel_shuffle")
        if not isinstance(getattr(vlm_model, "mlp1", None), nn.Module):
            raise RuntimeError("InternVL model does not expose its mlp1 projector")

    def _encode_with_registers(
        self,
        vision_model: nn.Module,
        pixel_values: torch.Tensor,
    ):
        self.validate_vision_structure(vision_model)
        embedded = vision_model.embeddings(pixel_values)
        if embedded.ndim != 3 or embedded.shape[-1] != self.vision_hidden_dim:
            raise RuntimeError(
                "Unexpected InternViT embedding shape: "
                f"got {tuple(embedded.shape)}, expected [tiles,tokens,{self.vision_hidden_dim}]"
            )
        original_patch_count = embedded.shape[1] - 1
        if original_patch_count <= 0:
            raise RuntimeError("InternViT embeddings contain no patch tokens")

        registers = self.planning_registers.expand(embedded.shape[0], -1, -1)
        registers = registers.to(device=embedded.device, dtype=embedded.dtype)
        encoder_inputs = torch.cat(
            (embedded[:, :1], registers, embedded[:, 1:]), dim=1
        )
        encoder_outputs = vision_model.encoder(
            inputs_embeds=encoder_inputs,
            output_hidden_states=False,
            return_dict=True,
        )
        encoded = getattr(encoder_outputs, "last_hidden_state", None)
        if encoded is None:
            raise RuntimeError(
                "InternVL encoder(inputs_embeds=...) did not return last_hidden_state"
            )
        expected_tokens = 1 + self.num_registers + original_patch_count
        if encoded.shape[1] != expected_tokens:
            raise RuntimeError(
                "InternViT encoder changed the token count unexpectedly: "
                f"expected {expected_tokens}, got {encoded.shape[1]}"
            )

        encoded_registers = encoded[:, 1:1 + self.num_registers]
        encoded_patches = encoded[:, 1 + self.num_registers:]
        if encoded_patches.shape[1] != original_patch_count:
            raise RuntimeError(
                "Patch-token count changed after planning-register split: "
                f"expected {original_patch_count}, got {encoded_patches.shape[1]}"
            )
        return encoded_registers, encoded_patches

    def _project_registers(
        self,
        encoded_registers: torch.Tensor,
        num_patches_list: List[int],
    ):
        per_tile_registers = self.register_projection(
            self.register_norm(encoded_registers)
        )
        scene_registers = self._aggregate_tiles(
            per_tile_registers, num_patches_list
        )
        return per_tile_registers, scene_registers

    def encode_registers_only(
        self,
        vision_model: nn.Module,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
    ) -> torch.Tensor:
        """EMA-facing path that does not retain patch or language features."""
        encoded_registers, _ = self._encode_with_registers(
            vision_model, pixel_values
        )
        _, scene_registers = self._project_registers(
            encoded_registers, num_patches_list
        )
        return scene_registers

    def _aggregate_tiles(
        self,
        per_tile_registers: torch.Tensor,
        num_patches_list: List[int],
    ) -> torch.Tensor:
        counts = [int(count) for count in num_patches_list]
        if not counts or any(count <= 0 for count in counts):
            raise ValueError(
                f"num_patches_list must contain positive tile counts, got {counts}"
            )
        if sum(counts) != per_tile_registers.shape[0]:
            raise ValueError(
                "InternVL tile/register aggregation mismatch: "
                f"sum(num_patches_list)={sum(counts)} but encoded "
                f"{per_tile_registers.shape[0]} tiles"
            )
        if self.tile_aggregation != "mean":
            raise RuntimeError(
                f"Unsupported tile register aggregation {self.tile_aggregation!r}"
            )
        return torch.stack(
            [tile_group.mean(dim=0) for tile_group in per_tile_registers.split(counts)],
            dim=0,
        )

    def forward(
        self,
        vlm_model: nn.Module,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
    ) -> InternVLPlanningOutput:
        self.validate_runtime_structure(vlm_model)
        vision_model = vlm_model.vision_model
        encoded_registers, encoded_patches = self._encode_with_registers(
            vision_model, pixel_values
        )
        original_patch_count = encoded_patches.shape[1]

        grid_size = math.isqrt(original_patch_count)
        if grid_size * grid_size != original_patch_count:
            raise RuntimeError(
                "InternVL patch tokens cannot be square-reshaped after register "
                f"removal: {original_patch_count} is not a perfect square"
            )
        patch_grid = encoded_patches.reshape(
            encoded_patches.shape[0],
            grid_size,
            grid_size,
            encoded_patches.shape[-1],
        )
        shuffled_patches = vlm_model.pixel_shuffle(
            patch_grid,
            scale_factor=vlm_model.downsample_ratio,
        )
        shuffled_patches = shuffled_patches.reshape(
            shuffled_patches.shape[0], -1, shuffled_patches.shape[-1]
        )
        patch_features = vlm_model.mlp1(shuffled_patches)

        per_tile_registers, scene_registers = self._project_registers(
            encoded_registers, num_patches_list
        )
        return InternVLPlanningOutput(
            patch_features=patch_features,
            per_tile_registers=per_tile_registers,
            scene_registers=scene_registers,
            encoded_patches=encoded_patches,
        )
