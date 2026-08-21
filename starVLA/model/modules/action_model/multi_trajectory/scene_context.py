"""Typed scene-memory contract for the DDP-DRS planning stack."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SceneContext:
    """Qwen-derived global and dense scene memory.

    ``memory_key_padding_mask`` follows PyTorch attention semantics: ``True``
    denotes a token that must be ignored.
    """

    global_scene_tokens: torch.Tensor
    dense_scene_memory: torch.Tensor
    memory_key_padding_mask: torch.Tensor

    def validate(
        self,
        *,
        expected_num_queries: int = 16,
        expected_scene_dim: int = 2048,
        check_finite: bool = False,
    ) -> None:
        global_tokens = self.global_scene_tokens
        dense_memory = self.dense_scene_memory
        padding_mask = self.memory_key_padding_mask
        if global_tokens.ndim != 3:
            raise ValueError("global_scene_tokens must have shape [B, Q, scene_dim]")
        if dense_memory.ndim != 3:
            raise ValueError("dense_scene_memory must have shape [B, L, scene_dim]")
        if padding_mask.ndim != 2:
            raise ValueError("memory_key_padding_mask must have shape [B, L]")
        if global_tokens.shape[1] != expected_num_queries:
            raise ValueError(
                f"expected {expected_num_queries} global scene queries, got "
                f"{global_tokens.shape[1]}"
            )
        if global_tokens.shape[2] != expected_scene_dim:
            raise ValueError(
                f"global scene width must be {expected_scene_dim}, got "
                f"{global_tokens.shape[2]}"
            )
        if dense_memory.shape[2] != expected_scene_dim:
            raise ValueError(
                f"dense scene width must be {expected_scene_dim}, got "
                f"{dense_memory.shape[2]}"
            )
        if global_tokens.shape[0] != dense_memory.shape[0]:
            raise ValueError("global and dense scene-memory batch sizes differ")
        if tuple(padding_mask.shape) != tuple(dense_memory.shape[:2]):
            raise ValueError("dense scene length and padding-mask length differ")
        if padding_mask.dtype is not torch.bool:
            raise TypeError("memory_key_padding_mask must have dtype torch.bool")
        if padding_mask.all(dim=1).any():
            raise ValueError("every sample must contain at least one valid scene token")
        if not (
            global_tokens.device == dense_memory.device == padding_mask.device
        ):
            raise ValueError("all SceneContext tensors must be on the same device")
        if check_finite and (
            not torch.isfinite(global_tokens).all()
            or not torch.isfinite(dense_memory).all()
        ):
            raise ValueError("SceneContext contains NaN or Inf")
