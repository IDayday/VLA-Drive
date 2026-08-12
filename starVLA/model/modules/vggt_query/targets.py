"""Convert DPT-pre VGGT aggregator tokens into compact query targets."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from starVLA.model.modules.vggt_query.resolution_probe import (
    crop_and_pool_valid_patches,
)


def select_vggt_global_teacher_layer(
    aggregated_tokens: Sequence[Optional[torch.Tensor]],
    *,
    layer_index: int = 11,
    branch_dim: int = 1024,
) -> torch.Tensor:
    """Select pure global-attention features ``[B,V,N,1024]`` from VGGT.

    Official VGGT cached DPT inputs concatenate frame and global branches as
    ``[..., 2*C]``. The global branch is the second half. ``layer_index=11``
    matches the DPT cached layer convention used by VGGT.
    """

    assert layer_index >= 0 and branch_dim > 0
    if layer_index >= len(aggregated_tokens) or aggregated_tokens[layer_index] is None:
        raise RuntimeError(f"VGGT aggregator did not cache requested layer {layer_index}")
    combined = aggregated_tokens[layer_index]
    assert combined is not None
    assert combined.ndim == 4, "VGGT cached tokens must be [B,V,N,2C]"
    assert combined.shape[-1] == 2 * branch_dim, (
        f"VGGT cached feature dim must be frame+global={2 * branch_dim}, "
        f"found {combined.shape[-1]}"
    )
    selected = combined[..., branch_dim:].contiguous()
    assert selected.shape == (*combined.shape[:-1], branch_dim)
    return selected


def extract_vggt_spatial_query_targets(
    layer_tokens: torch.Tensor,
    *,
    spatial_validity: torch.Tensor,
    patch_start_idx: int = 5,
    patch_grid_size: int = 37,
    output_size: tuple[int, int] = (6, 10),
    minimum_valid_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 180 pure-spatial teacher queries from layer-11 global tokens.

    Args:
        layer_tokens: Pure global VGGT features ``[B,V,5+G*G,D]``.
        spatial_validity: Source content coverage ``[B,V,G,G]``.

    Returns:
        View-major spatial features ``[B,V*R*C,D]`` and validity
        ``[B,V*R*C]``. Camera/register tokens are intentionally excluded.
    """

    assert layer_tokens.ndim == 4, "layer_tokens must be [B,V,N,D]"
    batch, views, token_count, feature_dim = layer_tokens.shape
    assert token_count == patch_start_idx + patch_grid_size**2
    assert spatial_validity.shape == (batch, views, patch_grid_size, patch_grid_size)
    rows, cols = int(output_size[0]), int(output_size[1])
    assert rows > 0 and cols > 0
    patches = layer_tokens[:, :, patch_start_idx:].reshape(
        batch, views, patch_grid_size, patch_grid_size, feature_dim
    )
    pooled, valid_grid = crop_and_pool_valid_patches(
        patches,
        spatial_validity,
        output_size=(rows, cols),
        minimum_coverage=minimum_valid_ratio,
    )
    features = pooled.reshape(batch, views * rows * cols, feature_dim).contiguous()
    valid_mask = valid_grid.reshape(batch, views * rows * cols).contiguous()
    assert features.shape == (batch, views * rows * cols, feature_dim)
    assert valid_mask.shape == features.shape[:2]
    return features, valid_mask


def extract_vggt_layer11_memory_targets(
    layer_tokens: torch.Tensor,
    *,
    spatial_validity: torch.Tensor,
    patch_start_idx: int = 5,
    patch_grid_size: int = 37,
    output_size: tuple[int, int] = (6, 10),
    minimum_valid_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the V2 teacher memory ``[B,15+V*R*C,D]``.

    The first 15 slots are the five pure global-branch special tokens from
    each of three views. The remaining 180 slots are cropped, pooled spatial
    features in view-major order.
    """

    assert layer_tokens.ndim == 4, "layer_tokens must be [B,V,N,D]"
    batch, views, _, feature_dim = layer_tokens.shape
    assert views == 3, "V2 teacher memory requires front/left/right views"
    assert patch_start_idx == 5, "V2 global contract requires five tokens per view"
    special = layer_tokens[:, :, :patch_start_idx].reshape(
        batch, views * patch_start_idx, feature_dim
    )
    spatial, spatial_mask = extract_vggt_spatial_query_targets(
        layer_tokens,
        spatial_validity=spatial_validity,
        patch_start_idx=patch_start_idx,
        patch_grid_size=patch_grid_size,
        output_size=output_size,
        minimum_valid_ratio=minimum_valid_ratio,
    )
    special_mask = torch.ones(
        batch,
        views * patch_start_idx,
        device=layer_tokens.device,
        dtype=torch.bool,
    )
    features = torch.cat((special, spatial), dim=1).contiguous()
    valid_mask = torch.cat((special_mask, spatial_mask), dim=1).contiguous()
    expected = views * patch_start_idx + views * output_size[0] * output_size[1]
    assert features.shape == (batch, expected, feature_dim)
    assert valid_mask.shape == features.shape[:2]
    return features, valid_mask


def extract_vggt_query_targets(
    final_tokens: torch.Tensor,
    *,
    patch_start_idx: int = 5,
    patch_grid_size: int = 37,
    pooled_grid_size: int = 4,
    spatial_validity: Optional[torch.Tensor] = None,
    minimum_valid_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact teacher features from final DPT-pre VGGT tokens.

    Args:
        final_tokens: VGGT aggregator output ``[B, V, N, D]``.
        spatial_validity: Optional source patch validity ``[B, V, G, G]``.

    Returns:
        features ``[B, V*5 + V*P*P, D]`` and boolean valid mask
        ``[B, V*5 + V*P*P]``. Special tokens precede spatial tokens.
    """

    assert final_tokens.ndim == 4, "final_tokens must be [B,V,N,D]"
    batch, views, token_count, feature_dim = final_tokens.shape
    expected_tokens = patch_start_idx + patch_grid_size * patch_grid_size
    assert token_count == expected_tokens, (
        f"VGGT token count mismatch: expected {expected_tokens}, found {token_count}"
    )
    assert 0 < pooled_grid_size <= patch_grid_size

    special = final_tokens[:, :, :patch_start_idx].reshape(
        batch, views * patch_start_idx, feature_dim
    )
    patches = final_tokens[:, :, patch_start_idx:]
    patches = patches.reshape(batch * views, patch_grid_size, patch_grid_size, feature_dim)
    patches = patches.permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(patches, (pooled_grid_size, pooled_grid_size))
    pooled = pooled.permute(0, 2, 3, 1).reshape(
        batch, views * pooled_grid_size * pooled_grid_size, feature_dim
    )
    pooled = pooled.to(dtype=final_tokens.dtype)
    features = torch.cat((special, pooled), dim=1).contiguous()

    special_mask = torch.ones(
        batch,
        views * patch_start_idx,
        dtype=torch.bool,
        device=final_tokens.device,
    )
    if spatial_validity is None:
        spatial_mask = torch.ones(
            batch,
            views * pooled_grid_size * pooled_grid_size,
            dtype=torch.bool,
            device=final_tokens.device,
        )
    else:
        assert spatial_validity.shape == (
            batch,
            views,
            patch_grid_size,
            patch_grid_size,
        ), "spatial_validity must be [B,V,G,G]"
        ratios = F.adaptive_avg_pool2d(
            spatial_validity.reshape(batch * views, 1, patch_grid_size, patch_grid_size).float(),
            (pooled_grid_size, pooled_grid_size),
        )
        spatial_mask = ratios.reshape(batch, -1) >= float(minimum_valid_ratio)
    valid_mask = torch.cat((special_mask, spatial_mask), dim=1)
    assert features.shape[:2] == valid_mask.shape
    return features, valid_mask
