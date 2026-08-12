"""CPU-friendly statistics for comparing VGGT spatial pooling layouts."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def crop_and_pool_valid_patches(
    patches: torch.Tensor,
    spatial_validity: torch.Tensor,
    *,
    output_size: tuple[int, int],
    minimum_coverage: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop padding and pool VGGT patches into a rectangular content grid.

    Args:
        patches: Dense patch features ``[B,V,G,G,D]``.
        spatial_validity: Source-patch content coverage ``[B,V,G,G]``.
        output_size: Target spatial layout ``(rows, cols)``.

    Returns:
        Pooled features ``[B,V,R,C,D]`` and validity ``[B,V,R,C]``.
    """

    assert patches.ndim == 5, "patches must be [B,V,G,G,D]"
    assert spatial_validity.shape == patches.shape[:4]
    assert spatial_validity.dtype.is_floating_point
    rows, cols = (int(output_size[0]), int(output_size[1]))
    assert rows > 0 and cols > 0
    batch, views, _, _, feature_dim = patches.shape
    pooled_batches = []
    mask_batches = []
    for batch_index in range(batch):
        pooled_views = []
        mask_views = []
        for view_index in range(views):
            validity = spatial_validity[batch_index, view_index].float()
            occupied = validity > 0
            assert occupied.any(), "every image needs at least one valid source patch"
            occupied_rows = occupied.any(dim=1).nonzero(as_tuple=False).flatten()
            occupied_cols = occupied.any(dim=0).nonzero(as_tuple=False).flatten()
            row_start, row_end = int(occupied_rows[0]), int(occupied_rows[-1]) + 1
            col_start, col_end = int(occupied_cols[0]), int(occupied_cols[-1]) + 1

            cropped_features = patches[
                batch_index, view_index, row_start:row_end, col_start:col_end
            ].permute(2, 0, 1).float()
            cropped_validity = validity[
                row_start:row_end, col_start:col_end
            ].unsqueeze(0)
            numerator = F.adaptive_avg_pool2d(
                cropped_features * cropped_validity,
                (rows, cols),
            )
            denominator = F.adaptive_avg_pool2d(
                cropped_validity,
                (rows, cols),
            )
            pooled = numerator / denominator.clamp_min(1e-6)
            pooled_views.append(pooled.permute(1, 2, 0))
            mask_views.append(denominator.squeeze(0) >= float(minimum_coverage))
        pooled_batches.append(torch.stack(pooled_views))
        mask_batches.append(torch.stack(mask_views))
    pooled_tensor = torch.stack(pooled_batches).to(dtype=patches.dtype)
    mask_tensor = torch.stack(mask_batches)
    assert pooled_tensor.shape == (batch, views, rows, cols, feature_dim)
    assert mask_tensor.shape == (batch, views, rows, cols)
    return pooled_tensor, mask_tensor


class NormalizedSlotStatistics:
    """Stream statistics over directional features ``[B,Q,D]``.

    ``update`` L2-normalizes the supplied features. Callers that need the
    training aligner's exact contract should apply per-token LayerNorm first.
    The implementation stores only per-slot sums, not per-sample features.
    """

    def __init__(self, slot_count: int, feature_dim: int) -> None:
        assert slot_count > 0 and feature_dim > 0
        self.feature_dim = int(feature_dim)
        self.count = torch.zeros(slot_count, dtype=torch.float64)
        self.unit_sum = torch.zeros(slot_count, feature_dim, dtype=torch.float64)

    def update(self, features: torch.Tensor, valid_mask: torch.Tensor) -> None:
        assert features.ndim == 3, "features must be [B,Q,D]"
        assert features.shape[1:] == self.unit_sum.shape
        assert valid_mask.shape == features.shape[:2]
        assert valid_mask.dtype == torch.bool
        unit = F.normalize(features.detach().float(), dim=-1, eps=1e-6).cpu().double()
        valid = valid_mask.detach().cpu()
        self.count += valid.double().sum(dim=0)
        self.unit_sum += (unit * valid.unsqueeze(-1)).sum(dim=0)

    def summary(self) -> Dict[str, float | int]:
        active = self.count > 0
        assert active.any(), "statistics contain no valid feature observations"
        counts = self.count[active]
        sums = self.unit_sum[active]
        observations = counts.sum()
        sum_norm_squared = sums.square().sum(dim=-1)

        # Sum_i cos(x_i, normalized(sum_i x_i)) equals ||sum_i x_i||.
        template_cosine = sum_norm_squared.sqrt().sum() / observations
        pair_counts = counts * (counts - 1)
        pair_mask = pair_counts > 0
        if pair_mask.any():
            pair_similarity_sum = (sum_norm_squared[pair_mask] - counts[pair_mask]).sum()
            pair_similarity = pair_similarity_sum / pair_counts[pair_mask].sum()
        else:
            pair_similarity = torch.tensor(float("nan"), dtype=torch.float64)

        # For unit vectors: sum_i ||x_i - mean||^2 = n - ||sum||^2 / n.
        residual_energy = (counts - sum_norm_squared / counts).sum()
        residual_rms = torch.sqrt(
            residual_energy.clamp_min(0) / (observations * self.feature_dim)
        )
        return {
            "valid_slots": int(active.sum()),
            "observations": int(observations),
            "slot_template_cosine": float(template_cosine),
            "same_slot_cross_scene_cosine": float(pair_similarity),
            "cross_scene_residual_rms": float(residual_rms),
        }


def summarize_scene_descriptors(
    descriptors: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> Dict[str, float | int]:
    """Summarize scene discrimination from descriptors ``[N,D]``.

    Pairwise similarities are evaluated in chunks so a 4k-scene probe does
    not materialize a full similarity matrix.
    """

    assert descriptors.ndim == 2, "descriptors must be [N,D]"
    assert descriptors.shape[0] >= 2, "at least two scenes are required"
    assert chunk_size > 0
    values = F.normalize(descriptors.detach().float().cpu(), dim=-1, eps=1e-6)
    sample_count = values.shape[0]
    cross_sum = 0.0
    nearest_values = []
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        similarities = values[start:end] @ values.transpose(0, 1)
        local_rows = torch.arange(end - start)
        global_rows = torch.arange(start, end)
        cross_sum += float(similarities.sum() - similarities[local_rows, global_rows].sum())
        similarities[local_rows, global_rows] = -torch.inf
        nearest_values.append(similarities.max(dim=1).values)
    nearest = torch.cat(nearest_values)
    pair_count = sample_count * (sample_count - 1)
    return {
        "scene_count": int(sample_count),
        "cross_scene_cosine_mean": cross_sum / pair_count,
        "nearest_other_cosine_mean": float(nearest.mean()),
        "nearest_other_cosine_p95": float(torch.quantile(nearest, 0.95)),
        "self_to_nearest_margin_mean": float((1.0 - nearest).mean()),
    }
