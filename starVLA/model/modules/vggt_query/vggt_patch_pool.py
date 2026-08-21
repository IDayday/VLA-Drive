"""Parameter-free per-view pooling for dense final-layer VGGT patch tokens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _pool_one_payload(
    payload: Mapping[str, Tensor],
    output_hw: tuple[int, int],
    device: torch.device,
) -> Tensor:
    required = {"features", "valid_mask", "patch_grid_hw"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Dense VGGT payload is missing keys: {sorted(missing)}")

    features = payload["features"]
    valid_mask = payload["valid_mask"]
    patch_grid_hw = payload["patch_grid_hw"]
    if features.ndim != 2:
        raise ValueError("Dense VGGT features must be [N,Dv]")
    token_count, feature_dim = features.shape
    if valid_mask.shape != (token_count,) or valid_mask.dtype != torch.bool:
        raise ValueError("Dense VGGT valid_mask must be bool[N]")
    if not valid_mask.all():
        raise ValueError("Dense VGGT pooling does not accept invalid or padded tokens")
    if patch_grid_hw.shape != (3, 2):
        raise ValueError("Dense VGGT patch_grid_hw must be [3,2]")
    grids = [[int(value) for value in row] for row in patch_grid_hw.tolist()]
    if any(height <= 0 or width <= 0 for height, width in grids):
        raise ValueError("Every dense VGGT view must have a positive patch grid")
    expected_count = sum(height * width for height, width in grids)
    if expected_count != token_count:
        raise ValueError(
            f"Dense VGGT patch grids sum to {expected_count}, found N={token_count}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("Dense VGGT features must be finite")

    features = features.detach().to(device=device, dtype=torch.float32, non_blocking=True)
    pooled_views = []
    offset = 0
    for height, width in grids:
        count = height * width
        view = features[offset : offset + count].reshape(height, width, feature_dim)
        view = view.permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(view, output_size=output_hw)
        pooled_views.append(pooled.flatten(2).transpose(1, 2).squeeze(0))
        offset += count
    return torch.cat(pooled_views, dim=0)


def pool_dense_vggt_per_view(
    payloads: Sequence[Mapping[str, Tensor]],
    output_hw: tuple[int, int] = (6, 10),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pool front, left, and right patch grids independently, then stack a batch."""

    if not payloads:
        raise ValueError("Dense VGGT pooling requires a non-empty payload batch")
    if len(output_hw) != 2 or any(int(value) <= 0 for value in output_hw):
        raise ValueError("output_hw must contain two positive integers")
    first_features = payloads[0].get("features")
    if not torch.is_tensor(first_features):
        raise TypeError("Dense VGGT features must be tensors")
    target_device = torch.device(device) if device is not None else first_features.device
    feature_dims = set()
    for payload in payloads:
        features = payload.get("features")
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError("Every dense VGGT feature payload must be [N,Dv]")
        feature_dims.add(int(features.shape[-1]))
    if len(feature_dims) != 1:
        raise ValueError("Dense VGGT batch must share one feature dimension")

    pooled = torch.stack(
        [
            _pool_one_payload(
                payload,
                (int(output_hw[0]), int(output_hw[1])),
                target_device,
            )
            for payload in payloads
        ],
        dim=0,
    )
    target_dtype = dtype if dtype is not None else first_features.dtype
    return pooled.to(dtype=target_dtype)
