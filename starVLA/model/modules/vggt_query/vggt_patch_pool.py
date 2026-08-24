"""Parameter-free per-view pooling for dense final-layer VGGT patch tokens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


GEOMETRY_VIEW_ORDER = ("cam_f0", "cam_l0", "cam_r0")


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


def _pool_channels(
    values: Tensor,
    height: int,
    width: int,
    output_hw: tuple[int, int],
) -> Tensor:
    channels = int(values.shape[-1])
    image = values.reshape(height, width, channels).permute(2, 0, 1).unsqueeze(0)
    return F.adaptive_avg_pool2d(image, output_hw).flatten(2).transpose(1, 2).squeeze(0)


def _pool_one_geometry_payload(
    payload: Mapping[str, Tensor],
    output_hw: tuple[int, int],
    device: torch.device,
) -> dict[str, Tensor]:
    required = {
        "features",
        "valid_mask",
        "view_ids",
        "uv_coords",
        "ray_features",
        "patch_grid_hw",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Dense VGGT payload is missing keys: {sorted(missing)}")
    features = torch.as_tensor(payload["features"])
    valid_mask = torch.as_tensor(payload["valid_mask"])
    view_ids = torch.as_tensor(payload["view_ids"])
    uv_coords = torch.as_tensor(payload["uv_coords"])
    ray_features = torch.as_tensor(payload["ray_features"])
    grids_tensor = torch.as_tensor(payload["patch_grid_hw"])
    if features.ndim != 2:
        raise ValueError("Dense VGGT features must be [N,Dv]")
    count = int(features.shape[0])
    if valid_mask.shape != (count,) or valid_mask.dtype != torch.bool:
        raise ValueError("Dense VGGT valid_mask must be bool[N]")
    if not valid_mask.all():
        raise ValueError("Geometry pooling does not accept invalid or padded tokens")
    if view_ids.shape != (count,):
        raise ValueError("Dense VGGT view_ids must be [N]")
    if uv_coords.shape != (count, 2):
        raise ValueError("Dense VGGT uv_coords must be [N,2]")
    if ray_features.shape != (count, 6):
        raise ValueError("Dense VGGT ray_features must be [N,6]")
    if grids_tensor.shape != (3, 2):
        raise ValueError("Dense VGGT patch_grid_hw must be [3,2]")
    grids = [(int(row[0]), int(row[1])) for row in grids_tensor.tolist()]
    if any(height <= 0 or width <= 0 for height, width in grids):
        raise ValueError("All three views must have positive patch grids")
    if sum(height * width for height, width in grids) != count:
        raise ValueError("patch_grid_hw does not match the dense token count")
    for name, value in (
        ("features", features),
        ("uv_coords", uv_coords),
        ("ray_features", ray_features),
    ):
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Dense VGGT {name} must be finite")

    features = features.detach().to(device=device, dtype=torch.float32)
    uv_coords = uv_coords.detach().to(device=device, dtype=torch.float32)
    ray_features = ray_features.detach().to(device=device, dtype=torch.float32)
    view_ids = view_ids.detach().to(device=device, dtype=torch.long)
    outputs = {"features": [], "view_ids": [], "uv_coords": [], "ray_features": []}
    offset = 0
    slots_per_view = int(output_hw[0] * output_hw[1])
    for view_index, (height, width) in enumerate(grids):
        view_count = height * width
        sl = slice(offset, offset + view_count)
        if not view_ids[sl].eq(view_index).all():
            raise ValueError(
                "Dense VGGT tokens must be ordered cam_f0 -> cam_l0 -> cam_r0"
            )
        pooled_features = _pool_channels(features[sl], height, width, output_hw)
        pooled_uv = _pool_channels(uv_coords[sl], height, width, output_hw)
        pooled_origins = _pool_channels(
            ray_features[sl, :3], height, width, output_hw
        )
        pooled_directions = _pool_channels(
            ray_features[sl, 3:], height, width, output_hw
        )
        pooled_directions = F.normalize(pooled_directions, dim=-1, eps=1e-12)
        outputs["features"].append(pooled_features)
        outputs["uv_coords"].append(pooled_uv)
        outputs["ray_features"].append(
            torch.cat((pooled_origins, pooled_directions), dim=-1)
        )
        outputs["view_ids"].append(
            torch.full(
                (slots_per_view,), view_index, device=device, dtype=torch.long
            )
        )
        offset += view_count
    return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def pool_dense_vggt_geometry_per_view(
    payloads: Sequence[Mapping[str, Tensor]],
    output_hw: tuple[int, int] = (6, 10),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """Pool features and spatial metadata independently per camera view."""

    if not payloads:
        raise ValueError("Dense VGGT geometry pooling requires a non-empty batch")
    if tuple(int(value) for value in output_hw) != (6, 10):
        raise ValueError("GP-SQ3D-Mix pooling layout must be 3x6x10")
    first = torch.as_tensor(payloads[0]["features"])
    target_device = torch.device(device) if device is not None else first.device
    pooled = [
        _pool_one_geometry_payload(payload, (6, 10), target_device)
        for payload in payloads
    ]
    feature_dim = int(pooled[0]["features"].shape[-1])
    if any(int(item["features"].shape[-1]) != feature_dim for item in pooled):
        raise ValueError("Dense VGGT batch must share one feature dimension")
    result = {
        key: torch.stack([item[key] for item in pooled], dim=0)
        for key in ("features", "view_ids", "uv_coords", "ray_features")
    }
    result["features"] = result["features"].to(dtype=dtype or first.dtype)
    result["uv_coords"] = result["uv_coords"].to(dtype=dtype or torch.float32)
    result["ray_features"] = result["ray_features"].to(
        dtype=dtype or torch.float32
    )
    expected_slots = len(GEOMETRY_VIEW_ORDER) * 6 * 10
    if result["features"].shape[1] != expected_slots:
        raise RuntimeError("Geometry pooling did not produce exactly 180 slots")
    return result
