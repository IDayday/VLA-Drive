"""Explicit camera-coordinate transforms and differentiable projection."""

from __future__ import annotations

import torch


def make_ego_bev_anchors(
    field_size,
    x_range_m,
    y_range_m,
    height_anchors_m,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return ego anchors ``[Ny,Nx,Z,3]`` in float32.

    The spatial convention is x-forward on Ny, y-left on Nx and z-up on Z.
    Cell centers are used so writer and geometry supervision share exactly the
    same coordinate lattice.
    """

    if len(field_size) != 2 or min(field_size) <= 0:
        raise ValueError("field_size must be positive [Ny,Nx]")
    if len(x_range_m) != 2 or x_range_m[0] >= x_range_m[1]:
        raise ValueError("x_range_m must be increasing")
    if len(y_range_m) != 2 or y_range_m[0] >= y_range_m[1]:
        raise ValueError("y_range_m must be increasing")
    if len(height_anchors_m) < 1:
        raise ValueError("height_anchors_m cannot be empty")
    ny, nx = int(field_size[0]), int(field_size[1])
    dx = (float(x_range_m[1]) - float(x_range_m[0])) / ny
    dy = (float(y_range_m[1]) - float(y_range_m[0])) / nx
    x = torch.linspace(
        float(x_range_m[0]) + 0.5 * dx,
        float(x_range_m[1]) - 0.5 * dx,
        ny,
        device=device,
        dtype=torch.float32,
    )
    y = torch.linspace(
        float(y_range_m[0]) + 0.5 * dy,
        float(y_range_m[1]) - 0.5 * dy,
        nx,
        device=device,
        dtype=torch.float32,
    )
    z = torch.as_tensor(
        height_anchors_m, device=device, dtype=torch.float32
    )
    return torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)


def center_crop_xywh(raw_image_hw: torch.Tensor, output_hw: torch.Tensor) -> torch.Tensor:
    """Return center-crop boxes ``[...,4]`` matching aspect-ratio resize."""

    if raw_image_hw.shape != output_hw.shape or raw_image_hw.shape[-1] != 2:
        raise ValueError("raw_image_hw and output_hw must share shape [...,2]")
    raw = raw_image_hw.to(dtype=torch.float32)
    output = output_hw.to(device=raw.device, dtype=torch.float32)
    raw_h, raw_w = raw[..., 0], raw[..., 1]
    dst_h, dst_w = output[..., 0], output[..., 1]
    src_aspect, dst_aspect = raw_w / raw_h, dst_w / dst_h
    crop_w = torch.where(src_aspect > dst_aspect, raw_h * dst_aspect, raw_w)
    crop_h = torch.where(src_aspect < dst_aspect, raw_w / dst_aspect, raw_h)
    left = (raw_w - crop_w) * 0.5
    top = (raw_h - crop_h) * 0.5
    return torch.stack((left, top, crop_w, crop_h), dim=-1)


def scale_intrinsics_for_crop_resize(
    intrinsics: torch.Tensor,
    crop_xywh: torch.Tensor,
    output_hw: torch.Tensor,
) -> torch.Tensor:
    """Transform K after center/general crop and resize.

    Args:
        intrinsics: ``[...,3,3]`` in raw pixel coordinates.
        crop_xywh: ``[...,4]`` as left, top, width, height in raw pixels.
        output_hw: ``[...,2]`` as output height, width.
    """

    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in [3,3]")
    if crop_xywh.shape != (*intrinsics.shape[:-2], 4):
        raise ValueError("crop_xywh leading dimensions must match intrinsics")
    if output_hw.shape != (*intrinsics.shape[:-2], 2):
        raise ValueError("output_hw leading dimensions must match intrinsics")
    work = intrinsics.to(dtype=torch.float32).clone()
    crop = crop_xywh.to(device=work.device, dtype=torch.float32)
    output = output_hw.to(device=work.device, dtype=torch.float32)
    sx = output[..., 1] / crop[..., 2]
    sy = output[..., 0] / crop[..., 3]
    work[..., 0, 0] *= sx
    work[..., 1, 1] *= sy
    work[..., 0, 2] = (work[..., 0, 2] - crop[..., 0]) * sx
    work[..., 1, 2] = (work[..., 1, 2] - crop[..., 1]) * sy
    return work


def sensor_to_lidar_to_ego_to_camera(
    sensor_to_lidar: torch.Tensor,
    lidar_to_planning_ego: torch.Tensor,
) -> torch.Tensor:
    """Build planning-ego→camera from explicit transform chain.

    ``sensor_to_lidar`` is ``[B,V,4,4]`` and maps camera/sensor coordinates
    into lidar. ``lidar_to_planning_ego`` is ``[B,4,4]``. No identity-frame
    assumption is made inside this function.
    """

    if sensor_to_lidar.ndim != 4 or sensor_to_lidar.shape[-2:] != (4, 4):
        raise ValueError("sensor_to_lidar must have shape [B,V,4,4]")
    if lidar_to_planning_ego.shape != (sensor_to_lidar.shape[0], 4, 4):
        raise ValueError("lidar_to_planning_ego must have shape [B,4,4]")
    work = sensor_to_lidar.to(dtype=torch.float32)
    lidar_to_ego = lidar_to_planning_ego.to(device=work.device, dtype=torch.float32)
    sensor_to_ego = lidar_to_ego[:, None] @ work
    return torch.linalg.inv(sensor_to_ego)


def project_ego_points(
    points_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    ego_to_camera: torch.Tensor,
    image_hw: torch.Tensor,
    min_depth: float = 1e-4,
):
    """Project ego points into all views using float32 geometry math.

    Args:
        points_ego: ``[B,...,3]``.
        intrinsics: ``[B,V,3,3]``.
        ego_to_camera: ``[B,V,4,4]`` (camera z forward).
        image_hw: ``[B,V,2]`` ordered height, width.

    Returns:
        pixels ``[B,V,...,2]``, valid mask ``[B,V,...]`` and camera depth
        ``[B,V,...]``.
    """

    if points_ego.ndim < 3 or points_ego.shape[-1] != 3:
        raise ValueError("points_ego must have shape [B,...,3]")
    batch = points_ego.shape[0]
    if intrinsics.ndim != 4 or intrinsics.shape[0] != batch or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B,V,3,3]")
    if ego_to_camera.shape != (*intrinsics.shape[:2], 4, 4):
        raise ValueError("ego_to_camera must have shape [B,V,4,4]")
    if image_hw.shape != (*intrinsics.shape[:2], 2):
        raise ValueError("image_hw must have shape [B,V,2]")

    spatial_shape = points_ego.shape[1:-1]
    points = points_ego.reshape(batch, -1, 3).to(dtype=torch.float32)
    ones = torch.ones((*points.shape[:-1], 1), device=points.device, dtype=points.dtype)
    homogeneous = torch.cat((points, ones), dim=-1)
    extrinsic = ego_to_camera.to(device=points.device, dtype=torch.float32)
    camera = torch.einsum("bvij,bnj->bvni", extrinsic, homogeneous)[..., :3]
    depth = camera[..., 2]
    safe_depth = depth.clamp_min(min_depth)
    normalized = camera / safe_depth[..., None]
    k = intrinsics.to(device=points.device, dtype=torch.float32)
    pixel_h = torch.einsum("bvij,bvnj->bvni", k, normalized)
    pixels = pixel_h[..., :2]
    hw = image_hw.to(device=points.device, dtype=torch.float32)
    height, width = hw[..., 0, None], hw[..., 1, None]
    valid = (
        (depth > min_depth)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < height)
    )
    views = intrinsics.shape[1]
    pixels = pixels.reshape(batch, views, *spatial_shape, 2)
    valid = valid.reshape(batch, views, *spatial_shape)
    depth = depth.reshape(batch, views, *spatial_shape)
    return pixels, valid, depth
