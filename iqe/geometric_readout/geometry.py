"""Current-observation camera geometry, independent of labels and learned heads.

``navsim_pinhole`` deliberately follows the locked NAVSIM visualization function.
``brown_conrady`` is available for the calibration audit, not an implicit switch.
The presence of distortion metadata does not prove that supplied images need a
second distortion correction. No road height, future actor or score is inferred.
"""
import torch
from iqe.contracts import require


@torch.no_grad()
def project_points(points, intrinsics, camera_to_ego, image_sizes, *,
                   camera_model="navsim_pinhole", distortion=None):
    """B,K,Q,3 ego points -> B,K,V,Q,2 pixels and true visibility mask.

    image_sizes is B,V,2 in width,height order. Calibration must already include
    the original crop/resize. Points are in the Base rear-axle ego coordinate
    frame; the audited data source has identity lidar2ego. Behind-camera and
    out-of-frame points stay invalid, rather than being clamped into the image.
    """
    require(points.ndim == 4 and points.shape[-1] == 3, "ego point shape")
    b = len(points)
    require(intrinsics.ndim == 4 and intrinsics.shape[0] == b
            and intrinsics.shape[-2:] == (3, 3), "intrinsics shape")
    v = intrinsics.shape[1]
    require(camera_to_ego.shape == (b, v, 4, 4)
            and image_sizes.shape == (b, v, 2), "camera calibration shape")
    require(all(torch.isfinite(x).all().item() for x in
                (points, intrinsics, camera_to_ego, image_sizes)), "nonfinite geometry")
    require((image_sizes > 1).all().item(), "image size")
    rotation = camera_to_ego[..., :3, :3].detach()
    eye = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    require(torch.allclose(rotation.transpose(-1, -2) @ rotation, eye.expand_as(rotation),
                           atol=1e-5, rtol=1e-5)
            and torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]),
                               atol=1e-5, rtol=1e-5), "camera rotation must be proper")
    bottom = camera_to_ego[..., 3, :]
    require(torch.equal(bottom, bottom.new_tensor([0, 0, 0, 1]).expand_as(bottom)),
            "camera homogeneous transform")
    delta = points[:, :, None] - camera_to_ego[:, None, :, None, :3, 3]
    # Row-vector inverse transform: (ego - translation) @ camera_to_ego.R.
    camera = torch.einsum("bkvqi,bvij->bkvqj", delta, rotation)
    z = camera[..., 2]
    normalized = camera[..., :2] / z.clamp_min(1e-3)[..., None]
    visible = z > 1e-3
    if camera_model == "brown_conrady":
        require(distortion is not None and distortion.shape == (b, v, 5)
                and torch.isfinite(distortion).all().item(), "Caltech distortion required")
        k1, k2, p1, p2, k3 = (distortion[:, None, :, None, i] for i in range(5))
        x, y = normalized.unbind(-1)
        r2 = (x.square() + y.square()).clamp_max(1e4)
        radial = 1 + k1*r2 + k2*r2.square() + k3*r2.pow(3)
        derivative = 1 + 3*k1*r2 + 5*k2*r2.square() + 7*k3*r2.pow(3)
        normalized = torch.stack((x*radial + 2*p1*x*y + p2*(r2 + 2*x.square()),
                                  y*radial + p1*(r2 + 2*y.square()) + 2*p2*x*y), -1)
        visible = visible & (radial > 0) & (derivative > 0)
    else:
        require(camera_model == "navsim_pinhole", "unsupported camera model")
    if camera_model == "navsim_pinhole":
        projected = torch.einsum("bkvqi,bvji->bkvqj", camera, intrinsics)
        pixel = projected[..., :2] / projected[..., 2:].clamp_min(1e-3)
    else:
        homogeneous = torch.cat((normalized, torch.ones_like(z[..., None])), -1)
        pixel = torch.einsum("bkvqi,bvji->bkvqj", homogeneous, intrinsics)[..., :2]
    bounds = image_sizes[:, None, :, None] - 1
    visible &= (pixel > 0).all(-1) & (pixel < bounds).all(-1) & torch.isfinite(pixel).all(-1)
    return pixel, visible


@torch.no_grad()
def candidate_anchors(trajectory, heights=(0., 1.)):
    """Two explicit height hypotheses per time point, not measured road height."""
    require(trajectory.ndim == 4 and trajectory.shape[-1] == 3
            and len(heights) > 0, "physical trajectory/height contract")
    z = trajectory.new_tensor(heights)
    xy = trajectory[..., None, :2].expand(*trajectory.shape[:-1], len(z), 2)
    z = z.expand(*trajectory.shape[:-1], len(z))[..., None]
    return torch.cat((xy, z), -1).flatten(-3, -2)


@torch.no_grad()
def neighborhood_indices(pixel, visible, image_sizes, merged_grid, *, radius=1):
    """Indices/masks of a bounded square neighborhood in row-major merged V.

    The pixel-center convention is explicit: patch centers occur at
    (column+.5)*W/grid_width-.5. Never use the old display linspace endpoints as
    patch centers. Invalid pixels produce an all-false mask, not a border token.
    """
    require(pixel.ndim == 5 and pixel.shape[-1] == 2
            and visible.shape == pixel.shape[:-1], "pixel/mask contract")
    b, _, v, _, _ = pixel.shape
    require(image_sizes.shape == merged_grid.shape == (b, v, 2)
            and merged_grid.dtype == torch.long and (merged_grid > 0).all().item(), "grid width,height contract")
    require(type(radius) is int and 0 <= radius <= 2, "bounded visual neighborhood")
    center = ((pixel + .5) / image_sizes[:, None, :, None]
              * merged_grid[:, None, :, None]).floor().long()
    offsets = torch.arange(-radius, radius+1, device=pixel.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    location = center[..., None, :] + torch.stack((ox.flatten(), oy.flatten()), -1)
    bounds = merged_grid[:, None, :, None, None]
    valid = visible[..., None] & (location >= 0).all(-1) & (location < bounds).all(-1)
    safe = location.clamp_min(0).minimum(bounds-1)
    index = safe[..., 1]*bounds[..., 0] + safe[..., 0]
    return index, valid


def gather_neighborhood(memory, indices):
    """Project V once per scene, then gather candidate-local tokens with gradients."""
    require(memory.ndim == 4 and indices.ndim == 5 and indices.dtype == torch.long,
            "memory/indices contract")
    b, v, n, _ = memory.shape
    require(indices.shape[0] == b and indices.shape[2] == v
            and (indices >= 0).all().item() and (indices < n).all().item(), "token index out of bounds")
    batch = torch.arange(b, device=memory.device)[:, None, None, None, None]
    view = torch.arange(v, device=memory.device)[None, None, :, None, None]
    return memory[batch, view, indices]
