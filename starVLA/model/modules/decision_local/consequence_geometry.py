"""Offline physical consequences, using the existing grid/body/box operators.

Distances are raster approximations, not exact polygon penetration depths.
Unknown cells bound the distance domain rather than being labelled free.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points


def carry_static_ground_support(occupancy, valid, instances, future_valid, height_quality):
    """Label-only support for annotated free cells on CURRENT known ground.

    A missing future LiDAR return under the logged ego is not evidence of an
    unknown road height: the same fixed ego(t0) cell can already have measured
    ground at t0. Reuse that support only if the unmasked, complete future box
    raster contains NO object. Never fill an unsupported annotated object,
    missing frame, invalid LiDAR record, or unknown current ground. No ego GT
    coordinate/footprint is used. This is a documented static-surface assumption,
    not a change to the historical G caches or an assertion that no return=free.
    """
    occ, mask = np.array(occupancy, copy=True), np.asarray(valid,bool).copy()
    current_free_ground = mask[0] & (occ[0] == 0)
    counts=[]
    for t in range(1,len(occ)):
        complete = bool(future_valid[t-1]) and 'reason' not in height_quality[t]
        add = (~mask[t]) & current_free_ground & (instances[t] == 0) & complete
        mask[t,add]=True;occ[t,add]=0;counts.append(int(add.sum()))
    return occ,mask,{'added_current_ground_free_cells':counts,
        'rule':'known current free ground + complete future annotation raster empty; unknown objects stay unknown',
        'assumption':'static surface at the same fixed xy; never propagate across an annotated unknown object'}


def obstacle_distance(occupied, valid, grid=GridSpec(), clip_m=10.):
    occupied, valid = np.asarray(occupied), np.asarray(valid, bool)
    if occupied.shape != (grid.height, grid.width) or valid.shape != occupied.shape:
        raise ValueError('Occupancy grid shape mismatch')
    valid = valid & (occupied != 255)
    inside, outside = (occupied == 1) & valid, (occupied == 0) & valid
    half = .5*min(grid.dx, grid.dy)
    to_obstacle = distance_transform_edt(~inside, sampling=(grid.dy, grid.dx)) if inside.any() else np.full(inside.shape, clip_m+half)
    to_free = distance_transform_edt(~outside, sampling=(grid.dy, grid.dx)) if outside.any() else np.full(inside.shape, clip_m+half)
    signed = np.where(inside, -np.maximum(to_free-half, 0.), np.maximum(to_obstacle-half, 0.))
    signed = np.clip(signed, -clip_m, clip_m)
    # The map exterior is unknown too. A closer unknown region could contain
    # an unseen obstacle/free boundary, so its distance cannot be supervised.
    uncertainty = distance_transform_edt(np.pad(valid, 1, constant_values=False), sampling=(grid.dy, grid.dx))[1:-1, 1:-1]
    reliable = valid & (uncertainty >= np.minimum(np.abs(signed)+half, clip_m+half)-1e-7)
    return signed.astype(np.float32), reliable


def read_consequences(road, road_valid, occupancy, occupancy_valid, poses, body, *, grid=GridSpec(), clip_m=10.):
    """Return K,T,3 values/masks; every read covers the same complete body."""
    poses = torch.as_tensor(poses, dtype=torch.float32)
    K, T, _ = poses.shape
    if len(occupancy) != T or occupancy_valid.shape != occupancy.shape:
        raise ValueError('Environment fields must follow real physical query times')
    # Reference signature also exists in the pinned original G runtime; this
    # read-only operator must not require its later execution-cache additions.
    points = body_points(poses, body)
    P = points.shape[-2]
    xy = points.permute(1, 0, 2, 3).reshape(T, K*P, 2)
    distances, distance_valid = zip(*(obstacle_distance(o, v, grid, clip_m) for o, v in zip(occupancy, occupancy_valid)))
    fields = np.stack((np.repeat(np.asarray(road)[None], T, axis=0), np.asarray(distances)), axis=1)
    supports = np.stack((np.repeat(np.asarray(road_valid)[None], T, axis=0), np.asarray(distance_valid)), axis=1)
    sampled, in_grid = grid.sample(torch.from_numpy(fields).float(), xy)
    supported, _ = grid.sample(torch.from_numpy(supports).float(), xy)
    known_occ = np.asarray(occupancy_valid, bool) & (occupancy != 255)
    ob, _ = grid.sample(torch.from_numpy(((occupancy == 1) & known_occ)[:, None]).float(), xy, mode='nearest')
    ov, _ = grid.sample(torch.from_numpy(known_occ[:, None]).float(), xy, mode='nearest')
    sampled = sampled.reshape(T, K, P, 2)
    support = (supported.reshape(T, K, P, 2) >= 1.-1e-6) & in_grid.reshape(T, K, P, 1)
    margins = sampled.amin(2)
    margin_valid = support.all(2)
    ob = ob.reshape(T, K, P) > .5
    ov = (ov.reshape(T, K, P) > .5) & in_grid.reshape(T, K, P)
    overlap = (ob & ov).any(-1)
    # A known overlap is positive even if another body region is unknown. A
    # negative overlap requires complete valid support, including map bounds.
    overlap_valid = overlap | ov.all(-1)
    result = torch.cat((margins, overlap[..., None].float()), -1).permute(1, 0, 2)
    valid = torch.cat((margin_valid, overlap_valid[..., None]), -1).permute(1, 0, 2)
    return torch.where(valid, result, torch.zeros_like(result)).numpy(), valid.numpy(), {
        'body_in_grid': in_grid.reshape(T, K, P).all(-1).T.numpy(),
        'body_known_occupancy_fraction': ov.float().mean(-1).T.numpy(),
        'distance_saturated': (margins.abs() >= clip_m-1e-6).permute(1, 0, 2).numpy()}


def current_cv_occupancy(frame, current_occupancy, current_valid, times, *, grid=GridSpec()):
    """Privileged CV using provided CURRENT annotation velocity only.

    This function receives one current frame, never a future frame/track list.
    Geometry reuses the pinned centered UniAD rasterizer. Ground support is
    held at the current known domain; unsupported/multilevel objects stay unknown.
    """
    import cv2
    from starVLA.dataloader.structured_world.adapters import NAV_CLASSES, CLASS_NAMES, box_container, transform_boxes_to_ego, pose_matrix
    from starVLA.dataloader.structured_world.centered_boxes import CenteredOccFlowLabels
    e2g = pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'], planar=True)
    attitude = np.linalg.inv(e2g)@pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'])
    l2e = attitude@pose_matrix(frame['lidar2ego_translation'], frame['lidar2ego_rotation'])
    anns = frame['anns']
    keep = np.array([name in NAV_CLASSES for name in anns['gt_names']], dtype=bool)
    names = np.asarray(anns['gt_names'])[keep]
    labels = np.asarray([CLASS_NAMES.index(NAV_CLASSES[n]) for n in names], dtype=np.int64)
    boxes = transform_boxes_to_ego(box_container(anns['gt_boxes'][keep]), l2e)
    provided = np.asarray(anns.get('gt_velocity_3d', np.full((len(keep), 3), np.nan)), dtype=np.float64)[keep]
    good_velocity = np.isfinite(provided).all(-1)
    velocity = np.where(good_velocity[:, None], provided, 0.)@l2e[:3, :3].T
    stationary_class = np.isin(names, ['traffic_cone', 'barrier', 'generic_object'])
    velocity[stationary_class] = 0.
    # Box height and yaw are held; vertical velocity is not ground travel.
    velocity[:, 2] = 0.
    identity = np.eye(3, dtype=np.float32)
    phases = np.r_[0., np.asarray(times, dtype=np.float64)]
    results = {k: [] for k in ('future_gt_bboxes_3d', 'future_gt_labels_3d', 'future_gt_inds',
                               'future_gt_vis_tokens', 'occ_l2e_r_mats', 'occ_l2e_t_vecs',
                               'occ_e2g_r_mats', 'occ_e2g_t_vecs')}
    for t in phases:
        moved = boxes.clone()
        moved.translate(velocity.astype(np.float32)*t)
        results['future_gt_bboxes_3d'].append(moved)
        results['future_gt_labels_3d'].append(labels.copy())
        results['future_gt_inds'].append(np.arange(1, len(boxes)+1))
        results['future_gt_vis_tokens'].append(np.full(len(boxes), -1))
        for name in ('occ_l2e_r_mats', 'occ_e2g_r_mats'): results[name].append(identity)
        for name in ('occ_l2e_t_vecs', 'occ_e2g_t_vecs'): results[name].append(np.zeros(3, dtype=np.float32))
    results.update(occ_has_invalid_frame=False, occ_img_is_valid=np.ones(len(phases), bool))
    generator = CenteredOccFlowLabels(dict(xbound=[grid.xmin, grid.xmax, grid.dx],
        ybound=[grid.ymin, grid.ymax, grid.dy], zbound=[-4., 4., 8.]),
        only_vehicle=False, filter_invisible=False, ignore_index=255, compute_flow=False)
    generator.filter_cls_ids = np.arange(10)
    generated = generator(results)
    occ = generated['gt_segmentation'].numpy().astype(np.uint8)
    current_valid = np.asarray(current_valid, bool) & (current_occupancy != 255)
    mismatches = int(((occ[0] != current_occupancy) & current_valid).sum())
    if mismatches:
        raise ValueError(f'CV current raster differs from existing known labels in {mismatches} cells')
    valid = np.repeat(current_valid[None], len(phases), axis=0)
    current_corners = boxes.corners[:, [0, 3, 7, 4], :2].numpy() if len(boxes) else np.empty((0, 4, 2))
    reliable = []
    for corners in current_corners:
        mask = np.zeros(current_valid.shape, np.uint8)
        cells = np.round((corners-[grid.xmin, grid.ymin])/[grid.dx, grid.dy]).astype(np.int32)
        cv2.fillPoly(mask, [cells], 1)
        pixels = mask.astype(bool)
        reliable.append(bool(pixels.any() and current_valid[pixels].all()))
    for t, moved in enumerate(results['future_gt_bboxes_3d']):
        corners = moved.corners[:, [0, 3, 7, 4], :2].numpy() if len(moved) else np.empty((0, 4, 2))
        for ok, polygon in zip(reliable, corners):
            if not ok:
                cells = np.round((polygon-[grid.xmin, grid.ymin])/[grid.dx, grid.dy]).astype(np.int32)
                mask = np.zeros(current_valid.shape, np.uint8); cv2.fillPoly(mask, [cells], 1)
                valid[t, mask.astype(bool)] = False
    occ[~valid] = 255
    return occ[1:], valid[1:], {'objects': len(boxes), 'provided_velocity_valid': int(good_velocity.sum()),
        'zero_velocity_fallback': int((~good_velocity & ~stationary_class).sum()),
        'static_class_zero_velocity': int(stationary_class.sum()), 'known_current_raster_mismatches': mismatches,
        'velocity_source': 'current anns.gt_velocity_3d; same source-frame rotation as current boxes; missing=>zero+flag; no future differences'}
