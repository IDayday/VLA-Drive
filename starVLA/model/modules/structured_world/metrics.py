"""Offline current-centre evaluation, independent of training assignment.

Radius uses strict distance < radius (the original 2 m boundary). A dummy
assignment costs more than all valid distances combined: maximum cardinality
is primary, minimum total distance secondary. No future/size/yaw in matching.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .legacy_metrics import legacy_diagnostics


@torch.no_grad()
def match_geometry(pred_xy, gt_xy, radius=2., pred_classes=None, gt_classes=None):
    if radius <= 0:
        raise ValueError('radius must be positive')
    p = pred_xy.detach().double().cpu().numpy()
    g = gt_xy.detach().double().cpu().numpy()
    if p.ndim != 2 or g.ndim != 2 or p.shape[1] != 2 or g.shape[1] != 2:
        raise ValueError('Expected N x 2 centres')
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Nonfinite centres')
    if (pred_classes is None) != (gt_classes is None):
        raise ValueError('Both class arrays required for class-constrained matching')
    n, m = len(p), len(g)
    empty = torch.empty(0, dtype=torch.long, device=pred_xy.device)
    if not n or not m:
        return empty, empty
    distance = np.linalg.norm(p[:, None] - g[None], axis=-1)
    valid = distance < radius
    if pred_classes is not None:
        valid &= pred_classes.cpu().numpy()[:, None] == gt_classes.cpu().numpy()[None]
    penalty = float(n + 1)
    cost = np.full((n, m + n), penalty, dtype=np.float64)
    cost[:, :m] = np.where(valid, distance / radius, 2 * penalty)
    rows, cols = linear_sum_assignment(cost)
    keep = cols < m
    rows, cols = rows[keep], cols[keep]
    if not valid[rows, cols].all():
        raise AssertionError('Invalid edge chosen despite available dummy columns')
    return torch.as_tensor(rows, device=pred_xy.device), torch.as_tensor(cols, device=pred_xy.device)


def prediction_filters(pred, target):
    xy = pred['boxes'][:, :2]
    lo, hi = target.supervision_bounds[:2], target.supervision_bounds[2:]
    roi = ((xy >= lo) & (xy <= hi)).all(-1)
    support = roi.clone()
    if target.supervision_grid is not None:
        grid = target.supervision_grid
        cell = ((xy - lo) / target.supervision_resolution).floor().long()
        inside = (cell >= 0).all(-1) & (cell[:, 0] < grid.shape[0]) & (cell[:, 1] < grid.shape[1])
        support &= inside & grid[cell[:, 0].clamp(0, grid.shape[0]-1), cell[:, 1].clamp(0, grid.shape[1]-1)]
    obj = pred['logits'].argmax(-1) != pred['logits'].shape[-1]-1
    return roi, support, obj


def motion_diagnostics(pred, target, rows, cols):
    mask = target.future_valid_mask[cols].bool()
    future = target.future_xy_in_ego_t0[cols]
    # Mask before arithmetic: missing points may contain NaNs.
    safe = future.masked_fill(~mask[..., None], 0.)
    errors = torch.linalg.vector_norm(pred['future_xy'][rows] - safe, dim=-1)
    stationary = torch.linalg.vector_norm(pred['boxes'][rows, :2, None].transpose(-1, -2) - safe, dim=-1)
    movement = torch.linalg.vector_norm(safe - target.current_boxes[cols, :2].unsqueeze(1), dim=-1).masked_fill(~mask, 0.)
    dynamic = movement.amax(-1) >= 1.  # diagnostic, >=1m displacement at any valid time
    out = {}
    for name, group in [('all', torch.ones(len(rows), device=rows.device, dtype=torch.bool)), ('dynamic', dynamic), ('static', ~dynamic)]:
        valid = mask & group[:, None]
        terminal = mask[:, -1] & group
        out.update({f'{name}_motion_instances': int((valid.any(-1)).sum()),
                    f'{name}_motion_points': int(valid.sum()), f'{name}_fde_targets': int(terminal.sum()),
                    f'{name}_ade_sum': float(errors[valid].sum()), f'{name}_fde_sum': float(errors[:, -1][terminal].sum()),
                    f'{name}_stationary_ade_sum': float(stationary[valid].sum()),
                    f'{name}_stationary_fde_sum': float(stationary[:, -1][terminal].sum())})
    return out


@torch.no_grad()
def diagnostics(pred, target, include_legacy=True):
    if any(not torch.isfinite(pred[k]).all() for k in ['boxes', 'logits', 'future_xy']):
        raise ValueError('Nonfinite prediction')
    gt_ids = torch.where(target.current_supervision_mask.bool())[0]
    roi, support, obj = prediction_filters(pred, target)
    probs = pred['logits'].softmax(-1)[:, -1]
    result = {'annotation_valid': bool(target.annotation_valid_mask), 'gt_targets': len(gt_ids) + target.overflow,
              'gt_cached_targets': len(gt_ids), 'gt_overflow': target.overflow, 'fixed_k': len(roi),
              'roi_predictions': int(roi.sum()), 'support_predictions': int(support.sum()),
              'objectness_predictions': int(obj.sum()), 'filtered_predictions': int((obj & support).sum()),
              'outside_roi': int((~roi).sum()), 'inside_roi_outside_support': int((roi & ~support).sum()),
              'no_object_probability_sum': float(probs.sum()),
              'gt_motion_instances': int(target.future_valid_mask[gt_ids].any(-1).sum()),
              'gt_motion_points': int(target.future_valid_mask[gt_ids].sum())}
    for axis, values in [('x', pred['boxes'][:, 0]), ('y', pred['boxes'][:, 1]), ('no_object', probs)]:
        for name, value in zip(['min', 'p25', 'median', 'p75', 'max'], torch.quantile(values.float(), values.new_tensor([0., .25, .5, .75, 1.], dtype=torch.float32))):
            result[f'raw_{axis}_{name}'] = float(value)
    if not bool(target.annotation_valid_mask):
        result['evaluation_status'] = 'missing_annotations'
        return result
    all_slots = torch.ones_like(roi)
    modes = [('raw_k', all_slots, False), ('roi', roi, False), ('support', support, False),
             ('objectness', obj, False), ('filtered', support & obj, False),
             ('class_raw_k', all_slots, True), ('class_filtered', support & obj, True)]
    for name, active, same_class in modes:
        ids = torch.where(active)[0]
        classes = dict(pred_classes=pred['logits'][ids].argmax(-1), gt_classes=target.current_classes[gt_ids]) if same_class else {}
        rows, cols = match_geometry(pred['boxes'][ids, :2], target.current_boxes[gt_ids, :2], **classes)
        rows, cols = ids[rows], gt_ids[cols]
        distances = (pred['boxes'][rows, :2] - target.current_boxes[cols, :2]).norm(dim=-1)
        boxmask = target.box_valid_mask[cols]
        yaw_valid = boxmask[:, 6:].all(-1)
        py = torch.atan2(pred['boxes'][rows, 6], pred['boxes'][rows, 7])
        gy = torch.atan2(target.current_boxes[cols, 6], target.current_boxes[cols, 7])
        yaw = torch.atan2((py-gy).sin(), (py-gy).cos()).abs()
        stats = {'tp': len(rows), 'predictions': len(ids), 'fp': len(ids)-len(rows),
                 'centre_error_sum': float(distances.sum()), 'yaw_error_sum': float(yaw[yaw_valid].sum()), 'yaw_targets': int(yaw_valid.sum())}
        stats.update(motion_diagnostics(pred, target, rows, cols))
        for group, low, high in [('near', 0, 10), ('middle', 10, 30), ('far', 30, float('inf'))]:
            distance = target.current_boxes[:, :2].norm(dim=-1)
            in_group = (distance >= low) & (distance < high) & target.current_supervision_mask
            stats[group+'_gt'] = int(in_group.sum()); stats[group+'_tp'] = int(in_group[cols].sum())
        result.update({f'{name}_{key}': val for key, val in stats.items()})
    if include_legacy:
        result.update({'legacy_'+k: v for k, v in legacy_diagnostics(pred, target).items()})
    result['evaluation_status'] = 'ok'
    return result
