import torch
import torch.distributed as dist
from torch.nn import functional as F
from .matching import match_current


def global_mean(numerator, count):
    """DDP averages gradients: compensate by world_size/global_count, incl. empty ranks."""
    denominator = numerator.new_tensor(float(count)).detach()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominator)
        world_size = dist.get_world_size()
    return numerator * world_size / denominator.clamp_min(1)


def world_losses(prediction, targets, return_sums=False):
    zero = sum(x.sum()*0 for x in prediction.values())
    cls_sum, box_sum, motion_sum = zero, zero, zero
    cls_count = box_count = motion_count = 0
    matches = []
    for b, target in enumerate(targets):
        pred = {k:v[b] for k,v in prediction.items()}
        rows, cols = match_current(pred,target)
        matches.append((rows,cols))
        if not bool(target.annotation_valid_mask):
            continue
        xy = pred['boxes'][...,:2].detach()
        lo, hi = target.supervision_bounds[:2], target.supervision_bounds[2:]
        supervised = ((xy >= lo) & (xy <= hi)).all(-1)
        if target.supervision_grid is not None:
            grid = target.supervision_grid
            cell = ((xy - lo) / target.supervision_resolution).floor().long()
            in_grid = (cell >= 0).all(-1) & (cell[:,0] < grid.shape[0]) & (cell[:,1] < grid.shape[1])
            safe = cell.clamp_min(0)
            supported = grid[safe[:,0].clamp_max(grid.shape[0]-1),safe[:,1].clamp_max(grid.shape[1]-1)]
            supervised &= in_grid & supported
        # Overflow means unmatched slots cannot safely receive no-object labels.
        if target.overflow:
            supervised[:] = False
        supervised[rows] = True
        labels = torch.full_like(supervised, pred['logits'].shape[-1]-1,dtype=torch.long)
        labels[rows] = target.current_classes[cols]
        cls_sum = cls_sum + F.cross_entropy(pred['logits'],labels,reduction='none')[supervised].sum()
        cls_count += int(supervised.sum())
        if len(rows):
            box_mask = target.box_valid_mask[cols].bool()
            # NaNs in absent channels must be removed before arithmetic.
            gt_boxes = target.current_boxes[cols].masked_fill(~box_mask,0.)
            scale = gt_boxes.new_tensor([20.,20.,20.,5.,5.,5.,1.,1.])
            error = F.smooth_l1_loss(pred['boxes'][rows]/scale,gt_boxes/scale,reduction='none')
            box_sum = box_sum + error[box_mask].sum()
            box_count += int(box_mask.sum())
            valid = target.future_valid_mask[cols].bool()
            gt_future = target.future_xy_in_ego_t0[cols].masked_fill(~valid[...,None],0.)
            error = F.smooth_l1_loss(pred['future_xy'][rows]/20.,gt_future/20.,reduction='none').sum(-1)
            motion_sum = motion_sum + error[valid].sum()
            motion_count += int(valid.sum())
    sums={'cls':cls_sum,'box':box_sum,'motion':motion_sum}
    counts={'cls':cls_count,'box':box_count,'motion':motion_count}
    if return_sums:
        return (sums,counts),matches
    # All ranks perform exactly these three collectives, including empty ranks.
    return normalize_accumulated_world_losses([(sums,counts)]),matches


def normalize_accumulated_world_losses(microbatches):
    # Accumulate numerators AND denominators, not per-microbatch means.
    # Retains microbatch graphs until backward; callers must budget memory accordingly.
    return {name:global_mean(sum(sums[name] for sums,_ in microbatches),
                             sum(counts[name] for _,counts in microbatches))
            for name in ('cls','box','motion')}
