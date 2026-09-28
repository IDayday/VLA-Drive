"""Finite, globally element-normalized supervision; no label in the student."""
import torch
from torch import distributed as dist
from torch.nn import functional as F


def masked_regression(prediction, target, valid, *, kind='mse', global_count=None):
    if target.shape != prediction.shape or valid.dtype != torch.bool:
        raise ValueError('Regression shape/mask contract')
    valid = torch.broadcast_to(valid, prediction.shape)
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite valid regression value')
    # Sanitize BEFORE subtraction/loss, including backward on invalid padding.
    p = torch.where(valid, prediction.float(), 0.)
    t = torch.where(valid, target.detach().float(), 0.)
    if kind == 'mse': errors = (p-t).square()
    elif kind == 'smooth_l1': errors = F.smooth_l1_loss(p, t, reduction='none')
    else: raise ValueError('Unknown regression loss')
    total = errors.sum()
    count = valid.sum().detach().to(device=total.device, dtype=torch.float64)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if global_count is None:
        if world > 1: dist.all_reduce(count)
    else:
        count = torch.as_tensor(global_count, device=total.device, dtype=torch.float64)
    if count.ndim or count < 0: raise ValueError('Invalid global denominator')
    # DDP averages parameter gradients; compensate by world size.
    loss = total * world / count.clamp_min(1).to(total.dtype)
    return loss, count


def interaction_loss(prediction, target, scene_valid, eps=1e-5, global_count=None):
    if scene_valid.shape != (len(prediction),) or scene_valid.dtype != torch.bool:
        raise ValueError('Interaction validity must be per scene')
    mask = scene_valid[:, None, None].expand_as(prediction)
    if target.shape != prediction.shape or not torch.isfinite(target[mask]).all():
        raise ValueError('Invalid valid teacher latent')
    p = torch.where(mask, prediction.float(), 0.)
    t = torch.where(mask, target.detach().float(), 0.)
    p = F.layer_norm(p, (p.shape[-1],), eps=eps)
    t = F.layer_norm(t, (t.shape[-1],), eps=eps)
    return masked_regression(p, t, mask, global_count=global_count)
