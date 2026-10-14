"""Scene means, fixed physical scales, common absolute targets and valid pairs."""
import torch
from torch.nn import functional as F
from .consequence_contract import CONTINUOUS, DISCRETE


def per_scene_mean(value, valid):
    valid = valid.bool()
    value = torch.where(valid, value, torch.zeros_like(value))
    dims = tuple(range(1, value.ndim))
    count = valid.sum(dims)
    return value.sum(dims)/count.clamp_min(1), count


def consequence_loss(raw, target, valid, *, difference_weight=0., distance_scale_m=5., positive_weights=(1., 1.)):
    if raw.shape != target.shape or raw.shape != valid.shape or raw.shape[-1] != 5:
        raise ValueError('Consequence channel/mask shape mismatch')
    if difference_weight < 0 or distance_scale_m <= 0:
        raise ValueError('Invalid registered weights')
    losses, counts = {}, {}
    for channel in CONTINUOUS:
        truth = torch.where(valid[..., channel], target[..., channel], torch.zeros_like(target[..., channel])).detach()
        value = F.smooth_l1_loss(raw[..., channel].float()/distance_scale_m,
                                 truth.float()/distance_scale_m, reduction='none')
        means, count = per_scene_mean(value, valid[..., channel])
        losses['abs_'+str(channel)] = means.mean(); counts['abs_'+str(channel)] = count.detach()
    for k, channel in enumerate(DISCRETE):
        safe_target = torch.where(valid[..., channel], target[..., channel], torch.zeros_like(target[..., channel]))
        value = F.binary_cross_entropy_with_logits(raw[..., channel].float(), safe_target.float(), reduction='none',
                                                    pos_weight=raw.new_tensor(positive_weights[k]).float())
        means, count = per_scene_mean(value, valid[..., channel])
        losses['abs_'+str(channel)] = means.mean(); counts['abs_'+str(channel)] = count.detach()
    absolute = sum(losses.values())/5.
    delta_losses = []
    for channel in CONTINUOUS:
        # Six anchor-to-variant pairs once each, never all ordered duplicates.
        prediction = raw[:, 1:, :, channel]-raw[:, :1, :, channel]
        pair_valid = valid[:, 1:, :, channel] & valid[:, :1, :, channel]
        safe = torch.where(valid[..., channel], target[..., channel], torch.zeros_like(target[..., channel]))
        truth = (safe[:, 1:]-safe[:, :1]).detach()
        value = F.smooth_l1_loss(prediction.float()/distance_scale_m, truth.float()/distance_scale_m, reduction='none')
        means, count = per_scene_mean(value, pair_valid)
        delta_losses.append(means.mean()); counts['pair_'+str(channel)] = count.detach()
    difference = sum(delta_losses)/len(CONTINUOUS)
    return {'loss': absolute+difference_weight*difference, 'absolute': absolute,
            'difference': difference, 'channels': losses, 'counts': counts}
