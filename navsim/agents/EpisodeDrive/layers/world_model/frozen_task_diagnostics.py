"""Frozen predictor controls; never changes the formal correct-future task."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def frozen_predictor_controls(predictor, current, trajectory, future_target, current_speed):
    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError('Controls require an eval-mode, frozen predictor; do not mutate a training model')
    if current.shape[0] < 2:
        raise ValueError('shuffle-current diagnostic requires at least two scenes')
    target = predictor.normalize_register_state(future_target)
    trajectory = trajectory.to(device=current.device, dtype=current.dtype)
    inputs = {'correct': current, 'action_only': torch.zeros_like(current),
              'shuffle_current': current.roll(1, 0)}
    results = {}
    for name, values in inputs.items():
        output = predictor(values, trajectory[:,None], (.5,1.5,3.), current_speed=current_speed)[:,0]
        results[name] = float((1-F.cosine_similarity(output, target, dim=-1)).mean())
    copy = predictor.normalize_register_state(current)[:,None].expand_as(target)
    results['copy_current'] = float((1-F.cosine_similarity(copy, target, dim=-1)).mean())
    results['target_variance'] = float(target.var(dim=0, unbiased=False).mean())
    return results
