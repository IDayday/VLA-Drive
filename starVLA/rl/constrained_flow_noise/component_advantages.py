"""Scalarize component advantages, never scalarize then normalize the reward.

GDPO's independent group moments are the source reference. Deliberate DDP
changes: reject NaN, use FP64 centered moments, population std and fixed floors;
no final batch whitening, nansum, clamp5 or differentiation through multipliers.
"""
import torch
from starVLA.rl.flow_grpo.math import centered_group_rewards
from .contracts import COSTS


@torch.no_grad()
def component_advantages(rewards, costs, multipliers, floors, *, mode='scadv_floor'):
    if rewards.ndim != 2 or rewards.shape[1] != 16 or costs.shape != rewards.shape + (5,):
        raise ValueError('Expected separate [scene,16] reward and [scene,16,5] costs')
    if len(floors) != 6 or any(not 0 < x <= 1 for x in floors):
        raise ValueError('Six fixed positive component floors required')
    if multipliers.shape != (5,) or not torch.isfinite(multipliers).all() or (multipliers < 0).any():
        raise ValueError('Five finite nonnegative detached multipliers required')
    if not torch.isfinite(costs).all() or (costs < 0).any() or (costs > 1).any():
        raise ValueError('Costs must be finite bounded observations')
    components = torch.cat([rewards[..., None], costs], -1)
    zs, stds = [], []
    for j in range(6):
        delta, std, _ = centered_group_rewards(components[..., j])
        if mode == 'scadv_floor':
            denom = std.clamp_min(floors[j])
        elif mode == 'official_scadv':
            # GDPO component arithmetic: sample std +1e-4, before its optional
            # batch whitening. This mode is a tensor oracle, never a formal run.
            denom = std*(16/15)**.5 + 1e-4
        else:
            raise ValueError('Unknown component advantage mode')
        z = torch.where(std == 0, 0., delta/denom)
        zs.append(z); stds.append(std[:, 0])
    z = torch.stack(zs, -1)
    weights = torch.cat([multipliers.new_ones(1), -multipliers.detach()]).double()
    advantage = (z*weights).sum(-1)
    if not torch.isfinite(advantage).all() or (advantage.abs() > 1e4).any():
        raise FloatingPointError('Extreme/nonfinite scalarized advantage; no silent clipping')
    return advantage.float().detach(), {'component_z': z.detach(), 'component_std': torch.stack(stds, -1).detach(),
            'zero_reward_groups': (z[..., 0] == 0).all(-1), 'zero_final_groups': (advantage == 0).all(-1),
            'cost_names': COSTS, 'mode': mode, 'post_scalarization_whitening': False}
