"""Adapt the previous GDPO-style component moments and OmniSafe dual to G8/C3.

Reuses the validated FP64 centered_group_rewards and the vendored, licensed
OmniSafe Lagrange implementation. The historical G16/five-cost contract is not
relaxed. Event costs and actor advantages never carry an autograd graph.
"""
import copy
import torch
from starVLA.rl.flow_grpo.math import centered_group_rewards
from starVLA.rl.constrained_flow_noise.vendor.omnisafe_lagrange import Lagrange
from .safe_recipe import COSTS


@torch.no_grad()
def event_costs(records):
    if len(records) != 8 or any(r['status'] != 'OK' for r in records):
        raise ValueError('Eight successful canonical feedback records are required')
    values = torch.tensor([[r['metrics'][k] for k in COSTS] for r in records], dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Invalid official safety components')
    return (values < 1).double().unsqueeze(0)


@torch.no_grad()
def component_advantages(rewards, costs, multipliers, floors=(.01, 1e-8, 1e-8, 1e-8)):
    if rewards.ndim != 2 or rewards.shape[1] != 8 or costs.shape != (*rewards.shape, 3):
        raise ValueError('Native safety objective requires separate [scene,8,3] costs')
    if (len(floors) != 4 or any(not 0 < f <= 1 for f in floors) or
            multipliers.shape != (3,) or not torch.isfinite(multipliers).all() or (multipliers < 0).any()):
        raise ValueError('Invalid fixed floors or detached nonnegative multipliers')
    if not torch.isfinite(costs).all() or not ((costs == 0) | (costs == 1)).all():
        raise ValueError('Only the three official binary event costs enter this version')
    values = torch.cat((rewards.double()[..., None], costs.double()), -1)
    zs, stds = [], []
    for j, floor in enumerate(floors):
        delta, std, _ = centered_group_rewards(values[..., j])
        zs.append(torch.where(std == 0, 0., delta / std.clamp_min(floor)))
        stds.append(std[:, 0])
    z = torch.stack(zs, -1)
    w = torch.cat((multipliers.new_ones(1), -multipliers.detach())).double()
    advantage = (z * w).sum(-1)
    if not torch.isfinite(advantage).all() or (advantage.abs() > 1e4).any():
        raise FloatingPointError('Scalarized advantage alarm; no silent clamp or whitening')
    return advantage.float().detach(), dict(component_z=z, component_std=torch.stack(stds, -1))


class EventMultipliers:
    """Three independent original OmniSafe optimizers; no actor parameter here."""
    def __init__(self, budgets):
        if len(budgets) != 3 or any(not 0 <= x <= 1 for x in budgets):
            raise ValueError('Three frozen SFT expected event costs required')
        self.budgets = tuple(map(float, budgets))
        self.parts = [Lagrange(d, 1., .01, 'Adam') for d in self.budgets]
        self.collections = 0

    def snapshot(self):
        return torch.stack([p.lagrangian_multiplier.detach().clone() for p in self.parts])

    def update(self, means, collection):
        if collection != self.collections or means.shape != (3,) or not torch.isfinite(means).all():
            raise ValueError('Exactly one unfiltered global cost update per fresh collection')
        for p, mean in zip(self.parts, means.detach().cpu()):
            p.update_lagrange_multiplier(float(mean))
        self.collections += 1
        if not torch.isfinite(self.snapshot()).all() or (self.snapshot() > 1000).any():
            raise FloatingPointError('Dual multiplier alarm; budgets remain fixed')

    def state_dict(self):
        optimizers=[]
        for p in self.parts:
            optimizer=copy.deepcopy(p.lambda_optimizer.state_dict())
            for row in optimizer['state'].values():
                for k,v in row.items():
                    if not torch.is_tensor(v) or v.numel()!=1 or v.dtype!=torch.float32:
                        raise ValueError('Expected FP32 scalar Adam state for an independent scalar multiplier')
                    row[k]=float(v)
            optimizers.append(optimizer)
        # Exact FP32 scalar values are representable as Python/JSON doubles.
        # This keeps shared progress/result journals serializable while the
        # native checkpoint carries every independent Adam moment and step.
        return dict(budgets=self.budgets, collections=self.collections, values=self.snapshot().tolist(),
                    optimizers=optimizers)

    def load_state_dict(self, state):
        if tuple(state['budgets']) != self.budgets or len(state['optimizers']) != 3:
            raise ValueError('Changed immutable cost budgets/optimizer closure')
        for p, value, optimizer in zip(self.parts, state['values'], state['optimizers']):
            with torch.no_grad(): p.lagrangian_multiplier.copy_(torch.tensor(value,dtype=torch.float32))
            optimizer=copy.deepcopy(optimizer)
            for row in optimizer['state'].values():
                for k,v in row.items():row[k]=torch.tensor(v,dtype=torch.float32)
            p.lambda_optimizer.load_state_dict(optimizer)
        self.collections = state['collections']
