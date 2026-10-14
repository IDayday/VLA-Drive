"""Five unchanged OmniSafe Lagrange instances with checkpointed independent Adam."""
import copy
import torch
from .contracts import COSTS
from .vendor.omnisafe_lagrange import Lagrange


class CostMultipliers:
    def __init__(self, budgets, *, lr=.01, initial=(1., 1., 1., .1, .1)):
        if len(budgets) != 5 or any(not 0 <= d <= 1 for d in budgets):
            raise ValueError('Fixed bounded SFT/C_SDE costs required')
        self.budgets = tuple(map(float, budgets)); self.lr = lr
        self.components = [Lagrange(d, value, lr, 'Adam') for d, value in zip(self.budgets, initial)]
        self.collections = 0

    def snapshot(self, device=None):
        return torch.stack([x.lagrangian_multiplier.detach().clone() for x in self.components]).to(device=device)

    def update(self, first_batch_costs, *, collection):
        if collection != self.collections or first_batch_costs.ndim != 3 or first_batch_costs.shape[1:] != (16, 5):
            raise ValueError('One dual update per fresh collection, first unfiltered scenes only')
        if not 0 < len(first_batch_costs) <= 16 or not torch.isfinite(first_batch_costs).all():
            raise ValueError('Missing/nonfinite first-batch costs')
        means = first_batch_costs.detach().double().mean((0, 1)).cpu()
        for component, mean in zip(self.components, means):
            component.update_lagrange_multiplier(float(mean))
        self.collections += 1
        if not torch.isfinite(self.snapshot()).all() or (self.snapshot() > 1000).any():
            raise FloatingPointError('Multiplier numerical alarm; fixed safety budgets remain unchanged')
        return means

    def state_dict(self):
        return {'budgets': self.budgets, 'lr': self.lr, 'collections': self.collections,
                'values': self.snapshot(), 'optimizers': [copy.deepcopy(x.lambda_optimizer.state_dict()) for x in self.components]}

    def load_state_dict(self, state):
        if tuple(state['budgets']) != self.budgets or state['lr'] != self.lr:
            raise ValueError('Resume changed fixed safety budgets/dual optimizer')
        for c, value, optimizer in zip(self.components, state['values'], state['optimizers']):
            with torch.no_grad(): c.lagrangian_multiplier.copy_(value)
            c.lambda_optimizer.load_state_dict(optimizer)
        self.collections = state['collections']
