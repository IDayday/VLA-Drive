"""Optional current-only residual; disabled in the first A/B comparisons."""
import torch
from torch import nn
from .future_latent_head import CrossReadout


class PlannerResidualW(nn.Module):
    def __init__(self, hidden_dim, dim=512):
        super().__init__()
        self.action = nn.Linear(hidden_dim, dim)
        self.world = nn.Linear(hidden_dim, dim)
        self.readout = CrossReadout(dim)
        self.output = nn.Linear(dim, hidden_dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, action, world):
        if action.ndim != 3 or world.ndim != 3 or action.shape[0] != world.shape[0] or not world.shape[1]:
            raise ValueError('Current H_A/W required by residual planner')
        if not torch.isfinite(action).all() or not torch.isfinite(world).all():
            raise ValueError('Nonfinite residual planner input')
        residual = self.output(self.readout(self.action(action), self.world(world)))
        return action + self.alpha.tanh() * residual
