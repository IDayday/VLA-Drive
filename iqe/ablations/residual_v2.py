"""True unrestricted raw-domain residual. Legacy residual is algebraically degenerate."""
from copy import deepcopy
import torch
from torch import nn
from ..contracts import TrajectoryBundle, require


class ResidualExpertV2(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.initial = deepcopy(source).eval().requires_grad_(False)
        self.delta = deepcopy(source).requires_grad_(True)
        self.trajectory_contract = source.trajectory_contract
        # Each supervised raw head predicts a delta directly, without subtracting Base.
        for head in self.delta.heads:
            linear = [m for m in head.modules() if isinstance(m, nn.Linear)]
            require(linear, 'residual_v2 requires an explicit final linear raw head')
            nn.init.zeros_(linear[-1].weight)
            if linear[-1].bias is not None:
                nn.init.zeros_(linear[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.initial.eval()
        return self

    def forward(self, features):
        with torch.no_grad():
            base = self.initial(features)
        delta = self.delta(features)
        raw = tuple(b + d for b, d in zip(base.intermediates, delta.intermediates))
        return TrajectoryBundle(raw[-1], self.trajectory_contract.physical(raw[-1]), raw, base.dt)

    def ablation_identity(self):
        return {'variant': 'residual_v2', 'domain': 'normalized_xy_sincos',
                'yaw': 'unrestricted sin/cos deltas; inherited atan2 conversion',
                'zero_initialized_output': True, 'reads_frozen_base': True,
                'private_parameters': sum(p.numel() for p in self.delta.parameters()),
                'decoder_calls_per_candidate': 2, 'clipping_or_residual_penalty': False}
