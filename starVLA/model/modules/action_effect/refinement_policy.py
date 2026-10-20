"""A0/A1 identical coefficient refiner and differentiable canonical basis.

No simulator, true feedback, future labels or external scorer is imported here.
The torch reconstruction mirrors intervention_basis.reconstruct; numerical
parity and the zero-initialization Jacobian are tested explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from .effect_model import ContextTrajectoryBackbone, EffectModelConfig, module_initialization
from .intervention_basis import BasisSpec


def _wrap(x):
    return torch.atan2(x.sin(), x.cos())


def torch_basis_matrix(base, spec: BasisSpec):
    spec.validate()
    if base.ndim != 3 or base.shape[1:] != (spec.horizon, 3) or not torch.isfinite(base).all():
        raise ValueError("Expected finite canonical physical [B,T,3] trajectory")
    times = torch.arange(1, spec.horizon + 1, device=base.device, dtype=base.dtype) * spec.dt
    directions = (torch.stack((base[..., 2].cos(), base[..., 2].sin()), -1),
                  torch.stack((-base[..., 2].sin(), base[..., 2].cos()), -1))
    columns = []
    for direction in directions:
        for begin, end in spec.windows:
            u = ((times - begin) / (end - begin)).clamp(0., 1.)
            weight = u ** 3 * (10. - 15. * u + 6. * u ** 2)
            columns.append(direction * weight[None, :, None])
    return torch.stack(columns, -1)


def reconstruct_torch(base, coefficients, spec: BasisSpec):
    """Exact zero values with the LIMITING nonzero Jacobian retained.

    A naive where(coefficients==0, base, reconstructed) kills all learning from
    the zero-initialized output head. The detached value correction below only
    makes the zero value bit-exact; derivatives follow the same smooth basis.
    """
    matrix = torch_basis_matrix(base, spec)
    if coefficients.shape != (len(base), spec.size) or not torch.isfinite(coefficients).all():
        raise ValueError("Expected finite canonical physical coefficient batch")
    xy = base[..., :2] + torch.einsum("btdk,bk->btd", matrix, coefficients)
    origin = torch.zeros_like(xy[:, :1])
    old_d = torch.diff(torch.cat((origin, base[..., :2]), 1), dim=1)
    new_d = torch.diff(torch.cat((origin, xy), 1), dim=1)
    defined = (old_d.norm(dim=-1) > 1e-6) & (new_d.norm(dim=-1) > 1e-6)
    # atan2(0,0) would have NaN derivatives even when subsequently masked.
    safe_direction = torch.zeros_like(old_d); safe_direction[..., 0] = 1.
    old_safe = torch.where(defined[..., None], old_d, safe_direction)
    new_safe = torch.where(defined[..., None], new_d, safe_direction)
    change = _wrap(torch.atan2(new_safe[..., 1], new_safe[..., 0])
                   - torch.atan2(old_safe[..., 1], old_safe[..., 0]))
    yaw = _wrap(base[..., 2] + torch.where(defined, change, torch.zeros_like(change)))
    result = torch.cat((xy, yaw[..., None]), -1)
    zero = (coefficients == 0.).all(-1)[:, None, None]
    exact_value = torch.where(zero, base, result)
    return result + (exact_value - result).detach()


@dataclass(frozen=True)
class RefinerConfig:
    model: EffectModelConfig = EffectModelConfig()
    basis: BasisSpec = BasisSpec()
    coefficient_scales: tuple[float, ...] = (2., 2., 2., 2., .6, .6, .6, .6)
    active_coefficients: tuple[bool, ...] = (True,) * 8
    init_seed: int = 20261021

    def validate(self):
        self.model.validate(); self.basis.validate()
        if (self.model.horizon != self.basis.horizon or self.model.dt != self.basis.dt
                or len(self.coefficient_scales) != self.basis.size
                or len(self.active_coefficients) != self.basis.size
                or any(type(active) is not bool for active in self.active_coefficients)
                or not any(self.active_coefficients)
                or any(not torch.isfinite(torch.tensor(s)) or s <= 0 for s in self.coefficient_scales)):
            raise ValueError("Refiner/basis horizon, time or physical coefficient bounds mismatch")
        if self.model.mode not in ("delta_feedback", "trajectory_only"):
            raise ValueError("A0/A1 share the canonical delta-context refiner, not a G1 ablation mode")
        return self


class CoefficientRefiner(nn.Module):
    """Predict K8 coefficients in metres, not an unconstrained 8x3 target."""
    def __init__(self, config: RefinerConfig = RefinerConfig()):
        super().__init__()
        self.config = config.validate()
        with module_initialization(config.init_seed):
            self.backbone = ContextTrajectoryBackbone(replace(config.model, init_seed=config.init_seed))
            self.coefficient_head = nn.Linear(config.model.hidden_dim, config.basis.size)
            nn.init.zeros_(self.coefficient_head.weight)
            nn.init.zeros_(self.coefficient_head.bias)
        self.register_buffer("coefficient_scales", torch.tensor(config.coefficient_scales, dtype=torch.float32))
        self.register_buffer("active_coefficients", torch.tensor(config.active_coefficients, dtype=torch.bool))

    def forward(self, context, base_physical):
        base = base_physical.detach()
        memory, valid = self.backbone.context_memory(context)
        if len(base) != len(memory):
            raise ValueError("Refiner scene/trajectory batch mismatch")
        features = self.backbone.ordered(memory, valid, base, base)
        coefficients = self.coefficient_head(features).tanh() * self.coefficient_scales * self.active_coefficients
        trajectory = reconstruct_torch(base, coefficients.to(base.dtype), self.config.basis)
        delta = torch.cat((trajectory[..., :2] - base[..., :2],
                           _wrap(trajectory[..., 2:3] - base[..., 2:3])), -1)
        return {"coefficients": coefficients, "trajectory": trajectory, "delta": delta}


ResidualTrajectoryRefiner = CoefficientRefiner
