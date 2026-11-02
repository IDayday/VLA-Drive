"""Conservative advantage-weighted flow matching with behavior convexifier.

Inspired by AWM and AdvantageFlow; this registered pilot is not an exact
reproduction of either paper. Its Hessian guarantee is in velocity space,
not in nonlinear network parameters and not a performance guarantee.
"""
from __future__ import annotations

import copy
import torch

from starVLA.rl.qwen_world_research_algorithms.fpo_training import _physical_anchor
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from .fpo_adapter import (attach_cfm_material, validate_cfm_material,
                          cfm_energy, state_and_target)

SCHEMA = 'world_performance_conservative_AWFM_v1'
MATERIAL_KEY = 'performance_awfm'
CONFIG = dict(schema=SCHEMA, mc_samples=4, sigma_min=.2, sigma_max=.8,
              advantage_scale=.25, advantage_cap=5., behavior_floor=1.,
              stateless_seed=42, energy_reduction='mean24')
PROTOCOL = dict(CONFIG, objective='A*mean24(v-target)^2+(relu(-A)+1)*mean24(v-v_behavior)^2',
                A='.25*clip(finalized_A,-5,5)*source_mask; detached',
                old='collection-start detached behavior velocity at exactly the same state/sigma/noise',
                reduction='mean over MC and all candidates; zero-A candidates retain consistency regularizer',
                anchor='exact original IL70 W3 physical mean MSE, beta50',
                curvature='A+relu(-A)+1 >=1 in velocity space, not network parameter Hessian')


def attach_awfm_material(actor, rollout, *, rank, mc_samples=4):
    if mc_samples != 4:
        raise ValueError('AWFM MC4 is independent from the trainer inner-update count')
    return attach_cfm_material(actor, rollout, rank=rank, protocol=PROTOCOL, key=MATERIAL_KEY)


def validate_awfm_material(rollout, *, rank):
    return validate_cfm_material(rollout, rank=rank, protocol=PROTOCOL, key=MATERIAL_KEY)


def conservative_awfm_objective(velocity, target, behavior_velocity, advantages, mask):
    """Inputs [MC,G,8,3]; literal conservative quadratic, no exponential A."""
    if (velocity.ndim != 4 or velocity.shape[-2:] != (8, 3)
            or target.shape != velocity.shape or behavior_velocity.shape != velocity.shape
            or advantages.shape != velocity.shape[1:2] or mask.shape != advantages.shape
            or not bool(torch.isfinite(velocity).all()) or not bool(torch.isfinite(target).all())
            or not bool(torch.isfinite(behavior_velocity).all()) or not bool(torch.isfinite(advantages).all())
            or not bool(((mask == 0) | (mask == 1)).all())):
        raise ValueError('Finite aligned MC/G velocity material and binary mask required')
    a = (.25*advantages.detach().clamp(-5, 5)*mask.detach()).to(velocity)
    coefficient = torch.relu(-a)+1
    error = (velocity.float()-target.detach().float()).square().mean((-1, -2))
    behavior_error = (velocity.float()-behavior_velocity.detach().float()).square().mean((-1, -2))
    signed = (a[None, :]*error).mean()
    consistency = (coefficient[None, :]*behavior_error).mean()
    total = signed+consistency
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite conservative AWFM objective')
    return total, dict(awfm_signed_energy=signed.detach(), awfm_behavior_consistency=consistency.detach(),
                       awfm_effective_A_abs_mean=a.abs().mean().detach(),
                       awfm_velocity_curvature_coefficient_min=(a+coefficient).min().detach(),
                       awfm_behavior_error_mean=behavior_error.mean().detach())


class AWFMLoss(ResearchLoss):
    def __init__(self, cfg, *, mc_samples=4):
        super().__init__(cfg)
        if mc_samples != 4 or cfg['grpo']['train']['kl_beta'] != 50:
            raise ValueError('AWFM MC4 and original beta50 IL anchor required')
        self.protocol = copy.deepcopy(PROTOCOL)

    def _policy_loss(self, model, rollout):
        payload = rollout[MATERIAL_KEY]
        if payload['protocol'] != self.protocol:
            raise ValueError('AWFM material protocol differs')
        velocities, targets = [], []
        with model._context_call(rollout['cond_bg']) as condition:
            for j in range(self.protocol['mc_samples']):
                state, target = state_and_target(payload['clean'], payload['sigma'][j], payload['noise'][j])
                velocities.append(model.action_velocity(state, payload['sigma'][j]*1000, condition, use_ref=False))
                targets.append(target)
        current = torch.stack(velocities)
        pg, metrics = conservative_awfm_objective(current, torch.stack(targets), payload['behavior_velocity'],
                                                  rollout['advantages'], rollout['adv_mask'])
        fresh_key = '_performance_fresh_'+MATERIAL_KEY
        fresh = (getattr(model,'_performance_fresh_inner',None)==0 or getattr(model, fresh_key, None) == payload['material_sha256'])
        error = (current-payload['behavior_velocity']).abs().max().detach()
        if fresh:
            if not bool(torch.isfinite(error)) or float(error) > 1e-4:
                raise FloatingPointError('Fresh AWFM behavior velocity differs')
            object.__setattr__(model, fresh_key, None)
        anchor, extra = _physical_anchor(model, rollout, self._effective_kl_beta(model))
        observer = getattr(self, 'record_policy_components', None)
        if observer is not None:
            observer(pg, anchor)
        metrics.update(awfm_fresh_checked=pg.detach().new_tensor(float(fresh)),
                       awfm_fresh_velocity_max_abs_error=error)
        return pg+anchor, dict(policy_loss=pg.detach(), bc_loss=pg.detach().new_zeros(()),
                               kl_loss=anchor.detach(), **metrics, **extra)
