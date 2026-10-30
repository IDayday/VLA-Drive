"""PPO/CISPO controls with unchanged World transition and IL-reference math.

Reference: MiniMax-M1, https://arxiv.org/abs/2506.13585 (CISPO).
The ratio here remains the source clamped, dimension-mean transition proxy.
It is not an exact final-action or full-chain importance ratio.

The reference-mean penalty below follows the MIT-licensed SimWAM extraction
in simwam_recipe/vendor/trainer_math.py. Keep its operations and reduction so
the PPO control can be compared by value and gradient to the original loss.
"""
from __future__ import annotations

import math
import torch

from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss


def _masked_mean(value, mask):
    denominator = mask.sum().clamp(min=1.0) * float(value.shape[1])
    return (value * mask[:, None]).sum() / denominator


def policy_gradient_objective(new_logp, old_logp, advantages, mask, *,
                              algorithm='ppo', clip_range=.005, cispo_upper=1.005):
    """Return the PG term and effective derivative weights on source logprobs.

    Inputs are [candidate, stochastic_step], except mask [candidate].
    Weights are detached diagnostics of d(-L)/d(logp), before mean reduction.
    Masked/tied groups do not contribute. PPO uses the exact historical
    maximum formulation; CISPO clips only the detached IS weight from above.
    """
    if new_logp.ndim != 2 or old_logp.shape != new_logp.shape:
        raise ValueError('Aligned [candidate, stochastic_step] logprobs required')
    if advantages.shape != new_logp.shape or mask.shape != new_logp.shape[:1]:
        raise ValueError('Advantages and group mask are misaligned')
    if not math.isfinite(float(clip_range)) or clip_range <= 0:
        raise ValueError('Positive finite source PPO clip range required')
    if not math.isfinite(float(cispo_upper)) or cispo_upper < 1:
        raise ValueError('CISPO upper IS cap must be finite and >= 1')
    if algorithm not in ('ppo', 'cispo'):
        raise ValueError('Registered policy objective must be ppo or cispo')
    old_logp = old_logp.detach()
    advantages = advantages.detach()
    mask = mask.detach().to(new_logp)
    ratio = torch.exp(new_logp - old_logp)
    if not bool(torch.isfinite(ratio).all()):
        raise FloatingPointError('Nonfinite source transition importance ratio')
    ppo_blocked = (((advantages > 0) & (ratio > 1 + clip_range)) |
                   ((advantages < 0) & (ratio < 1 - clip_range)))
    if algorithm == 'ppo':
        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
        loss = _masked_mean(torch.maximum(unclipped, clipped), mask)
        derivative_weight = advantages * ratio.detach() * (~ppo_blocked)
    else:
        weight = ratio.detach().clamp(max=float(cispo_upper))
        loss = -_masked_mean(weight * advantages * new_logp, mask)
        derivative_weight = advantages * weight
    derivative_weight = derivative_weight.detach() * mask[:, None]
    return loss, dict(ratio=ratio.detach(), ppo_blocked=ppo_blocked.detach(),
                      derivative_weight=derivative_weight)


@torch.no_grad()
def ratio_diagnostics(values, advantages, mask, steps, *, clip_range, cispo_upper):
    ratio = values['ratio']
    eligible = mask[:, None].bool().expand_as(ratio) & advantages.ne(0)
    positive = eligible & advantages.gt(0)
    negative = eligible & advantages.lt(0)
    blocked = values['ppo_blocked'] & eligible
    effective = values['derivative_weight'].ne(0) & eligible

    def fraction(event, population):
        return (event & population).sum().float()/population.sum().clamp_min(1)

    out = {
        'ratio_mean': ratio.mean(),
        'clipfrac': (ratio.sub(1).abs() > clip_range).float().mean(),
        'ppo_blocked_fraction': fraction(blocked, eligible),
        'positive_ppo_blocked_fraction': fraction(blocked, positive),
        'negative_ppo_blocked_fraction': fraction(blocked, negative),
        'positive_active_fraction': fraction(effective, positive),
        'negative_active_fraction': fraction(effective, negative),
        'cispo_upper_hit_fraction': fraction(ratio > cispo_upper, eligible),
        'pg_logprob_weight_abs_mean': values['derivative_weight'].abs().mean(),
    }
    for col, step in enumerate(steps):
        p = f'step{step}_'
        keep = eligible[:, col]
        r = ratio[:, col]
        weights = r * mask.to(r)
        ess = weights.sum().square()/weights.square().sum().clamp_min(1e-12)
        out.update({
            p+'ratio_mean': r.mean(),
            p+'ratio_min': r.min(),
            p+'ratio_max': r.max(),
            p+'ratio_std': r.std(unbiased=False),
            p+'ratio_ess_fraction': ess/mask.sum().clamp_min(1),
            p+'positive_ppo_blocked_fraction': fraction(blocked[:, col], positive[:, col]),
            p+'negative_ppo_blocked_fraction': fraction(blocked[:, col], negative[:, col]),
            p+'active_fraction': fraction(effective[:, col], keep),
            p+'pg_logprob_weight_abs_mean': values['derivative_weight'][:, col].abs().mean(),
        })
    return {k: v.detach() for k, v in out.items()}


class ResearchLoss(SourceLoss):
    """Drop-in actor._loss for root's separately registered runtime.

    cfg.research.policy_objective selects ppo/cispo. The original actor, reward,
    clamped probability, source advantages, reference mean penalty, per-rank
    reduction, optimizer, and sampled chain are unchanged. BC and KL gates are
    rejected because they are not present in the selected Memory baseline.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        research = cfg.get('research', {})
        self.algorithm = research.get('policy_objective', 'ppo')
        self.cispo_upper = float(research.get('cispo_upper', 1.005))
        if self.algorithm not in ('ppo', 'cispo'):
            raise ValueError('Unsupported research policy objective')
        if self.ppo_clip_range is None or self.kl_gate_enabled:
            raise ValueError('Research control requires the registered PPO clip and no KL gate')
        if cfg['grpo']['train']['use_bc_loss']:
            raise ValueError('BC is a separate experimental change, not this objective control')

    def _policy_loss(self, model, rollout):
        chain, times, deltas = (rollout[k] for k in ('chain', 'timesteps', 'deltas'))
        indices = rollout.get('stoch_idx')
        beta = self._effective_kl_beta(model)
        if beta > 0:
            new_logp, terms = model.action_chain_logprobs(
                rollout['cond_bg'], chain, times, deltas, stoch_idx=indices, return_terms=True)
        else:
            new_logp = model.action_chain_logprobs(
                rollout['cond_bg'], chain, times, deltas, stoch_idx=indices)
            terms = None
        steps = list(range(len(times))) if indices is None else indices.detach().cpu().tolist()
        denoise_idx = torch.tensor(steps, device=new_logp.device, dtype=torch.float32)
        discount = model.grpo_denoising_discount ** (len(times)-1-denoise_idx)
        adv = rollout['advantages'].detach()[:, None] * discount[None, :]
        if self.adv_clip_max is not None:
            adv = adv.clamp(-self.adv_clip_max, self.adv_clip_max)
        mask = rollout.get('adv_mask', torch.ones(new_logp.shape[0], device=new_logp.device))
        mask = mask.detach().to(new_logp)
        pg, values = policy_gradient_objective(new_logp, rollout['old_logp'], adv, mask,
            algorithm=self.algorithm, clip_range=self.ppo_clip_range, cispo_upper=self.cispo_upper)
        extra = ratio_diagnostics(values, adv, mask, steps,
                                  clip_range=self.ppo_clip_range, cispo_upper=self.cispo_upper)
        extra['approx_kl'] = (rollout['old_logp']-new_logp).mean().detach()
        extra['adv_kept_frac'] = mask.mean().detach()
        bc_loss = torch.zeros((), device=pg.device, dtype=pg.dtype)
        kl_loss = torch.zeros((), device=pg.device, dtype=pg.dtype)
        if terms is not None:
            with torch.no_grad():
                _, reference = model.action_chain_logprobs(rollout['cond_bg'], chain, times, deltas,
                    stoch_idx=indices, use_ref=True, return_terms=True)
            d_mean_raw = (terms['mean']-reference['mean']).pow(2)
            d_vel_raw = (terms['vel']-reference['vel']).pow(2)
            weights = model.kl_weight_matrix(horizon=d_mean_raw.shape[2], action_dim=d_mean_raw.shape[3],
                                           device=d_mean_raw.device, dtype=d_mean_raw.dtype)
            if weights is None:
                d_mean_sq = d_mean_raw.mean(dim=(2, 3))
                d_vel_sq = d_vel_raw.mean(dim=(2, 3))
            else:
                denom = weights.sum().clamp(min=1e-12)
                d_mean_sq = (d_mean_raw*weights).sum(dim=(2, 3))/denom
                d_vel_sq = (d_vel_raw*weights).sum(dim=(2, 3))/denom
            kl_norm = d_mean_sq/(2.0*terms['std'].pow(2))[None, :]
            if model.grpo_kl_type == 'x_kl':
                kl_loss = kl_norm.mean()
            elif model.grpo_kl_type == 'x':
                kl_loss = d_mean_sq.mean()
            else:
                kl_loss = d_vel_sq.mean()
            extra.update(kl_div=kl_loss.detach(), kl_x_norm=kl_norm.mean().detach(),
                         kl_x=d_mean_sq.mean().detach(), kl_v=d_vel_sq.mean().detach())
            for i, value in enumerate(d_mean_raw.mean(dim=(0, 1, 2)).tolist()):
                extra[f'kl_x_dim{i}'] = torch.as_tensor(value, device=pg.device)
        total = pg + model.grpo_bc_coeff*bc_loss + beta*kl_loss
        return total, dict(policy_loss=pg.detach(), bc_loss=bc_loss.detach(),
                           kl_loss=(beta*kl_loss).detach(), **extra)
