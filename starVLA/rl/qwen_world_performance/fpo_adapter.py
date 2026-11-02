"""Per-MC CFM-energy PPO adapter; never an endpoint likelihood ratio.

Reuses ResearchLoss's constructor, ordinary PPO math and exact IL mean anchor.
The old implementation's G8-only material/validators are not used for G16/32.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_research_algorithms.fpo_training import _physical_anchor
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss, policy_gradient_objective
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest

SCHEMA = 'world_performance_FPO_perMC_v1'
MATERIAL_KEY = 'performance_fpo'
CONFIG = dict(schema=SCHEMA, mc_samples=2, sigma_min=.2, sigma_max=.8,
              clip_range=.01, stateless_seed=42, energy_reduction='mean_xyz_sum8')
PROTOCOL = dict(CONFIG, ratio='exp(E_old-E_current) separately per MC; CFM surrogate, not endpoint likelihood',
                old='collection-start behavior energy detached and fixed across native inner updates',
                anchor='exact original IL70 W3 physical mean MSE, beta50',
                exclusions='no ASPO, critic, entropy bonus, endpoint-density or full FPO++ claim')


def cfm_energy(velocity, target, *, reduction='mean_xyz_sum8'):
    """[candidate,8,3] velocity error in explicit units, preserving gradients."""
    if velocity.ndim != 3 or velocity.shape[-2:] != (8, 3) or target.shape != velocity.shape:
        raise ValueError('Matching [candidate,H8,D3] velocity/target required')
    error = (velocity.float()-target.detach().float()).square()
    if reduction == 'mean_xyz_sum8':
        value = error.mean(-1).sum(-1)
    elif reduction == 'mean24':
        value = error.mean((-1, -2))
    else:
        raise ValueError('Unregistered CFM energy reduction')
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite CFM energy')
    return value


def _fresh_clean(rollout):
    chain = rollout['chain']
    if (chain.ndim != 4 or tuple(chain.shape[1:]) != (11, 8, 3)
            or chain.shape[0] not in (8, 16, 32) or chain.dtype != torch.float32
            or chain.requires_grad or not bool(torch.isfinite(chain).all())):
        raise ValueError('Detached scored source [G,11,8,3] chain required')
    g = chain.shape[0]
    if (rollout['advantages'].shape != (g,) or rollout['adv_mask'].shape != (g,)
            or rollout['rewards'].shape != (1, g) or len(rollout['records']) != g):
        raise ValueError('Source endpoints and once-finalized credit disagree')
    clean = chain[:, -1].detach().float().clone()
    decoded = ActionCodec.denorm_odo(clean).detach().cpu().numpy().astype(np.float64)
    requests = []
    for i, record in enumerate(rollout['records']):
        request = np.asarray(record['requested_trajectory'], dtype=np.float64)
        if (record.get('status') != 'OK' or record['token'] != rollout['token']
                or record['log'] != rollout['log'] or record.get('protocol_sha256') != rollout['protocol_sha256']
                or request.shape != (8, 3) or not np.isfinite(request).all()):
            raise ValueError('Fresh scored endpoint identity differs')
        bound = 8*np.finfo(np.float32).eps*(np.abs(decoded[i])+np.asarray([66.74, 42., 3.53]))
        if np.any(np.abs(decoded[i]-request) > bound):
            raise ValueError('Clean endpoint is not its actually scored request')
        score = float(record['metrics']['score'])
        if not np.isfinite(score) or not 0 <= score <= 1 or float(rollout['rewards'][0, i]) != float(np.float32(score)):
            raise ValueError('Canonical scored reward differs')
        requests.append(hashlib.sha256(np.ascontiguousarray(request).tobytes()).hexdigest())
    return clean, dict(endpoint_sha256=tensor_digest(clean), requested_sha256=requests,
                       canonical_protocol_sha256=rollout['protocol_sha256'],
                       credit_sha256=tensor_digest({k: rollout[k] for k in ('advantages', 'adv_mask', 'rewards')}))


def _draws(rollout, rank, protocol):
    key = dict(schema=protocol['schema'], token=rollout['token'], behavior=rollout['behavior_version'], rank=int(rank), seed=42)
    seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':')).encode()).digest()[:8], 'big') & ((1 << 63)-1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    m, g = protocol['mc_samples'], rollout['chain'].shape[0]
    sigma = protocol['sigma_min']+(protocol['sigma_max']-protocol['sigma_min'])*torch.rand(m, g, generator=generator)
    noise = torch.randn(m, g, 8, 3, generator=generator)
    return seed, sigma.float(), noise.float()


def state_and_target(clean, sigma, noise):
    return (1-sigma[:, None, None])*clean+sigma[:, None, None]*noise, noise-clean


def material_digest(payload):
    return tensor_digest({k: v for k, v in payload.items() if k != 'material_sha256'})


@torch.no_grad()
def attach_cfm_material(actor, rollout, *, rank, protocol, key):
    """Fixed private MC material and actual behavior velocity, once/collection."""
    if key in rollout:
        raise ValueError('CFM material must be attached once before optimizer updates')
    clean, evidence = _fresh_clean(rollout)
    seed, sigma, noise = _draws(rollout, rank, protocol)
    clean, sigma, noise = (t.to(actor.device) for t in (clean, sigma, noise))
    velocities, energies = [], []
    with actor._context_call(rollout['cond_bg']) as condition:
        for j in range(protocol['mc_samples']):
            state, target = state_and_target(clean, sigma[j], noise[j])
            velocity = actor.action_velocity(state, sigma[j]*1000, condition, use_ref=False).float()
            velocities.append(velocity.detach().clone())
            energies.append(cfm_energy(velocity, target, reduction=protocol['energy_reduction']).detach().clone())
    payload = dict(protocol=copy.deepcopy(protocol), rank=int(rank), token=rollout['token'], log=rollout['log'],
                   behavior_version=copy.deepcopy(rollout['behavior_version']),
                   condition_sha256=tensor_digest(rollout['cond_bg']), seed=seed, evidence=evidence,
                   clean=clean, sigma=sigma, noise=noise, behavior_velocity=torch.stack(velocities),
                   old_energy=torch.stack(energies))
    payload['material_sha256'] = material_digest(payload)
    rollout[key] = payload
    object.__setattr__(actor, '_performance_fresh_'+key, payload['material_sha256'])
    return rollout


def validate_cfm_material(rollout, *, rank, protocol, key):
    payload = rollout.get(key, {})
    clean, evidence = _fresh_clean(rollout)
    clean=clean.detach().cpu()
    seed, sigma, noise = _draws(rollout, rank, protocol)
    if (payload.get('protocol') != protocol or payload.get('rank') != rank
            or payload.get('token') != rollout['token'] or payload.get('log') != rollout['log']
            or payload.get('behavior_version') != rollout['behavior_version']
            or payload.get('condition_sha256') != tensor_digest(rollout['cond_bg'])
            or payload.get('seed') != seed or payload.get('evidence') != evidence
            or payload.get('material_sha256') != material_digest(payload)):
        raise ValueError('CFM material identity, behavior or digest changed')
    for name, expected in [('clean', clean), ('sigma', sigma), ('noise', noise)]:
        value = payload[name]
        if (value.requires_grad or value.dtype != torch.float32 or value.shape != expected.shape
                or not torch.equal(value.cpu(), expected.cpu())):
            raise ValueError('CFM saved material differs: '+name)
    m, g = protocol['mc_samples'], clean.shape[0]
    for name, shape in [('behavior_velocity', (m, g, 8, 3)), ('old_energy', (m, g))]:
        value = payload[name]
        if value.requires_grad or value.dtype != torch.float32 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Detached finite behavior material required: '+name)
    for j in range(m):
        _, target = state_and_target(clean, sigma[j], noise[j])
        expected = cfm_energy(payload['behavior_velocity'][j].cpu(), target.cpu(), reduction=protocol['energy_reduction'])
        if not torch.allclose(payload['old_energy'][j].cpu(), expected, atol=1e-6, rtol=3e-6):
            raise ValueError('Old energy does not match saved behavior velocity')
    return payload


def attach_fpo_material(actor, rollout, *, rank, mc_samples=2):
    if mc_samples != CONFIG['mc_samples']:
        raise ValueError('Registered FPO uses MC2; native inner-update count is independent')
    return attach_cfm_material(actor, rollout, rank=rank, protocol=PROTOCOL, key=MATERIAL_KEY)


def validate_fpo_material(rollout, *, rank):
    return validate_cfm_material(rollout, rank=rank, protocol=PROTOCOL, key=MATERIAL_KEY)


def per_mc_fpo_objective(current_energy, old_energy, advantages, mask, *, clip_range=.01):
    if (current_energy.ndim != 2 or old_energy.shape != current_energy.shape
            or advantages.shape != current_energy.shape[1:] or mask.shape != advantages.shape
            or not bool(torch.isfinite(current_energy).all()) or not bool(torch.isfinite(old_energy).all())
            or not bool(torch.isfinite(advantages).all()) or not bool(((mask == 0) | (mask == 1)).all())):
        raise ValueError('Finite aligned [MC,G] energy and binary finalized mask required')
    a = advantages.detach().clamp(-5, 5)[:, None].expand(-1, current_energy.shape[0])
    loss, values = policy_gradient_objective(-current_energy.T, -old_energy.detach().T, a, mask,
                                             algorithm='ppo', clip_range=clip_range)
    eligible = mask.bool()[:, None] & a.ne(0)
    blocked = values['ppo_blocked'] & eligible
    return loss, dict(fpo_proxy_ratio_mean=values['ratio'].mean(),
                      fpo_proxy_ratio_min=values['ratio'].min(), fpo_proxy_ratio_max=values['ratio'].max(),
                      fpo_eligible_blocked_fraction=blocked.sum().float()/eligible.sum().clamp_min(1),
                      fpo_current_energy_mean=current_energy.detach().mean(), fpo_old_energy_mean=old_energy.detach().mean())


class FPOAdapterLoss(ResearchLoss):
    def __init__(self, cfg, *, mc_samples=2):
        super().__init__(cfg)
        if mc_samples != 2 or cfg['grpo']['train']['kl_beta'] != 50:
            raise ValueError('FPO MC2 and original beta50 IL anchor required')
        self.protocol = copy.deepcopy(PROTOCOL)

    def _policy_loss(self, model, rollout):
        payload = rollout[MATERIAL_KEY]
        if payload['protocol'] != self.protocol:
            raise ValueError('FPO material protocol differs')
        energies = []
        with model._context_call(rollout['cond_bg']) as condition:
            for j in range(self.protocol['mc_samples']):
                state, target = state_and_target(payload['clean'], payload['sigma'][j], payload['noise'][j])
                velocity = model.action_velocity(state, payload['sigma'][j]*1000, condition, use_ref=False)
                energies.append(cfm_energy(velocity, target))
        current = torch.stack(energies)
        pg, metrics = per_mc_fpo_objective(current, payload['old_energy'], rollout['advantages'], rollout['adv_mask'])
        fresh_key = '_performance_fresh_'+MATERIAL_KEY
        fresh = (getattr(model,'_performance_fresh_inner',None)==0 or getattr(model, fresh_key, None) == payload['material_sha256'])
        error = torch.expm1(payload['old_energy']-current).abs().max().detach()
        if fresh:
            if not bool(torch.isfinite(error)) or float(error) > 1e-4:
                raise FloatingPointError('Fresh FPO energy ratio differs from one')
            object.__setattr__(model, fresh_key, None)
        anchor, extra = _physical_anchor(model, rollout, self._effective_kl_beta(model))
        observer = getattr(self, 'record_policy_components', None)
        if observer is not None:
            observer(pg, anchor)
        metrics.update(fpo_fresh_checked=pg.detach().new_tensor(float(fresh)),
                       fpo_fresh_ratio_max_abs_error=error)
        return pg+anchor, dict(policy_loss=pg.detach(), bc_loss=pg.detach().new_zeros(()),
                               kl_loss=anchor.detach(), **metrics, **extra)
