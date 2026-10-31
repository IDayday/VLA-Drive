"""CPU-only, unregistered endpoint-CFM per-MC ordinary-PPO prototype.

Primary formula: arXiv:2602.02481v2 eq9, ordinary PPO (ASPO disabled).
An energy difference is a ratio SURROGATE, never exact endpoint likelihood.
There is deliberately no entry, runtime injection, anchor change or launcher.
Hyperparameters have no campaign defaults; a future pilot needs registration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from .policy import policy_gradient_objective
from .preference_auxiliary import array_digest
from .safety_efficiency import canonical_components

SCHEMA = 'world_endpoint_CFM_per_MC_PPO_CPU_prototype_v1'
REDUCTION = 'mean_xyz_then_sum_8poses'


@dataclass(frozen=True)
class PrototypeSpec:
    """Explicit toy/protocol choices, not a registered training recipe."""
    mc_samples: int
    clip_range: float
    sigma_min: float
    sigma_max: float
    stateless_seed: int = 42

    def __post_init__(self):
        if (type(self.mc_samples) is not int or self.mc_samples < 1
                or not math.isfinite(self.clip_range) or not 0 < self.clip_range < 1
                or not 0 < self.sigma_min < self.sigma_max < 1
                or self.stateless_seed != 42):
            raise ValueError('Explicit MC/clip/positive finite sigma interval and private seed42 required')

    def protocol(self):
        return dict(schema=SCHEMA, **asdict(self), energy_reduction=REDUCTION,
            ratio='rho[MC,candidate] = exp(ell_behavior_old - ell_current); proxy, not exact likelihood',
            trust_region='ordinary symmetric PPO clipping separately for each MC sample; same candidate A; no ASPO',
            velocity='decreasing-noise World convention: state=(1-sigma)*endpoint+sigma*eps, target=eps-endpoint',
            old_policy='CURRENT collection-start behavior actor under no_grad; never permanently frozen IL reference',
            credit='fresh once-finalized candidate advantage with source A cap5 and mask; no teacher or historical A',
            randomness='private stateless token/behavior/rank/42; saved and reused for four inners; no source-SDE/global RNG',
            execution='one native G8 current velocity forward per MC sample, one fresh differentiable Memory context per inner',
            scope='CPU executable prototype only; no runtime/entry/native-contract registration or GPU deployment')


def cfm_energy(velocity, target):
    """Per action+MC ell=sum_horizon mean_action_dim square error.

    Official fpo-control 8f460e44: flow_model.forward_fpo means action_dim;
    finetune_online_rl chunk-level PPO sums poses by default. Here H8,D3,
    so this is sum(all24)/3 = 8 * mean(all24), not Rescue's mean24 units.
    No Huber, loss weighting, temperature, old-energy or logratio clamp.
    """
    if velocity.shape != (8, 8, 3) or target.shape != velocity.shape:
        raise ValueError('Endpoint CFM uses native [G8,H8,D3] velocities')
    energy = (velocity.float()-target.detach().float()).square().mean(-1).sum(-1)
    if not bool(torch.isfinite(energy).all()):
        raise FloatingPointError('Nonfinite CFM proxy energy')
    return energy


def per_mc_ppo_objective(current_energy, old_energy, advantages, mask, *, clip_range):
    """Per-MC ratio clipping; signs match ordinary source PPO exactly."""
    if (current_energy.ndim != 2 or current_energy.shape[1] != 8
            or old_energy.shape != current_energy.shape or advantages.shape != (8,)
            or mask.shape != (8,) or not bool(torch.isfinite(current_energy).all())
            or not bool(torch.isfinite(old_energy).all())
            or not bool(torch.isfinite(advantages).all()) or not bool(torch.isfinite(mask).all())
            or not bool(((mask == 0) | (mask == 1)).all())):
        raise ValueError('Finite matching [MC,G8] CFM energies and finalized [G8] A/mask required')
    # Existing PPO math only, with negative CFM energy as a surrogate score.
    # This does not make that score a true action log probability.
    a = advantages.detach().clamp(-5, 5)[:, None].expand(8, current_energy.shape[0])
    loss, diagnostics = policy_gradient_objective(-current_energy.T, -old_energy.detach().T,
        a, mask.detach(), algorithm='ppo', clip_range=clip_range)
    ratio = diagnostics['ratio']
    return loss, dict(fpo_proxy_ratio_mean=ratio.mean().detach(),
        fpo_proxy_ratio_min=ratio.min().detach(), fpo_proxy_ratio_max=ratio.max().detach(),
        fpo_proxy_clip_fraction=(ratio.sub(1).abs() > clip_range).float().mean().detach(),
        fpo_proxy_PPO_blocked_fraction=diagnostics['ppo_blocked'].float().mean().detach(),
        fpo_current_energy_mean=current_energy.mean().detach(), fpo_old_energy_mean=old_energy.mean().detach())


def _stateless_draws(rollout, spec, rank):
    key = dict(schema=SCHEMA, token=rollout['token'], behavior=rollout['behavior_version'],
               rank=rank, seed=spec.stateless_seed)
    seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':')).encode()).digest()[:8], 'big') & ((1 << 63)-1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    sigma = spec.sigma_min+(spec.sigma_max-spec.sigma_min)*torch.rand(spec.mc_samples, 8, generator=generator, dtype=torch.float32)
    noise = torch.randn(spec.mc_samples, 8, 8, 3, generator=generator, dtype=torch.float32)
    return seed, sigma, noise


def _fresh_evidence(rollout):
    expected = canonical_components(rollout['records'])
    if (not torch.equal(rollout['rewards'].cpu(), expected['canonical_rewards'])
            or rollout['chain'].shape != (8, 11, 8, 3) or rollout['chain'].dtype != torch.float32
            or rollout['chain'].requires_grad or not bool(torch.isfinite(rollout['chain']).all())
            or rollout['advantages'].shape != (8,) or rollout['adv_mask'].shape != (8,)):
        raise ValueError('Use fresh original canonical endpoints and already-finalized candidate credit')
    clean = rollout['chain'][:, -1].detach().cpu().clone()
    decoded = ActionCodec.denorm_odo(clean).numpy().astype(np.float64)
    request_hashes = []
    for i, record in enumerate(rollout['records']):
        request = np.ascontiguousarray(record['requested_trajectory'], dtype=np.float64)
        if (request.shape != (8, 3) or not np.isfinite(request).all()
                or record['token'] != rollout['token'] or record['log'] != rollout['log']
                or record.get('protocol_sha256') != rollout['protocol_sha256']):
            raise ValueError('Endpoint/source canonical scoring identity differs')
        bound = 8*np.finfo(np.float32).eps*(np.abs(decoded[i])+np.array([66.74,42.,3.53]))
        if np.any(np.abs(decoded[i]-request) > bound):
            raise ValueError('Endpoint is not the actually scored fresh request')
        request_hashes.append(array_digest(request))
    credit = {k:rollout[k] for k in ('advantages', 'adv_mask', 'rewards')}
    return clean, dict(endpoint_sha256=tensor_digest(clean), requested_trajectory_sha256=request_hashes,
                       credit_sha256=tensor_digest(credit), protocol_sha256=rollout['protocol_sha256'])


def _state(clean, sigma, noise):
    return ((1-sigma[:, None, None])*clean+sigma[:, None, None]*noise,
            noise-clean)


def material_digest(material):
    return tensor_digest({k:v for k,v in material.items() if k != 'material_sha256'})


@torch.no_grad()
def cache_behavior_material(actor, rollout, spec, *, rank):
    """Call once after fresh collection/finalization, BEFORE any inner update.

    Saves old behavior velocity/energy, not IL-reference velocity. A future
    runtime must retain its full source checkpoint/recovery/rank validator too.
    """
    if 'fpo_prototype' in rollout:
        raise ValueError('Cache behavior CFM material once per collection')
    clean, evidence = _fresh_evidence(rollout)
    seed, sigma, noise = _stateless_draws(rollout, spec, rank)
    clean, sigma, noise = clean.to(actor.device), sigma.to(actor.device), noise.to(actor.device)
    velocity, energies = [], []
    with actor._context_call(rollout['cond_bg']) as condition:
        for j in range(spec.mc_samples):
            state, target = _state(clean, sigma[j], noise[j])
            v = actor.action_velocity(state, sigma[j]*1000, condition, use_ref=False)
            velocity.append(v.detach().float().clone())
            energies.append(cfm_energy(v, target).detach().clone())
    p = dict(protocol=spec.protocol(), token=rollout['token'], log=rollout['log'], rank=rank,
        behavior_version=copy.deepcopy(rollout['behavior_version']), condition=copy.deepcopy(rollout['cond_bg']),
        selection_seed=seed, evidence=evidence, clean=clean, sigma=sigma, noise=noise,
        behavior_velocity=torch.stack(velocity), old_energy=torch.stack(energies))
    p['material_sha256'] = material_digest(p)
    rollout['fpo_prototype'] = p
    return rollout


def validate_material(rollout, spec, *, rank):
    """Detached material/math validator ONLY, not a registered native boundary."""
    p = rollout.get('fpo_prototype', {})
    clean, evidence = _fresh_evidence(rollout)
    seed, sigma, noise = _stateless_draws(rollout, spec, rank)
    keys = {'protocol','token','log','rank','behavior_version','condition','selection_seed','evidence',
            'clean','sigma','noise','behavior_velocity','old_energy','material_sha256'}
    if (set(p) != keys or p.get('protocol') != spec.protocol()
            or p.get('token') != rollout['token'] or p.get('log') != rollout['log'] or p.get('rank') != rank
            or p.get('behavior_version') != rollout['behavior_version'] or p.get('condition') != rollout['cond_bg']
            or p.get('selection_seed') != seed or p.get('evidence') != evidence
            or p.get('material_sha256') != material_digest(p)):
        raise ValueError('FPO prototype material/source identity/digest changed')
    for key, want in (('clean',clean), ('sigma',sigma), ('noise',noise)):
        value=p[key]
        if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float32
                or value.shape != want.shape or not torch.equal(value.cpu(),want)):
            raise ValueError('FPO fresh endpoint/stateless material changed: '+key)
    for key, shape in [('behavior_velocity',(spec.mc_samples,8,8,3)),('old_energy',(spec.mc_samples,8))]:
        v=p[key]
        if (not torch.is_tensor(v) or v.requires_grad or v.shape != shape or v.dtype != torch.float32
                or not bool(torch.isfinite(v).all())):
            raise ValueError('FPO old behavior material must be finite and detached: '+key)
    for j in range(spec.mc_samples):
        _, target = _state(clean, sigma[j], noise[j])
        want = cfm_energy(p['behavior_velocity'][j].cpu(),target)
        if not torch.allclose(p['old_energy'][j].cpu(), want, atol=1e-6, rtol=3e-6):
            raise ValueError('FPO cached old velocity/energy audit differs')
    return p


def prototype_policy_loss(actor, rollout, spec):
    """PG term alone. A future runtime adds a separately pinned IL anchor.

    Never call ResearchLoss PG then add this PG again. This executable prototype
    is intentionally not an actor._loss and cannot be selected by existing arms.
    """
    p=rollout['fpo_prototype']
    if p['protocol'] != spec.protocol():
        raise ValueError('FPO prototype protocol changed')
    energy=[]
    # New scope each inner, current Memory differentiable; reuse only within
    # this MC loop. No current-memory graph is retained in the material.
    with actor._context_call(rollout['cond_bg']) as condition:
        for j in range(spec.mc_samples):
            state,target=_state(p['clean'],p['sigma'][j],p['noise'][j])
            v=actor.action_velocity(state,p['sigma'][j]*1000,condition,use_ref=False)
            energy.append(cfm_energy(v,target))
    return per_mc_ppo_objective(torch.stack(energy),p['old_energy'],rollout['advantages'],
        rollout['adv_mask'],clip_range=spec.clip_range)
