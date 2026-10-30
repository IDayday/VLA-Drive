"""Candidate-metric-conditioned safety/efficiency credit for an isolated pilot.

This does not add scene-risk inference, a critic, new observations or samples.
Canonical feedback stays intact. Only the once-finalized PPO advantage changes;
the source transition ratio, advantage cap and permanent IL mean anchor remain.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.distributed as dist

from starVLA.rl.simwam_recipe.recovery_state import (
    tensor_digest, validate_boundary as source_validate_boundary,
)

STRUCTURED_SCHEMA = 'candidate_metric_safety_feasible_EP_global_std_v1'
COMPONENT_NAMES = ('no_at_fault_collisions', 'drivable_area_compliance',
                   'time_to_collision_within_bound', 'ego_progress')
STRUCTURED_PROTOCOL = dict(
    schema=STRUCTURED_SCHEMA,
    kappa=0.0,
    lambda_EP=1.0,
    feasible='f = 1 iff official NC == DAC == TTC == 1',
    cost='c = mean(1-NC, 1-DAC, 1-TTC); indicator deficit, not accident severity',
    safety_advantage='if any f: f-mean(f); else: mean(c)-c',
    efficiency_advantage='if sum(f)>=2: f*(EP-mean_feasible(EP)); else: 0',
    raw_advantage='A_safe + A_EP; each group sums to zero',
    normalization='population_std of all 64 fresh A_raw, including zero groups, + 1e-8',
    mask='exactly all-zero A_raw group has PG mask 0; no direction or new sampling on ties',
    canonical_feedback='original rewards, records, score, protocol and readouts unchanged',
    component_names=list(COMPONENT_NAMES),
    inner_updates='four source inner updates reuse the same finalized advantage tensor',
    loss='source PPO, source local effective mean then equal rank mean, source A cap 5, beta50 IL mean anchor',
    scope='candidate metric conditioning only; no claim of scene-risk reasoning or safety guarantee',
)


@torch.no_grad()
def canonical_components(records, *, device='cpu'):
    """Extract unmodified canonical FP32 rewards and FP64 NC/DAC/TTC/EP."""
    if len(records) != 8:
        raise ValueError('Exactly eight successful canonical feedback records required')
    rewards, components = [], []
    names = COMPONENT_NAMES + ('comfort', 'score')
    for record in records:
        if record.get('status') != 'OK':
            raise ValueError('Scoring failure must not become a candidate-metric advantage')
        m = record['metrics']
        if any(not math.isfinite(float(m[k])) or not 0 <= float(m[k]) <= 1 for k in names):
            raise ValueError('Canonical components must be finite in [0,1]')
        nc, dac, ttc, ep, comfort, score = (float(m[k]) for k in names)
        if abs(nc*dac*(5*ep+5*ttc+2*comfort)/12-score) > 5e-15:
            raise ValueError('Canonical feedback aggregation changed')
        readout = float(record['readouts']['canonical_PDMS'])
        if not math.isfinite(readout) or abs(readout-score) > 5e-15:
            raise ValueError('Canonical readout changed')
        rewards.append(score)
        components.append((nc, dac, ttc, ep))
    return dict(
        canonical_rewards=torch.tensor([rewards], device=device, dtype=torch.float32),
        components=torch.tensor([components], device=device, dtype=torch.float64),
    )


@torch.no_grad()
def safety_efficiency_parts(components):
    """Pure group-local math; no whitening of either individual component."""
    if (components.ndim != 3 or tuple(components.shape[1:]) != (8, 4)
            or not torch.isfinite(components).all()
            or not bool(((components >= 0) & (components <= 1)).all())):
        raise ValueError('Finite [scene,8,NC/DAC/TTC/EP] components in [0,1] required')
    x = components.detach().double()
    feasible = (x[:, :, :3] == 1).all(-1)
    f = feasible.double()
    count = f.sum(1, keepdim=True)
    cost = (1-x[:, :, :3]).mean(-1)
    # Offset centering keeps exact constant-cost groups exactly zero.
    dc = cost-cost[:, :1]
    a_safe = torch.where(count > 0, f-f.mean(1, keepdim=True), dc.mean(1, keepdim=True)-dc)
    ep = x[:, :, 3]
    first = feasible.to(torch.int64).argmax(1, keepdim=True)
    ep_offsets = (ep-ep.gather(1, first))*f
    a_ep = (ep_offsets-ep_offsets.sum(1, keepdim=True)/count.clamp_min(1))*f
    a_ep = torch.where(count >= 2, a_ep, torch.zeros_like(a_ep))
    return dict(feasible=feasible.detach(), cost=cost.detach(),
                A_safe=a_safe.detach(), A_EP=a_ep.detach(), A_raw=(a_safe+a_ep).detach())


@torch.no_grad()
def safety_efficiency_advantages(rewards, components, *, all_material=None, eps=1e-8):
    """One gather over the fresh rollout; all four updates reuse its result.

    all_material is only an explicit math-test/restore injection. Its last axis
    is canonical reward followed by NC/DAC/TTC/EP. Runtime uses one all_gather.
    Historical global_std/raw_group_std keep their canonical-reward meanings.
    """
    if (rewards.ndim != 2 or rewards.shape[1] != 8
            or tuple(components.shape) != (*rewards.shape, 4)
            or not torch.isfinite(rewards).all()
            or not bool(((rewards >= 0) & (rewards <= 1)).all()) or eps != 1e-8):
        raise ValueError('Canonical [scene,8] rewards, matching components and source epsilon required')
    local_parts = safety_efficiency_parts(components)
    material = torch.cat((rewards.detach().double().unsqueeze(-1),
                          components.detach().to(device=rewards.device, dtype=torch.float64)), -1)
    if all_material is None:
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(material) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, material)
            all_material = torch.cat(gathered, 0)
        else:
            all_material = material
    all_material = all_material.detach().to(device=rewards.device, dtype=torch.float64)
    if (all_material.ndim != 3 or tuple(all_material.shape[1:]) != (8, 5)
            or not torch.isfinite(all_material).all()
            or not bool(((all_material >= 0) & (all_material <= 1)).all())):
        raise ValueError('Invalid global canonical/component material')
    global_parts = safety_efficiency_parts(all_material[:, :, 1:])
    denominator = global_parts['A_raw'].std(unbiased=False)
    raw = rewards.detach().double()
    offsets = raw-raw[:, :1]
    group_std = offsets.std(1, unbiased=False, keepdim=True)
    active = local_parts['A_raw'].ne(0).any(1, keepdim=True)
    mask = active.expand_as(raw).reshape(-1).float()
    count = local_parts['feasible'].sum(1)
    return dict(
        **local_parts,
        advantages=(local_parts['A_raw']/(denominator+eps)).reshape(-1).float().detach(),
        adv_mask=mask.detach(),
        global_std=all_material[:, :, 0].std(unbiased=False).detach(),
        raw_group_std=group_std.detach(), normalization_std=denominator.detach(),
        statistics=dict(
            structured_advantage_schema=STRUCTURED_SCHEMA,
            normalization_population_candidates=int(all_material.shape[0]*8),
            normalization_population_groups=int(all_material.shape[0]),
            normalization=STRUCTURED_PROTOCOL['normalization'],
            all_full=int((raw == 1).all(1).sum()), all_zero=int((raw == 0).all(1).sum()),
            tied=int((group_std <= eps).sum()),
            structured_zero_groups=int((~active).sum()),
            feasible_candidates=int(count.sum()), all_fail_groups=int((count == 0).sum()),
            singleton_feasible_groups=int((count == 1).sum()),
            effective_PG_candidate_fraction=float(mask.mean()),
            reduction='source local effective mean, then equal mean across ranks',
        ),
    )


def finalize_safety_efficiency_advantages(rollout):
    """Explicit runtime hook; canonical scalar reward is never transformed."""
    if any(k in rollout for k in ('advantages', 'structured_advantage', 'canonical_rewards', 'reward_transform')):
        raise ValueError('Finalize structured credit exactly once on original canonical feedback')
    material = canonical_components(rollout['records'], device=rollout['rewards'].device)
    if not torch.equal(rollout['rewards'], material['canonical_rewards']):
        raise ValueError('Structured credit requires original canonical rewards')
    stats = safety_efficiency_advantages(rollout['rewards'], material['components'])
    rollout['canonical_rewards'] = rollout['rewards'].detach().clone()
    for k in ('advantages', 'adv_mask', 'global_std', 'raw_group_std', 'normalization_std'):
        rollout[k] = stats[k]
    rollout['advantage_statistics'] = stats['statistics']
    rollout['structured_advantage'] = dict(
        protocol=copy.deepcopy(STRUCTURED_PROTOCOL), components=material['components'],
        **{k:stats[k] for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
    )
    return rollout


def validate_safety_efficiency_recipe(cfg):
    """Allow this fixed pilot only, with no simultaneous research transforms."""
    r = cfg.get('research', {})
    if (r.get('structured_advantage') != STRUCTURED_SCHEMA
            or r.get('policy_objective') != 'ppo' or r.get('data_policy') != 'uniform_full'
            or any(r.get(k) is not None for k in (
                'advantage_normalization', 'reward_transform', 'anchor_ablation', 'memory_lr_ablation'))):
        raise ValueError('Register a separate canonical PPO/full-data safety-efficiency pilot')
    if (cfg['learning_rate'] != 5e-5 or cfg['memory_learning_rate'] != 5e-6
            or cfg['grpo']['train']['kl_beta'] != 50 or cfg['variant'] != 'R_CANONICAL'):
        raise ValueError('Safety-efficiency pilot retains source learning rates, beta50 and SDE window')
    return cfg


def validate_safety_efficiency_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    """Keep source structure/optimizer semantics; replace only A semantics."""
    validate_safety_efficiency_recipe(identity['recipe'])
    if identity['world'] != 8:
        raise ValueError('Safety-efficiency pilot requires eight ranks and 64 fresh candidates')
    summary = source_validate_boundary(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    summary.update(structured_advantage_schema=STRUCTURED_SCHEMA, structured_world=identity['world'])
    if not summary['pending']:
        return summary
    if 'reward_transform' in rollout:
        raise ValueError('Canonical rewards must not be transformed')
    expected = canonical_components(rollout['records'])
    for k in ('canonical_rewards', 'rewards'):
        value = rollout.get(k)
        if (not torch.is_tensor(value) or tuple(value.shape) != (1, 8) or value.requires_grad
                or value.dtype != torch.float32 or not torch.equal(value.cpu(), expected['canonical_rewards'])):
            raise ValueError('Saved canonical reward material differs: '+k)
    saved = rollout.get('structured_advantage', {})
    if set(saved) != {'protocol', 'components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'} or saved.get('protocol') != STRUCTURED_PROTOCOL:
        raise ValueError('Missing or changed structured credit protocol')
    parts = safety_efficiency_parts(expected['components'])
    expected.update(parts)
    for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'):
        value, want = saved[k], expected[k]
        if (not torch.is_tensor(value) or value.shape != want.shape or value.requires_grad
                or value.dtype != want.dtype
                or not torch.allclose(value.cpu(), want, atol=1e-14, rtol=0)):
            raise ValueError('Saved structured credit material differs: '+k)
    scale, meta = rollout.get('normalization_std'), rollout['advantage_statistics']
    if (not torch.is_tensor(scale) or scale.ndim != 0 or scale.requires_grad
            or scale.dtype != torch.float64 or not bool(torch.isfinite(scale)) or float(scale) < 0
            or meta.get('structured_advantage_schema') != STRUCTURED_SCHEMA
            or meta.get('normalization_population_candidates') != identity['world']*8
            or meta.get('normalization_population_groups') != identity['world']):
        raise ValueError('Missing or changed global structured-normalization material')
    expected_raw_std = (expected['canonical_rewards'].double()-expected['canonical_rewards'].double()[:, :1]).std(1, unbiased=False, keepdim=True)
    raw_std = rollout['raw_group_std']
    if (not torch.is_tensor(raw_std) or raw_std.requires_grad or raw_std.shape != (1, 1)
            or not torch.allclose(raw_std.double().cpu(), expected_raw_std, atol=1e-12, rtol=0)):
        raise ValueError('Canonical raw-group std changed')
    summary.update(
        normalization_std=float(scale),
        canonical_rewards=rollout['canonical_rewards'].double().cpu().tolist(),
        structured_material={k:saved[k].cpu().tolist() for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
        structured_digests={k:tensor_digest(saved[k]) for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
        canonical_rewards_digest=tensor_digest(rollout['canonical_rewards']),
    )
    return summary


def validate_safety_efficiency_ranks(summaries):
    """Recompute all saved A from canonical components at commit and restore."""
    if not summaries or len(summaries) != 8:
        raise ValueError('Eight safety-efficiency rank summaries required')
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run',
              'pending', 'structured_advantage_schema', 'structured_world')
    for s in summaries:
        if s.get('structured_advantage_schema') != STRUCTURED_SCHEMA or s.get('structured_world') != len(summaries):
            raise ValueError('Missing safety-efficiency rank protocol')
        if {k:s[k] for k in common} != {k:summaries[0][k] for k in common}:
            raise ValueError('Ranks disagree on recovery boundary')
    if not summaries[0]['pending']:
        return summaries
    if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):
        raise ValueError('Ranks disagree on collection-start behavior')
    rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
    components = torch.tensor([s['structured_material']['components'][0] for s in summaries], dtype=torch.float64)
    material = torch.cat((rewards.unsqueeze(-1), components), -1)
    expected = safety_efficiency_advantages(rewards, components, all_material=material)
    adv = expected['advantages'].reshape(len(summaries), 8)
    mask = expected['adv_mask'].reshape(len(summaries), 8)
    for rank, s in enumerate(summaries):
        if (s['canonical_rewards'] != s['rewards']
                or abs(s['global_std']-float(expected['global_std'])) > 1e-12
                or abs(s['normalization_std']-float(expected['normalization_std'])) > 1e-12
                or not torch.allclose(torch.tensor(s['advantages']), adv[rank], atol=1e-7, rtol=1e-6)):
            raise ValueError('Structured advantage was not globally finalized')
        if s['mask'] != mask[rank].tolist():
            raise ValueError('Structured PG mask mismatch')
        for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'):
            want = expected[k][rank:rank+1]
            saved = torch.tensor(s['structured_material'][k], dtype=want.dtype)
            if not torch.allclose(saved, want, atol=1e-14, rtol=0):
                raise ValueError('Rank structured component mismatch: '+k)
    return summaries
