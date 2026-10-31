"""ABC-only SafetyEP credit on each own pruned deployment/LoRA architecture.

Credit equations are the proven original SafetyEP equations. The original
four-framework SourceLoss, architecture actor and frozen/current prefix lifetimes
are reused without importing historical52.7M-memory research packages.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import torch
import torch.distributed as dist
from starVLA.rl.qwen_world_compare.recovery_state import (validate_boundary as source_validate_boundary,tensor_digest)
from starVLA.rl.qwen_world_compare.runtime import expected_recipe
from starVLA.rl.qwen_world_compare.contracts import file_sha
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss

ABC_SCHEMA='qwen_world_abcd_ABC_SafetyEP_sourcePPO_v1'
ABC_OBJECTIVE_CONFIG=dict(schema=ABC_SCHEMA)


def validate_recipe(cfg):
    arch=cfg.get('architecture_id')
    if arch not in ('A','B','C'):
        raise ValueError('ABC Safety objective does not accept architectureD')
    science=copy.deepcopy(cfg);research=science.pop('research',{})
    if science!=expected_recipe(arch):
        raise ValueError('ABC objective retains complete own-architecture source science recipe')
    if research.get('abc_objective')!=ABC_OBJECTIVE_CONFIG:
        raise ValueError('Missing/changed explicit ABC SafetyEP objective schema')
    return cfg

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

def validate_safety_efficiency_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    """Keep source structure/optimizer semantics; replace only A semantics."""
    validate_recipe(identity['recipe'])
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
              'pending', 'architecture_id', 'structured_advantage_schema', 'structured_world')
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

class ABCSafetyLoss(SourceLoss):
    def __init__(self,cfg):
        validate_recipe(cfg)
        super().__init__(cfg)

    def _policy_loss(self,model,rollout):
        if rollout.get('structured_advantage',{}).get('protocol')!=STRUCTURED_PROTOCOL:
            raise ValueError('ABC source loss requires committed SafetyEP credit')
        return super()._policy_loss(model,rollout)


class ABCObjectiveMixin:
    finalize_advantages=staticmethod(finalize_safety_efficiency_advantages)
    validate_boundary=staticmethod(validate_safety_efficiency_boundary)
    validate_ranks=staticmethod(validate_safety_efficiency_ranks)

    def load_recipe(self,args):
        return validate_recipe(super().load_recipe(args))

    def make_actor(self,cfg,args,device):
        validate_recipe(cfg)
        actor,a,b=super().make_actor(cfg,args,device)
        arch=actor.action_expert.architecture_id
        if arch!=cfg['architecture_id'] or arch not in ('A','B','C'):
            raise ValueError('ABC current parent/scope architecture mismatch')
        if (not actor.closure['base_frozen'] or not actor.closure['reference_frozen']
                or actor.closure['qwen_text_changes']
                or actor.closure['current_prefix_recomputed_each_logprob_call']!=(arch in ('B','C'))):
            raise ValueError('ABC deployment LoRA/reference/prefix closure changed')
        actor._loss=ABCSafetyLoss(cfg)
        return actor,a,b

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['abc_objective']=dict(schema=ABC_SCHEMA,structured_credit=copy.deepcopy(STRUCTURED_PROTOCOL),
            source_PPO='unchanged SourceLoss/W3/clamped dimension-mean transition proxy, not exactIS',
            observation_scope='frozen Qwen+C160; current B P/C W+KV differentiably rebuilt each logp',
            trainable_scope='own actual architecture attention LoRA closure, never historical52.7M Memory full-parameter scope',
            active_IL_auxiliary_losses=False,training_data='full合法TRAIN90890 only; DEV/embargoed/NAVTEST remain excluded')
        return value

    def source_files(self):
        result=dict(super().source_files())
        for relative in ('starVLA/rl/qwen_world_abcd_research/abc.py',
                         'starVLA/rl/qwen_world_abcd_research/data.py'):
            result[relative]=file_sha(relative)
        return result


# Explicit module-level hooks for an independently constructed common runner.
finalize_advantages=finalize_safety_efficiency_advantages
validate_boundary=validate_safety_efficiency_boundary
validate_ranks=validate_safety_efficiency_ranks
