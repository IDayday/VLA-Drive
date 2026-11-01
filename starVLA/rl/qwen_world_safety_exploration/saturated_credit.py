"""Canonical credit only for strictly-safe, EP1 Safety-credit saturation.

This independent variant does not reweight progress in already-active groups,
does not transform canonical rewards, and does not create a successful label
for any failed candidate. Preserve the original world*G raw-A standard deviation
unless it is exactly zero, in which case every rank uses the same updated scale.
"""
import copy

import torch
import torch.distributed as dist

from .credit import GroupCredit, canonical_components, raw_advantage_parts, protocol as base_protocol
from .group_runtime import validate_recipe
from .recovery import validate_structure, tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss

SCHEMA = 'world_SafetyEP_success_saturation_canonical_delta_fixed_base_std_v1'


def protocol(g):
    p = base_protocol(g)
    p.update(schema=SCHEMA,
        eligibility='all G NC=DAC=TTC=EP exactly1; original Safety rawA exactly0; saved canonical R nonconstant',
        canonical_numeric='unchanged original FP32 canonical reward labels promoted to FP64 for offset-centering',
        delta='deltaA=R-mean(R), FP64 offset-centered; A_safe unchanged; add to A_EP and A_raw only in eligible groups',
        normalization='original Safety world*G rawA population std +1e-8; only exactzero uses updated world*G rawA std',
        other_groups='baseline normalized advantages and mask bitexact when baselineStd>0',
        extra_material='saturated_credit status/eligibility/delta/digest persisted and recomputed on native recovery',
        safety_labels='original strict-feasible labels unchanged; failed groups never become eligible',
        high_quality='diagnostic strict NC/DAC/TTC/Comfort1 and canonicalscore>=.9, not a new eligibility gate',
        exclusivity='cannot combine DenseDAC and success-saturation credit in the same runtime',
        activation_audit='one fresh collection is one decision; per-inner metrics reuse the same material')
    return p


def validate_saturated_recipe(cfg):
    validate_recipe(cfg)
    if cfg['research']['structured_advantage'] != SCHEMA:
        raise ValueError('Success-saturation credit needs its independent schema')
    return cfg


@torch.no_grad()
def decision(rewards, components):
    if (components.shape[0] != 1 or tuple(rewards.shape) != components.shape[:2]
            or components.shape[1] not in (8, 16)):
        raise ValueError('One matching G8/G16 scene per rank required')
    if not torch.isfinite(rewards).all() or not ((rewards >= 0) & (rewards <= 1)).all():
        raise ValueError('Finite canonical rewards in [0,1] required')
    old = raw_advantage_parts(components)
    subtype = bool((components[:, :, :3] == 1).all()
        and (components[:, :, 3] == 1).all() and old['A_raw'].eq(0).all())
    # Tiny CPU feedback calculations are identical before/after deserialization.
    raw = rewards.detach().cpu().double()
    offsets = raw-raw[:, :1]
    varied = bool(offsets.ne(0).any())
    info = dict(schema=SCHEMA, subtype=subtype, canonical_varied=varied, eligible=False,
        status='UNCHANGED_NOT_STRICT_SAFE_EP1_ZERO_CREDIT', delta=None,
        canonical_reward_digest=tensor_digest(raw))
    if not subtype:
        return info
    if not varied:
        info['status'] = 'UNCHANGED_CANONICAL_TIED'
        return info
    info.update(eligible=True, status='APPLIED_SATURATED_CANONICAL_CREDIT',
                delta=offsets-offsets.mean(1, keepdim=True))
    return info


def _parts(components, info):
    old = raw_advantage_parts(components)
    result = dict(old)
    if info['eligible']:
        delta = info['delta'].to(components.device)
        result['A_EP'] = old['A_EP']+delta
        result['A_raw'] = old['A_raw']+delta
    return old, result


class SaturatedCredit(GroupCredit):
    def __init__(self, group_size):
        super().__init__(group_size)
        self.protocol = protocol(group_size)

    @torch.no_grad()
    def advantages(self, rewards, components, *, info=None, all_material=None):
        if all_material is None:
            local = torch.cat((rewards.detach().double().unsqueeze(-1),
                               components.detach().to(rewards.device, torch.float64)), -1)
            if dist.is_available() and dist.is_initialized():
                if dist.get_world_size() != 8:
                    raise ValueError('Success-saturation requires eight-rank global material')
                gathered = [torch.empty_like(local) for _ in range(8)]
                dist.all_gather(gathered, local)
                all_material = torch.cat(gathered)
            else:
                all_material = local
        all_material = all_material.detach().to(rewards.device, torch.float64)
        base = GroupCredit(self.group_size).advantages(rewards, components, all_material=all_material)
        info = decision(rewards, components) if info is None else info
        old, parts = _parts(components, info)
        original = base['normalization_std']
        fallback = bool(original == 0)
        scale = original
        if fallback:
            all_new = raw_advantage_parts(all_material[:, :, 1:])['A_raw'].clone()
            for rank in range(len(all_material)):
                item = all_material[rank:rank+1]
                other_info = decision(item[:, :, 0], item[:, :, 1:])
                if other_info['eligible']:
                    all_new[rank:rank+1] += other_info['delta'].to(all_new.device)
            scale = all_new.std(unbiased=False)
        active = parts['A_raw'].ne(0).any(1, keepdim=True)
        out = dict(base, **parts, original_normalization_std=original,
                   normalization_std=scale, normalization_fallback=fallback)
        if info['eligible'] or fallback:
            out['advantages'] = (parts['A_raw']/(scale+1e-8)).reshape(-1).float()
            out['adv_mask'] = active.expand_as(parts['A_raw']).reshape(-1).float()
        meta = dict(base['statistics'])
        meta.update(structured_advantage_schema=SCHEMA, normalization=self.protocol['normalization'],
            structured_zero_groups=int((~active).sum()),
            effective_PG_candidate_fraction=float(out['adv_mask'].mean()),
            saturated_credit_status=info['status'],
            saturated_credit_subtype_groups=int(info['subtype']),
            saturated_credit_activated_groups=int(info['eligible']),
            saturated_credit_activated_candidates=self.group_size*int(info['eligible']),
            saturated_credit_skipped_not_subtype=int(not info['subtype']),
            saturated_credit_skipped_canonical_tied=int(info['subtype'] and not info['canonical_varied']),
            saturated_credit_original_normalization_std=float(original),
            saturated_credit_effective_normalization_std=float(scale),
            saturated_credit_normalization_fallback=int(fallback),
            saturated_credit_delta_abs_mean=float(info['delta'].abs().mean()) if info['eligible'] else 0.,
            saturated_credit_adv_cap5_candidates=int(out['advantages'].abs().gt(5).sum()))
        out['statistics'] = meta
        return out

    @torch.no_grad()
    def finalize(self, rollout):
        if any(k in rollout for k in ('advantages', 'structured_advantage', 'canonical_rewards',
                                     'reward_transform', 'saturated_credit', 'dense_dac')):
            raise ValueError('Finalize fresh, mutually exclusive saturation credit once')
        m = canonical_components(rollout['records'], self.group_size, device=rollout['rewards'].device)
        if not torch.equal(rollout['rewards'], m['canonical_rewards']):
            raise ValueError('Canonical reward labels changed')
        info = decision(rollout['rewards'], m['components'])
        result = self.advantages(rollout['rewards'], m['components'], info=info)
        rollout['canonical_rewards'] = rollout['rewards'].detach().clone()
        for k in ('advantages', 'adv_mask', 'global_std', 'raw_group_std', 'normalization_std', 'original_normalization_std'):
            rollout[k] = result[k]
        rollout['saturated_credit'] = info
        rollout['advantage_statistics'] = result['statistics']
        rollout['structured_advantage'] = dict(protocol=copy.deepcopy(self.protocol),
            components=m['components'], **{k:result[k] for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')})
        return rollout


def _equal(value, want, name, tolerance=0.):
    if (not torch.is_tensor(value) or value.requires_grad or value.dtype != want.dtype
            or value.shape != want.shape or not torch.allclose(value.cpu(), want.cpu(), atol=tolerance, rtol=0)):
        raise ValueError('Changed saturation-credit material: '+name)


def _scale(value, name):
    if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float64
            or value.ndim or not torch.isfinite(value) or value < 0):
        raise ValueError('Invalid saturation normalization: '+name)
    return float(value)


def validate_saturated_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, credit, legacy=False, _recipe_validator=None):
    (_recipe_validator or validate_saturated_recipe)(identity['recipe'])
    g = identity['recipe']['grpo']['sample']['group_size']
    if credit.protocol != protocol(g):
        raise ValueError('Saturation protocol/recipe mismatch')
    summary = validate_structure(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    summary.update(structured_advantage_schema=SCHEMA, structured_protocol=credit.protocol, group_size=g, structured_world=8)
    if not summary['pending']:
        return summary
    if 'dense_dac' in rollout or 'reward_transform' in rollout:
        raise ValueError('DenseDAC/reward transform cannot mix with saturation credit')
    material = canonical_components(rollout['records'], g)
    for k in ('rewards', 'canonical_rewards'):
        _equal(rollout[k], material['canonical_rewards'], k)
    info = decision(material['canonical_rewards'], material['components'])
    saved_info = rollout.get('saturated_credit', {})
    if set(saved_info) != set(info) or tensor_digest(saved_info) != tensor_digest(info):
        raise ValueError('Saturation eligibility/delta/digest changed')
    if info['delta'] is not None:
        _equal(saved_info['delta'], info['delta'], 'delta')
    old, parts = _parts(material['components'], info)
    saved = rollout.get('structured_advantage', {})
    if (set(saved) != {'protocol', 'components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'}
            or saved['protocol'] != credit.protocol):
        raise ValueError('Saturation structured protocol changed')
    _equal(saved['components'], material['components'], 'components')
    for k, value in parts.items():
        _equal(saved[k], value, k, 0. if value.dtype == torch.bool else 1e-14)
    scale = _scale(rollout.get('normalization_std'), 'effective std')
    original = _scale(rollout.get('original_normalization_std'), 'original std')
    if original > 0 and scale != original:
        raise ValueError('Nonzero original Safety std must be retained exactly')
    offsets = material['canonical_rewards'].double()-material['canonical_rewards'].double()[:, :1]
    _equal(rollout['raw_group_std'], offsets.std(1, unbiased=False, keepdim=True), 'raw reward std', 1e-12)
    meta = rollout['advantage_statistics']
    if (meta.get('structured_advantage_schema') != SCHEMA or meta.get('group_size') != g
            or meta.get('normalization_population_candidates') != 8*g
            or meta.get('normalization_population_groups') != 8):
        raise ValueError('Saturation normalization population changed')
    summary.update(normalization_std=scale, original_normalization_std=original,
        canonical_rewards=material['canonical_rewards'].double().tolist(),
        structured_material={k:saved[k].cpu().tolist() for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
        saturation_material={k:(v.cpu().tolist() if torch.is_tensor(v) else v) for k,v in info.items()},
        saturation_digest=tensor_digest(info), advantage_statistics=copy.deepcopy(meta),
        canonical_rewards_digest=tensor_digest(rollout['canonical_rewards']))
    return summary


def validate_saturated_ranks(summaries, *, credit):
    if len(summaries) != 8:
        raise ValueError('Eight saturation-credit rank summaries required')
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run', 'pending',
              'group_size', 'structured_world', 'structured_advantage_schema', 'structured_protocol')
    for s in summaries:
        if (s.get('structured_protocol') != credit.protocol or s.get('group_size') != credit.group_size
                or {k:s[k] for k in common} != {k:summaries[0][k] for k in common}):
            raise ValueError('Ranks disagree on saturation recovery identity')
    if not summaries[0]['pending']:
        return summaries
    if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):
        raise ValueError('Ranks disagree on collection-start policy')
    rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
    components = torch.tensor([s['structured_material']['components'][0] for s in summaries], dtype=torch.float64)
    material = torch.cat((rewards.unsqueeze(-1), components), -1)
    for rank, s in enumerate(summaries):
        info = decision(rewards[rank:rank+1], components[rank:rank+1])
        actual = copy.deepcopy(s['saturation_material'])
        if actual.get('delta') is not None:
            actual['delta'] = torch.tensor(actual['delta'], dtype=torch.float64)
        if tensor_digest(actual) != tensor_digest(info) or s['saturation_digest'] != tensor_digest(info):
            raise ValueError('Rank saturation decision changed')
        expected = credit.advantages(rewards[rank:rank+1], components[rank:rank+1], info=info, all_material=material)
        if (s['canonical_rewards'] != s['rewards']
                or abs(s['global_std']-float(expected['global_std'])) > 1e-12
                or abs(s['normalization_std']-float(expected['normalization_std'])) > 1e-12
                or abs(s['original_normalization_std']-float(expected['original_normalization_std'])) > 1e-12
                or not torch.allclose(torch.tensor(s['advantages']), expected['advantages'], atol=1e-7, rtol=1e-6)
                or s['mask'] != expected['adv_mask'].tolist()):
            raise ValueError('Saturation global std/advantages/mask changed')
        if set(s['advantage_statistics']) != set(expected['statistics']):
            raise ValueError('Saturation statistics fields changed')
        for k, want in expected['statistics'].items():
            got = s['advantage_statistics'][k]
            if (abs(got-want) > 1e-12 if k in ('saturated_credit_original_normalization_std',
                    'saturated_credit_effective_normalization_std') else got != want):
                raise ValueError('Saturation statistics changed: '+k)
        for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'):
            if not torch.allclose(torch.tensor(s['structured_material'][k], dtype=expected[k].dtype), expected[k], atol=1e-14, rtol=0):
                raise ValueError('Saturation raw component changed: '+k)
    return summaries


class SaturatedLoss(ResearchLoss):
    def _policy_loss(self, model, rollout):
        total, metrics = super()._policy_loss(model, rollout)
        for key, value in rollout['advantage_statistics'].items():
            if key.startswith('saturated_credit_') and isinstance(value, (int, float)):
                metrics[key] = torch.tensor(value, device=total.device, dtype=torch.float64)
        return total, metrics


class SaturatedRuntimeMixin:
    def make_credit(self, cfg):
        from .dense_credit import DenseRuntimeMixin
        if isinstance(self, DenseRuntimeMixin):
            raise ValueError('Saturation and DenseDAC runtime mixins are mutually exclusive')
        validate_saturated_recipe(cfg)
        return SaturatedCredit(cfg['grpo']['sample']['group_size'])

    def make_actor(self, cfg, args, device):
        actor, oldcfg, sft = super().make_actor(cfg, args, device)
        actor._loss = SaturatedLoss(cfg)
        return actor, oldcfg, sft

    def finalize_advantages(self, rollout):
        return self.credit.finalize(rollout)

    def validate_boundary(self, *args, **kwargs):
        return validate_saturated_boundary(*args, credit=self.credit, **kwargs)

    def validate_ranks(self, summaries):
        return validate_saturated_ranks(summaries, credit=self.credit)

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value['success_saturation_credit_protocol'] = copy.deepcopy(self.credit.protocol)
        return value
