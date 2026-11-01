"""Task-gated zeroing of negative credit on actually verified high candidates.

This is a custom biased credit rule, not unbiased GRPO or a Sign replication.
It supplies no new positive examples and does not guarantee deployment retention.
"""
import copy

import torch

from .credit import GroupCredit, canonical_components, raw_advantage_parts, protocol as base_protocol
from .group_runtime import validate_recipe
from .recovery import validate_structure, tensor_digest
from .saturated_credit import _equal, _scale
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss

SCHEMA = 'world_SafetyEP_verified_high_negative_A_zero_fixed_base_std_v1'
QUALITY_FIELDS = ('comfort', 'score')


def protocol(g):
    p = base_protocol(g)
    p.update(schema=SCHEMA,
        eligibility='candidate official NC=DAC=TTC=Comfort1 and FP64 canonical score>=.9 and original Safety rawA<0',
        canonical_numeric='quality gate uses original FP64 official score, not rounded FP32 rollout reward',
        original_sign='gate uses stored actual original FP64 rawA; CPU recovery validates it within1e-14 before reusing its exact sign',
        delta='eligible delta=-original rawA; A_safe unchanged; A_EP +=delta and rawA eligible becomes0; all others unchanged',
        normalization='original world*G Safety rawA std +1e-8 strictly retained; no recenter, no fallback, no new global normalization',
        mask='entire original group PGmask retained even when a protected candidate becomes zero',
        group_mean='updated group mean may be positive: explicit task-gated baseline bias, not unbiased GRPO',
        preserved='ineligible rawA/normalizedA bitexact; eligible negative normalizedA replaced by0 only',
        high_quality='strict four metric passes with canonical score>=.9; not a new positive advantage',
        sign_relation='inspired by fixed-baseline freedom from group mean; not Sign/NGRPO algorithm replication',
        exclusivity='independent from success-saturation and DenseDAC; no combined runtime',
        retention_limit='no negative self-credit on this candidate; shared parameters, anchor and other states may still harm it',
        activation_audit='one decision per fresh collection; native metrics repeat for four inner uses')
    return p


def validate_protected_recipe(cfg):
    validate_recipe(cfg)
    if cfg['research']['structured_advantage'] != SCHEMA:
        raise ValueError('Protected high credit requires its independent schema')
    return cfg


@torch.no_grad()
def decision(records, components, original_raw):
    g = components.shape[1]
    if components.shape != (1, g, 4) or g not in (8, 16):
        raise ValueError('One G8/G16 scene per rank required')
    m = canonical_components(records, g)
    _equal(components.detach().cpu(), m['components'], 'record/component binding')
    raw = original_raw.detach().cpu()
    if (raw.dtype != torch.float64 or raw.shape != (1, g) or original_raw.requires_grad
            or not torch.isfinite(raw).all()):
        raise ValueError('Actual detached original FP64 raw advantage required')
    quality = torch.tensor([[[float(r['metrics'][k]) for k in QUALITY_FIELDS] for r in records]], dtype=torch.float64)
    feasible = m['components'][:, :, :3].eq(1).all(-1)
    high = feasible & quality[:, :, 0].eq(1) & quality[:, :, 1].ge(.9)
    eligible = high & raw.lt(0)
    delta = torch.where(eligible, -raw, torch.zeros_like(raw))
    return dict(schema=SCHEMA, status='APPLIED_VERIFIED_HIGH_NEGATIVE_ZERO' if eligible.any() else 'UNCHANGED_NO_NEGATIVE_VERIFIED_HIGH',
        activated=bool(eligible.any()), quality_fields=list(QUALITY_FIELDS), quality_material=quality,
        quality_flags=high, eligible=eligible, original_raw_A=raw.clone(), delta=delta,
        quality_digest=tensor_digest(quality), original_raw_digest=tensor_digest(raw))


def updated_parts(components, info):
    old = raw_advantage_parts(components)
    delta = info['delta'].to(components.device)
    out = dict(old, A_EP=old['A_EP']+delta, A_raw=old['A_raw'].clone())
    out['A_raw'][info['eligible'].to(components.device)] = 0.
    return old, out


class ProtectedCredit(GroupCredit):
    make_decision = staticmethod(decision)
    update_parts = staticmethod(updated_parts)

    def __init__(self, group_size):
        super().__init__(group_size)
        self.protocol = protocol(group_size)

    @torch.no_grad()
    def advantages(self, rewards, components, *, records=None, info=None, all_material=None):
        base = GroupCredit(self.group_size).advantages(rewards, components, all_material=all_material)
        if info is None:
            if records is None:raise ValueError('Verified original canonical records required')
            info = self.make_decision(records, components, base['A_raw'])
        if info['activated'] and not bool(base['normalization_std'] > 0):
            raise ValueError('Protecting genuinely negative rawA requires nonzero original global std; no fabricated fallback')
        old, parts = self.update_parts(components, info)
        out = dict(base, **parts, original_normalization_std=base['normalization_std'])
        if info['activated']:
            out['advantages'] = base['advantages'].clone()
            out['advantages'][info['eligible'].reshape(-1).to(rewards.device)] = 0.
        # Mask and every ineligible normalized candidate are the original tensors.
        meta = dict(base['statistics'])
        meta.update(structured_advantage_schema=SCHEMA, normalization=self.protocol['normalization'],
            protected_credit_status=info['status'], protected_credit_activated_groups=int(info['activated']),
            protected_credit_quality_candidates=int(info['quality_flags'].sum()),
            protected_credit_activated_candidates=int(info['eligible'].sum()),
            protected_credit_unchanged_candidates=int(info['eligible'].numel()-info['eligible'].sum()),
            protected_credit_removed_negative_raw_mass=float(info['delta'].sum()),
            protected_credit_raw_group_mean_before=float(info['original_raw_A'].mean()),
            protected_credit_raw_group_mean_after=float(parts['A_raw'].mean()),
            protected_credit_original_normalization_std=float(base['normalization_std']),
            protected_credit_normalized_removed_mass=float((out['advantages']-base['advantages']).detach().cpu().double().sum()),
            protected_credit_adv_cap5_candidates=int(out['advantages'].abs().gt(5).sum()))
        out['statistics'] = meta
        out['protection'] = info
        return out

    @torch.no_grad()
    def finalize(self, rollout):
        if any(k in rollout for k in ('advantages', 'structured_advantage', 'canonical_rewards', 'reward_transform',
                                     'protected_credit', 'saturated_credit', 'dense_dac')):
            raise ValueError('Finalize fresh mutually exclusive protected credit once')
        m = canonical_components(rollout['records'], self.group_size, device=rollout['rewards'].device)
        if not torch.equal(rollout['rewards'], m['canonical_rewards']):
            raise ValueError('Canonical rewards changed')
        out = self.advantages(rollout['rewards'], m['components'], records=rollout['records'])
        rollout['canonical_rewards'] = rollout['rewards'].detach().clone()
        for k in ('advantages', 'adv_mask', 'global_std', 'raw_group_std', 'normalization_std', 'original_normalization_std'):
            rollout[k] = out[k]
        rollout['protected_credit'] = out['protection']
        rollout['advantage_statistics'] = out['statistics']
        rollout['structured_advantage'] = dict(protocol=copy.deepcopy(self.protocol), components=m['components'],
            **{k:out[k] for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')})
        return rollout


def validate_protected_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, credit, legacy=False):
    validate_protected_recipe(identity['recipe'])
    g = identity['recipe']['grpo']['sample']['group_size']
    if credit.protocol != protocol(g):raise ValueError('Protected credit/recipe mismatch')
    summary = validate_structure(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    summary.update(structured_advantage_schema=SCHEMA, structured_protocol=credit.protocol, group_size=g, structured_world=8)
    if not summary['pending']:return summary
    if any(k in rollout for k in ('reward_transform', 'saturated_credit', 'dense_dac')):
        raise ValueError('Protected credit cannot mix transformations')
    material = canonical_components(rollout['records'], g)
    for k in ('rewards', 'canonical_rewards'):_equal(rollout[k], material['canonical_rewards'], k)
    actual = rollout.get('protected_credit', {})
    raw = actual.get('original_raw_A')
    old = raw_advantage_parts(material['components'])
    _equal(raw, old['A_raw'], 'original raw Safety credit', 1e-14)
    info = decision(rollout['records'], material['components'], raw)
    if set(actual) != set(info) or tensor_digest(actual) != tensor_digest(info):
        raise ValueError('Protected quality/gate/original rawA/delta material changed')
    for k in ('quality_material','quality_flags','eligible','original_raw_A','delta'):
        _equal(actual[k], info[k], k)
    _, parts = updated_parts(material['components'], info)
    saved = rollout.get('structured_advantage', {})
    if (set(saved) != {'protocol','components','feasible','cost','A_safe','A_EP','A_raw'}
            or saved.get('protocol') != credit.protocol):raise ValueError('Protected structured protocol changed')
    _equal(saved['components'], material['components'], 'components')
    for k,v in parts.items():_equal(saved[k], v, k, 0. if v.dtype == torch.bool else 1e-14)
    scale = _scale(rollout.get('normalization_std'), 'normalization std')
    original = _scale(rollout.get('original_normalization_std'), 'original normalization std')
    if scale != original or info['activated'] and original <= 0:
        raise ValueError('Protected credit strictly retains nonzero base std; no fallback')
    offsets = material['canonical_rewards'].double()-material['canonical_rewards'].double()[:, :1]
    _equal(rollout['raw_group_std'], offsets.std(1, unbiased=False, keepdim=True), 'raw group std', 1e-12)
    meta = rollout['advantage_statistics']
    if (meta.get('structured_advantage_schema') != SCHEMA or meta.get('group_size') != g
            or meta.get('normalization_population_candidates') != 8*g
            or meta.get('normalization_population_groups') != 8):raise ValueError('Protected population changed')
    summary.update(normalization_std=scale, original_normalization_std=original,
        canonical_rewards=material['canonical_rewards'].double().tolist(),
        structured_material={k:saved[k].cpu().tolist() for k in ('components','feasible','cost','A_safe','A_EP','A_raw')},
        protection_material={k:(v.cpu().tolist() if torch.is_tensor(v) else v) for k,v in info.items()},
        protection_digest=tensor_digest(info), advantage_statistics=copy.deepcopy(meta),
        canonical_rewards_digest=tensor_digest(rollout['canonical_rewards']))
    return summary


def _restore_info(value):
    out = copy.deepcopy(value)
    for k,dtype in (('quality_material',torch.float64),('quality_flags',torch.bool),('eligible',torch.bool),
                    ('original_raw_A',torch.float64),('delta',torch.float64)):
        out[k] = torch.tensor(out[k], dtype=dtype)
    return out


def validate_protected_ranks(summaries, *, credit):
    if len(summaries) != 8:raise ValueError('Eight protected-credit rank summaries required')
    common = ('phase','updates','collections','scene_cursor','next_inner','run','pending','group_size',
              'structured_world','structured_advantage_schema','structured_protocol')
    for s in summaries:
        if (s.get('structured_protocol') != credit.protocol or s.get('structured_world') != 8
                or s.get('group_size') != credit.group_size
                or {k:s[k] for k in common} != {k:summaries[0][k] for k in common}):raise ValueError('Ranks disagree on protected identity')
    if not summaries[0]['pending']:return summaries
    if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):raise ValueError('Collection-start behavior differs')
    rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
    components = torch.tensor([s['structured_material']['components'][0] for s in summaries], dtype=torch.float64)
    material = torch.cat((rewards.unsqueeze(-1), components), -1)
    old = raw_advantage_parts(components)
    for rank,s in enumerate(summaries):
        info = _restore_info(s['protection_material'])
        if tensor_digest(info) != s['protection_digest']:raise ValueError('Protected rank material digest changed')
        _equal(info['original_raw_A'], old['A_raw'][rank:rank+1], 'rank original rawA', 1e-14)
        # Quality scalars were joined to the actual canonical records at boundary.
        quality = info['quality_material']
        expected_high = components[rank:rank+1,:,:3].eq(1).all(-1)&quality[:,:,0].eq(1)&quality[:,:,1].ge(.9)
        eligible = expected_high&info['original_raw_A'].lt(0)
        if (not torch.equal(info['quality_flags'], expected_high) or not torch.equal(info['eligible'], eligible)
                or not torch.equal(info['delta'], torch.where(eligible,-info['original_raw_A'],torch.zeros_like(info['original_raw_A'])))
                or info['activated'] != bool(eligible.any())):raise ValueError('Protected rank eligibility/delta changed')
        expected = credit.advantages(rewards[rank:rank+1], components[rank:rank+1], info=info, all_material=material)
        if (s['canonical_rewards'] != s['rewards'] or abs(s['global_std']-float(expected['global_std'])) > 1e-12
                or abs(s['normalization_std']-float(expected['normalization_std'])) > 1e-12
                or s['normalization_std'] != s['original_normalization_std']
                or not torch.allclose(torch.tensor(s['advantages']), expected['advantages'], atol=1e-7, rtol=1e-6)
                or s['mask'] != expected['adv_mask'].tolist()):raise ValueError('Protected global std/advantage/mask changed')
        if set(s['advantage_statistics']) != set(expected['statistics']):raise ValueError('Protected statistic fields changed')
        numeric_tolerance = ('protected_credit_original_normalization_std','protected_credit_raw_group_mean_before',
            'protected_credit_raw_group_mean_after','protected_credit_removed_negative_raw_mass','protected_credit_normalized_removed_mass')
        for k,want in expected['statistics'].items():
            got = s['advantage_statistics'][k]
            if (abs(got-want) > (1e-6 if k=='protected_credit_normalized_removed_mass' else 1e-12) if k in numeric_tolerance else got != want):
                raise ValueError('Protected statistics changed: '+k)
        for k in ('feasible','cost','A_safe','A_EP','A_raw'):
            if not torch.allclose(torch.tensor(s['structured_material'][k],dtype=expected[k].dtype),expected[k],atol=1e-14,rtol=0):
                raise ValueError('Protected raw component changed: '+k)
    return summaries


class ProtectedLoss(ResearchLoss):
    def _policy_loss(self, model, rollout):
        total, metrics = super()._policy_loss(model, rollout)
        for k,v in rollout['advantage_statistics'].items():
            if k.startswith('protected_credit_') and isinstance(v,(int,float)):
                metrics[k] = torch.tensor(v,device=total.device,dtype=torch.float64)
        return total, metrics


class ProtectedRuntimeMixin:
    def make_credit(self,cfg):
        from .dense_credit import DenseRuntimeMixin
        from .saturated_credit import SaturatedRuntimeMixin
        if isinstance(self,(DenseRuntimeMixin,SaturatedRuntimeMixin)):
            raise ValueError('Protected, saturated and DenseDAC credit mixins are mutually exclusive')
        validate_protected_recipe(cfg)
        return ProtectedCredit(cfg['grpo']['sample']['group_size'])
    def make_actor(self,cfg,args,device):
        actor,oldcfg,sft = super().make_actor(cfg,args,device)
        actor._loss = ProtectedLoss(cfg)
        return actor,oldcfg,sft
    def finalize_advantages(self,rollout):return self.credit.finalize(rollout)
    def validate_boundary(self,*args,**kwargs):return validate_protected_boundary(*args,credit=self.credit,**kwargs)
    def validate_ranks(self,summaries):return validate_protected_ranks(summaries,credit=self.credit)
    def recipe_identity(self,cfg,args,condition_sha):
        value = super().recipe_identity(cfg,args,condition_sha)
        value['verified_high_protection_protocol'] = copy.deepcopy(self.credit.protocol)
        return value
