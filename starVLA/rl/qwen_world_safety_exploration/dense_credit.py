"""Initially-legal, DAC-only failed ties: real offroad duration credit.

Failed least-cost candidates remain failed. Canonical scoring and actor inputs
are unchanged. The original Safety raw-A scale is retained whenever nonzero.
"""
import copy

import torch
import torch.distributed as dist

from .credit import GroupCredit, canonical_components, raw_advantage_parts, protocol as base_protocol
from .group_runtime import validate_recipe
from .recovery import validate_structure, tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss

SCHEMA = 'world_SafetyEP_DenseDAC_initial_legal_fixed_base_std_v1'
FIELD = 'non_drivable_body_by_time'


def protocol(g):
    value = base_protocol(g)
    value.update(schema=SCHEMA,
        eligibility='all G NC=TTC=1,DAC=0; original Safety raw A exactly zero; all t0 drivable; real Gx41 binary fields; duration differs',
        dense_field='records[candidate].non_drivable_body_by_time, t=0,.1,...,4seconds',
        dense_cost='d_i=mean(flags_i[1:41]); fraction of failed offroad time, not geometric margin',
        initial_legal_gate='all flags[:,0] == false; initially-invalid groups unchanged',
        dense_delta='mean(d)-d, offset-centered FP64; add to A_safe and A_raw, keep A_EP',
        normalization='original Safety raw-A population std over world*G +1e-8; only exact zero uses updated raw-A global std',
        extra_material='eligible real flags/duration/delta saved; ineligible explicitly None; missing is not zero',
        interpretation='positive A is failed least-cost guidance, never a verified successful trajectory',
        other_groups='original normalized advantages bit-exact when original global std >0; no global renormalization',
        activation_audit='one decision per fresh collection; repeated per-inner metrics must not be counted as new groups')
    return value


def validate_dense_recipe(cfg):
    validate_recipe(cfg)
    if cfg['research']['structured_advantage'] != SCHEMA:
        raise ValueError('DenseDAC requires its independent credit schema')
    return cfg


@torch.no_grad()
def decision(records, components):
    """No synthetic flags; unavailable/malformed/initially-illegal are distinct."""
    if components.shape[0] != 1 or len(records) != components.shape[1]:
        raise ValueError('Exactly one G-candidate scene per rank')
    g = components.shape[1]; old = raw_advantage_parts(components)
    x = components[0]
    subtype = bool((x[:, 0] == 1).all() and (x[:, 1] == 0).all()
                   and (x[:, 2] == 1).all() and old['A_raw'].eq(0).all())
    out = dict(schema=SCHEMA, subtype=subtype, eligible=False,
        status='UNCHANGED_NOT_DAC_ONLY_ZERO_CREDIT', temporal_available=None,
        initial_legal=None, duration_varied=None, flags=None, duration=None,
        delta=None, source_field_digest=None)
    if not subtype:
        return out
    if any(r.get(FIELD) is None for r in records):
        out.update(status='UNCHANGED_MISSING_REAL_TIME_FIELD', temporal_available=False)
        return out
    if any(torch.is_tensor(r[FIELD]) and r[FIELD].requires_grad for r in records):
        raise ValueError('Scorer time flags must be detached feedback')
    # Feedback is CPU material. Keeping its tiny exact-count arithmetic on CPU
    # makes the extra payload identical before/after CPU-native deserialization.
    flags = torch.stack([torch.as_tensor(r[FIELD]).detach().cpu() for r in records])
    if (flags.shape != (g, 41) or not torch.isfinite(flags).all()
            or not ((flags == 0) | (flags == 1)).all()):
        raise ValueError('Actual finite binary Gx41 offroad flags required')
    flags = flags.bool().detach()
    out['temporal_available'] = True
    out['initial_legal'] = not bool(flags[:, 0].any())
    if not out['initial_legal']:
        out['status'] = 'UNCHANGED_INITIALLY_NON_DRIVABLE'
        return out
    duration = flags[:, 1:41].double().mean(1).unsqueeze(0)
    out['duration_varied'] = bool(duration.ne(duration[:, :1]).any())
    if not out['duration_varied']:
        out['status'] = 'UNCHANGED_OFFROAD_DURATION_TIED'
        return out
    offsets = duration-duration[:, :1]
    delta = offsets.mean(1, keepdim=True)-offsets
    out.update(status='APPLIED_FAILED_DAC_DURATION_CREDIT', eligible=True,
        flags=flags, duration=duration, delta=delta, source_field_digest=tensor_digest(flags))
    return out


def _updated_parts(components, info):
    old = raw_advantage_parts(components)
    parts = dict(old)
    if info['eligible']:
        parts['A_safe'] = old['A_safe']+info['delta'].to(components.device)
        parts['A_raw'] = old['A_raw']+info['delta'].to(components.device)
    return old, parts


class DenseCredit(GroupCredit):
    def __init__(self, group_size):
        super().__init__(group_size)
        self.protocol = protocol(group_size)

    @torch.no_grad()
    def advantages(self, rewards, components, *, info, all_material=None, all_new_raw=None):
        # Do not let this subclass's protocol or updated A alter the base scale.
        base = GroupCredit(self.group_size).advantages(rewards, components, all_material=all_material)
        old, parts = _updated_parts(components, info)
        original_scale = base['normalization_std']
        fallback = bool(original_scale == 0)
        scale = original_scale
        if fallback:
            if all_new_raw is None:
                if dist.is_available() and dist.is_initialized():
                    if dist.get_world_size() != 8:
                        raise ValueError('Dense fallback requires eight ranks')
                    local = parts['A_raw'].contiguous()
                    gathered = [torch.empty_like(local) for _ in range(8)]
                    dist.all_gather(gathered, local)
                    all_new_raw = torch.cat(gathered)
                else:
                    all_new_raw = parts['A_raw']
            all_new_raw = all_new_raw.detach().to(rewards.device, torch.float64)
            if (all_new_raw.ndim != 2 or all_new_raw.shape[1] != self.group_size
                    or not torch.isfinite(all_new_raw).all()):
                raise ValueError('Invalid complete updated raw-A fallback material')
            scale = all_new_raw.std(unbiased=False)
        active = parts['A_raw'].ne(0).any(1, keepdim=True)
        result = dict(base, **parts, original_normalization_std=original_scale,
            normalization_std=scale, normalization_fallback=fallback)
        # Unrelated groups use the already finalized baseline tensor unchanged.
        if info['eligible'] or fallback:
            result['advantages'] = (parts['A_raw']/(scale+1e-8)).reshape(-1).float()
            result['adv_mask'] = active.expand_as(parts['A_raw']).reshape(-1).float()
        meta = dict(base['statistics'])
        meta.update(structured_advantage_schema=SCHEMA, normalization=self.protocol['normalization'],
            structured_zero_groups=int((~active).sum()),
            effective_PG_candidate_fraction=float(result['adv_mask'].mean()),
            dense_dac_status=info['status'], dense_dac_subtype_groups=int(info['subtype']),
            dense_dac_activated_groups=int(info['eligible']),
            dense_dac_activated_candidates=self.group_size*int(info['eligible']),
            dense_dac_missing_fields=int(info['status'] == 'UNCHANGED_MISSING_REAL_TIME_FIELD'),
            dense_dac_initially_invalid=int(info['status'] == 'UNCHANGED_INITIALLY_NON_DRIVABLE'),
            dense_dac_duration_tied=int(info['status'] == 'UNCHANGED_OFFROAD_DURATION_TIED'),
            dense_dac_original_normalization_std=float(original_scale),
            dense_dac_effective_normalization_std=float(scale),
            dense_dac_normalization_fallback=int(fallback),
            dense_dac_adv_cap5_candidates=int(result['advantages'].abs().gt(5).sum()),
            dense_dac_failed_positive_candidates=int((parts['A_raw'] > 0).sum()) if info['eligible'] else 0,
            dense_dac_delta_abs_mean=float(info['delta'].abs().mean()) if info['eligible'] else 0.)
        result['statistics'] = meta
        return result

    @torch.no_grad()
    def finalize(self, rollout):
        if any(k in rollout for k in ('advantages', 'structured_advantage', 'canonical_rewards', 'reward_transform', 'dense_dac')):
            raise ValueError('Finalize fresh DenseDAC credit once only')
        material = canonical_components(rollout['records'], self.group_size, device=rollout['rewards'].device)
        if not torch.equal(rollout['rewards'], material['canonical_rewards']):
            raise ValueError('Original canonical rewards changed')
        info = decision(rollout['records'], material['components'])
        stats = self.advantages(rollout['rewards'], material['components'], info=info)
        rollout['canonical_rewards'] = rollout['rewards'].detach().clone()
        for k in ('advantages', 'adv_mask', 'global_std', 'raw_group_std', 'normalization_std', 'original_normalization_std'):
            rollout[k] = stats[k]
        rollout['dense_dac'] = info
        rollout['advantage_statistics'] = stats['statistics']
        rollout['structured_advantage'] = dict(protocol=copy.deepcopy(self.protocol),
            components=material['components'],
            **{k:stats[k] for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')})
        return rollout


def _tensor_equal(value, want, field, *, tolerance=0.):
    if (not torch.is_tensor(value) or value.requires_grad or value.dtype != want.dtype
            or value.shape != want.shape
            or not (torch.equal(value.cpu(), want.cpu()) if tolerance == 0 else
                    torch.allclose(value.cpu(), want.cpu(), atol=tolerance, rtol=0))):
        raise ValueError('Changed detached DenseDAC material: '+field)


def _scalar(value, field):
    if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float64
            or value.ndim != 0 or not bool(torch.isfinite(value)) or float(value) < 0):
        raise ValueError('Invalid DenseDAC normalization material: '+field)
    return float(value)


def validate_dense_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, credit, legacy=False):
    validate_dense_recipe(identity['recipe'])
    g = identity['recipe']['grpo']['sample']['group_size']
    if credit.protocol != protocol(g):
        raise ValueError('DenseDAC credit/recipe protocol mismatch')
    summary = validate_structure(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    summary.update(structured_advantage_schema=SCHEMA, structured_protocol=credit.protocol,
                   group_size=g, structured_world=8)
    if not summary['pending']:
        return summary
    if 'reward_transform' in rollout:
        raise ValueError('DenseDAC cannot transform canonical rewards')
    material = canonical_components(rollout['records'], g)
    for k in ('rewards', 'canonical_rewards'):
        _tensor_equal(rollout.get(k), material['canonical_rewards'], k)
    info = decision(rollout['records'], material['components'])
    actual = rollout.get('dense_dac', {})
    if set(actual) != set(info):
        raise ValueError('DenseDAC material fields changed')
    for k in ('flags', 'duration', 'delta'):
        if info[k] is None:
            if actual[k] is not None:
                raise ValueError('Ineligible DenseDAC material must be None: '+k)
        else:
            _tensor_equal(actual[k], info[k], k)
    if tensor_digest(actual) != tensor_digest(info):
        raise ValueError('Real time fields / eligibility / DenseDAC delta changed')
    old, parts = _updated_parts(material['components'], info)
    saved = rollout.get('structured_advantage', {})
    if (set(saved) != {'protocol', 'components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'}
            or saved.get('protocol') != credit.protocol):
        raise ValueError('DenseDAC structured protocol changed')
    _tensor_equal(saved['components'], material['components'], 'components')
    for k, v in parts.items():
        _tensor_equal(saved[k], v, k, tolerance=0. if v.dtype == torch.bool else 1e-14)
    scale = _scalar(rollout.get('normalization_std'), 'normalization_std')
    original = _scalar(rollout.get('original_normalization_std'), 'original_normalization_std')
    if original > 0 and scale != original:
        raise ValueError('Nonzero base Safety normalization must remain unchanged')
    raw_std = (material['canonical_rewards'].double()-material['canonical_rewards'].double()[:, :1]).std(1, unbiased=False, keepdim=True)
    _tensor_equal(rollout['raw_group_std'], raw_std, 'raw_group_std', tolerance=1e-12)
    meta = rollout['advantage_statistics']
    if (meta.get('structured_advantage_schema') != SCHEMA or meta.get('group_size') != g
            or meta.get('normalization_population_candidates') != 8*g
            or meta.get('normalization_population_groups') != 8):
        raise ValueError('DenseDAC population identity changed')
    # Global scale and the complete per-rank statistics are checked together.
    summary.update(normalization_std=scale, original_normalization_std=original,
        canonical_rewards=material['canonical_rewards'].double().tolist(),
        structured_material={k:saved[k].cpu().tolist() for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
        dense_material={k:(v.cpu().tolist() if torch.is_tensor(v) else v) for k,v in info.items()},
        dense_digest=tensor_digest(info), advantage_statistics=copy.deepcopy(meta),
        canonical_rewards_digest=tensor_digest(rollout['canonical_rewards']))
    return summary


def validate_dense_ranks(summaries, *, credit):
    if len(summaries) != 8:
        raise ValueError('Eight DenseDAC rank summaries required')
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run', 'pending',
              'group_size', 'structured_world', 'structured_advantage_schema', 'structured_protocol')
    for s in summaries:
        if (s.get('structured_protocol') != credit.protocol or s.get('group_size') != credit.group_size
                or s.get('structured_world') != 8
                or {k:s[k] for k in common} != {k:summaries[0][k] for k in common}):
            raise ValueError('Ranks disagree on DenseDAC recovery identity')
    if not summaries[0]['pending']:
        return summaries
    if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):
        raise ValueError('Ranks disagree on collection-start policy')
    rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
    components = torch.tensor([s['structured_material']['components'][0] for s in summaries], dtype=torch.float64)
    material = torch.cat((rewards.unsqueeze(-1), components), -1)
    all_new = raw_advantage_parts(components)['A_raw'].clone()
    infos = []
    for rank, s in enumerate(summaries):
        info = copy.deepcopy(s['dense_material'])
        for k, dtype in (('flags', torch.bool), ('duration', torch.float64), ('delta', torch.float64)):
            if info[k] is not None:
                info[k] = torch.tensor(info[k], dtype=dtype)
        if tensor_digest(info) != s['dense_digest']:
            raise ValueError('DenseDAC rank material digest changed')
        if info['eligible']:
            flags = info['flags']
            x = components[rank]
            if (flags is None or flags.shape != (credit.group_size, 41) or flags[:, 0].any()
                    or not bool((x[:, 0] == 1).all() and (x[:, 1] == 0).all() and (x[:, 2] == 1).all())
                    or not raw_advantage_parts(components[rank:rank+1])['A_raw'].eq(0).all()):
                raise ValueError('Rank DenseDAC eligibility changed')
            duration = flags[:, 1:41].double().mean(1).unsqueeze(0)
            offsets = duration-duration[:, :1]; delta = offsets.mean(1, keepdim=True)-offsets
            if (not duration.ne(duration[:, :1]).any() or not torch.equal(info['duration'], duration)
                    or not torch.equal(info['delta'], delta)):
                raise ValueError('Rank DenseDAC time reduction changed')
        elif any(info[k] is not None for k in ('flags', 'duration', 'delta')):
            raise ValueError('Ineligible rank must not contain synthetic DenseDAC material')
        infos.append(info)
        if info['eligible']:
            all_new[rank:rank+1] += info['delta']
    for rank, s in enumerate(summaries):
        expected = credit.advantages(rewards[rank:rank+1], components[rank:rank+1], info=infos[rank],
                                     all_material=material, all_new_raw=all_new)
        if (s['canonical_rewards'] != s['rewards']
                or abs(s['global_std']-float(expected['global_std'])) > 1e-12
                or abs(s['normalization_std']-float(expected['normalization_std'])) > 1e-12
                or abs(s['original_normalization_std']-float(expected['original_normalization_std'])) > 1e-12
                or not torch.allclose(torch.tensor(s['advantages']), expected['advantages'], atol=1e-7, rtol=1e-6)
                or s['mask'] != expected['adv_mask'].tolist()):
            raise ValueError('DenseDAC global normalization/activation/A/mask changed')
        if set(s['advantage_statistics']) != set(expected['statistics']):
            raise ValueError('DenseDAC statistics fields changed')
        for k, want in expected['statistics'].items():
            got = s['advantage_statistics'][k]
            if (abs(got-want) > 1e-12 if k in ('dense_dac_original_normalization_std',
                                              'dense_dac_effective_normalization_std') else got != want):
                raise ValueError('DenseDAC activation/statistics changed: '+k)
        for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'):
            if not torch.allclose(torch.tensor(s['structured_material'][k], dtype=expected[k].dtype), expected[k], atol=1e-14, rtol=0):
                raise ValueError('DenseDAC rank raw component changed: '+k)
    return summaries


class DenseDACLoss(ResearchLoss):
    """Native journal activation readouts only; source loss/gradients unchanged."""
    def _policy_loss(self, model, rollout):
        total, metrics = super()._policy_loss(model, rollout)
        meta = rollout['advantage_statistics']
        for k, value in meta.items():
            if k.startswith('dense_dac_') and isinstance(value, (int, float)):
                metrics[k] = torch.tensor(value, device=total.device, dtype=torch.float64)
        return total, metrics


class DenseRuntimeMixin:
    def make_credit(self, cfg):
        validate_dense_recipe(cfg)
        return DenseCredit(cfg['grpo']['sample']['group_size'])

    def make_actor(self, cfg, args, device):
        actor, oldcfg, sft = super().make_actor(cfg, args, device)
        actor._loss = DenseDACLoss(cfg)
        return actor, oldcfg, sft

    def finalize_advantages(self, rollout):
        return self.credit.finalize(rollout)

    def validate_boundary(self, *args, **kwargs):
        return validate_dense_boundary(*args, credit=self.credit, **kwargs)

    def validate_ranks(self, summaries):
        return validate_dense_ranks(summaries, credit=self.credit)

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value['dense_credit_protocol'] = copy.deepcopy(self.credit.protocol)
        return value
