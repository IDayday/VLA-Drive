"""Verified positive TRAIN trajectory CFM, separate from fresh SafetyEP PPO.

No negative/reference-energy, endpoint likelihood, new parameters or historical
PG advantages. The teacher is metric-verified, not a human optimum. Scalar loss
statistics do not establish a separate auxiliary gradient magnitude.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.contracts import registered_stochastic_steps_for_recipe
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest, validate_boundary as source_boundary
from .policy import ResearchLoss
from .preference_auxiliary import file_sha, fingerprint, array_digest, current_rank, METRICS
from .safety_efficiency import (
    STRUCTURED_SCHEMA, STRUCTURED_PROTOCOL, canonical_components,
    safety_efficiency_parts, validate_safety_efficiency_ranks,
)

AUX_SCHEMA = 'world_verified_positive_CFM_teacher_aux_v1'
BANK_SCHEMA = 'world_TRAIN_rescue_retention_positive_teacher_manifest_v1'
DISTRIBUTION = 'full80_low10_high10_static_v1'
AUX_CONFIG = dict(schema=AUX_SCHEMA, lambda_low=1., lambda_high=.2,
                  sigma_min=.2, sigma_max=.8, stateless_seed=42)
AUX_PROTOCOL = dict(
    **AUX_CONFIG,
    objective='lambda_kind * mean((v_theta((1-sigma)*teacher+sigma*noise,sigma,current_scene)-noise+teacher)^2)',
    velocity='native World clean-x conversion v=(state-pred_x0)/sigma exactly once',
    draws='8 independent sigma U[.2,.8] and Gaussian noise per fresh scene; stateless token/update/rank/seed42; reuse four inners',
    teacher='static same-TRAIN-scene strict NC=DAC=TTC=Comfort=1 verified endpoint, upgraded only when a fresh strict4 candidate score exceeds static by >1e-6; highest fresh score, earliest index tie; fixed four inners',
    target_origin='static_teacher or online_upgrade with chain endpoint/scored requested/behavior/cache hashes; verified positive self-imitation CFM, not the AWR algorithm; never target advantages',
    skip='fresh eight NC=DAC=TTC=Comfort=EP=1, or no teacher for fresh token',
    policy_gradient='fresh canonical records only; original SafetyEP kappa0/lambdaEP1 with global64 A_raw population std once',
    anchor='permanent IL70 source mean anchor averaged over W2[7,8], beta50; never step9',
    memory='current scene/predictive Memory rebuilt in separate context per inner; no retained graphs',
    interpretation='positive CFM supervision, not DPO; no negative/reference energy; scalar loss is not auxiliary gradient magnitude',
)


def validate_rescue_recipe(cfg):
    r = cfg.get('research', {})
    if (r.get('arm') not in ('PPO_RESCUE_REPLAY', 'PPO_RESCUE_CFM')
            or registered_stochastic_steps_for_recipe(cfg) != [7, 8]
            or r.get('structured_advantage') != STRUCTURED_SCHEMA
            or cfg['learning_rate'] != 5e-5 or cfg['memory_learning_rate'] != 5e-6
            or cfg['grpo']['train']['kl_beta'] != 50 or cfg['max_steps'] != 50000
            or cfg.get('seed') != 42
            or r.get('warmstart', {}).get('schema') != 'world_verified_RL_weights_only_warmstart_v1'):
        raise ValueError('Independent Rescue matched SafetyEP/W2/LR/beta50/seed42/scheduler50000 recipe required')
    return cfg


class RescueTeacherBank:
    """One-time hash-pinned compact bank validation; no cache/weight rescans."""
    def __init__(self, manifest_path, expected_sha256, eligible_tokens):
        path = Path(manifest_path)
        if file_sha(path) != expected_sha256:
            raise ValueError('Rescue teacher manifest file SHA changed')
        self.sha256 = expected_sha256
        self.manifest = m = json.loads(path.read_text())
        body = dict(m); seal = body.pop('sha256', None)
        if (seal != fingerprint(body) or m.get('schema') != BANK_SCHEMA
                or m.get('status') != 'READY_OFFLINE_VERIFIED_TEACHERS'
                or m.get('offline_only') is not True or m.get('on_policy_GRPO_eligible') is not False
                or m.get('NAVTEST_tokens_or_logs_included') is not False
                or m.get('metric_keys') != list(METRICS)):
            raise ValueError('Rescue bank seal/schema/offline boundary changed')
        for name in ('teachers.npz', 'teachers.jsonl'):
            spec = m['artifacts'][name]
            if (Path(spec['path']).resolve() != (path.parent/name).resolve()
                    or file_sha(path.parent/name) != spec['sha256']):
                raise ValueError('Rescue teacher artifact SHA/path changed: '+name)
        with np.load(path.parent/'teachers.npz', allow_pickle=False) as archive:
            self.arrays = a = {k: archive[k] for k in archive.files}
        rows = [json.loads(line) for line in (path.parent/'teachers.jsonl').read_text().splitlines()]
        n = len(rows)
        if (n != m['count'] or not n or a['metric_keys'].tolist() != list(METRICS)
                or a['clean_normalized'].shape != (n, 8, 3) or a['clean_normalized'].dtype != np.float32
                or a['requested_trajectory'].shape != (n, 8, 3) or a['requested_trajectory'].dtype != np.float64
                or a['metrics'].shape != (n, 7) or a['metrics'].dtype != np.float64
                or any(a[k].shape != (n,) for k in ('tokens', 'logs', 'bank_kind'))):
            raise ValueError('Rescue teacher dimensions/dtypes/count changed')
        for k in ('clean_normalized', 'requested_trajectory', 'metrics'):
            if not np.isfinite(a[k]).all():
                raise ValueError('Nonfinite rescue teacher: '+k)
        x = a['metrics']
        if (not np.all((x >= 0) & (x <= 1))
                or not np.allclose(x[:, 1]*x[:, 2]*(5*x[:, 4]+5*x[:, 3]+2*x[:, 5])/12,
                                   x[:, 0], atol=5e-15, rtol=0)
                or not np.all(x[:, [1, 2, 3, 5]] == 1)):
            raise ValueError('Teacher lacks strict canonical NC/DAC/TTC/Comfort witness')
        decoded = ActionCodec.denorm_odo(torch.from_numpy(a['clean_normalized'])).numpy().astype(np.float64)
        eligible = set(eligible_tokens); self.by_token = {}
        for i, row in enumerate(rows):
            token, kind = row['token'], row['bank_kind']
            if (token not in eligible or token in self.by_token or kind not in ('low', 'high')
                    or row['normalized_index'] != i or str(a['tokens'][i]) != token
                    or str(a['logs'][i]) != row['log'] or str(a['bank_kind'][i]) != kind
                    or row['protocol_sha256'] != m['protocol_sha256']
                    or row.get('offline_only') is not True or row.get('on_policy_GRPO_eligible') is not False
                    or row.get('metric_verified') is not True or row.get('human_optimal') is not False
                    or row['metrics'] != dict(zip(METRICS, x[i].tolist()))):
                raise ValueError('Teacher outside registered TRAIN/protocol or offline boundary')
            requested_sha, clean_sha = array_digest(a['requested_trajectory'][i]), array_digest(a['clean_normalized'][i])
            expected_id = fingerprint(dict(token=token, bank_kind=kind, source=row['source'], requested_trajectory_sha256=requested_sha))
            if (row['teacher_id'] != expected_id or row['requested_trajectory_sha256'] != requested_sha
                    or row['normalized_sha256'] != clean_sha
                    or not isinstance(row['source_scene'], str) or len(row['source_scene_sha256']) != 64
                    or len(row['cache_sha256']) != 64):
                raise ValueError('Teacher endpoint/source/provenance identity changed')
            error = np.abs(decoded[i]-a['requested_trajectory'][i]).max(0)
            codec = row['codec_roundtrip']; bound = np.asarray(codec['bound_by_dimension'], dtype=np.float64)
            if (bound.shape != (3,) or not np.isfinite(bound).all() or np.any(bound < 0)
                    or np.any(bound > .001) or np.any(error > bound)
                    or not np.array_equal(error, np.asarray(codec['max_abs_by_dimension']))):
                raise ValueError('Teacher clean endpoint does not bind the actually scored requested trajectory')
            if kind == 'low':
                baseline = row['baseline_SAFETY10k_score']
                if not (0 <= baseline < .95 and x[i, 4] >= .5 and x[i, 0] > baseline):
                    raise ValueError('Low teacher lacks improving efficient canonical witness')
            elif x[i, 0] < .95:
                raise ValueError('High retention teacher below registered canonical threshold')
            self.by_token[token] = row
        for kind in ('low', 'high'):
            if set(m[kind+'_tokens']) != {t for t, r in self.by_token.items() if r['bank_kind'] == kind}:
                raise ValueError('Rescue pool token identity changed')

    def clean_teacher(self, token):
        return torch.from_numpy(self.arrays['clean_normalized'][self.by_token[token]['normalized_index']]).detach().clone()


def fresh_saturated(rollout):
    canonical_components(rollout['records'])
    return all(all(r['metrics'][key] == 1 for key in (
        'no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
        'comfort', 'ego_progress')) for r in rollout['records'])


def stateless_material(token, update, rank):
    key = dict(schema=AUX_SCHEMA, token=token, behavior_update=update, rank=rank, seed=42)
    seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':')).encode()).digest()[:8], 'big') & ((1 << 63)-1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    sigma = .2+.6*torch.rand(8, generator=generator, dtype=torch.float32)
    noise = torch.randn(8, 8, 3, generator=generator, dtype=torch.float32)
    return seed, sigma.detach(), noise.detach()


def forward_material(clean, sigma, noise):
    if clean.shape != (8, 3) or sigma.shape != (8,) or noise.shape != (8, 8, 3):
        raise ValueError('Rescue positive teacher requires native G8 CFM samples')
    teacher = clean.detach().expand(8, -1, -1)
    return (1-sigma[:, None, None])*teacher+sigma[:, None, None]*noise, sigma, noise-teacher


def positive_cfm_objective(velocity, target):
    if velocity.shape != (8, 8, 3) or target.shape != velocity.shape:
        raise ValueError('Rescue CFM velocity/target requires [8,8,3]')
    loss = (velocity.float()-target.detach().float()).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite positive CFM teacher loss')
    return loss


@torch.no_grad()
def select_verified_target(rollout, bank):
    """Use actually scored fresh evidence to avoid regression to a static label.

    No model call, scoring, RNG or new PG sample. Selection is once/collection;
    replay recomputes it from the immutable chain/feedback, not current weights.
    """
    canonical_components(rollout['records'])
    row = bank.by_token[rollout['token']]
    if row['log'] != rollout['log']:
        raise ValueError('Teacher and fresh physical log differ')
    target = dict(origin='static_teacher', teacher_id=row['teacher_id'],
        metrics=copy.deepcopy(row['metrics']), normalized_sha256=row['normalized_sha256'],
        requested_trajectory_sha256=row['requested_trajectory_sha256'])
    strict = [i for i, record in enumerate(rollout['records']) if all(record['metrics'][k] == 1 for k in (
        'no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound', 'comfort'))]
    best = max(strict, key=lambda i: rollout['records'][i]['metrics']['score']) if strict else None
    if best is None or rollout['records'][best]['metrics']['score'] <= row['metrics']['score']+1e-6:
        return bank.clean_teacher(rollout['token']), target
    record = rollout['records'][best]
    if (record['token'] != rollout['token'] or record['log'] != rollout['log']
            or record.get('protocol_sha256') != bank.manifest['protocol_sha256']
            or record.get('cache', {}).get('sha256') != row['cache_sha256']):
        raise ValueError('Fresh upgraded target scene/protocol/cache differs from verified bank')
    clean = rollout['chain'][best, -1].detach().float().cpu().contiguous().clone()
    requested = np.asarray(record['requested_trajectory'])
    if clean.shape != (8, 3) or not bool(torch.isfinite(clean).all()) or requested.shape != (8, 3) or not np.isfinite(requested).all():
        raise ValueError('Invalid fresh upgraded target endpoint')
    # Both original rollout decoder and this validation use native FP32 element
    # operations. Allow only conservative FP32 rounding across CPU/CUDA devices;
    # the actual scored requested bits are separately hashed, never relabelled.
    decoded = ActionCodec.denorm_odo(clean).numpy().astype(np.float64)
    requested = np.ascontiguousarray(requested, dtype=np.float64)
    bound = 8*np.finfo(np.float32).eps*(np.abs(decoded)+np.asarray([66.74, 42., 3.53]))
    error = np.abs(decoded-requested)
    if np.any(error > bound):
        raise ValueError('Fresh clean endpoint is not the actually scored requested target')
    target = dict(origin='online_upgrade', teacher_id=row['teacher_id'], candidate_index=best,
        token=rollout['token'], log=rollout['log'],
        behavior_version=copy.deepcopy(rollout['behavior_version']),
        cache=copy.deepcopy(record['cache']), protocol_sha256=record['protocol_sha256'],
        metrics=copy.deepcopy(record['metrics']), normalized_sha256=array_digest(clean.numpy()),
        requested_trajectory_sha256=array_digest(requested),
        codec_roundtrip=dict(max_abs_by_dimension=error.max(0).tolist(), bound_by_dimension=bound.max(0).tolist(),
                             bound_is_not_metric_invariance=True))
    return clean, target


def payload_digest(payload):
    return tensor_digest({k: v for k, v in payload.items() if k != 'material_sha256'})


@torch.no_grad()
def attach_rescue_material(actor, rollout, bank, *, enabled, rank=None):
    if 'rescue_auxiliary' in rollout:
        raise ValueError('Attach immutable rescue material exactly once per collection')
    if rollout.get('protocol_sha256') != bank.manifest['protocol_sha256']:
        raise ValueError('Fresh canonical scoring protocol differs from verified teachers')
    rank = current_rank() if rank is None else rank
    skip = 'replay_control' if not enabled else 'fresh_safety_comfort_EP_saturated' if fresh_saturated(rollout) else 'no_teacher' if rollout['token'] not in bank.by_token else None
    p = dict(protocol=copy.deepcopy(AUX_PROTOCOL), bank_manifest_sha256=bank.sha256,
        token=rollout['token'], behavior_update=rollout['behavior_version']['update'], rank=rank,
        enabled=skip is None, skip_reason=skip)
    if skip is None:
        row = bank.by_token[rollout['token']]
        if row['log'] != rollout['log']:
            raise ValueError('Teacher and fresh physical log differ')
        seed, sigma, noise = stateless_material(p['token'], p['behavior_update'], rank)
        clean, target = select_verified_target(rollout, bank)
        p.update(teacher=copy.deepcopy(row), target=target, selection_seed=seed,
            clean=clean.to(actor.device),
            sigma=sigma.to(actor.device), noise=noise.to(actor.device),
            condition_identity=rollout['cond_bg']['identity'], canonical_protocol_sha256=rollout['protocol_sha256'],
            coefficient=1. if row['bank_kind'] == 'low' else .2)
    p['material_sha256'] = payload_digest(p)
    rollout['rescue_auxiliary'] = p
    return rollout


class RescueCFMLoss(ResearchLoss):
    def __init__(self, cfg):
        validate_rescue_recipe(cfg)
        if cfg['research']['arm'] != 'PPO_RESCUE_CFM':
            raise ValueError('Positive teacher loss is exclusively PPO_RESCUE_CFM')
        super().__init__(cfg)

    def _policy_loss(self, model, rollout):
        total, metrics = super()._policy_loss(model, rollout)
        p = rollout.get('rescue_auxiliary', {})
        if p.get('protocol') != AUX_PROTOCOL or p.get('skip_reason') == 'replay_control':
            raise ValueError('Rescue loss requires once-finalized teacher material')
        zero = total.detach().new_zeros(())
        metrics.update(rescue_aux_enabled=zero+float(p['enabled']),
            rescue_aux_skip_saturated=zero+float(p['skip_reason'] == 'fresh_safety_comfort_EP_saturated'),
            rescue_aux_skip_no_teacher=zero+float(p['skip_reason'] == 'no_teacher'),
            rescue_aux_loss=zero, rescue_aux_weighted_loss=zero, rescue_aux_coefficient=zero,
            rescue_aux_pool_low=zero, rescue_aux_pool_high=zero, rescue_aux_online_upgrade=zero)
        if not p['enabled']:
            return total, metrics
        state, sigma, target = forward_material(p['clean'], p['sigma'], p['noise'])
        # Source PG/ref context has ended. Rebuild differentiable current Memory
        # here each inner; no frozen reference or old differentiable graph used.
        with model._context_call(rollout['cond_bg']) as condition:
            velocity = model.action_velocity(state, sigma*1000, condition)
        raw = positive_cfm_objective(velocity, target)
        weighted = p['coefficient']*raw
        tm = p['target']['metrics']; best = max(r['metrics']['score'] for r in rollout['records'])
        metrics.update(rescue_aux_loss=raw.detach(), rescue_aux_weighted_loss=weighted.detach(),
            rescue_aux_coefficient=zero+p['coefficient'],
            rescue_aux_pool_low=zero+float(p['teacher']['bank_kind'] == 'low'),
            rescue_aux_pool_high=zero+float(p['teacher']['bank_kind'] == 'high'),
            rescue_aux_sigma_min=sigma.min().detach(), rescue_aux_sigma_max=sigma.max().detach(),
            rescue_aux_teacher_score=zero+tm['score'], rescue_aux_teacher_EP=zero+tm['ego_progress'],
            rescue_aux_online_upgrade=zero+float(p['target']['origin'] == 'online_upgrade'),
            rescue_aux_static_teacher_score=zero+p['teacher']['metrics']['score'],
            rescue_aux_teacher_minus_fresh_best_score=zero+tm['score']-best)
        return total+weighted, metrics


def validate_rescue_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, bank, rank=None, legacy=False):
    validate_rescue_recipe(identity['recipe'])
    if identity['world'] != 8 or identity['recipe']['research']['rescue_manifest_sha256'] != bank.sha256:
        raise ValueError('Rescue requires registered eight-rank teacher bank')
    summary = validate_structured_boundary(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    rank = current_rank() if rank is None else rank
    summary.update(rescue_schema=AUX_SCHEMA, rescue_manifest_sha256=bank.sha256, rescue_rank=rank,
                   rescue_arm=identity['recipe']['research']['arm'])
    if not summary['pending']:
        return summary
    if rollout.get('protocol_sha256') != bank.manifest['protocol_sha256']:
        raise ValueError('Fresh canonical scoring protocol differs from verified teachers')
    p = rollout.get('rescue_auxiliary', {})
    enabled = identity['recipe']['research']['arm'] == 'PPO_RESCUE_CFM'
    skip = 'replay_control' if not enabled else 'fresh_safety_comfort_EP_saturated' if fresh_saturated(rollout) else 'no_teacher' if rollout['token'] not in bank.by_token else None
    if (p.get('protocol') != AUX_PROTOCOL or p.get('bank_manifest_sha256') != bank.sha256
            or p.get('token') != rollout['token'] or p.get('rank') != rank
            or p.get('behavior_update') != rollout['behavior_version']['update']
            or p.get('enabled') != (skip is None) or p.get('skip_reason') != skip
            or p.get('material_sha256') != payload_digest(p)):
        raise ValueError('Saved rescue protocol/digest/skip identity changed')
    base = {'protocol', 'bank_manifest_sha256', 'token', 'behavior_update', 'rank', 'enabled', 'skip_reason', 'material_sha256'}
    if skip is not None:
        if set(p) != base:
            raise ValueError('Skipped rescue group must not retain teacher/aux tensors')
    else:
        extras = {'teacher', 'target', 'selection_seed', 'clean', 'sigma', 'noise', 'condition_identity', 'canonical_protocol_sha256', 'coefficient'}
        if set(p) != base | extras:
            raise ValueError('Missing or unexpected rescue auxiliary material')
        row = bank.by_token[rollout['token']]
        seed, sigma, noise = stateless_material(p['token'], p['behavior_update'], rank)
        clean, target = select_verified_target(rollout, bank)
        if (p['target'] != target or p['teacher'] != row or row['log'] != rollout['log'] or p['selection_seed'] != seed
                or p['condition_identity'] != rollout['cond_bg']['identity']
                or p['canonical_protocol_sha256'] != rollout['protocol_sha256']
                or p['coefficient'] != (1. if row['bank_kind'] == 'low' else .2)):
            raise ValueError('Saved rescue teacher/stateless/source identity changed')
        for key, want in (('clean', clean), ('sigma', sigma), ('noise', noise)):
            value = p[key]
            if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float32
                    or value.shape != want.shape or not torch.equal(value.cpu(), want)):
                raise ValueError('Saved rescue endpoint/stateless randomness changed: '+key)
    summary.update(rescue_material_sha256=p['material_sha256'], rescue_enabled=p['enabled'], rescue_skip=skip)
    return summary


def validate_rescue_ranks(summaries):
    if (not summaries or len(summaries) != 8
            or [s.get('rescue_rank') for s in summaries] != list(range(8))
            or any(s.get('rescue_schema') != AUX_SCHEMA for s in summaries)
            or len({s.get('rescue_manifest_sha256') for s in summaries}) != 1
            or len({s.get('rescue_arm') for s in summaries}) != 1):
        raise ValueError('Rescue rank/bank/arm protocol differs')
    # Retain the original SafetyEP global A_raw math, masks and source counters;
    # offline teacher targets are never arguments to this rank validator.
    return validate_safety_efficiency_ranks(summaries)


# Isolated structural adapter: same SafetyEP component/recovery mathematics;
# only recipe validation uses the new exact Rescue roles, never old guards.
def validate_structured_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    """Keep source structure/optimizer semantics; replace only A semantics."""
    validate_rescue_recipe(identity['recipe'])
    if identity['world'] != 8:
        raise ValueError('Safety-efficiency pilot requires eight ranks and 64 fresh candidates')
    summary = source_boundary(state, rollout, identity, optimizer, scheduler, legacy=legacy)
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

