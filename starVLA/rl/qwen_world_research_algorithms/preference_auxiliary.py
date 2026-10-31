"""Offline TRAIN preference auxiliary, preserving canonical on-policy PPO.

Negative squared conditional-flow velocity error is a ranking SURROGATE, not
endpoint log likelihood or exact diffusion DPO. Historical samples never enter
the fresh PPO reward/advantage calculation. No head or parameter is added.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.recovery_state import (
    tensor_digest, validate_boundary as source_boundary, validate_ranks as source_ranks,
)
from .policy import ResearchLoss
from .safety_efficiency import canonical_components

AUX_SCHEMA = 'world_offline_cfm_preference_aux_v1'
BANK_SCHEMA = 'world_TRAIN_metric_preference_manifest_v1'
DISTRIBUTION = 'full80_pref20_static_v1'
AUX_CONFIG = dict(schema=AUX_SCHEMA, beta=.1, coefficient=.001, margin=0.,
                  sigma_min=.2, sigma_max=.8, stateless_seed=42)
AUX_PROTOCOL = dict(
    **AUX_CONFIG,
    score='negative squared CFM velocity error; surrogate, not exact endpoint likelihood',
    objective='softplus(-beta*((-E_current_w+E_ref_w)-(-E_current_l+E_ref_l)))',
    reference='permanently frozen IL70 action + scene/predictive memory via use_ref=True',
    pair_choice='uniform over valid immutable pairs for fresh token; stateless token/update/rank/seed42',
    randomness='four common winner/loser sigma and Gaussian noise draws per collection; reuse across four inner updates',
    batch='winner4 then loser4; one current G8 velocity per inner, one cached reference G8 per collection',
    source_PPO='canonical fresh rewards/global raw std advantages/mask/ratio/beta50 IL anchor unchanged',
    skip='fresh eight NC=DAC=TTC=EP=1, or fresh token absent from bank',
    information='historical scored TRAIN trajectories as auxiliary targets only; no new actor input, head, or trainable parameter',
)
METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'time_to_collision_within_bound', 'ego_progress', 'comfort', 'driving_direction_compliance')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_digest(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str((str(x.dtype), x.shape)).encode()+x.tobytes()).hexdigest()


def validate_preference_recipe(cfg):
    r = cfg.get('research', {})
    if (r.get('policy_objective') != 'ppo' or r.get('data_policy') != 'preference_mixture'
            or r.get('preference_distribution') != DISTRIBUTION
            or r.get('preference_auxiliary') != AUX_CONFIG
            or not isinstance(r.get('preference_manifest'), str)
            or not isinstance(r.get('preference_manifest_sha256'), str)
            or len(r['preference_manifest_sha256']) != 64
            or any(r.get(k) is not None for k in ('advantage_normalization', 'structured_advantage',
                'reward_transform', 'anchor_ablation', 'memory_lr_ablation'))):
        raise ValueError('Register the independent canonical PPO/pref-mixture CFM auxiliary arm')
    if (cfg['learning_rate'] != 5e-5 or cfg['memory_learning_rate'] != 5e-6
            or cfg['grpo']['train']['kl_beta'] != 50 or cfg['variant'] != 'R_CANONICAL'
            or cfg['max_steps'] != 50000 or cfg.get('seed') != 42):
        raise ValueError('Preference pilot retains source LR/beta50/seed42/scheduler50000')
    return cfg


class PreferenceBank:
    """Hash-pinned compact real-scoring bank; no original bank/weight rescans."""
    def __init__(self, manifest_path, expected_sha256, eligible_tokens):
        path = Path(manifest_path)
        if file_sha(path) != expected_sha256:
            raise ValueError('Preference manifest file SHA changed')
        self.sha256 = expected_sha256
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        body = dict(m); seal = body.pop('sha256', None)
        if (seal != fingerprint(body) or m.get('schema') != BANK_SCHEMA
                or m.get('status') != 'READY_OFFLINE_RESEARCH_ONLY'):
            raise ValueError('Preference bank schema/seal/status changed')
        for name in ('candidates.npz', 'groups.jsonl', 'pairs.jsonl'):
            spec = m['artifacts'][name]
            if Path(spec['path']).resolve() != (path.parent/name).resolve() or file_sha(path.parent/name) != spec['sha256']:
                raise ValueError('Preference bank artifact SHA/path changed: '+name)
        with np.load(path.parent/'candidates.npz', allow_pickle=False) as values:
            self.arrays = {k:values[k] for k in values.files}
        self.groups = [json.loads(line) for line in (path.parent/'groups.jsonl').read_text().splitlines()]
        pairs = [json.loads(line) for line in (path.parent/'pairs.jsonl').read_text().splitlines()]
        a = self.arrays; n = len(self.groups)
        if (n != m['groups'] or m['candidates'] != n*8 or len(pairs) != m['pairs']
                or a['metric_keys'].tolist() != list(METRICS)
                or a['metrics'].shape != (n, 8, 7) or a['metrics'].dtype != np.float64
                or a['clean_normalized'].shape != (n, 8, 8, 3) or a['clean_normalized'].dtype != np.float32
                or a['requested_trajectory'].shape != (n, 8, 8, 3)
                or a['raw_progress_meters'].shape != (n, 8)):
            raise ValueError('Preference bank dimensions/dtypes/counts changed')
        for k in ('metrics', 'clean_normalized', 'requested_trajectory', 'raw_progress_meters'):
            if not np.isfinite(a[k]).all():
                raise ValueError('Nonfinite preference material: '+k)
        x = a['metrics']
        if not np.all((x >= 0) & (x <= 1)):
            raise ValueError('Canonical preference components outside [0,1]')
        score = x[:, :, 1]*x[:, :, 2]*(5*x[:, :, 4]+5*x[:, :, 3]+2*x[:, :, 5])/12
        if not np.allclose(score, x[:, :, 0], atol=5e-15, rtol=0):
            raise ValueError('Preference canonical aggregation changed')
        decoded = ActionCodec.denorm_odo(torch.from_numpy(a['clean_normalized'])).numpy()
        if not np.allclose(decoded, a['requested_trajectory'], atol=1e-5, rtol=0):
            raise ValueError('Preference clean endpoints do not bind actually scored trajectories')
        eligible_tokens = set(eligible_tokens); self.by_token = {}; self.pairs = {}
        group_tokens = defaultdict_list(self.groups)
        for i, g in enumerate(self.groups):
            if (g['group'] != i or g['token'] not in eligible_tokens or g['token'] != str(a['tokens'][i])
                    or g['log'] != str(a['logs'][i]) or g['protocol_sha256'] != m['protocol_sha256']):
                raise ValueError('Preference group outside registered TRAIN/protocol')
        for ids in group_tokens.values():
            if len({self.groups[i]['cache_sha256'] for i in ids}) != 1:
                raise ValueError('Preference same-scene canonical environment differs')
        for p in pairs:
            w, l = p['winner'], p['loser']
            for ref in (w, l):
                g, i = ref['group'], ref['candidate']
                if type(g) is not int or type(i) is not int or not (0 <= g < n and 0 <= i < 8):
                    raise ValueError('Preference candidate index invalid')
                if self.groups[g]['token'] != p['token'] or self.groups[g]['log'] != p['log']:
                    raise ValueError('Preference pair crosses scenes')
                if ref['trajectory_sha256'] != array_digest(a['requested_trajectory'][g, i]):
                    raise ValueError('Preference scored trajectory hash changed')
            expected_id = fingerprint(dict(token=p['token'], winner=w['trajectory_sha256'],
                loser=l['trajectory_sha256'], reason=p['reason']))
            if (p['pair_id'] != expected_id or p['pair_id'] in self.pairs
                    or p.get('on_policy_GRPO_eligible') is not False or p.get('offline_only') is not True
                    or w['trajectory_sha256'] == l['trajectory_sha256']):
                raise ValueError('Preference pair identity/offline boundary changed')
            wm, lm = x[w['group'], w['candidate']], x[l['group'], l['candidate']]
            if p['winner_metrics'] != dict(zip(METRICS, wm.tolist())) or p['loser_metrics'] != dict(zip(METRICS, lm.tolist())):
                raise ValueError('Preference pair metric labels changed')
            wf, lf = bool((wm[1:4] == 1).all()), bool((lm[1:4] == 1).all())
            reason = p['reason']
            if reason == 'pass_over_fail':
                legal = wf and not lf
            elif reason == 'feasible_efficiency':
                legal = (wf and lf and wm[4]-lm[4] >= .05 and wm[5] >= lm[5]
                    and a['raw_progress_meters'][w['group'], w['candidate']] > a['raw_progress_meters'][l['group'], l['candidate']]+1e-6)
            elif reason == 'allfail_deficit':
                pool = [w['group']] if p['origin'] == 'same_collection' else group_tokens[p['token']]
                legal = (not np.any(np.all(x[pool, :, 1:4] == 1, axis=-1))
                    and np.all(wm[1:4] >= lm[1:4]) and (lm[1:4]-wm[1:4]).mean() <= -1/6+1e-12)
            else:
                legal = False
            if not legal:
                raise ValueError('Preference label has no strict canonical metric witness')
            self.pairs[p['pair_id']] = p
            self.by_token.setdefault(p['token'], []).append(p['pair_id'])
        if len(self.by_token) > m['unique_scenes'] or not self.pairs:
            raise ValueError('Preference scene coverage changed/empty bank')
        for ids in self.by_token.values():
            ids.sort()

    def clean_pair(self, pair):
        return torch.from_numpy(np.stack([self.arrays['clean_normalized'][pair[k]['group'], pair[k]['candidate']]
                                         for k in ('winner', 'loser')])).detach().clone()


def defaultdict_list(groups):
    result = {}
    for i, g in enumerate(groups):
        result.setdefault(g['token'], []).append(i)
    return result


def current_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get('RANK', 0))


def fresh_saturated(rollout):
    canonical_components(rollout['records'])
    return all(all(r['metrics'][k] == 1 for k in
        ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound', 'ego_progress')) for r in rollout['records'])


def stateless_material(bank, token, update, rank):
    seed = int.from_bytes(hashlib.sha256(json.dumps(dict(token=token, behavior_update=update,
        rank=rank, seed=42), sort_keys=True, separators=(',', ':')).encode()).digest()[:8], 'big') & ((1 << 63)-1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = bank.by_token[token]
    pair = bank.pairs[ids[int(torch.randint(len(ids), (1,), generator=generator))]]
    sigma = .2+.6*torch.rand(4, generator=generator, dtype=torch.float32)
    noise = torch.randn(4, 8, 3, generator=generator, dtype=torch.float32)
    return pair, seed, sigma.detach(), noise.detach()


def forward_material(clean, sigma, noise):
    clean8 = torch.cat((clean[0:1].expand(4, -1, -1), clean[1:2].expand(4, -1, -1)), 0)
    sigma8 = torch.cat((sigma, sigma)); noise8 = torch.cat((noise, noise), 0)
    state = (1-sigma8[:, None, None])*clean8+sigma8[:, None, None]*noise8
    return state, sigma8, noise8-clean8


def velocity_energies(velocity, target):
    if velocity.shape != (8, 8, 3) or target.shape != velocity.shape:
        raise ValueError('Preference CFM requires winner4 + loser4 native G8 velocities')
    return (velocity.float()-target.detach().float()).square().mean((1, 2))


def preference_objective(velocity, target, reference_energy):
    energies = velocity_energies(velocity, target)
    reference = reference_energy.detach().to(energies)
    logits = .1*((reference[:4]-energies[:4])-(reference[4:]-energies[4:]))
    loss = F.softplus(-logits).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite offline CFM preference auxiliary')
    return loss, dict(pref_aux_current_w_energy=energies[:4].mean().detach(),
        pref_aux_current_l_energy=energies[4:].mean().detach(), pref_aux_logit_mean=logits.mean().detach())


def payload_digest(payload):
    return tensor_digest({k:v for k,v in payload.items() if k != 'material_sha256'})


@torch.no_grad()
def attach_preference_material(actor, rollout, bank, *, rank=None):
    if 'preference_auxiliary' in rollout:
        raise ValueError('Attach immutable preference material exactly once per collection')
    if rollout.get('protocol_sha256') != bank.manifest['protocol_sha256']:
        raise ValueError('Fresh canonical scoring protocol differs from historical preference bank')
    rank = current_rank() if rank is None else rank
    saturated = fresh_saturated(rollout)
    reason = 'fresh_safety_EP_saturated' if saturated else 'no_bank_pair' if rollout['token'] not in bank.by_token else None
    payload = dict(protocol=copy.deepcopy(AUX_PROTOCOL), bank_manifest_sha256=bank.sha256,
        token=rollout['token'], behavior_update=rollout['behavior_version']['update'], rank=rank,
        enabled=reason is None, skip_reason=reason)
    if reason is None:
        pair, seed, sigma, noise = stateless_material(bank, payload['token'], payload['behavior_update'], rank)
        clean = bank.clean_pair(pair).to(actor.device)
        sigma, noise = sigma.to(actor.device), noise.to(actor.device)
        state, sigma8, target = forward_material(clean, sigma, noise)
        with actor._context_call(rollout['cond_bg']) as condition:
            ref = actor.action_velocity(state, sigma8*1000, condition, use_ref=True).detach().clone()
        if not bool(torch.isfinite(ref).all()):
            raise FloatingPointError('Nonfinite permanent IL70 preference reference velocity')
        payload.update(pair_id=pair['pair_id'], pair_reason=pair['reason'], origin=pair['origin'], selection_seed=seed,
            winner=copy.deepcopy(pair['winner']), loser=copy.deepcopy(pair['loser']), clean=clean.detach().clone(),
            clean_sha256=tensor_digest(clean), sigma=sigma.detach().clone(), noise=noise.detach().clone(),
            reference_velocity=ref, reference_energy=velocity_energies(ref, target).detach().clone(),
            reference_condition_identity=rollout['cond_bg']['identity'],
            canonical_protocol_sha256=rollout['protocol_sha256'])
    payload['material_sha256'] = payload_digest(payload)
    rollout['preference_auxiliary'] = payload
    return rollout


class PreferenceAuxiliaryLoss(ResearchLoss):
    def __init__(self, cfg):
        validate_preference_recipe(cfg)
        super().__init__(cfg)

    def _policy_loss(self, model, rollout):
        total, metrics = super()._policy_loss(model, rollout)
        material = rollout.get('preference_auxiliary')
        if not material or material.get('protocol') != AUX_PROTOCOL:
            raise ValueError('Preference loss requires its once-finalized native material')
        enabled = bool(material['enabled'])
        metrics.update(pref_aux_enabled=total.detach().new_tensor(float(enabled)),
            pref_aux_skip_saturated=total.detach().new_tensor(float(material['skip_reason'] == 'fresh_safety_EP_saturated')),
            pref_aux_skip_no_bank=total.detach().new_tensor(float(material['skip_reason'] == 'no_bank_pair')))
        if not enabled:
            metrics['pref_aux_loss'] = total.detach().new_zeros(())
            metrics['pref_aux_weighted_loss'] = total.detach().new_zeros(())
            return total, metrics
        state, sigma, target = forward_material(material['clean'], material['sigma'], material['noise'])
        # The source PG scope has ended. Build this auxiliary's differentiable
        # memory/KV once; no previous inner-update graph is kept or reused.
        with model._context_call(rollout['cond_bg']) as condition:
            velocity = model.action_velocity(state, sigma*1000, condition)
        auxiliary, diagnostics = preference_objective(velocity, target, material['reference_energy'])
        metrics.update(diagnostics, pref_aux_loss=auxiliary.detach(),
            pref_aux_weighted_loss=(.001*auxiliary).detach(),
            pref_aux_ref_w_energy=material['reference_energy'][:4].mean().detach(),
            pref_aux_ref_l_energy=material['reference_energy'][4:].mean().detach(),
            pref_aux_sigma_min=material['sigma'].min().detach(), pref_aux_sigma_max=material['sigma'].max().detach())
        return total+.001*auxiliary, metrics


def validate_preference_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, bank, rank=None, legacy=False):
    validate_preference_recipe(identity['recipe'])
    if identity['recipe']['research']['preference_manifest_sha256'] != bank.sha256:
        raise ValueError('Preference runtime bank/recipe identity differs')
    summary = source_boundary(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    rank = current_rank() if rank is None else rank
    summary.update(preference_auxiliary_schema=AUX_SCHEMA, preference_manifest_sha256=bank.sha256, preference_rank=rank)
    if not summary['pending']:
        return summary
    if rollout.get('protocol_sha256') != bank.manifest['protocol_sha256']:
        raise ValueError('Fresh canonical scoring protocol differs from historical preference bank')
    p = rollout.get('preference_auxiliary', {})
    skip = 'fresh_safety_EP_saturated' if fresh_saturated(rollout) else 'no_bank_pair' if rollout['token'] not in bank.by_token else None
    if (p.get('protocol') != AUX_PROTOCOL or p.get('bank_manifest_sha256') != bank.sha256
            or p.get('token') != rollout['token'] or p.get('rank') != rank
            or p.get('behavior_update') != rollout['behavior_version']['update']
            or p.get('enabled') != (skip is None) or p.get('skip_reason') != skip
            or p.get('material_sha256') != payload_digest(p)):
        raise ValueError('Saved preference protocol/digest/skip identity changed')
    expected_rewards = canonical_components(rollout['records'])['canonical_rewards']
    if not torch.equal(rollout['rewards'].cpu(), expected_rewards):
        raise ValueError('Preference auxiliary must preserve raw canonical fresh rewards')
    base_keys = {'protocol', 'bank_manifest_sha256', 'token', 'behavior_update', 'rank', 'enabled', 'skip_reason', 'material_sha256'}
    if skip is not None:
        if set(p) != base_keys:
            raise ValueError('Skipped preference group must not retain auxiliary tensors')
    else:
        extra_keys = {'pair_id', 'pair_reason', 'origin', 'selection_seed', 'winner', 'loser', 'clean', 'clean_sha256',
            'sigma', 'noise', 'reference_velocity', 'reference_energy', 'reference_condition_identity', 'canonical_protocol_sha256'}
        if set(p) != base_keys | extra_keys:
            raise ValueError('Missing or unexpected preference auxiliary material')
        pair, seed, sigma, noise = stateless_material(bank, rollout['token'], p['behavior_update'], rank)
        clean = bank.clean_pair(pair)
        for key, expected in (('pair_id', pair['pair_id']), ('pair_reason', pair['reason']), ('origin', pair['origin']),
                ('selection_seed', seed), ('winner', pair['winner']), ('loser', pair['loser']),
                ('clean_sha256', tensor_digest(clean)), ('reference_condition_identity', rollout['cond_bg']['identity']),
                ('canonical_protocol_sha256', rollout['protocol_sha256'])):
            if p[key] != expected:
                raise ValueError('Saved preference pair/stateless identity changed: '+key)
        for key, expected in (('clean', clean), ('sigma', sigma), ('noise', noise)):
            value = p[key]
            if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float32
                    or value.shape != expected.shape or not torch.equal(value.cpu(), expected)):
                raise ValueError('Saved preference endpoint/randomness changed: '+key)
        for key, shape in (('reference_velocity', (8, 8, 3)), ('reference_energy', (8,))):
            value = p[key]
            if (not torch.is_tensor(value) or value.shape != shape or value.requires_grad
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid detached preference reference material: '+key)
        _, _, target = forward_material(clean, sigma, noise)
        expected = velocity_energies(p['reference_velocity'].cpu(), target)
        if not torch.allclose(p['reference_energy'].cpu(), expected, atol=1e-6, rtol=3e-6):
            raise ValueError('Preference reference velocity/energy audit differs')
    summary.update(preference_material_sha256=p['material_sha256'], preference_enabled=p['enabled'], preference_skip=skip)
    return summary


def validate_preference_ranks(summaries):
    if (not summaries or [s.get('preference_rank') for s in summaries] != list(range(len(summaries)))
            or any(s.get('preference_auxiliary_schema') != AUX_SCHEMA for s in summaries)
            or len({s.get('preference_manifest_sha256') for s in summaries}) != 1):
        raise ValueError('Preference rank/bank protocol differs')
    # Crucially retain the SOURCE RAW reward std/A/mask semantics unchanged.
    return source_ranks(summaries)
