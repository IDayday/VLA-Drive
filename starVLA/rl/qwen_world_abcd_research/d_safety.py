"""Unregistered D credit-only ablation: JaT Gaussian objective + ABC SafetyEP.

Only the advantage definition changes. This is a falsifiable CPU prototype,
not an improvement claim or a registration/authorization to run a GPU pilot.
The original D default and every old entry remain unchanged.
"""
import copy
import math

import torch

from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
from . import abc
from . import d_objective as gaussian

SCHEMA = 'abcd_D_action_JaT_SafetyEP_joint_gaussian_v1'
CONFIG = dict(gaussian.CONFIG, schema=SCHEMA, advantage=abc.STRUCTURED_SCHEMA)
PROTOCOL = copy.deepcopy(gaussian.PROTOCOL)
PROTOCOL.update(CONFIG)
PROTOCOL.update(
    credit=copy.deepcopy(abc.STRUCTURED_PROTOCOL),
    lifetime='oldmean/std/recorded_next/innovation and original ABC SafetyEP A fixed for all4inners; currentD C/P rebuilt each current call',
    ablation='only canonical-centered advantage replaced by original ABC SafetyEP credit; no sampling/loss/scope/LR/deployment change',
    credit_record_compatibility='Original ABC descriptor retained for exact credit-material validation only; its historical source-PPO/beta50 loss text is not executed. Actual objective is D jointGaussian PPO with nativeGaussian IL KL.001 and no physical mean anchor.',
)


def validate_recipe(cfg):
    research = cfg.get('research', {})
    if (research.get('d_objective') != CONFIG
            or research.get('structured_advantage') != abc.STRUCTURED_PROTOCOL
            or 'abc_objective' in research):
        raise ValueError('D SafetyEP requires its independent exact objective/structured-credit schema')
    # Check every original D restriction using the known sole-credit delta;
    # this copy is only a recipe assertion, never a saved/runtime protocol.
    baseline = copy.deepcopy(cfg)
    baseline['research']['d_objective'] = copy.deepcopy(gaussian.CONFIG)
    gaussian.validate_recipe(baseline)
    return cfg


@torch.no_grad()
def finalize_advantages(rollout):
    if 'd_gaussian' in rollout:
        raise ValueError('Finalize fresh D SafetyEP Gaussian material exactly once')
    abc.finalize_safety_efficiency_advantages(rollout)
    mean, std, nxt, innovation = gaussian.old_terms(rollout)
    payload = dict(protocol=copy.deepcopy(PROTOCOL), token=rollout['token'], log=rollout['log'],
        behavior_version=copy.deepcopy(rollout['behavior_version']), condition=copy.deepcopy(rollout['cond_bg']),
        old_mean=mean.clone(), std=std.clone(), recorded_next=nxt.clone(), innovation=innovation.clone(),
        source_sha256=tensor_digest(dict(mean=mean, std=std, next=nxt, stoch_idx=rollout['stoch_idx'])))
    payload['sha256'] = gaussian.material_digest(payload)
    rollout['d_gaussian'] = payload
    return rollout


def validate_material(rollout):
    p = rollout.get('d_gaussian', {})
    if (set(p) != {'protocol','token','log','behavior_version','condition','old_mean','std',
                   'recorded_next','innovation','source_sha256','sha256'}
            or p.get('protocol') != PROTOCOL or p.get('token') != rollout['token']
            or p.get('log') != rollout['log'] or p.get('behavior_version') != rollout['behavior_version']
            or p.get('condition') != rollout['cond_bg'] or p.get('sha256') != gaussian.material_digest(p)):
        raise ValueError('D SafetyEP Gaussian protocol/source/digest changed')
    mean, std, nxt, innovation = gaussian.old_terms(rollout)
    for key, expected in (('old_mean',mean), ('std',std), ('recorded_next',nxt), ('innovation',innovation)):
        value = p[key]
        if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float32
                or not torch.equal(value.cpu(), expected.cpu())):
            raise ValueError('D SafetyEP saved Gaussian source differs: '+key)
    expected_source = tensor_digest(dict(mean=mean, std=std, next=nxt, stoch_idx=rollout['stoch_idx']))
    if p['source_sha256'] != expected_source:
        raise ValueError('D SafetyEP Gaussian source ancestry changed')
    return p


class DSafetyJaTLoss(SourceLoss):
    def __init__(self, cfg):
        validate_recipe(cfg)
        super().__init__(cfg)

    def _policy_loss(self, model, rollout):
        if (rollout.get('structured_advantage', {}).get('protocol') != abc.STRUCTURED_PROTOCOL
                or not math.isinf(model.grpo_randn_clip)):
            raise ValueError('D SafetyEP requires finalized original credit and unclipped Gaussian innovations')
        p = rollout['d_gaussian']
        if p['protocol'] != PROTOCOL:
            raise ValueError('Wrong D SafetyEP Gaussian loss protocol')
        args = (rollout['cond_bg'], rollout['chain'], rollout['timesteps'], rollout['deltas'])
        _, terms = model.action_chain_logprobs(*args, stoch_idx=rollout['stoch_idx'], return_terms=True)
        with torch.no_grad():
            _, ref = model.action_chain_logprobs(*args, stoch_idx=rollout['stoch_idx'], use_ref=True, return_terms=True)
        if not torch.equal(terms['std'], p['std']) or not torch.equal(ref['std'], p['std']):
            raise ValueError('D SafetyEP policy-independent Gaussian covariance changed')
        logr, kl = gaussian.gaussian_logratio_kl(terms['mean'], p['old_mean'], ref['mean'], p['std'], p['recorded_next'])
        fresh = getattr(model, '_d_expect_fresh_material', None) == p['sha256']
        if fresh:
            error = logr.detach().expm1().abs().max()
            if not bool(torch.isfinite(error)) or float(error) > 1e-4:
                raise FloatingPointError('D SafetyEP fresh joint Gaussian ratio differs from1')
            object.__setattr__(model, '_d_expect_fresh_material', None)
        total, metrics = gaussian.clipped_objective(logr, kl, rollout['advantages'], rollout['adv_mask'])
        metrics['d_fresh_joint_ratio_checked'] = torch.as_tensor(float(fresh), device=total.device)
        return total, metrics


def validate_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    summary = abc.validate_safety_efficiency_boundary(state, rollout, identity, optimizer, scheduler,
        legacy=legacy, _recipe_validator=validate_recipe)
    if identity.get('architecture_id') != 'D' or identity.get('d_objective_protocol') != PROTOCOL:
        raise ValueError('D SafetyEP native objective identity mismatch')
    summary.update(d_objective_schema=SCHEMA, d_objective_protocol=copy.deepcopy(PROTOCOL))
    if summary['pending']:
        p = validate_material(rollout)
        summary.update(d_material_sha256=p['sha256'], d_behavior_update=p['behavior_version']['update'])
    return summary


def validate_ranks(summaries):
    abc.validate_safety_efficiency_ranks(summaries)
    for summary in summaries:
        if (summary.get('architecture_id') != 'D' or summary.get('d_objective_schema') != SCHEMA
                or summary.get('d_objective_protocol') != PROTOCOL):
            raise ValueError('D SafetyEP rank objective/protocol mismatch')
        if summary['pending'] and (len(summary.get('d_material_sha256', '')) != 64
                or summary.get('d_behavior_update') != summary['updates']-summary['next_inner']):
            raise ValueError('D SafetyEP Gaussian material/rank behavior changed')
    return summaries


class DSafetyRuntimeMixin:
    validate_ranks = staticmethod(validate_ranks)

    def load_recipe(self, args):
        return validate_recipe(super().load_recipe(args))

    def make_actor(self, cfg, args, device):
        validate_recipe(cfg)
        actor, oldcfg, sft = super().make_actor(cfg, args, device)
        actor._loss = DSafetyJaTLoss(cfg)
        return actor, oldcfg, sft

    def finalize_advantages(self, rollout):
        finalize_advantages(rollout)
        object.__setattr__(self.actor, '_d_expect_fresh_material', rollout['d_gaussian']['sha256'])
        return rollout

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value.update(d_objective_protocol=copy.deepcopy(PROTOCOL),
            policy_ratio='D joint24-dimensional Gaussian transition ratio, unclipped innovation, JaT algebra/sourceW3 kernel adaptation',
            reference='own selectedD IL native GaussianKL.001; no beta50 physicalanchor; no CoT RL')
        return value

    def validate_boundary(self, state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
        summary = validate_boundary(state, rollout, identity, optimizer, scheduler, legacy=legacy)
        if summary['pending']:
            expected = rollout['d_gaussian']['sha256'] if state['next_inner'] == 0 else None
            object.__setattr__(self.actor, '_d_expect_fresh_material', expected)
        return summary
