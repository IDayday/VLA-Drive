"""Four-framework adaptation of the audited SimWAM recovery-state contract.

Counter arithmetic, committed phases, Adam/scheduler checks, rank consensus and
global-std advantage validation retain the existing implementation's semantics.
Only the new family/schema, native8x3 condition ancestry and dataclass-aware
diagnostic hashing are added; historical validators are not modified.
"""
from dataclasses import fields, is_dataclass
import hashlib

import numpy as np
import torch

from starVLA.rl.simwam_recipe import recovery_state as source
from .contracts import fingerprint

RL_SCHEMA = 'qwen_world_compare_RL_v1'
OBSERVATION_SCHEMA = 'qwen_world_compare_current_descriptor_v1'
PHASES = source.PHASES
boundary_phase = source.boundary_phase


def _digest_payload(value):
    if is_dataclass(value) and not isinstance(value, type):
        return dict(__dataclass__=type(value).__module__+'.'+type(value).__qualname__,
                    fields={f.name: _digest_payload(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _digest_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_digest_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_digest_payload(v) for v in value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return dict(__numpy__=str(array.dtype), shape=list(array.shape),
                    sha256=hashlib.sha256(array.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_digest(value):
    return source.tensor_digest(_digest_payload(value))


def rollout_digest(rollout):
    # The descriptor includes frozen input/provider ancestry, not mutable current
    # prefix/KVs. Old probabilities and all saved transition coordinates remain
    # fixed across the four inner optimizer updates.
    keys = ('chain', 'old_logp', 'old_probability', 'timesteps', 'deltas', 'stoch_idx',
            'rewards', 'advantages', 'adv_mask', 'behavior_version', 'cond_bg')
    return {key: tensor_digest(rollout[key]) for key in keys}


def validate_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    if legacy or identity.get('schema') != RL_SCHEMA or identity.get('architecture_id') not in ('A','B','C','D'):
        raise ValueError('Not a registered four-framework RL recovery identity')
    body = dict(identity); expected = body.pop('sha256')
    if fingerprint(body) != expected:
        raise ValueError('Four-framework scientific identity checksum changed')
    cfg = identity['recipe']; world = identity['world']
    if (cfg.get('action_shape') != [8,3] or cfg.get('prediction_type') != 'x0'
            or type(world) is not int or world < 1):
        raise ValueError('Changed four-framework native clean-x/world contract')
    u,c,i = (state[k] for k in ('updates','collections','next_inner'))
    if not (0 <= u <= cfg['max_steps'] and 0 <= i < 4 and c >= 0):
        raise ValueError('Invalid update/collection/inner counters')
    phase = state.get('phase')
    if phase not in PHASES or phase == 'COLLECTING':
        raise ValueError('Not a committed recovery phase: '+str(phase))
    pending = phase in ('ROLLOUT_COMMITTED','INNER_COMMITTED')
    expected_c = u//4+int(pending)
    if c != expected_c or i != u%4:
        raise ValueError('Collection/optimizer boundary mismatch')
    if phase == 'ROLLOUT_COMMITTED' and i != 0:
        raise ValueError('Uncommitted inner')
    if phase == 'INNER_COMMITTED' and i == 0:
        raise ValueError('Missing inner commit')
    if phase == 'COMPLETE' and u != cfg['max_steps']:
        raise ValueError('Premature COMPLETE')
    if any(state[k] != c*world for k in ('scene_cursor','raw_scene_exposures')):
        raise ValueError('Scene cursor/exposure inconsistent with committed collections')
    if state['new_candidates'] != c*world*8:
        raise ValueError('Candidate budget inconsistent with committed collections')
    if pending != (rollout is not None):
        raise ValueError('Phase/rollout presence mismatch')
    summary = dict(phase=phase,updates=u,collections=c,scene_cursor=state['scene_cursor'],
        next_inner=i,run=identity['sha256'],pending=pending,architecture_id=identity['architecture_id'])
    if pending:
        required = ('cond_bg','chain','timesteps','deltas','stoch_idx','old_logp','rewards',
                    'old_probability','records','token','log','behavior_version','advantages','adv_mask',
                    'global_std','raw_group_std','advantage_statistics')
        missing = set(required)-set(rollout)
        if missing:
            raise ValueError('Unfinalized rollout: '+repr(sorted(missing)))
        descriptor = rollout['cond_bg']
        if (not isinstance(descriptor,dict) or set(descriptor) != {'schema','token','identity','repeats'}
                or descriptor['schema'] != OBSERVATION_SCHEMA or descriptor['token'] != rollout['token']
                or descriptor['repeats'] != 8
                or descriptor['identity'] != identity['projected_condition_identity']):
            raise ValueError('Four-framework frozen current descriptor ancestry mismatch; prefix graphs are not recoverable data')
        shapes = dict(chain=(8,11,8,3),old_logp=(8,3),rewards=(1,8),advantages=(8,),adv_mask=(8,))
        for key, shape in shapes.items():
            value = rollout[key]
            if (not torch.is_tensor(value) or tuple(value.shape) != shape
                    or not torch.isfinite(value).all() or value.requires_grad):
                raise ValueError('Invalid detached rollout field: '+key)
        indices = rollout['stoch_idx']
        if torch.is_tensor(indices):
            indices = indices.tolist()
        if list(indices) != [7,8,9]:
            raise ValueError('Changed registered stochastic window')
        behavior = rollout['behavior_version']
        if behavior['run'] != identity['sha256'] or behavior['update'] != u-i:
            raise ValueError('Behavior is not the collection-start policy')
        old = rollout['old_probability'].get('current',rollout['old_probability'])
        if not torch.allclose(old['source_clamped_logprob'].mean((-1,-2)),rollout['old_logp'],atol=5e-7,rtol=0):
            raise ValueError('Old probability/chain audit mismatch')
        summary.update(digests=rollout_digest(rollout),behavior=behavior,
            rewards=rollout['rewards'].double().cpu().tolist(),
            advantages=rollout['advantages'].double().cpu().tolist(),
            mask=rollout['adv_mask'].cpu().tolist(),global_std=float(rollout['global_std']))
    if optimizer is not None:
        opt = optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer
        steps = [float(v['step']) for v in opt['state'].values()]
        if (u and not steps) or any(s != u for s in steps):
            raise ValueError('Adam step does not match committed update')
        if u and len(steps) != sum(len(g['params']) for g in opt['param_groups']):
            raise ValueError('Missing optimizer parameter state')
    if scheduler is not None:
        sched = scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler
        if sched['last_epoch'] != u:
            raise ValueError('Scheduler progress mismatch')
    return summary


def validate_ranks(summaries):
    if not summaries or any(s['architecture_id'] != summaries[0]['architecture_id'] for s in summaries):
        raise ValueError('Ranks disagree on four-framework architecture')
    return source.validate_ranks(summaries)
