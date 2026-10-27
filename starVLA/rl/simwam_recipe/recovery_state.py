"""Pure validation of committed training boundaries; no policy mathematics.

COLLECTING is deliberately not recoverable. A failed collection is replayed
from the preceding sealed checkpoint, including its RNG and optimizer state.
"""
import hashlib
import json
import torch
from .contracts import registered_stochastic_steps

PHASES = {'READY', 'COLLECTING', 'ROLLOUT_COMMITTED', 'INNER_COMMITTED', 'COMPLETE'}


def tensor_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(str((str(y.dtype), tuple(y.shape))).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x: visit(y)
        else: h.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def rollout_digest(rollout):
    keys = ['chain', 'old_logp', 'advantages', 'adv_mask', 'behavior_version']
    if rollout.get('cond_bg', {}).get('schema') in (
            'native_frozen_observation_v1','world_postcot_frozen_observation_v1'):
        keys.append('cond_bg')
    return {k: tensor_digest(rollout[k]) for k in keys}


def boundary_phase(state, max_steps):
    """Called only after a completed collection/update, never on exceptions."""
    if state['updates'] == max_steps: return 'COMPLETE'
    if state['next_inner']: return 'INNER_COMMITTED'
    if state['collections'] * 4 == state['updates']: return 'READY'
    return 'ROLLOUT_COMMITTED'


def validate_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    world = identity['world']; cfg = identity['recipe']
    u, c, i = (state[k] for k in ('updates', 'collections', 'next_inner'))
    if not (0 <= u <= cfg['max_steps'] and 0 <= i < 4 and c >= 0):
        raise ValueError('Invalid update/collection/inner counters')
    phase = state.get('phase')
    # Legacy qualification is an explicit read-only audit, not a permissive load.
    if phase is None and legacy: phase = boundary_phase(state, cfg['max_steps'])
    if phase not in PHASES or phase == 'COLLECTING':
        raise ValueError('Not a committed recovery phase: ' + str(phase))
    pending = phase in ('ROLLOUT_COMMITTED', 'INNER_COMMITTED')
    expected_c = u // 4 + int(pending)
    if c != expected_c or i != u % 4:
        raise ValueError('Collection/optimizer boundary mismatch')
    if phase == 'ROLLOUT_COMMITTED' and i != 0: raise ValueError('Uncommitted inner')
    if phase == 'INNER_COMMITTED' and i == 0: raise ValueError('Missing inner commit')
    if phase == 'COMPLETE' and u != cfg['max_steps']: raise ValueError('Premature COMPLETE')
    if any(state[k] != c * world for k in ('scene_cursor', 'raw_scene_exposures')):
        raise ValueError('Scene cursor/exposure inconsistent with committed collections')
    if state['new_candidates'] != c * world * 8:
        raise ValueError('Candidate budget inconsistent with committed collections')
    if pending != (rollout is not None): raise ValueError('Phase/rollout presence mismatch')
    summary = dict(phase=phase, updates=u, collections=c, scene_cursor=state['scene_cursor'],
                   next_inner=i, run=identity['sha256'], pending=pending)
    if pending:
        required = ('cond_bg', 'chain', 'timesteps', 'deltas', 'stoch_idx', 'old_logp',
                    'rewards', 'old_probability', 'records', 'token', 'log', 'behavior_version',
                    'advantages', 'adv_mask', 'global_std', 'raw_group_std', 'advantage_statistics')
        missing = set(required) - set(rollout)
        if missing: raise ValueError('Unfinalized rollout: ' + repr(sorted(missing)))
        action_dim = 4
        native = identity.get('schema') == 'native_qwen_sourceflow_RL_v1'
        world_cleanx = identity.get('schema') == 'world_postcot_cleanx_canonical_RL_v1'
        if native or world_cleanx:
            if cfg['action_shape'] != [8, 3]:
                raise ValueError('Changed native 3D action contract')
            if world_cleanx and (cfg.get('model_family')!='QwenWorldPostCoT'
                                or cfg.get('prediction_type')!='x0'):
                raise ValueError('Changed World clean-x recovery contract')
            descriptor = rollout['cond_bg']
            condition_schema=('world_postcot_frozen_observation_v1' if world_cleanx
                              else 'native_frozen_observation_v1')
            if (set(descriptor) != {'schema', 'token', 'identity', 'repeats'} or
                    descriptor['schema'] != condition_schema or
                    descriptor['token'] != rollout['token'] or descriptor['repeats'] != 8 or
                    descriptor['identity'] != identity['projected_condition_identity']):
                raise ValueError('Native frozen condition ancestry mismatch')
            action_dim = 3
        shapes = {'chain': (8, 11, 8, action_dim), 'old_logp': (8, 3), 'rewards': (1, 8),
                  'advantages': (8,), 'adv_mask': (8,)}
        for k, shape in shapes.items():
            t = rollout[k]
            if tuple(t.shape) != shape or not torch.isfinite(t).all() or t.requires_grad:
                raise ValueError('Invalid detached rollout field: ' + k)
        indices = rollout['stoch_idx']
        if torch.is_tensor(indices): indices = indices.tolist()
        if list(indices) != registered_stochastic_steps(cfg['variant']):
            raise ValueError('Changed registered stochastic window')
        v = rollout['behavior_version']
        if v['run'] != identity['sha256'] or v['update'] != u - i:
            raise ValueError('Behavior is not the collection-start policy')
        old = rollout['old_probability']['current'] if 'current' in rollout['old_probability'] else rollout['old_probability']
        # Restore validates CPU tensors. CPU and CUDA FP32 reduction trees differ
        # by up to 2.384e-7 on the actual eight-rank chain. Payload hashes still
        # enforce exact stored bits; this check only validates their semantics.
        if not torch.allclose(old['source_clamped_logprob'].mean((-1, -2)), rollout['old_logp'],atol=5e-7,rtol=0):
            raise ValueError('Old probability/chain audit mismatch')
        summary.update(digests=rollout_digest(rollout), behavior=v,
                       rewards=rollout['rewards'].double().cpu().tolist(),
                       advantages=rollout['advantages'].double().cpu().tolist(),
                       mask=rollout['adv_mask'].cpu().tolist(), global_std=float(rollout['global_std']))
    if optimizer is not None:
        opt = optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer
        steps = [float(v['step']) for v in opt['state'].values()]
        if (u and not steps) or any(s != u for s in steps):
            raise ValueError('Adam step does not match committed update')
        if u and len(steps) != sum(len(g['params']) for g in opt['param_groups']):
            raise ValueError('Missing optimizer parameter state')
    if scheduler is not None:
        sched = scheduler.state_dict() if hasattr(scheduler, 'state_dict') else scheduler
        if sched['last_epoch'] != u: raise ValueError('Scheduler progress mismatch')
    return summary


def validate_ranks(summaries):
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run', 'pending')
    for item in summaries:
        if {k: item[k] for k in common} != {k: summaries[0][k] for k in common}:
            raise ValueError('Ranks disagree on recovery boundary')
    if summaries[0]['pending']:
        if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):
            raise ValueError('Ranks disagree on collection-start behavior')
        rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
        std = rewards.std(unbiased=False)
        offsets = rewards - rewards[:, :1]
        expected = ((offsets - offsets.mean(1, keepdim=True)) / (std + 1e-8)).float()
        mask = (offsets.std(1, unbiased=False) > 1e-8).float()
        for rank, s in enumerate(summaries):
            if abs(s['global_std'] - float(std)) > 1e-12 or not torch.allclose(
                    torch.tensor(s['advantages']), expected[rank], atol=1e-7, rtol=1e-6):
                raise ValueError('Advantage was not globally finalized')
            if s['mask'] != [float(mask[rank])] * 8: raise ValueError('PG mask mismatch')
    return summaries
