"""Pure validation of committed training boundaries; no policy mathematics.

COLLECTING is deliberately not recoverable. A failed collection is replayed
from the preceding sealed checkpoint, including its RNG and optimizer state.
"""
import hashlib
import json
import torch
from starVLA.rl.simwam_recipe.contracts import registered_stochastic_steps_for_recipe
from .credit import canonical_components

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
            'native_frozen_observation_v1','world_postcot_frozen_observation_v1',
            'world_postcot_trainable_memory_observation_v1'):
        keys.append('cond_bg')
    return {k: tensor_digest(rollout[k]) for k in keys}


def boundary_phase(state, max_steps):
    """Called only after a completed collection/update, never on exceptions."""
    if state['updates'] == max_steps: return 'COMPLETE'
    if state['next_inner']: return 'INNER_COMMITTED'
    if state['collections'] * 4 == state['updates']: return 'READY'
    return 'ROLLOUT_COMMITTED'


def validate_structure(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    world = identity['world']; cfg = identity['recipe']
    g = cfg['grpo']['sample']['group_size']
    if world != 8 or type(g) is not int or g not in (8, 16):
        raise ValueError('Only eight ranks with registered G8/G16')
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
    if state['new_candidates'] != c * world * g:
        raise ValueError('Candidate budget inconsistent with committed collections')
    if pending != (rollout is not None): raise ValueError('Phase/rollout presence mismatch')
    summary = dict(phase=phase, updates=u, collections=c, scene_cursor=state['scene_cursor'],
                   next_inner=i, run=identity['sha256'], pending=pending)
    if pending:
        if rollout.get('group_size') != g:
            raise ValueError('Saved rollout candidate G changed')
        required = ('cond_bg', 'chain', 'timesteps', 'deltas', 'stoch_idx', 'old_logp',
                    'rewards', 'old_probability', 'records', 'token', 'log', 'behavior_version',
                    'advantages', 'adv_mask', 'global_std', 'raw_group_std', 'advantage_statistics')
        missing = set(required) - set(rollout)
        if missing: raise ValueError('Unfinalized rollout: ' + repr(sorted(missing)))
        action_dim = 4
        native = identity.get('schema') == 'native_qwen_sourceflow_RL_v1'
        world_cleanx = identity.get('schema') in ('world_postcot_cleanx_canonical_RL_v1','world_postcot_cleanx_scope_RL_v1')
        if native or world_cleanx:
            if cfg['action_shape'] != [8, 3]:
                raise ValueError('Changed native 3D action contract')
            if world_cleanx and (cfg.get('model_family')!='QwenWorldPostCoT'
                                or cfg.get('prediction_type')!='x0'):
                raise ValueError('Changed World clean-x recovery contract')
            descriptor = rollout['cond_bg']
            condition_schema=('world_postcot_frozen_observation_v1' if world_cleanx
                              else 'native_frozen_observation_v1')
            if identity.get('schema')=='world_postcot_cleanx_scope_RL_v1':
                scope=cfg.get('adaptation')
                if scope not in ('ATTN_LORA_MEMORY','TIME_MOD_LORA'):raise ValueError('Unregistered World scope')
                if scope=='ATTN_LORA_MEMORY':condition_schema='world_postcot_trainable_memory_observation_v1'
            if (set(descriptor) != {'schema', 'token', 'identity', 'repeats'} or
                    descriptor['schema'] != condition_schema or
                    descriptor['token'] != rollout['token'] or descriptor['repeats'] != g or
                    descriptor['identity'] != identity['projected_condition_identity']):
                raise ValueError('Native frozen condition ancestry mismatch')
            action_dim = 3
        registered_steps=registered_stochastic_steps_for_recipe(cfg)
        shapes = {'chain': (g, 11, 8, action_dim), 'old_logp': (g, len(registered_steps)), 'rewards': (1, g),
                  'advantages': (g,), 'adv_mask': (g,)}
        for k, shape in shapes.items():
            t = rollout[k]
            if tuple(t.shape) != shape or not torch.isfinite(t).all() or t.requires_grad:
                raise ValueError('Invalid detached rollout field: ' + k)
        for k in ('chain', 'old_logp', 'rewards', 'advantages', 'adv_mask'):
            if rollout[k].dtype != torch.float32:
                raise ValueError('Source FP32 rollout field changed: '+k)
        for k in ('timesteps', 'deltas'):
            if (not torch.is_tensor(rollout[k]) or rollout[k].shape != (10,)
                    or rollout[k].requires_grad or not torch.isfinite(rollout[k]).all()):
                raise ValueError('Invalid scheduler material: '+k)
        indices = rollout['stoch_idx']
        if torch.is_tensor(indices): indices = indices.tolist()
        if list(indices) != registered_steps:
            raise ValueError('Changed registered stochastic window')
        v = rollout['behavior_version']
        if v['run'] != identity['sha256'] or v['update'] != u - i:
            raise ValueError('Behavior is not the collection-start policy')
        old = rollout['old_probability']['current'] if 'current' in rollout['old_probability'] else rollout['old_probability']
        if registered_steps != [7,8,9] or tuple(old['source_clamped_logprob'].shape)!=(g,3,8,3):
            raise ValueError('Group exploration retains exactly W3 probability material')
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


def validate_boundary(state, rollout, identity, optimizer=None, scheduler=None, *,
                      credit, recipe_validator, legacy=False, _structure_validator=None):
    """Original transaction semantics, dynamic G and the registered raw-A math."""
    recipe_validator(identity['recipe'])
    g = identity['recipe']['grpo']['sample']['group_size']
    if (credit.group_size != g
            or identity['recipe']['research']['structured_advantage'] != credit.protocol['schema']):
        raise ValueError('Credit/recipe G or schema mismatch')
    structure = validate_structure if _structure_validator is None else _structure_validator
    summary = structure(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    summary.update(structured_advantage_schema=credit.protocol['schema'],
                   structured_protocol=credit.protocol, group_size=g, structured_world=8)
    if not summary['pending']:
        return summary
    if 'reward_transform' in rollout:
        raise ValueError('Canonical feedback is untransformed')
    expected = canonical_components(rollout['records'], g)
    for k in ('canonical_rewards', 'rewards'):
        value = rollout.get(k)
        if (not torch.is_tensor(value) or tuple(value.shape) != (1, g) or value.requires_grad
                or value.dtype != torch.float32 or not torch.equal(value.cpu(), expected['canonical_rewards'])):
            raise ValueError('Saved canonical reward material changed: '+k)
    saved = rollout.get('structured_advantage', {})
    if (set(saved) != {'protocol', 'components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'}
            or saved.get('protocol') != credit.protocol):
        raise ValueError('Structured credit protocol/material changed')
    expected.update(credit.parts(expected['components']))
    for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'):
        value, want = saved[k], expected[k]
        if (not torch.is_tensor(value) or value.shape != want.shape or value.requires_grad
                or value.dtype != want.dtype
                or not torch.allclose(value.cpu(), want, atol=1e-14, rtol=0)):
            raise ValueError('Saved structured component changed: '+k)
    scale, meta = rollout.get('normalization_std'), rollout['advantage_statistics']
    if (not torch.is_tensor(scale) or scale.ndim != 0 or scale.requires_grad
            or scale.dtype != torch.float64 or not bool(torch.isfinite(scale)) or float(scale) < 0
            or meta.get('structured_advantage_schema') != credit.protocol['schema']
            or meta.get('group_size') != g
            or meta.get('normalization_population_candidates') != 8*g
            or meta.get('normalization_population_groups') != 8):
        raise ValueError('Global normalization identity changed')
    raw_std = rollout['raw_group_std']
    want_std = (expected['canonical_rewards'].double()-expected['canonical_rewards'].double()[:, :1]).std(1, unbiased=False, keepdim=True)
    if (not torch.is_tensor(raw_std) or raw_std.requires_grad or raw_std.dtype != torch.float64
            or raw_std.shape != (1, 1)
            or not torch.allclose(raw_std.cpu(), want_std, atol=1e-12, rtol=0)):
        raise ValueError('Canonical raw group std changed')
    summary.update(normalization_std=float(scale),
        canonical_rewards=rollout['canonical_rewards'].double().cpu().tolist(),
        structured_material={k:saved[k].cpu().tolist() for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
        structured_digests={k:tensor_digest(saved[k]) for k in ('components', 'feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')},
        canonical_rewards_digest=tensor_digest(rollout['canonical_rewards']))
    return summary


def validate_ranks(summaries, *, credit):
    if len(summaries) != 8:
        raise ValueError('Exactly eight committed rank summaries required')
    g = credit.group_size
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run',
              'pending', 'group_size', 'structured_world', 'structured_advantage_schema', 'structured_protocol')
    for s in summaries:
        if (s.get('group_size') != g or s.get('structured_world') != 8
                or s.get('structured_protocol') != credit.protocol
                or {k:s[k] for k in common} != {k:summaries[0][k] for k in common}):
            raise ValueError('Ranks disagree on G/credit/recovery boundary')
    if not summaries[0]['pending']:
        return summaries
    if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):
        raise ValueError('Ranks disagree on collection-start behavior')
    rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
    components = torch.tensor([s['structured_material']['components'][0] for s in summaries], dtype=torch.float64)
    material = torch.cat((rewards.unsqueeze(-1), components), -1)
    expected = credit.advantages(rewards, components, all_material=material)
    adv = expected['advantages'].reshape(8, g); mask = expected['adv_mask'].reshape(8, g)
    for rank, s in enumerate(summaries):
        if (s['canonical_rewards'] != s['rewards']
                or abs(s['global_std']-float(expected['global_std'])) > 1e-12
                or abs(s['normalization_std']-float(expected['normalization_std'])) > 1e-12
                or not torch.allclose(torch.tensor(s['advantages']), adv[rank], atol=1e-7, rtol=1e-6)
                or s['mask'] != mask[rank].tolist()):
            raise ValueError('Saved credit was not globally finalized across world*G')
        for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw'):
            want = expected[k][rank:rank+1]
            if not torch.allclose(torch.tensor(s['structured_material'][k], dtype=want.dtype), want, atol=1e-14, rtol=0):
                raise ValueError('Rank structured component mismatch: '+k)
    return summaries
