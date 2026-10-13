"""Frozen-source loading and resumable, scene-bound original policy sampling."""
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.action_head_policy import FrozenFeatureStore, feature_identity, freeze_for_action_head
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.rollout import SamplingSpec
from .contracts import file_sha, seed_for
from .policy_adapter import encode_condition, sample_ode, sample_legacy_sde, decode_request


def load_cached_policy(config_path, source_config, *, behavior=None, behavior_sha256=None):
    cfg, sft = resolve_config(config_path, overrides=['checkpoint_contract.source_config=' + str(source_config)])
    identity = feature_identity(cfg, sft)  # before QwenOFT adds derived DiTConfig
    store = FrozenFeatureStore(cfg['frozen_feature_cache']['root'], identity, readonly=True)
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    freeze_for_action_head(policy)
    behavior_identity = cfg['checkpoint_contract']['sha256']
    if behavior is not None:
        if not behavior_sha256 or file_sha(behavior) != behavior_sha256:
            raise ValueError('Behavior checkpoint hash must match an explicit identity')
        weights = torch.load(behavior, map_location='cpu', mmap=True, weights_only=True)
        weights = weights.get('module', weights)
        current = policy.state_dict()
        if set(current) != set(weights):
            raise ValueError('Behavior export and original full policy keys differ')
        for name, value in current.items():
            if not name.startswith('action_model.') and not torch.equal(value.cpu(), weights[name].to(dtype=value.dtype)):
                raise ValueError('Behavior changed the frozen SFT prefix: ' + name)
        head = {name.removeprefix('action_model.'): value for name, value in weights.items() if name.startswith('action_model.')}
        policy.action_model.load_state_dict(head, strict=True)
        behavior_identity = behavior_sha256
        del weights, head, current
    return policy, store, cfg, sft, behavior_identity


@torch.no_grad()
def sample_scene(policy, store, token, seeds, *, mode, role, act_norm):
    value = store.get(token)
    raw = value['raw'].cuda()
    condition = encode_condition(policy, raw)
    # Standard deployment retains BF16 initial draws. Paired sampler diagnostics
    # explicitly use the same FP32 Gaussian roots in both SDE and ODE; that
    # controlled diagnostic is kept separate from deployment evaluations.
    root_dtype = torch.float32 if role == 'sampler_gap' else raw.dtype
    root_namespace = 'sampler_gap_initial' if role == 'sampler_gap' else 'original_ode_initial'
    roots, keys = [], []
    for seed in seeds:
        key = seed_for(root_namespace, seed, token)
        roots.append(torch.randn((1, 8, 4), device=raw.device, dtype=root_dtype,
                                 generator=torch.Generator(device=raw.device).manual_seed(key)))
        keys.append({'seed': seed, 'root_key': key, 'root_namespace': root_namespace})
    result = {'token': token, 'noise_keys': keys, 'sampler': mode, 'role': role,
              'initial_noise_dtype': str(root_dtype), 'initial_noise': torch.cat(roots).cpu()}
    if mode == 'ode':
        actions = torch.cat([sample_ode(policy, condition, root) for root in roots], dim=0)
    elif mode == 'legacy_sde':
        if role != 'sampler_gap':
            raise ValueError('This legacy SDE entrypoint is restricted to the paired diagnostic')
        key = seed_for('sampler_gap_sde_innovations', 4101, token)
        spec = SamplingSpec(group_size=len(seeds), num_steps=10, temporal_noise_correlation=.8)
        sampled = sample_legacy_sde(policy, condition, torch.stack(roots, dim=1),
                                    innovation_generator=torch.Generator(device=raw.device).manual_seed(key), spec=spec)
        actions = sampled['actions'][0]
        result.update(chain=sampled['chain'][0].cpu(), elementwise_logprob=sampled['elementwise_logprob'][0].cpu(),
                      innovation_key=key, innovation_namespace='sampler_gap_sde_innovations', sampling_spec=vars(spec))
    else:
        raise ValueError('Unknown registered sampler')
    actions = actions.float().cpu()
    result['normalized_actions'] = actions
    result['requested_trajectories'] = torch.from_numpy(decode_request(actions.numpy(), act_norm=act_norm))
    return result
