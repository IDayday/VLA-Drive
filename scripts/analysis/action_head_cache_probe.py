"""Real CUDA cached/uncached head contract and source-oracle comparisons."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.flow_grpo.config import resolve_config, config_hash
from starVLA.rl.flow_grpo.loading import load_policy, file_sha
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.action_head_policy import (
    freeze_for_action_head, install_frozen_features, FrozenFeatureStore, feature_identity, prefix_parameters,
)
from starVLA.rl.flow_grpo.data import KeyedDataset, to_device
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.model import FlowGRPOActor, make_reference
from starVLA.rl.flow_grpo.rollout import evaluate_transitions, sample_chain
from starVLA.rl.flow_grpo.math import clipped_surrogate, reduce_dimensions
from starVLA.rl.flow_grpo.checkpoint import capture_rng, restore_rng
from starVLA.rl.flow_grpo.source_oracle import action_only_source
from starVLA.rl.flow_grpo.acceptance import executable_identity
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.contracts import tensor_hashes


def restore_current_images(chain, example):
    """Hydrate cache-only observations for the uncached oracle, never resample.

    The token, instruction and ego history must agree with the saved behavior.
    Only deployment observations are reconstructed; future labels are excluded.
    The resulting regenerated chain must still match the saved chain exactly.
    """
    observation = prepare_policy_observation([example])
    old = chain.observation
    if (observation.tokens != old.tokens or observation.instructions != old.instructions
            or len(observation.states) != len(old.states)
            or any(not np.array_equal(a, b) for a, b in zip(observation.states, old.states))
            or not all(observation.images)):
        raise ValueError('raw/current observation differs from saved behavior')
    return replace(chain, observation=observation)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True); p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); cfg, sft = resolve_config(a.config); configure_numerics()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'device_type': 'cuda', 'device': torch.cuda.get_device_name(),
              'checkpoint_sha256': cfg['checkpoint_contract']['sha256'], 'config_hash': config_hash(cfg),
              'executable_sha256': executable_identity(), 'script_sha256': file_sha(__file__), 'scenes': []}
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    policy._inference_qwen_forward_mode = 'optimized'
    freeze_for_action_head(policy)
    report['trainable_names'] = [n for n,p in policy.named_parameters() if p.requires_grad]
    report['trainable_numel'] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    reference = make_reference(policy).cuda().bfloat16().eval()
    report['prefix_matches_own_reference'] = tensor_hashes(policy, lambda n,p: not p.requires_grad) == {
        n:h for n,h in tensor_hashes(reference).items() if n in dict(prefix_parameters(policy))}
    original = policy.encode_policy_features
    store = FrozenFeatureStore(out/'features', feature_identity(cfg, sft))
    dataset = KeyedDataset(sft)
    for rank in range(4):
        chain = to_device(torch.load(Path(a.bank)/f'rollout_rank{rank}_v0.pt', map_location='cpu', weights_only=False)[0], 'cuda')
        example = dataset[(chain.observation.tokens[0], 42)]
        chain = restore_current_images(chain, example)
        row = {'token': example['token'], 'official_reward_nonzero_advantage': bool(chain.advantages.abs().max()>0)}
        policy.encode_policy_features = original
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = evaluate_transitions(policy, chain.observation, chain, layout=cfg['runtime'].get('transition_evaluation', 'serial'))
            condition = policy.encode_policy_condition(chain.observation)
            state = capture_rng(policy)
            loss = policy.compute_sft_losses([example])
            if rank == 0:
                restore_rng(state, policy)
                with action_only_source(policy, cfg['checkpoint_contract']): old_loss = policy([example])
                row['source_sft_max_abs'] = max(abs(float(loss[k]-old_loss[k])) for k in old_loss)
                restore_rng(state, policy)
                with action_only_source(policy, cfg['checkpoint_contract']): old_ode = policy.predict_action_infer_1d([example])['normalized_actions']
                restore_rng(state, policy)
                new_ode = policy.predict_action_infer_1d([example])['normalized_actions']
                row['source_ode_max_abs'] = float(np.max(np.abs(new_ode-old_ode)))
        def backward():
            policy.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                stats = evaluate_transitions(policy, chain.observation, chain, checkpoint=cfg['runtime']['activation_checkpointing'], layout=cfg['runtime'].get('transition_evaluation', 'serial'))
                current = reduce_dimensions(stats['elementwise_logprob'], chain.dimension_mask, chain.spec.reduction)
                pg,_ = clipped_surrogate(current, chain.old_logprob, chain.advantages[:,:,None], cfg['algorithm']['ppo_clip_range'])
                objective = (pg*chain.transition_mask).sum()/chain.transition_mask.sum()
            objective.backward()
            return {n:p.grad.detach().cpu().clone() if p.grad is not None else None
                    for n,p in policy.named_parameters() if p.requires_grad}
        start = time.monotonic(); grads = backward(); row['uncached_backward_seconds'] = time.monotonic()-start
        install_frozen_features(policy, store)
        policy.encode_policy_features([example])  # populate with original replay labels
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            cached_condition = policy.encode_policy_condition(chain.observation)
            actual = evaluate_transitions(policy, chain.observation, chain, layout=cfg['runtime'].get('transition_evaluation', 'serial'))
            restore_rng(state, policy); cached_loss = policy.compute_sft_losses([example])
            generated = sample_chain(policy, chain.observation, chain.spec, 0, chain.noise_seed, chain.provenance)
        row['condition_equal'] = torch.equal(condition, cached_condition)
        row['transitions_equal'] = {k:torch.equal(v,actual[k]) for k,v in expected.items()}
        row['sft_equal'] = {k:torch.equal(v,cached_loss[k]) for k,v in loss.items()}
        row['chain_equal'] = torch.equal(chain.chain, generated.chain)
        row['old_equal'] = torch.equal(chain.old_elementwise_logprob, generated.old_elementwise_logprob)
        start = time.monotonic(); other = backward(); row['cached_backward_seconds'] = time.monotonic()-start
        row['gradients'] = {}
        for name,g in grads.items():
            h = other[name]; equal = g is not None and h is not None and torch.equal(g,h)
            if equal:
                norm = float(g.float().norm())
                detail = {'present':True,'equal':True,'max_abs':0.,'relative_l2':0.,
                          'cosine':1. if norm else None,'norm_ratio':1. if norm else None,
                          'outside_tolerance':0,'norm':norm}
            else:
                from starVLA.rl.flow_grpo.comparison import tensor_comparison
                detail = tensor_comparison(g,h,atol=0.,rtol=0.); detail['equal']=False
            row['gradients'][name] = detail
        row['frozen_no_grad'] = all(p.grad is None for _,p in prefix_parameters(policy))
        report['scenes'].append(row); atomic_json(out/'probe.json',report)
        print(json.dumps({k:v for k,v in row.items() if k!='gradients'}),flush=True)
        del grads, other, expected, actual
    report['reference_no_grad'] = all(p.grad is None and not p.requires_grad for p in reference.parameters())
    report['status'] = 'PASS' if (report['prefix_matches_own_reference'] and report['reference_no_grad']
        and all(r['condition_equal'] and r['chain_equal'] and r['old_equal'] and r['frozen_no_grad']
                and all(r['transitions_equal'].values()) and all(r['sft_equal'].values())
                and all(g['equal'] for g in r['gradients'].values()) for r in report['scenes'])
        and report['scenes'][0]['source_sft_max_abs']<=2e-6
        and report['scenes'][0]['source_ode_max_abs']<=2e-5) else 'FAIL'
    atomic_json(out/'probe.json',report)
    if report['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
