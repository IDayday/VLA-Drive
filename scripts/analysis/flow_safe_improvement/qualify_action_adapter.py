"""Real cached action-head forward/backward qualification; no method score claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.flow_grpo.action_head_policy import FrozenFeatureStore, feature_identity, freeze_for_action_head, install_frozen_features, prefix_parameters
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import SamplingSpec, sample_chain
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.source_oracle import action_only_source
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.policy_adapter import encode_condition, sample_ode, sample_legacy_sde, original_fm_loss, decode_request, encode_external_request
from starVLA.rl.safe_flow_improvement.weighted_flow_objective import interpolation, positive_flow_loss


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-config', help='Read-only absolute path to the same hash-locked source YAML')
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    overrides = ['checkpoint_contract.source_config=' + a.source_config] if a.source_config else []
    cfg, sft = resolve_config(a.config, overrides=overrides)
    # QwenOFT initializes derived DiTConfig in-place. Bind the unmodified source
    # configuration before construction, as the existing training path does.
    identity = feature_identity(cfg, sft)
    started = time.monotonic()
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    freeze_for_action_head(policy)
    store = FrozenFeatureStore(cfg['frozen_feature_cache']['root'], identity, readonly=True)
    install_frozen_features(policy, store)
    manifest = json.loads(Path(a.population).read_text())
    tokens = manifest['calibration_tokens'][:2]
    values = [store.get(t) for t in tokens]
    raw = torch.cat([v['raw'] for v in values]).cuda()
    targets = torch.stack([v['action'] for v in values]).cuda().float()
    examples = [{'image': (), 'lang': v['lang'], 'state': v['state'].numpy(), 'token': v['token'],
                 'action': v['action'].numpy()} for v in values]
    report = {'status': 'RUNNING', 'sft_sha256': cfg['checkpoint_contract']['sha256'],
              'feature_identity_sha256': store.identity_sha, 'scenes': len(tokens),
              'stage': 'ADAPTER_QUALIFICATION_NOT_LEARNING_RESULT',
              'load_seconds': time.monotonic() - started, 'comparisons': {}}
    with torch.no_grad():
        condition = encode_condition(policy, raw)
        torch.cuda.manual_seed_all(18271)
        state = torch.cuda.get_rng_state()
        roots = torch.randn((len(tokens), 8, 4), device='cuda', dtype=raw.dtype)
        t0 = time.monotonic()
        ours = sample_ode(policy, condition, roots)
        torch.cuda.synchronize()
        report['ode_two_scene_seconds'] = time.monotonic() - t0
        original = policy.predict_action_infer_1d(examples, initial_noise=roots)['normalized_actions']
        report['comparisons']['original_export_ode_max_abs'] = float(np.max(np.abs(ours.float().cpu().numpy() - original)))
        torch.cuda.set_rng_state(state)
        with action_only_source(policy, cfg['checkpoint_contract']), torch.autocast('cuda', dtype=torch.float32):
            source = policy.action_model.predict_action(raw)
        report['comparisons']['independent_source_ode_max_abs'] = float((ours - source).abs().max())
        # Both functions consume precisely the same draws. The new adapter owns
        # separate generators in research; here the innovation generator is
        # advanced past the old root draw solely to compare the legacy kernel.
        observation = prepare_policy_observation(examples[:1])
        spec = SamplingSpec(group_size=16, num_steps=10, temporal_noise_correlation=.8)
        legacy = sample_chain(policy, observation, spec, 0, 17831, {'purpose': 'kernel_oracle'})
        gen = torch.Generator(device='cuda').manual_seed(17831)
        shared_roots = torch.randn((1, 16, 8, 4), device='cuda', generator=gen)
        adapted = sample_legacy_sde(policy, condition[:1], shared_roots, innovation_generator=gen, spec=spec)
        report['comparisons']['legacy_chain_equal'] = torch.equal(adapted['chain'], legacy.chain)
        report['comparisons']['legacy_logprob_equal'] = torch.equal(adapted['elementwise_logprob'], legacy.old_elementwise_logprob)
        physical = decode_request(ours.float().cpu().numpy(), act_norm=int(sft.datasets.vla_data.act_norm))
        encoded = encode_external_request(physical, act_norm=int(sft.datasets.vla_data.act_norm))
        roundtrip = decode_request(encoded, act_norm=int(sft.datasets.vla_data.act_norm))
        report['comparisons']['external_pose_roundtrip_max_abs'] = float(np.max(np.abs(roundtrip - physical)))
    repeat = int(sft.framework.action_model.get('flow_train_repeats', sft.framework.action_model.get('repeated_diffusion_steps', 8)))
    raw_repeat, target_repeat = raw.repeat(repeat, 1, 1), targets.repeat(repeat, 1, 1)
    noise = torch.randn_like(target_repeat)
    time_draw = policy.action_model.sample_time(len(target_repeat), target_repeat.device, target_repeat.dtype)
    xt, target_velocity = interpolation(target_repeat, noise, time_draw)
    t0 = time.monotonic()
    original_loss = original_fm_loss(policy, raw_repeat, target_repeat, noise, time_draw)
    with torch.autocast('cuda', dtype=torch.float32):
        prediction = policy.action_model.predict_velocity(xt, (time_draw * policy.action_model.num_timestep_buckets).long(), encode_condition(policy, raw_repeat))
    explicit_loss, _ = positive_flow_loss(prediction, target_velocity, scenes=len(tokens), repeats=repeat)
    report['comparisons']['original_fm_loss_abs'] = float((original_loss - explicit_loss).abs().detach())
    explicit_loss.backward()
    torch.cuda.synchronize()
    report['fm_two_scene_forward_backward_seconds'] = time.monotonic() - t0
    gradients = {name: {'present': value.grad is not None,
                        'finite': bool(value.grad is not None and torch.isfinite(value.grad).all()),
                        'norm': float(value.grad.float().norm()) if value.grad is not None else None}
                 for name, value in policy.named_parameters() if value.requires_grad}
    report['trainable_tensor_count'] = len(gradients)
    report['trainable_parameter_count'] = sum(v.numel() for v in policy.parameters() if v.requires_grad)
    report['all_trainable_gradients_valid'] = all(v['present'] and v['finite'] and v['norm'] > 0 for v in gradients.values())
    report['frozen_no_gradient'] = all(value.grad is None and not value.requires_grad for _, value in prefix_parameters(policy))
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    atomic_json(output / 'gradients.json', gradients)
    c = report['comparisons']
    passed = (c['original_export_ode_max_abs'] == 0 and c['independent_source_ode_max_abs'] <= 2e-5
              and c['legacy_chain_equal'] and c['legacy_logprob_equal']
              and c['external_pose_roundtrip_max_abs'] <= 2e-5 and c['original_fm_loss_abs'] <= 2e-6
              and report['trainable_tensor_count'] == 359 and report['all_trainable_gradients_valid']
              and report['frozen_no_gradient'])
    report.update(status='PASS' if passed else 'FAIL', total_seconds=time.monotonic() - started)
    atomic_json(output / 'report.json', report)
    np.savez_compressed(output / 'requests.npz', tokens=np.asarray(tokens), normalized_actions=ours.float().cpu().numpy(), requested_trajectories=physical)
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
