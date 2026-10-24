"""Real frozen-base root-policy profile; GPU placement is owned by the main agent."""
import argparse
import json
import time
from pathlib import Path
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, group_advantages, require_scene
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy, condition_features
from starVLA.rl.qwen_flow_exploration.latent.collection import collect_behavior
from starVLA.rl.qwen_flow_exploration.latent.training import train_collected_batch
from starVLA.rl.qwen_flow_exploration.latent.frozen import frozen_digest
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.flow_grpo.transactions import atomic_json


def tokens_from(path):
    value = json.loads(Path(path).read_text())
    return value if isinstance(value, list) else value['tokens']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--tokens', help='Optional explicit TRAIN smoke JSON; defaults to registered protocol.smoke.TRAIN')
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['L1', 'L2', 'L3'], default='L1')
    parser.add_argument('--scenes', type=int, default=32)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scorer-workers', type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous profile; new attempt requires a new output')
    protocol = read_protocol(args.protocol)
    tokens = (tokens_from(args.tokens) if args.tokens else protocol['smoke']['TRAIN'])[:args.scenes]
    if len(tokens) != args.scenes:
        raise ValueError('Insufficient fixed smoke TRAIN scenes')
    for token in tokens:
        require_scene(protocol, token, 'TRAIN', update=True)
    env = NativeEnvironment(protocol, trainable='frozen', scorer_workers=args.scorer_workers,
                            ledger_path=output.with_name(output.stem+'_ledger.json'),
                            experiment_id='latent_'+args.arm+'_real_profile_seed'+str(args.seed))
    try:
        tick = time.monotonic()
        generator_before = frozen_digest(env.model)
        hash_seconds = time.monotonic()-tick
        initial_condition = env.condition(tokens[0], 'TRAIN')
        feature_dim = condition_features(initial_condition).shape[-1]
        torch.manual_seed(args.seed)
        policy = LatentNoisePolicy(feature_dim, conditional=args.arm != 'L2').to(env.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        policy_before = tensor_digest(policy.state_dict())
        rng = torch.Generator(device='cpu').manual_seed(args.seed+19073)
        rollout, records = collect_behavior(env, policy, tokens, split='TRAIN', candidates=args.candidates,
            generator=rng, proposal_source=args.arm+'_PROFILE', policy_version='profile_initial_q0',
            advantage_callback=group_advantages, training_seed=args.seed)
        tick = time.monotonic()
        history = train_collected_batch(policy, optimizer, rollout, clip_range=.2,
                                        prior_coefficient=0.0 if args.arm == 'L3' else .01)
        env.ledger.add(optimizer_updates=len(history), update_seconds=time.monotonic()-tick,
                       latent_network_scene_forwards=len(tokens)*len(history))
        tick = time.monotonic()
        generator_after = frozen_digest(env.model)
        hash_seconds += time.monotonic()-tick
        if generator_after != generator_before:
            raise ValueError('Frozen VLM/provider/DiT parameters or buffers changed')
        changed = policy_before != tensor_digest(policy.state_dict())
        report = dict(status='PASS' if changed else 'NO_PG_SIGNAL_IN_FIXED_SMOKE',
            qualification_only=True, driving_efficacy=False, base_model_sha256=protocol['base_model_sha256'],
            common_protocol_sha256=protocol['sha256'], feature_dim=feature_dim, root_dim=24,
            scenes=len(tokens), candidates=len(records), actual_optimizer_updates=len(history),
            same_actual_sampling_and_recompute_Gaussian=True, first_inner_ratio_min=history[0]['ratio_min'],
            first_inner_ratio_max=history[0]['ratio_max'], frozen_parameters_and_all_buffers_unchanged=True,
            trainable_policy_changed=changed, effective_PG_groups=int(rollout.advantages.ne(0).any(1).sum()),
            max_GPU_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            frozen_hash_seconds=hash_seconds, invocation_wall_seconds=time.monotonic()-started,
            metrics=history, ledger=env.ledger.snapshot(), policy_identity=policy.identity())
        atomic_json(output, report)
        print(json.dumps({k:report[k] for k in ['status','scenes','candidates','actual_optimizer_updates','max_GPU_memory_allocated_bytes']}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
