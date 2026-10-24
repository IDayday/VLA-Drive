"""Finite L1/L2/L3 root-policy learning using the common native environment."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import traceback
import uuid
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, group_advantages, require_scene, require_budget_lock
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state, load_state
from starVLA.rl.qwen_flow_exploration.latent.policy import LatentNoisePolicy, condition_features
from starVLA.rl.qwen_flow_exploration.latent.collection import collect_behavior
from starVLA.rl.qwen_flow_exploration.latent.training import train_collected_batch
from starVLA.rl.qwen_flow_exploration.latent.frozen import frozen_digest
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import seed_for
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


def scene_batch(tokens, seed, round_index, B):
    selected = []
    for position in range(round_index*B, (round_index+1)*B):
        epoch, offset = divmod(position, len(tokens))
        order = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        selected.append(tokens[order[offset]])
    return selected


def source_identity():
    paths = list(Path('starVLA/rl/qwen_flow_exploration/latent').glob('*.py'))
    paths += [Path(name) for name in ['scripts/analysis/qwen_flow_exploration/latent_train.py',
        'starVLA/rl/qwen_flow_exploration/native.py', 'starVLA/rl/qwen_flow_exploration/protocol.py',
        'starVLA/rl/qwen_flow_exploration/checkpoint.py', 'starVLA/rl/flow_grpo/math.py',
        'starVLA/model/modules/qwen_sourceflow/source_flow_policy.py',
        'starVLA/model/modules/qwen_sourceflow/cached_action_runner.py',
        'starVLA/model/modules/qwen_sourceflow/action_codec.py',
        'starVLA/model/modules/qwen_sourceflow/qwen_encoder.py',
        'starVLA/model/modules/qwen_sourceflow/condition_provider.py']]
    return {str(path):file_sha(path) for path in sorted(set(paths))}


def validate_progress(state, rounds, inner_epochs):
    pending = state['phase'] in ('ROLLOUT_COMMITTED', 'INNER_COMMITTED')
    expected = (state['round']-int(pending))*inner_epochs+state['next_inner']
    if state['updates'] != expected or state['round'] > rounds or state['round'] < int(pending):
        raise ValueError('Latent round/inner/optimizer ancestry mismatch')
    if state['phase'] == 'COMPLETE' and state['round'] != rounds:
        raise ValueError('Incomplete budget cannot be a COMPLETE root-policy run')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['L1', 'L2', 'L3'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', help='Exact same run or its explicit committed checkpoint')
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--limit', type=int, help='Qualification only: number of fresh rollout rounds')
    parser.add_argument('--scorer-workers', type=int, default=1)
    args = parser.parse_args()
    protocol = read_protocol(args.protocol)
    budget_lock = None if args.qualification else require_budget_lock(protocol)
    if args.limit and not args.qualification:
        raise ValueError('Never shorten one formal method silently')
    rounds = args.limit if args.limit else protocol['screen']['rollout_rounds']
    if not 0 < rounds <= protocol['screen']['rollout_rounds']:
        raise ValueError('Only the preregistered finite window is authorized')
    B, G, inner_epochs = (protocol['screen'][key] for key in ['B', 'G', 'inner_epochs'])
    run = Path(args.output).resolve()
    if args.resume:
        if not run.is_dir():
            raise ValueError('Resume must preserve the original output run')
    else:
        run.mkdir(parents=True, exist_ok=False)
    invocation = run/'invocations'/uuid.uuid4().hex
    invocation.mkdir(parents=True, exist_ok=False)
    phase = 'INITIALIZING'
    env = None
    info = dict(status='RUNNING', start_unix=time.time(), resume=args.resume,
                arm=args.arm, seed=args.seed, qualification=args.qualification)
    atomic_json(invocation/'invocation.json', info)
    try:
        env = NativeEnvironment(protocol, trainable='frozen', scorer_workers=args.scorer_workers,
            ledger_path=invocation/'ledger.json', experiment_id=args.arm+'_seed'+str(args.seed))
        tokens = protocol['splits']['TRAIN']
        for token in tokens:
            require_scene(protocol, token, 'TRAIN', update=True)
        features = condition_features(env.condition(tokens[0], 'TRAIN'))
        torch.manual_seed(args.seed)
        policy = LatentNoisePolicy(features.shape[-1], conditional=args.arm != 'L2').to(env.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        warmup = max(1, math.ceil(.03*rounds*inner_epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: min(1., (index+1)/warmup))
        root_generator = torch.Generator(device='cpu').manual_seed(seed_for('three_routes_latent_train', args.seed, 'root_stream'))
        beta = 0.0 if args.arm == 'L3' else .01
        identity = dict(schema='frozen_native_root_policy_run_v1', protocol_sha256=protocol['sha256'],
            base_model_sha256=protocol['base_model_sha256'], arm=args.arm, seed=args.seed,
            budget_lock_sha256=budget_lock['sha256'] if budget_lock else None,
            qualification=args.qualification, rounds=rounds, B=B, G=G, inner_epochs=inner_epochs,
            optimizer='Adam', learning_rate=1e-4, joint_PPO_clip=.2, prior_KL_coefficient=beta,
            scheduler='3percent optimizer update warmup then constant', warmup_updates=warmup,
            policy_identity=policy.identity(), data_order='source-style seed+epoch permutation of shared fixed TRAIN512',
            root_RNG='CPU three_routes_latent_train namespace; state checkpointed',
            common_advantage=protocol['advantages'], source_files=source_identity())
        identity['sha256'] = fingerprint(identity)
        identity_path = run/'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Changed root-policy scientific identity')
        else:
            atomic_json(identity_path, identity)
        state = dict(phase='READY', round=0, updates=0, next_inner=0, arm=args.arm, seed=args.seed,
                     raw_scene_exposures=0, new_candidates=0)
        batch, history, batch_records_path = None, [], None
        if args.resume:
            state, saved = load_state(args.resume, protocol['sha256'], restore_rng=True)
            if saved['identity'] != identity:
                raise ValueError('Wrong arm/seed/base for exact resume')
            policy.load_state_dict(saved['policy'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            root_generator.set_state(saved['root_generator_state'])
            batch = saved['batch'].to(env.device) if saved['batch'] is not None else None
            history = saved['history']
            batch_records_path = saved['batch_records_path']
            if batch_records_path and file_sha(batch_records_path) != saved['batch_records_sha256']:
                raise ValueError('Saved feedback evidence changed')
        validate_progress(state, rounds, inner_epochs)
        frozen_before = frozen_digest(env.model)
        last_checkpoint = None

        def publish():
            nonlocal last_checkpoint
            validate_progress(state, rounds, inner_epochs)
            payload = dict(identity=identity, policy=copy.deepcopy(policy.state_dict()),
                           optimizer=copy.deepcopy(optimizer.state_dict()),
                           scheduler=copy.deepcopy(scheduler.state_dict()),
                           root_generator_state=root_generator.get_state(),
                           batch=batch.to('cpu') if batch is not None else None,
                           batch_records_path=batch_records_path,
                           batch_records_sha256=file_sha(batch_records_path) if batch_records_path else None,
                           history=history)
            last_checkpoint = save_state(run, copy.deepcopy(state), payload, protocol['sha256'])
            atomic_json(run/'progress.json', dict(state, checkpoint=str(last_checkpoint), status='RUNNING'))

        if not args.resume:
            publish()
        info['invocation_start_updates'] = state['updates']
        info['invocation_start_round'] = state['round']
        atomic_json(invocation/'invocation.json', info)
        while state['phase'] != 'COMPLETE':
            if state['phase'] == 'READY':
                phase = 'COLLECTING'
                chosen = scene_batch(tokens, args.seed, state['round'], B)
                batch, records = collect_behavior(env, policy, chosen, split='TRAIN', candidates=G,
                    generator=root_generator, proposal_source=args.arm,
                    policy_version=f"round{state['round']+1:04d}_update{state['updates']:06d}",
                    advantage_callback=group_advantages, training_seed=args.seed)
                record_path = invocation/f"collection{state['round']+1:04d}_records.json"
                atomic_json(record_path, records)
                batch_records_path = str(record_path)
                state.update(phase='ROLLOUT_COMMITTED', round=state['round']+1, next_inner=0,
                             raw_scene_exposures=state['raw_scene_exposures']+B,
                             new_candidates=state['new_candidates']+B*G)
                publish()
            phase = 'UPDATING'
            active_batch = batch

            def inner_commit(inner, _, metrics):
                nonlocal batch, batch_records_path
                state['updates'] += 1
                scheduler.step()
                complete_round = inner == inner_epochs
                state['next_inner'] = 0 if complete_round else inner
                state['phase'] = ('COMPLETE' if state['round'] == rounds else 'READY') if complete_round else 'INNER_COMMITTED'
                metrics.update(update=state['updates'], collection=state['round'],
                               raw_scene_exposures=state['raw_scene_exposures'], new_candidates=state['new_candidates'])
                history.append(metrics)
                with (invocation/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics)+'\n')
                env.ledger.add(optimizer_updates=1, latent_network_scene_forwards=B)
                if complete_round:
                    batch = None
                    batch_records_path = None
                publish()

            tick = time.monotonic()
            train_collected_batch(policy, optimizer, active_batch, inner_epochs=inner_epochs,
                start_inner=state['next_inner'], clip_range=.2, prior_coefficient=beta,
                checkpoint_callback=inner_commit)
            env.ledger.add(update_seconds=time.monotonic()-tick)
        if frozen_digest(env.model) != frozen_before:
            raise ValueError('Frozen generator parameters or buffers changed during learning')
        result = dict(status='COMPLETE', identity=identity['sha256'], **state,
                      checkpoint=str(last_checkpoint) if last_checkpoint else args.resume,
                      frozen_generator_hash_before=frozen_before, frozen_generator_unchanged=True,
                      trainable_noise_policy_sha256=tensor_digest(policy.state_dict()),
                      qualification=args.qualification, ledger=env.ledger.snapshot())
        atomic_json(run/'COMPLETE.json', result)
        atomic_json(run/'metrics_effective.json', history)
        atomic_json(run/'progress.json', result)
        info['status'] = 'COMPLETE'
        print(json.dumps({key:result[key] for key in ['status','arm','seed','round','updates','new_candidates']}))
    except BaseException:
        info.update(status='FAILED', phase=phase, error=traceback.format_exc())
        atomic_json(invocation/'failure.json', info)
        # In particular, no COLLECTING failure publishes a new latest pointer.
        raise
    finally:
        info['end_unix'] = time.time()
        info['wall_seconds'] = info['end_unix']-info['start_unix']
        atomic_json(invocation/'invocation.json', info)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
