"""Frozen-SFT C_SDE G16 census of every RL_TRAIN scene, with atomic resumption.

This reuses the qualified condition, real chain and canonical geometric scorer.
It performs no learning, no scene filtering and no deployment evaluation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import traceback

import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, fingerprint, SFT_SHA256
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy, legacy_std_range
from starVLA.rl.constrained_flow_noise.effective_group_collector import EffectiveGroupCollector
from starVLA.rl.constrained_flow_noise.runtime import OrderedScenes, collect_batch
from starVLA.rl.constrained_flow_noise.safety_costs import ConstrainedRewardService
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.constrained_flow_noise.diagnostics import summarize_rows
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from .temporary_guard import worker_authorization

STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--max-new', type=int)
    parser.add_argument('--lease')
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.lease:
        receipt = Path(os.environ['CFN_TEMP_GUARD_RECEIPT'])
        for _ in range(100):
            if receipt.exists():
                break
            time.sleep(.05)
    host, deadline = worker_authorization(args.lease)
    registration = json.loads(Path(args.registration).read_text())
    if registration['sha256'] != fingerprint({k: v for k, v in registration.items() if k != 'sha256'}):
        raise ValueError('Census registration changed')
    for name, digest in registration['sources'].items():
        if file_sha(name) != digest:
            raise ValueError('Census source changed: ' + name)
    if not 0 <= args.shard < registration['shards']:
        raise ValueError('Invalid immutable shard assignment')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/f'shard{args.shard:02d}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    configure_numerics()
    torch.cuda.set_device(0)
    actor, _, _ = make_noise_policy('C_SDE', legacy_std_range()[:2])
    actor.requires_grad_(False).eval()
    configure(actor, True)
    root = Path(registration['run_root'])
    store = ConditionStore(root/'condition_cache')
    pop = json.loads(Path(registration['population']).read_text())
    calibration = json.loads((root/'calibration/calibration_lock.json').read_text())
    if (store.identity_sha != registration['condition_identity'] or
            pop['identity_sha256'] != registration['population_identity'] or
            file_sha(root/'calibration/calibration_lock.json') != registration['calibration_sha256']):
        raise ValueError('Census asset identity changed')
    stream = OrderedScenes(pop['train_tokens'], 42)
    version = dict(audit_identity=registration['sha256'], behavior_SFT_sha256=SFT_SHA256,
                   collection='frozen_full_TRAIN_census', optimizer_updates=0)
    # Exact real-chain comparison against the original pre-update behavior.
    # This is a host/execution check, not another scoring call.
    evidence = Path(registration['initial_behavior_archive'])
    if file_sha(evidence) != registration['initial_behavior_archive_sha256']:
        raise ValueError('Initial real behavior archive changed')
    saved = torch.load(evidence, map_location='cpu', weights_only=False)['rows'][0]
    token = stream.at(0)
    if saved['position'] != 0 or saved['token'] != token:
        raise ValueError('Initial real-chain population mismatch')
    actual = actor.sample_chain(store.get(token).cuda(), [token], [0], version=version)
    for key in ('chain', 'old_mean', 'old_std', 'old_elementwise_logp'):
        if not torch.equal(actual[key].cpu(), saved['chain'][key]):
            raise ValueError('Host real-chain qualification failed: ' + key)
    atomic_json(out/f'qualification_shard{args.shard:02d}.json',
                dict(status='PASS', host=host, exact_saved_SFT_chain=True,
                     comparison_keys=['chain', 'old_mean', 'old_std', 'old_elementwise_logp'],
                     registration_sha256=registration['sha256'], extra_velocity_calls=164))
    spec = dict(repo=str(Path.cwd()), cache_roots=registration['cache_roots'], maps_root='/mnt/navsim/maps',
                metric_manifest=str(root/'full_TRAIN_assets.json'),
                metric_manifest_sha256=registration['metric_manifest_sha256'])
    service = ConstrainedRewardService(spec, workers=1)
    fresh, reused, gpu_seconds, cpu_seconds = 0, 0, 0., 0.
    last = None
    status_path = out/f'status_shard{args.shard:02d}.json'
    def status(state):
        atomic_json(status_path, dict(status=state, host=host, pid=os.getpid(), shard=args.shard,
                    registration_sha256=registration['sha256'], new_scenes=fresh, reused_scenes=reused,
                    last_position=last, actual_new_candidate_scores=16*fresh,
                    measured_rollout_seconds=gpu_seconds, measured_scoring_CPU_seconds=cpu_seconds,
                    wall_seconds=time.monotonic()-started, deadline=deadline, observed_unix=time.time()))
    try:
        for position in range(args.shard, len(pop['train_tokens']), registration['shards']):
            if STOP or (deadline and time.time() >= deadline) or (args.max_new is not None and fresh >= args.max_new):
                status('PAUSED_AT_ATOMIC_SCENE_BOUNDARY')
                break
            worker_authorization(args.lease)
            path = out/'scenes'/f'{position//1000:03d}'/f'{position:06d}.pt'
            seal = path.with_suffix('.json')
            if seal.exists():
                existing = json.loads(seal.read_text())
                if existing['registration_sha256'] != registration['sha256'] or file_sha(path) != existing['payload_sha256']:
                    raise ValueError('Saved census record changed')
                reused += 1
                last = position
                continue
            collector = EffectiveGroupCollector(version, torch.tensor([1., 1., 1., .1, .1]))
            _, row = collect_batch(actor, None, store, service, stream, pop, collector, position, 1,
                                   calibration, 0, 1, failure_dir=out/'failures', include_reference=False)[0]
            summary = summarize_rows([row], actor.spec)
            reward = row['rewards']
            summary['all_PDMS_one'] = bool((reward == 1).all())
            summary['tied_PDMS_value'] = float(reward[0]) if bool((reward == reward[0]).all()) else None
            summary['initial_lambda_semantics'] = 'fixed [1,1,1,.1,.1], diagnostic only; no dual update'
            payload = dict(token=row['token'], position=position, requests=row['requests'],
                           normalized_actions=row['chain']['chain'][0, :, -1], records=row['records'],
                           costs=row['costs'], rewards=reward,
                           root_keys=row['chain']['root_keys'], innovation_keys=row['chain']['innovation_keys'],
                           behavior=version, not_eligible_for_policy_updates=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.incomplete')
            torch.save(payload, temporary)
            os.replace(temporary, path)
            atomic_json(seal, dict(position=position, token=row['token'], log=pop['token_logs'][row['token']],
                        registration_sha256=registration['sha256'], payload_sha256=file_sha(path), summary=summary))
            fresh += 1
            last = position
            gpu_seconds += row['rollout_seconds']
            cpu_seconds += row['scoring_cpu_seconds']
            status('RUNNING')
        else:
            status('ASSIGNED_POPULATION_COMPLETE')
    except BaseException:
        atomic_json(out/f'failure_shard{args.shard:02d}_{int(time.time())}.json',
                    dict(last_committed_position=last, traceback=traceback.format_exc(), no_reward_zero_substitution=True))
        raise
    finally:
        service.close()
    print(status_path.read_text(), flush=True)


if __name__ == '__main__':
    main()
