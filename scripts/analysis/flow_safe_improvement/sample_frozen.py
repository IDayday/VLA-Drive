"""Generate actual original ODE/SDE samples with per-scene atomic resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import uuid

import torch

from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy, sample_scene


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    p.add_argument('--population', required=True)
    p.add_argument('--cohort', choices=['reference_screen_tokens', 'diagnostic_random_tokens', 'diagnostic_tokens', 'calibration_tokens', 'dev_tokens'], required=True)
    p.add_argument('--cohorts', help='Locked directed/random TRAIN cohort manifest, required for diagnostic_tokens')
    p.add_argument('--mode', choices=['ode', 'legacy_sde'], default='ode')
    p.add_argument('--role', choices=['reference', 'evaluation', 'sampler_gap'], default='reference')
    p.add_argument('--behavior')
    p.add_argument('--behavior-sha256')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-for-qualification', type=int)
    a = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    configure_numerics()
    manifest = json.loads(Path(a.population).read_text())
    cohorts = json.loads(Path(a.cohorts).read_text()) if a.cohorts else None
    if a.cohort == 'diagnostic_tokens':
        if cohorts is None or cohorts['population_identity'] != manifest['identity_sha256']:
            raise ValueError('The full TRAIN diagnostic requires its locked cohort identity')
        tokens = cohorts['diagnostic_tokens']
        if len(tokens) != 512 or not set(tokens) <= set(manifest['train_tokens']):
            raise ValueError('Diagnostic inference population changed')
    else:
        tokens = manifest[a.cohort]
    assert tokens and not set(tokens).intersection(manifest['navtest_tokens'])
    seeds = manifest[{'reference': 'reference_seeds', 'evaluation': 'evaluation_seeds', 'sampler_gap': 'diagnostic_root_seeds'}[a.role]]
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    policy, store, cfg, sft, behavior_id = load_cached_policy(a.config, a.source_config,
                                                            behavior=a.behavior, behavior_sha256=a.behavior_sha256)
    from starVLA.rl.safe_flow_improvement import sampling, policy_adapter
    identity = {'schema': 'flow_safe_frozen_samples_v1', 'population_identity': manifest['identity_sha256'],
                'tokens_sha256': fingerprint(tokens), 'scenes': len(tokens), 'cohort': a.cohort,
                'SFT_sha256': cfg['checkpoint_contract']['sha256'], 'behavior_sha256': behavior_id,
                'feature_identity': store.identity_sha, 'seeds': seeds, 'mode': a.mode, 'role': a.role,
                'root_precision': 'float32_paired_diagnostic' if a.role == 'sampler_gap' else 'original_bfloat16',
                'source_files': {Path(path).name: file_sha(path) for path in (__file__, sampling.__file__, policy_adapter.__file__)}}
    if a.cohort == 'diagnostic_tokens':
        identity.update(explicit_tokens=tokens, cohort_identity=cohorts['sha256'], cohort_manifest_sha256=file_sha(a.cohorts))
    identity['sha256'] = fingerprint(identity)
    with (output / 'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = output / 'manifest.json'
        if target.exists():
            if json.loads(target.read_text()) != identity:
                raise ValueError('Resume changes model/population/seed/sampler/source identity')
            # All ranks of the first torchrun may observe rank0's manifest.
            if not a.resume and (output / f'rank{rank}.json').exists():
                raise ValueError('Existing samples require explicit --resume')
        else:
            atomic_json(target, identity)
    status = {'rank': rank, 'world_size': world, 'host': socket.gethostname(), 'pid': os.getpid(),
              'identity_sha256': identity['sha256'], 'status': 'RUNNING', 'new_scenes': 0,
              'reused_scenes': 0, 'new_candidates': 0, 'velocity_forward_calls': 0,
              'GPU_generation_seconds': 0., 'wall_seconds': 0.}
    started = time.monotonic()
    assigned = tokens[rank::world]
    for token in assigned:
        path = output / 'scenes' / token[:2] / (token + '.pt')
        seal = path.with_suffix('.json')
        if seal.exists():
            marker = json.loads(seal.read_text())
            if marker['identity_sha256'] != identity['sha256'] or marker['payload_sha256'] != file_sha(path):
                raise ValueError('Previously committed scene has changed')
            status['reused_scenes'] += 1
            continue
        t0 = time.monotonic()
        sampled = sample_scene(policy, store, token, seeds, mode=a.mode, role=a.role,
                               act_norm=int(sft.datasets.vla_data.act_norm))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        sampled.update(log_name=manifest['token_logs'][token], behavior_sha256=behavior_id,
                       identity_sha256=identity['sha256'], generation_seconds=elapsed)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.attempt-' + uuid.uuid4().hex)
        with temporary.open('xb') as stream:
            torch.save(sampled, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            # Keep an interrupted unsealed payload as evidence.
            path.rename(path.with_suffix('.unsealed-' + uuid.uuid4().hex))
        os.replace(temporary, path)
        atomic_json(seal, {'token': token, 'identity_sha256': identity['sha256'],
                           'payload_sha256': file_sha(path), 'candidates': len(seeds)})
        status['new_scenes'] += 1
        status['new_candidates'] += len(seeds)
        status['velocity_forward_calls'] += len(seeds) * 10
        status['GPU_generation_seconds'] += elapsed
        status['wall_seconds'] = time.monotonic() - started
        if status['new_scenes'] % 8 == 0:
            atomic_json(output / f'rank{rank}.json', status)
            print(json.dumps(status, sort_keys=True), flush=True)
        if a.stop_after_for_qualification and status['new_scenes'] >= a.stop_after_for_qualification:
            status['status'] = 'QUALIFICATION_WINDOW_COMPLETE_FULL_POPULATION_PENDING'
            break
    else:
        status['status'] = 'ASSIGNED_POPULATION_COMPLETE'
    status['wall_seconds'] = time.monotonic() - started
    status['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    atomic_json(output / f'rank{rank}.json', status)
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
