"""Score sealed policy samples, preserving every scene, label and simulation."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import fingerprint, file_sha
from starVLA.rl.safe_flow_improvement.v1_reward import V1RewardService, protocol_identity


def tensors(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensors(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--samples', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--cache-receipt', help='Wait for a log-complete cache receipt before opening its files')
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--wait-seconds', type=float, default=7200)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-for-qualification', type=int)
    a = p.parse_args()
    if not 1 <= a.workers <= 32:
        raise ValueError('Explicit bounded CPU worker count required')
    samples, output = Path(a.samples), Path(a.output)
    manifest = json.loads((samples / 'manifest.json').read_text())
    population = json.loads(Path(a.population).read_text())
    if population['identity_sha256'] != manifest['population_identity']:
        raise ValueError('Sample and population identities differ')
    tokens = population[manifest['cohort']]
    assert fingerprint(tokens) == manifest['tokens_sha256']
    spec = {'repo': str(Path(__file__).resolve().parents[3]), 'cache_roots': [a.cache_root], 'maps_root': '/mnt/navsim/maps'}
    protocol = protocol_identity(spec['repo'])
    from starVLA.rl.safe_flow_improvement import v1_reward
    identity = {'schema': 'flow_safe_scored_samples_v1', 'sample_manifest': manifest,
                'protocol_sha256': protocol['sha256'], 'script_sha256': file_sha(__file__),
                'reward_adapter_sha256': file_sha(v1_reward.__file__)}
    identity['sha256'] = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists():
        if not a.resume or json.loads(path.read_text()) != identity:
            raise ValueError('Explicit resume with unchanged scoring identity required')
    else:
        atomic_json(path, identity)
    pending = []
    reused = 0
    for token in tokens:
        rowpath = output / 'scenes' / token[:2] / (token + '.pt')
        seal = rowpath.with_suffix('.json')
        if seal.exists():
            marker = json.loads(seal.read_text())
            if marker['identity_sha256'] != identity['sha256'] or marker['payload_sha256'] != file_sha(rowpath):
                raise ValueError('Committed score identity/content changed')
            reused += 1
        else:
            pending.append(token)
    status = {'status': 'RUNNING', 'identity_sha256': identity['sha256'], 'requested_scenes': len(tokens),
              'reused_scenes': reused, 'new_scenes': 0, 'actual_new_scoring_calls': 0,
              'CPU_scoring_seconds_sum': 0., 'errors': [], 'workers': a.workers}
    started, last_progress = time.monotonic(), time.monotonic()
    try:
        with V1RewardService(spec, a.workers) as service:
            while pending:
                cache_ready = None
                if a.cache_receipt:
                    receipt = json.loads(Path(a.cache_receipt).read_text())
                    if receipt['status'].startswith('FAILED'):
                        raise RuntimeError('Cache builder failed; stop affected scoring')
                    cache_ready = {row['token'] for log in receipt['logs'] for row in log['entries']}
                ready = []
                for token in pending:
                    source = samples / 'scenes' / token[:2] / (token + '.pt')
                    seal = source.with_suffix('.json')
                    if seal.exists() and (cache_ready is None or token in cache_ready):
                        marker = json.loads(seal.read_text())
                        if marker['identity_sha256'] != manifest['sha256'] or marker['payload_sha256'] != file_sha(source):
                            raise ValueError('Input sample publication changed')
                        row = torch.load(source, map_location='cpu', weights_only=True)
                        ready.append((token, row, marker['payload_sha256']))
                        if len(ready) >= a.workers * 2:
                            break
                if not ready:
                    if time.monotonic() - last_progress > a.wait_seconds:
                        raise TimeoutError('Required samples/cache incomplete; missing scenes remain in denominator')
                    status.update(status='WAITING_FOR_SEALED_SAMPLES_OR_CACHE', pending_scenes=len(pending),
                                  wall_seconds=time.monotonic() - started)
                    atomic_json(output / 'status.json', status)
                    time.sleep(5)
                    continue
                jobs = [(token, row['log_name'], row['requested_trajectories'].numpy()) for token, row, _ in ready]
                scored = service.score(jobs)
                for (token, row, input_sha), feedback in zip(ready, scored):
                    if len(feedback) != len(row['normalized_actions']):
                        raise ValueError('Scorer returned fewer candidates; no denominator exclusion')
                    payload = {'sample': row, 'feedback': tensors(feedback), 'identity_sha256': identity['sha256'],
                               'sample_payload_sha256': input_sha}
                    target = output / 'scenes' / token[:2] / (token + '.pt')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.attempt-' + uuid.uuid4().hex)
                    with temporary.open('xb') as stream:
                        torch.save(payload, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if target.exists():
                        target.rename(target.with_suffix('.unsealed-' + uuid.uuid4().hex))
                    os.replace(temporary, target)
                    atomic_json(target.with_suffix('.json'), {'token': token, 'identity_sha256': identity['sha256'],
                                                              'payload_sha256': file_sha(target), 'candidates': len(feedback)})
                    pending.remove(token)
                    status['new_scenes'] += 1
                    status['actual_new_scoring_calls'] += len(feedback)
                    status['CPU_scoring_seconds_sum'] += sum(x['scoring_seconds'] for x in feedback)
                last_progress = time.monotonic()
                status.update(status='RUNNING', pending_scenes=len(pending), wall_seconds=time.monotonic() - started)
                atomic_json(output / 'status.json', status)
                print(json.dumps(status, sort_keys=True), flush=True)
                if a.stop_after_for_qualification and status['new_scenes'] >= a.stop_after_for_qualification:
                    status['status'] = 'QUALIFICATION_WINDOW_COMPLETE_FULL_POPULATION_PENDING'
                    break
            else:
                status['status'] = 'COMPLETE'
    except Exception as error:
        status.update(status='FAILED_SCENES_RETAINED_IN_DENOMINATOR', pending_scenes=len(pending))
        status['errors'].append({'type': type(error).__name__, 'message': str(error)})
        atomic_json(output / 'status.json', status)
        raise
    status['wall_seconds'] = time.monotonic() - started
    atomic_json(output / 'status.json', status)
    if not pending:
        atomic_json(output / 'COMPLETE', {'identity_sha256': identity['sha256'], 'scenes': len(tokens),
                                         'candidates': len(tokens) * len(manifest['seeds'])})


if __name__ == '__main__':
    main()
