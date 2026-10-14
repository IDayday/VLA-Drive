"""Read completed native head checkpoints and export small reproducibility receipts.

This does not resume a trainer or export model tensors. It verifies the fixed
full-population endpoint, native continuation fields, and uninterrupted log IDs.
"""
import argparse
import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, identity, read, source_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    jobs = [j for j in read(a.artifacts/'HEAD_TASKS.json')['jobs'] if j['scope'] == 'formal']
    jobs += read(a.artifacts/'information_seed43_v1/HEAD_TASKS.json')['jobs']
    if len(jobs) != 17 or len({j['output'] for j in jobs}) != 17:
        raise ValueError('Expected the registered 17 distinct primary/repeated heads')
    rows = []
    for job in jobs:
        root = Path(job['output'])
        complete, meta = read(root/'COMPLETE.json'), read(root/'identity.json')
        checkpoint = root/'epoch_24.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        required = {'model', 'optimizer', 'scheduler', 'rng', 'order', 'identity',
                    'completed', 'epoch', 'offset', 'exposure', 'GPU_hours'}
        if not required.issubset(saved) or saved['identity'] != meta:
            raise ValueError('Missing native continuation state or changed identity')
        expected = {'completed': math.ceil(101592/128)*24, 'epoch': 24,
                    'offset': 0, 'exposure': 101592*24}
        if any(saved[k] != v for k, v in expected.items()):
            raise ValueError('Incomplete common full-population endpoint')
        if any(complete[k] != v for k, v in expected.items() if k != 'offset'):
            raise ValueError('Completion receipt disagrees with checkpoint')
        if not complete['full_training_population'] or meta['frozen_base_optimizer_updates']:
            raise ValueError('Wrong population or foundation was trained')
        if saved['scheduler']['last_epoch'] != saved['completed']:
            raise ValueError('Scheduler boundary mismatch')
        if not torch.equal(saved['order'].sort().values, torch.arange(101592)):
            raise ValueError('Saved data order is not the full registered population')
        inventory = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                     for k, v in saved['model'].items()}
        if not all(torch.isfinite(v).all() for v in saved['model'].values()):
            raise ValueError('Nonfinite checkpoint tensor')
        steps = sorted({int(v['step']) for v in saved['optimizer']['state'].values()})
        if steps != [saved['completed']]:
            raise ValueError('Optimizer update boundary mismatch')
        # One record per actual optimizer update, including across native resumes.
        import json
        with (root/'train.jsonl').open() as stream:
            logs = [json.loads(line) for line in stream if line.strip()]
        key = 'update' if 'update' in logs[0] else 'completed'
        if [row[key] for row in logs] != list(range(1, saved['completed']+1)):
            raise ValueError('Duplicate, missing or discontinuous training update IDs')
        rows.append({'run_id': job['id'], 'output': str(root), 'identity': meta,
            'complete': complete, 'checkpoint': str(checkpoint),
            'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': digest(checkpoint),
            'native_state_keys': sorted(saved), 'rng_keys': sorted(saved['rng']),
            'parameter_inventory_sha256': identity(inventory), 'model_state_tensors': len(inventory),
            'optimizer_state_tensors': len(saved['optimizer']['state']), 'optimizer_steps': steps,
            'optimizer_group_lr': [g['lr'] for g in saved['optimizer']['param_groups']],
            'scheduler_last_epoch': saved['scheduler']['last_epoch'], 'data_order_exact_permutation': True,
            'training_records': len(logs), 'training_log_sha256': digest(root/'train.jsonl'),
            'attempts': [read(x) for x in sorted(root.glob('attempt_*.json'))],
            'command': job['command']})
        del saved, logs
    atomic_json(a.output, {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'scope': 'Native final head checkpoints; no weights uploaded and no resumed training',
        'base_training_seeds': 1, 'formal_head_runs': len(rows), 'runs': rows,
        'limitations': 'State receipt complements existing strict-load and A/A-versus-resume evidence; it is not a new long-run numerical-equivalence claim.'})


if __name__ == '__main__':
    main()
