"""Bounded read-only check of the first105 actual matched training updates."""
from pathlib import Path
import hashlib
import json
import math
import resource
import time

ROOT = Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs')
RUNS = ('C1_fulltrain_seed42_official_4096_v1',
        'C1_rollout_std_fulltrain_seed42_official_4096_v1')
STEPS = 105
READ_CAP = 8 * 2**20


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def process_memory():
    # Linux getrusage.ru_maxrss can retain an ancestor's high-water mark
    # across fork/exec. /proc/self/status reports this executable's high water.
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        name, _, value = line.partition(':')
        if name in ('VmRSS', 'VmHWM'):
            amount, unit = value.split()
            if unit != 'kB':
                raise ValueError('Unsupported /proc memory unit')
            fields[name] = int(amount) * 1024
    if set(fields) != {'VmRSS', 'VmHWM'}:
        raise ValueError('Missing process memory accounting')
    return {'peak_process_RSS_bytes': fields['VmHWM'],
            'current_process_RSS_bytes': fields['VmRSS'],
            'memory_source': '/proc/self/status VmHWM/VmRSS',
            'rusage_inherited_high_water_observation_bytes':
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'rusage_observation_is_isolated_process_peak': False}


def main():
    started = time.monotonic()
    cpu = time.process_time()
    used = 0
    rows = []
    refs = []
    for name in RUNS:
        path = ROOT / name / 'steps.jsonl'
        prefix = []
        with path.open('rb') as stream:
            for _ in range(STEPS):
                line = stream.readline(min(256 * 2**10, READ_CAP - used) + 1)
                if not line.endswith(b'\n') or len(line) > min(256 * 2**10, READ_CAP - used):
                    raise ValueError('Incomplete prefix or read cap exceeded')
                used += len(line)
                prefix.append(line)
        data = b''.join(prefix)
        refs.append({'path': str(path), 'prefix_bytes': len(data),
                     'prefix_sha256': hashlib.sha256(data).hexdigest(),
                     'scope': 'first105completeJSONLrows; source may append later'})
        rows.append([json.loads(line) for line in prefix])
    old, new = rows
    data_order = []
    lrs = []
    for step, (left, right) in enumerate(zip(old, new), 1):
        if left['step'] != step or right['step'] != step:
            raise ValueError('Missing/duplicate optimizer step')
        if left['lr'] != right['lr'] or left['exposure'] != right['exposure']:
            raise ValueError('Learning-rate time axis or exposure diverged')
        if left['counts'] != right['counts'] or right['exposure'] != step * 32:
            raise ValueError('Actual supervision/rollout exposure diverged')
        ids_old = [rank['sample_ids'] for rank in left['per_rank']]
        ids_new = [rank['sample_ids'] for rank in right['per_rank']]
        if ids_old != ids_new or len(ids_new) != 4 or any(len(ids) != 8 for ids in ids_new):
            raise ValueError('Real four-rank sample order differs')
        if (sum(rank['official']['valid_candidates'] for rank in right['per_rank']) != 128
                or right['official_admitted_calls'] != step * 128
                or right['fp32_master_update_verified'] is not True
                or not math.isfinite(right['gradient_norm'])
                or not all(math.isfinite(value) for value in right['losses'].values())):
            raise ValueError('Invalid actual feedback/update in running arm')
        data_order.append(ids_new)
        lrs.append(right['lr'])
    ids = [scene for batch in data_order for rank in batch for scene in rank]
    if len(ids) != len(set(ids)) or len(ids) != 3360:
        raise ValueError('Warmup population repeats unexpectedly')
    result = {'schema': 'st_rollout_scale_actual_warmup_audit_v1',
              'status': 'PASS_FIRST105_DATA_LR_EXPOSURE_AND_FINITE_UPDATES',
              'scope': 'TRAIN execution through warmup100 and five cosine steps only; not final4096 or planning quality',
              'sources': refs, 'steps_checked': STEPS, 'unique_scenes': len(ids),
              'data_order_sha256': digest(data_order), 'learning_rate_sequence_sha256': digest(lrs),
              'learning_rate_markers': {str(s): lrs[s - 1] for s in (1, 99, 100, 101, 105)},
              'scene_presentations': 3360, 'BC_repeat8_presentations': 26880,
              'valid_official_feedback': 13440, 'all_real_master_updates_verified': True,
              'reads_bytes': used, 'read_cap_bytes': READ_CAP,
              'wall_seconds': time.monotonic() - started,
              'cpu_seconds': time.process_time() - cpu,
              **process_memory(),
              'new_score_calls': 0, 'model_image_cache_or_DEV_reads': False,
              'training_modified': False, 'scientific_gain': 'UNTESTED'}
    target = Path(__file__).with_name('RESULT.json')
    with target.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
