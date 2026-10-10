"""Read completed native planning records; report both time aggregations.

No model inference or evaluator modification is performed. Original metric
arrays, masks, collision exclusions and scene weights remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root, expected_samples, expected_scenes):
    complete_path = root / 'COMPLETE.json'
    complete_raw = complete_path.read_bytes()
    complete = json.loads(complete_raw)
    files = sorted((root / 'records').glob('*.json'))
    if len(files) != expected_samples or complete['scenes'] != expected_samples:
        raise ValueError(f'Incomplete or different population: {root}')
    variants = ('q0', 'q_final')
    names = {'L2': 'L2', 'obj_box_col': 'box_collision', 'obj_col': 'point_collision'}
    sums = {v: {k: np.zeros(6, dtype=np.float64) for k in names} for v in variants}
    tokens, scenes, record_digest = set(), set(), hashlib.sha256()
    for path in files:
        raw = path.read_bytes()
        record = json.loads(raw)
        if record['identity'] != complete['identity'] or record['token'] != path.stem:
            raise ValueError(f'Foreign record identity: {path}')
        if record['token'] in tokens:
            raise ValueError(f'Duplicate token: {path}')
        tokens.add(record['token'])
        scenes.add(record['scene'])
        record_digest.update(path.name.encode())
        record_digest.update(hashlib.sha256(raw).digest())
        for variant in variants:
            for name in names:
                values = np.asarray(record['metrics'][variant]['per_timestep'][name], dtype=np.float64)
                if values.shape != (6,) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid six-step metrics: {path}')
                sums[variant][name] += values
    if len(scenes) != expected_scenes:
        raise ValueError(f'Different official scene population: {root}')
    result = {
        'planning_root': str(root.resolve()),
        'original_identity': complete['identity'],
        'source_complete_sha256': hashlib.sha256(complete_raw).hexdigest(),
        'source_records_digest': record_digest.hexdigest(),
        'samples': expected_samples, 'scenes': expected_scenes,
        'original_protocol': complete['protocol'],
    }
    for variant in variants:
        result[variant] = {'prefix': {}, 'endpoint': {}}
        for name, metric in names.items():
            values = sums[variant][name] / expected_samples
            prefix = [float(values[:end].mean()) for end in (2, 4, 6)]
            endpoint = values[[1, 3, 5]].tolist()
            for seconds, value in zip((1, 2, 3), prefix):
                if abs(value - complete[variant][f'{metric}_{seconds}s']) > 1e-7:
                    raise ValueError(f'Saved prefix result mismatch: {root}, {variant}, {metric}')
            for aggregation, values in (('prefix', prefix), ('endpoint', endpoint)):
                result[variant][aggregation][metric] = {
                    'at_1_2_3s': values, 'average': float(np.mean(values)),
                    'unit': 'm' if name == 'L2' else 'fraction (multiply by 100 for percent)',
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning-root', type=Path, action='append', required=True)
    parser.add_argument('--expected-samples', type=int, default=4969)
    parser.add_argument('--expected-scenes', type=int, default=150)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [summarize(root, args.expected_samples, args.expected_scenes) for root in args.planning_root]
    output = {
        'scope': 'Read-only supplemental aggregation of completed native metric arrays',
        'prefix': 'Mean first 2/4/6 positions, then mean three horizons',
        'endpoint': 'Position indices 1/3/5, then mean three horizons',
        'collision_core': 'Unchanged locked native evaluator; time aggregation alone does not establish full ST-P3 protocol equality',
        'summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'observations': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'completed_observations': len(results)}))


if __name__ == '__main__':
    main()
