"""Collect heterogeneous native TRAIN objectives; these are not comparable L1s."""
import argparse
import hashlib
import json
from pathlib import Path


def read_window(path, field, end, rank=None):
    values = {}; digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            update = row.get('updates', row.get('update'))
            if update > end:
                break
            if end-500 < update:
                if update in values or (rank is not None and row['rank'] != rank):
                    raise ValueError('Duplicate/misowned effective update')
                value = row
                for key in field:
                    value = value[key]
                values[update] = float(value); digest.update(line)
    if sorted(values) != list(range(end-499, end+1)):
        raise ValueError('Incomplete 500-update window')
    return dict(path=str(path), end_update=end, updates=500,
                selected_records_sha256=digest.hexdigest(), mean=sum(values.values())/500)


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--output', required=True)
    args = p.parse_args(); result = {}
    world = Path('/mnt/project/qwen-world-postcot-20261025/runs/IL_full_seed42')
    replicas = [read_window(world/f'metrics_rank{r}.jsonl', ['losses', 'x'], 53000) for r in range(8)]
    if len({r['mean'] for r in replicas}) != 1:
        raise ValueError('Already all-reduced World losses differ across ranks')
    result['world53'] = dict(objective='normalized clean-x L1, BEFORE coefficient10',
        mean=replicas[0]['mean'], reduction='Already all-reduced; use once, do not sum eight replicas', evidence=replicas)
    native = Path('/mnt/project/qwen-sourceflow-native-20261018/runs/IL_seed42_v2/invocations')
    rows = {}
    for path in native.glob('*/rank*.json'):
        invocation = json.loads(path.read_text())
        if invocation.get('end_update') == 98486:
            rank = int(path.stem[4:])
            if rank in rows:
                raise ValueError('Ambiguous native rank ancestry')
            rows[rank] = read_window(path.parent/f'metrics_rank{rank}.jsonl', ['loss'], 98000, rank)
    if set(rows) != set(range(8)):
        raise ValueError('Missing native effective rank journal')
    result['native_qwen98'] = dict(objective='time-weighted normalized velocity MSE',
        mean=sum(r['mean'] for r in rows.values())/8, reduction='equal-scene local rank means averaged across8', evidence=rows)
    path = Path('/mnt/project/action-video-foresight-artifacts/20261002/students/formal_S0_seed42_full100k_v2/steps.jsonl')
    row = read_window(path, ['raw_losses', 'ego_fm'], 100000)
    result['ddp_dino_s0_100k'] = dict(objective='normalized 8x4 velocity MSE', mean=row['mean'], evidence=row)
    result['warning'] = 'Different targets, units, action encodings and time distributions: numeric magnitudes MUST NOT be ranked as L1.'
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print({k: v['mean'] for k, v in result.items() if isinstance(v, dict)})


if __name__ == '__main__':
    main()
