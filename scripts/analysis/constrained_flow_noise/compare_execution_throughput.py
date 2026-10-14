"""Observed production throughput across a qualified execution-only continuation.

Uses consecutive archive intervals, including native update/IO time. It never
sums overlapping CPU/GPU timers or treats different scene windows as paired.
"""
import argparse
import json
from pathlib import Path
import statistics

from starVLA.rl.flow_grpo.transactions import atomic_json


def window(run, indices):
    records = []
    logs = []
    for rank in range(8):
        with (run / f'training_rank{rank}.jsonl').open() as stream:
            logs.append({x['update']: x for line in stream if line.endswith('\n')
                         for x in [json.loads(line)]})
    for index in indices:
        before = run / 'behavior_collections' / f'{index - 1:06d}.pt'
        after = run / 'behavior_collections' / f'{index:06d}.pt'
        updates = (index * 2 + 1, index * 2 + 2)
        if not all(u in rank for rank in logs for u in updates):
            raise ValueError(f'Collection {index} has not finished on all ranks')
        metadata = json.loads((run / 'collection_summaries' / f'{index:06d}.json').read_text())
        seconds = after.stat().st_mtime - before.stat().st_mtime
        if seconds <= 0:
            raise ValueError('Nonpositive archive interval')
        records.append(dict(collection=index, raw_scenes=metadata['scenes'],
                            accepted_scenes=logs[0][updates[0]]['accepted_groups'],
                            candidate_scores=metadata['candidates'], wall_seconds=seconds,
                            update_seconds=sum(max(rank[u]['seconds'] for rank in logs) for u in updates)))
    wall = sum(x['wall_seconds'] for x in records)
    raw = sum(x['raw_scenes'] for x in records)
    return dict(collections=len(records), raw_scenes=raw,
                accepted_scenes=sum(x['accepted_scenes'] for x in records),
                candidate_scores=sum(x['candidate_scores'] for x in records),
                wall_seconds=wall, raw_scenes_per_second=raw / wall,
                median_collection_interval_seconds=statistics.median(x['wall_seconds'] for x in records),
                seconds_per_raw_scene=wall / raw, records=records)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--first-new-collection', type=int, required=True)
    parser.add_argument('--before-count', type=int, default=64)
    parser.add_argument('--after-count', type=int, default=32)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    run = Path(args.run)
    # Exclude the boundary and next collection: these contain stop/save/start,
    # cold worker initialization and graph capture, not steady-state training.
    first = args.first_new_collection
    before = window(run, range(max(1, first - args.before_count - 1), first - 1))
    after = window(run, range(first + 2, first + 2 + args.after_count))
    result = dict(schema='cfn_production_execution_throughput_v1',
                  scientific_identity=json.loads((run / 'run_identity.json').read_text())['sha256'],
                  execution_identity=json.loads((run / 'execution_identity.json').read_text())['sha256'],
                  first_new_collection=first, before=before, after=after,
                  observed_raw_throughput_ratio=after['raw_scenes_per_second'] / before['raw_scenes_per_second'],
                  limitation='Different consecutive TRAIN windows and shared-host load; observational throughput, not a paired speed guarantee. Atomic archive mtime is coarse. Startup/checkpoint boundary excluded; report transition cost separately.')
    atomic_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('before', 'after')}, indent=2))


if __name__ == '__main__':
    main()
