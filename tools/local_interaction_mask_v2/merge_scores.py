"""Merge disjoint CPU log partitions without dropping failures or mixing export identities."""
import argparse
import csv
import json
import math
from pathlib import Path
from tools.local_interaction_mask_v2.score_async import digest, atomic_json
from tools.local_interaction_mask_v2.compare_pdms import METRICS, read


def merge_population(index, groups):
    expected = {row['token']: row['log'] for row in index}
    if not expected or len(expected) != len(index):
        raise ValueError('Empty or duplicate requested scoring population')
    result = {}
    for rows in groups:
        for token, row in rows.items():
            if token in result: raise ValueError('Overlapping score partitions')
            if token not in expected or row['log'] != expected[token]: raise ValueError('Unexpected score token/log')
            if row['status'] not in ('ok', 'failed'): raise ValueError('Unknown score status')
            normalized = dict(row)
            for metric in METRICS:
                value = float(row.get(metric) or 0.) if row['status'] == 'ok' else 0.
                if not math.isfinite(value) or not -1e-7 <= value <= 1 + 1e-7:
                    raise ValueError('Invalid PDM factor')
                normalized[metric] = value
            result[token] = normalized
    if set(result) != set(expected): raise ValueError('Missing requested score rows')
    return [result[row['token']] for row in index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', nargs='+', required=True)
    for key in ('index', 'predictions', 'output'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--export-shards', type=int, required=True)
    parser.add_argument('--benchmark-navtest', action='store_true')
    args = parser.parse_args()
    index = json.loads(Path(args.index).read_text()); bank = Path(args.predictions)
    logs = sorted({row['log'] for row in index})
    if args.benchmark_navtest and (len(index) != 12146 or len(logs) != 136):
        raise ValueError('Incomplete full Navtest endpoint')
    groups, identities, source_csv = [], [], {}
    for part in map(Path, args.parts):
        identity = json.loads((part / 'identity.json').read_text()); arguments = identity['arguments']
        if identity['index_sha256'] != digest(args.index): raise ValueError('Score index differs')
        if Path(arguments['predictions']).resolve() != bank.resolve(): raise ValueError('Mixed proposal banks')
        if arguments['export_shards'] != args.export_shards: raise ValueError('Export shard count differs')
        identities.append({key: value for key, value in identity.items() if key != 'arguments'})
        rows = read(part / 'scenes.csv')
        assigned = set(logs[arguments['log_shard']::arguments['log_shards']])
        if set(rows) != {row['token'] for row in index if row['log'] in assigned}:
            raise ValueError('CPU partition does not cover its requested logs')
        groups.append(rows); source_csv[str(part / 'scenes.csv')] = digest(part / 'scenes.csv')
    if any(identity != identities[0] for identity in identities): raise ValueError('Mixed evaluator versions/protocols')
    export_identities = []
    for shard in range(args.export_shards):
        identity = json.loads((bank / f'identity_{shard}.json').read_text())
        status = json.loads((bank / f'shard_{shard}.json').read_text())
        if identity['shard'] != shard or identity['shards'] != args.export_shards or identity['expected_total'] != len(index):
            raise ValueError('Export population differs')
        if status['status'] != 'complete' or status['completed'] != len(index[shard::args.export_shards]):
            raise ValueError('Incomplete GPU export shard')
        export_identities.append({key: value for key, value in identity.items() if key != 'shard'})
    if any(identity != export_identities[0] for identity in export_identities):
        raise ValueError('Mixed model/config/source/noise within export bank')
    rows = merge_population(index, groups)
    for row in rows:
        if row['variant'] != bank.name: raise ValueError('Mixed score variants')
        exported = json.loads((bank / 'predictions' / (row['token'] + '.json')).read_text())
        if row['status'] == 'ok' and (exported['status'] != 'ok' or row['proposal_sha256'] != exported['proposal_sha256']):
            raise ValueError('Scored proposal provenance differs')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (output / 'scenes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, keys); writer.writeheader(); writer.writerows(rows)
    failed = sum(row['status'] != 'ok' for row in rows)
    summary = {'scenes': len(rows), 'logs': len(logs), 'failed': failed, 'valid': failed == 0,
        'PDMS': sum(row['score'] for row in rows) / len(rows), 'zero_fraction': sum(row['score'] == 0 for row in rows) / len(rows),
        'metrics': {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS},
        'full_navtest': args.benchmark_navtest, 'source_csv_sha256': source_csv,
        'export_identity': export_identities[0], 'evaluator_identity': identities[0],
        'failure_policy': 'all requested rows retained; failures zero and invalidate benchmark'}
    atomic_json(output / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('scenes', 'logs', 'failed', 'PDMS', 'valid')}), flush=True)
    if failed: raise RuntimeError('Failed scenes retained; merged benchmark invalid')


if __name__ == '__main__': main()
