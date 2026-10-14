"""Archive the completed fixed-C1 Euler sweep, with anonymized numeric scores only."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / 'evidence/fm_steps/RESULTS.json').read_text())
    with (root / 'evidence/fm_steps/RESULTS.csv').open() as stream:
        reported = {int(r['steps']): r for r in csv.DictReader(stream)}
    metrics = {'PDMS': 'score', 'NC': 'no_at_fault_collisions', 'DAC': 'drivable_area_compliance',
               'TTC': 'time_to_collision_within_bound', 'EP': 'ego_progress', 'Comfort': 'comfort',
               'Direction': 'driving_direction_compliance'}
    def anon(kind, value):
        return hashlib.sha256(f'cs-planning-results-v1:{kind}:navtest:{value}'.encode()).hexdigest()[:24]
    inventory, population = [], None
    path = root / 'data/C1_FM_SWEEP_SCENES.csv.gz'
    with path.open('xb') as binary:
        with gzip.GzipFile(filename='', fileobj=binary, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['steps', 'scene_id', 'log_id', *metrics, 'status'])
                writer.writeheader()
                for steps in reported:
                    p = args.source / f'steps_{steps:02d}' / 'scores'
                    raw = (p / 'scenes.csv').read_bytes()
                    values = list(csv.DictReader(io.StringIO(raw.decode())))
                    summary = json.loads((p / 'summary.json').read_text())
                    assert summary['valid'] and summary['failed'] == 0
                    assert summary['scenes'] == len(values) == 12146 and summary['logs'] == 136
                    assert summary['export_identity']['checkpoint']['sha256'] == results['checkpoint']['sha256']
                    membership = {r['token']: r['log'] for r in values}
                    assert len(membership) == 12146 and len(set(membership.values())) == 136
                    if population is None:
                        population = membership
                    assert membership == population
                    for name, metric in metrics.items():
                        average = float(np.mean([float(r[metric]) for r in values])) * 100
                        column = 'PDMS' if name == 'PDMS' else metric
                        assert abs(average - float(reported[steps][column])) < 1e-9
                    assert sum(float(r['score']) == 0 for r in values) == int(reported[steps]['zero_scenes'])
                    for r in values:
                        assert r['status'] == 'ok'
                        writer.writerow({'steps': steps, 'scene_id': anon('scene', r['token']),
                                         'log_id': anon('log', r['log']), 'status': r['status'],
                                         **{k: float(r[v]) * 100 for k, v in metrics.items()}})
                    inventory.append({'steps': steps, 'source': str(p),
                                      'scene_csv_sha256': hashlib.sha256(raw).hexdigest(),
                                      'summary_sha256': hashlib.sha256((p / 'summary.json').read_bytes()).hexdigest(),
                                      'scenes': 12146, 'logs': 136, 'failed': 0})
    (root / 'data/C1_FM_SWEEP_SOURCES.json').write_text(json.dumps({
        'checks': 'Full CSV means, zero counts, identical populations and checkpoint receipt; no weights loaded',
        'sources': inventory, 'scene_rows': 9 * 12146,
        'scope': 'Different inference steps at one C1 checkpoint, not nine training runs; 10-step bank repeats the historical baseline.'}, indent=2) + '\n')
    print(json.dumps({'banks': len(inventory), 'rows': 9 * 12146}))


if __name__ == '__main__':
    main()
