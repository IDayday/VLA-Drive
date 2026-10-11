"""Analyze complete canonical Navtest scores using existing paired diagnostics.

This consumes scored trajectories, never loads a model or changes evaluation.
Scene identifiers remain in controlled storage; only aggregates are exported.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from export_experiment_snapshot import METRICS, paired, require
from analyze_navsim_failure_transfer import compare, zero_patterns


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scenes(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 12146 and len({r['token'] for r in rows}) == 12146,
            'Full 12146-scene Navtest population required: '+str(path))
    require(len({r['log'] for r in rows}) == 136, 'Full 136-log Navtest population required')
    require(all(r['status'] == 'ok' for r in rows), 'Failed canonical rows: '+str(path))
    for row in rows:
        require(bool(row['log']), 'Missing cluster identity')
        require(all(np.isfinite(float(row[key])) and 0 <= float(row[key]) <= 1
                    for key in (*METRICS.values(), 'driving_direction_compliance')),
                'Invalid canonical metrics')
    return {r['token']: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--reference', action='append', default=[],
                        help='NAME=path/to/scenes.csv, same Navtest protocol and updates')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=20000)
    args = parser.parse_args()
    require(not args.output.exists(), 'Aggregate output already exists')
    require(args.bootstrap_replicates >= 1000, 'Insufficient bootstrap replicates')
    identity = json.loads((args.scores/'identity.json').read_text())
    complete = json.loads((args.scores/'COMPLETE.json').read_text())
    require(identity['population'] == 'navtest' and complete['identity'] == identity['identity'],
            'Foreign or non-Navtest canonical completion')
    for variant in ('q0', 'q_final'):
        require(complete[variant]['scenes'] == 12146 and complete[variant]['valid']
                and complete[variant]['failed'] == 0, 'Incomplete or invalid canonical score')
    paths = {'same_model_q0': args.scores/'q0/scenes.csv'}
    for value in args.reference:
        name, separator, path = value.partition('=')
        require(bool(separator) and bool(name) and name not in paths, 'Invalid or duplicate reference')
        paths[name] = Path(path)
    candidate_path = args.scores/'q_final/scenes.csv'
    candidate = read_scenes(candidate_path)
    q0 = read_scenes(paths['same_model_q0'])
    for variant, rows in (('q0', q0), ('q_final', candidate)):
        for metric, key in METRICS.items():
            mean = sum(float(row[key]) for row in rows.values())/len(rows)
            require(abs(mean-complete[variant]['metrics'][key]) < 1e-10,
                    'Summary and scored population differ: '+variant+'/'+metric)
    zeros = [token for token, row in candidate.items() if float(row['score']) == 0]
    rng = np.random.default_rng(42)
    comparisons = []
    for name, path in paths.items():
        baseline = read_scenes(path)
        comparisons.append({'name': name,
                            'paired': paired(candidate, baseline, name, rng, args.bootstrap_replicates),
                            'failure_transfer': compare(candidate, baseline, name)})
    report = {'schema': 'structured_world_complete_navtest_analysis_v1',
              'population': 'navtest', 'scenes': len(candidate),
              'logs': len({row['log'] for row in candidate.values()}),
              'canonical_identity': identity['identity'],
              'canonical_complete_sha256': digest(args.scores/'COMPLETE.json'),
              'analyzer_sha256': digest(__file__),
              'scene_csv_sha256': {'q_final': digest(candidate_path),
                                   **{name: digest(path) for name, path in paths.items()}},
              'planning': {variant: complete[variant] for variant in ('q0', 'q_final')},
              'q_final_zero_scenes': len(zeros),
              'q_final_zero_component_patterns': zero_patterns(candidate, zeros),
              'comparisons': comparisons,
              'bootstrap': {'unit': 'log', 'scene_weighting': 'pooled sampled log sums / sampled scene counts',
                            'replicates': args.bootstrap_replicates, 'seed': 42},
              'limits': ['One training seed; inference seed and log bootstrap are not training replicates',
                         'Same-model q0 is not independently trained original-DDP baseline',
                         'Historical architecture and compute differ; matched updates do not isolate supervision',
                         'The user explicitly requested Navtest at intermediate observations; it is not an untouched final holdout',
                         'No raw scene identifiers, images or model weights are exported']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'scenes': len(candidate),
                      'PDMS': {variant: complete[variant]['metrics']['score']*100
                               for variant in ('q0', 'q_final')}}))


if __name__ == '__main__':
    main()
