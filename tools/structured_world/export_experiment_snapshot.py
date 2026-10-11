"""Export completed checkpoints and metrics without model execution or raw data.

Training curves stop at each captured native checkpoint. Live log tails are
excluded; resumed duplicate update numbers use the last logged record. Scene
tokens remain local and are used only to validate paired NAVSIM populations.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np


METRICS = {'PDMS': 'score', 'NC': 'no_at_fault_collisions',
           'DAC': 'drivable_area_compliance', 'TTC': 'time_to_collision_within_bound',
           'EP': 'ego_progress', 'comfort': 'comfort'}


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scene_csv(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1696 and len({r['token'] for r in rows}) == 1696,
            'NAVSIM dev population differs: '+str(path))
    require(all(r['status'] == 'ok' for r in rows), 'Failed evaluation rows: '+str(path))
    table = {r['token']: r for r in rows}
    require(len({r['log'] for r in rows}) == 16, 'Expected sixteen NAVSIM dev logs')
    for row in rows:
        require(all(np.isfinite(float(row[key])) and 0 <= float(row[key]) <= 1
                    for key in METRICS.values()), 'Invalid planning metrics')
    return table


def paired(a, b, name, rng, replicates):
    require(set(a) == set(b) and all(a[t]['log'] == b[t]['log'] for t in a),
            'Paired scene/log populations differ')
    tokens = sorted(a)
    logs = sorted({a[t]['log'] for t in tokens})
    sizes = np.array([sum(a[t]['log'] == log for t in tokens) for log in logs])
    draws = rng.integers(0, len(logs), size=(replicates, len(logs)))
    result = {'comparison': name, 'scenes': len(tokens), 'logs': len(logs), 'metrics': {}}
    for metric, key in METRICS.items():
        delta = np.array([float(a[t][key])-float(b[t][key]) for t in tokens])
        sums = np.array([sum(d for t, d in zip(tokens, delta) if a[t]['log'] == log) for log in logs])
        boot = sums[draws].sum(1)/sizes[draws].sum(1)*100
        result['metrics'][metric] = {'delta_points': float(delta.mean()*100),
            'paired_log_95CI': np.quantile(boot, [.025, .975]).tolist()}
    az = np.array([float(a[t]['score']) == 0 for t in tokens])
    bz = np.array([float(b[t]['score']) == 0 for t in tokens])
    result['zero_failure_transfers'] = {'fixed_baseline_zero_scenes': int((bz & ~az).sum()),
        'introduced_zero_scenes': int((~bz & az).sum()), 'both_zero': int((bz & az).sum())}
    result['interpretation'] = 'Same-model or overall-recipe diagnostic; not isolated auxiliary-supervision causality'
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--curve-interval', type=int, default=100)
    parser.add_argument('--navsim-compare-update', type=int, default=50000)
    parser.add_argument('--navsim-reference', action='append', default=[], help='NAME=path/to/scenes.csv at the comparison update')
    parser.add_argument('--bootstrap-replicates', type=int, default=20000)
    args = parser.parse_args()
    require(args.curve_interval > 0 and args.bootstrap_replicates >= 1000, 'Invalid export interval/bootstrap size')
    require(not args.output.exists(), 'Snapshot output already exists')
    curves, planning, fields, runs, pairs, tables = [], [], [], [], [], {}
    references = []
    for run in args.run:
        identity, identity_sha = read(run/'identity.json')
        dataset, group = identity['dataset'], identity['config']['structured_world']['group']
        require(dataset in ('navsim', 'nuscenes') and identity['scope'] == 'formal', 'Formal two-dataset runs only')
        tag = (run/'checkpoints/latest').read_text().strip()
        require(re.fullmatch(r'(latest|milestone)_\d{7}', tag), 'Invalid checkpoint tag')
        checkpoint, checkpoint_sha = read(run/'checkpoints'/tag/'COMPLETE.json')
        require(checkpoint['identity'] == identity['identity'] and checkpoint['ranks'] == identity['world_size']
                and checkpoint['completed'] == int(tag.split('_')[1]), 'Native checkpoint identity differs')
        raw = (run/'metrics.jsonl').read_bytes()
        rows, digest = {}, hashlib.sha256()
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row['updates'] <= checkpoint['completed']:
                rows[row['updates']] = row
                digest.update(line)
        require(rows and max(rows) == checkpoint['completed'], 'Checkpoint has no matching logged update')
        first_step, last_step = min(rows), max(rows)
        for step, row in sorted(rows.items()):
            if step % args.curve_interval and step not in (first_step, last_step) and step not in identity['observation_plan']:
                continue
            curves.append({'dataset': dataset, 'group': group, 'updates': step,
                'fractional_epochs': row['fractional_epochs'], 'scene_exposure': row['scene_exposure'],
                'GPU_hours': row['GPU_hours'], 'step_seconds': row['seconds'],
                'global_gradient_norm': row['global_gradient_norm'], 'gradient_clipped': row['gradient_clipped'],
                'actual_parameter_group_lrs_json': json.dumps(row['actual_parameter_group_lrs']),
                'weighted_losses_json': json.dumps(row['weighted_losses'], sort_keys=True, allow_nan=False)})
        runs.append({'dataset': dataset, 'group': group, 'native_run': str(run),
            'training_source_sha': identity['training_source_sha'], 'native_identity': identity['identity'],
            'identity_sha256': identity_sha, 'checkpoint_tag': tag, 'checkpoint_complete_sha256': checkpoint_sha,
            'checkpoint_updates': checkpoint['completed'], 'registered_endpoint': identity['schedule_horizon'],
            'training_endpoint_reached': checkpoint['completed'] == identity['schedule_horizon'],
            'logged_prefix_sha256': digest.hexdigest(), 'logged_prefix_unique_updates': len(rows),
            'scene_exposure': rows[max(rows)]['scene_exposure'], 'logged_training_GPU_hours': rows[max(rows)]['GPU_hours'],
            'shared_geometry_identity': identity['shared_geometry_identity']})
        expected = 1696 if dataset == 'navsim' else 4969
        pattern = 'dev_*_canonical' if dataset == 'navsim' else 'val_*_planning'
        for folder in sorted(run.glob(pattern)):
            if not (folder/'COMPLETE.json').exists():
                continue
            update = int(folder.name.split('_')[1])
            require(update <= checkpoint['completed'], 'Evaluation exceeds native training snapshot')
            complete, complete_sha = read(folder/'COMPLETE.json')
            protocol, protocol_sha = read(folder/'identity.json')
            require(complete['identity'] == protocol['identity'], 'Evaluation identity differs')
            references.append({'kind': 'planning', 'path': str(folder), 'complete_sha256': complete_sha,
                               'identity_sha256': protocol_sha})
            for variant in ('q0', 'q_final'):
                values = complete[variant]
                require((values['scenes'] if dataset == 'navsim' else complete['scenes']) == expected,
                        'Evaluation population differs')
                row = {'dataset': dataset, 'group': group, 'updates': update, 'variant': variant,
                       'samples': expected, 'evaluation_identity': complete['identity']}
                if dataset == 'navsim':
                    require(values['valid'] and values['failed'] == 0, 'Invalid NAVSIM canonical result')
                    row.update({k: float(values['metrics'][v])*100 for k, v in METRICS.items()})
                    row['zero_fraction'] = values['zero_fraction']
                    if update == args.navsim_compare_update:
                        tables[group+'/'+variant] = scene_csv(folder/variant/'scenes.csv')
                else:
                    for seconds in (1, 2, 3):
                        row['L2_prefix_'+str(seconds)+'s_m'] = values['L2_'+str(seconds)+'s']
                        row['box_collision_prefix_'+str(seconds)+'s_percent'] = values['box_collision_'+str(seconds)+'s']*100
                    row['L2_prefix_average_m'] = float(np.mean([values['L2_'+str(s)+'s'] for s in (1, 2, 3)]))
                    row['box_collision_prefix_average_percent'] = float(np.mean([values['box_collision_'+str(s)+'s'] for s in (1, 2, 3)]))*100
                planning.append(row)
        for folder in sorted(run.glob('*_scene_metrics')):
            if not (folder/'COMPLETE.json').exists():
                continue
            update = int(folder.name.split('_')[1])
            require(update <= checkpoint['completed'], 'Scene evaluation exceeds native checkpoint')
            complete, complete_sha = read(folder/'COMPLETE.json')
            summary, summary_sha = read(folder/'SCENE_METRICS.json')
            protocol, protocol_sha = read(folder/'identity.json')
            require(complete['identity'] == summary['identity'] == protocol['identity'] and
                    complete['scenes'] == summary['scenes'] == expected, 'Scene-field identity/population differs')
            fields.append({'dataset': dataset, 'group': group, 'updates': update,
                'source_complete_sha256': complete_sha, 'source_summary_sha256': summary_sha,
                'source_protocol_sha256': protocol_sha, 'protocol': protocol, 'summary': summary})
    for reference in args.navsim_reference:
        name, path = reference.split('=', 1)
        table = scene_csv(path)
        tables[name] = table
        references.append({'kind': 'historical_NAVSIM', 'name': name, 'updates': args.navsim_compare_update,
                           'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        planning.append({'dataset': 'navsim', 'group': name, 'updates': args.navsim_compare_update,
            'variant': 'historical_final', 'samples': len(table),
            **{k: float(np.mean([float(r[v]) for r in table.values()]))*100 for k, v in METRICS.items()},
            'zero_fraction': sum(float(r['score']) == 0 for r in table.values())/len(table)})
    rng = np.random.default_rng(42)
    for name, table in tables.items():
        if not name.endswith('/q_final'):
            continue
        for baseline, other in tables.items():
            if baseline == name or ('/' in baseline and baseline != name.replace('/q_final', '/q0')):
                continue
            pairs.append(paired(table, other, name+' minus '+baseline, rng, args.bootstrap_replicates))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'TRAINING_CURVES.csv', curves)
    write_csv(args.output/'PLANNING_RESULTS.csv', planning)
    manifest = {'schema': 'structured_world_experiment_snapshot_v1',
        'captured_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_curve_interval': args.curve_interval, 'runs': runs, 'evaluation_sources': references,
        'bootstrap': {'seed': 42, 'replicates': args.bootstrap_replicates, 'unit': 'paired log clusters, actual scene weights'},
        'limits': ['Development observations, not NAVSIM Navtest', 'No new inference or evaluator changes',
                   'Historical architecture and compute differ; not isolated supervision ablations',
                   'Training seed42 only; bootstrap does not replace a second training seed',
                   'Scene metrics visibility is calibration/ROI support, not annotated object visibility']}
    for name, value in [('SCENE_FIELD_METRICS.json', fields), ('NAVSIM_PAIRED_COMPARISONS.json', pairs), ('MANIFEST.json', manifest)]:
        with (args.output/name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'output': str(args.output), 'runs': len(runs), 'planning_rows': len(planning),
                      'scene_observations': len(fields), 'training_curve_rows': len(curves)}))


if __name__ == '__main__':
    main()
