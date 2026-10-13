"""Read-only round-one result consolidation; no models, jobs or scene IDs exported.

Uses the existing canonical CSV validator and log-cluster bootstrap. Native
weights are never loaded. A live training log is captured as a bounded prefix;
G1's resumed tail is selected using its recorded byte boundary, not max(step).
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_navsim_navtest import read_scenes
from analyze_navsim_failure_transfer import compare
from export_experiment_snapshot import METRICS, paired, require, write_csv


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new snapshot directory')
    repo = Path(__file__).resolve().parents[2]
    sources = {}

    def source(path):
        path = Path(path)
        if not path.is_absolute():
            path = repo / path
        raw = path.read_bytes()
        sources[str(path)] = {'bytes': len(raw), 'sha256': digest(raw)}
        return raw

    def read(path):
        return json.loads(source(path))

    spec = read(args.inputs.resolve())
    started = utc()
    queue = Path(spec['queue'])
    run_meta, training, identities = [], [], {}
    scene_fields = []
    for run in spec['runs']:
        root = Path(run['path'])
        ident = read(root / 'identity.json')
        require(ident['identity'] == run['expected_identity'], 'Wrong native run')
        dataset, group = ident['dataset'], ident['config']['structured_world']['group']
        identities[(dataset, group)] = ident
        tag = source(root / 'checkpoints/latest').decode().strip()
        checkpoint = read(root / 'checkpoints' / tag / 'COMPLETE.json')
        require(checkpoint['identity'] == ident['identity'] and checkpoint['ranks'] == 8,
                'Wrong checkpoint identity/rank count')
        log = root / 'metrics.jsonl'
        with log.open('rb') as stream:
            raw = stream.read(log.stat().st_size)
        # An incomplete trailing line belongs to a future snapshot.
        raw = raw[:raw.rfind(b'\n') + 1]
        require(bool(raw), 'Empty training log')
        boundary = 0
        if run.get('metrics_segments'):
            segments = read(run['metrics_segments'])
            require(segments['run_identity'] == ident['identity'], 'Wrong resume lineage')
            boundary = segments['resumed_segment_start_byte']
            require(boundary < len(raw) and raw[boundary - 1:boundary] == b'\n',
                    'Invalid recorded resume boundary')
        tail = raw[boundary:].splitlines()[-32:]
        rows = [json.loads(line) for line in tail]
        last = rows[-1]
        require(all(a['updates'] + 1 == b['updates'] for a, b in zip(rows, rows[1:])),
                'Non-contiguous current training tail')
        complete_path = root / 'TRAINING_COMPLETE.json'
        complete = read(complete_path) if complete_path.exists() else None
        if complete:
            require(complete['identity'] == ident['identity'] and
                    complete['updates'] == ident['schedule_horizon'] == last['updates'],
                    'Training completion differs from native log')
        require(checkpoint['completed'] <= last['updates'], 'Saved state exceeds active lineage')
        training.append({'dataset': dataset, 'group': group,
                         'observed_UTC': utc(), 'status': 'complete' if complete else run['status'],
                         'last_logged_update': last['updates'], 'saved_update': checkpoint['completed'],
                         'registered_endpoint': ident['schedule_horizon'],
                         'fractional_epochs': last['fractional_epochs'],
                         'accepted_lineage_scene_exposure': last['scene_exposure'],
                         'logged_GPU_hours_counter': last['GPU_hours'],
                         'last_32_updates_median_seconds': float(np.median([r['seconds'] for r in rows])),
                         'last_32_updates_P90_seconds': float(np.quantile([r['seconds'] for r in rows], .9)),
                         'window_updates': len(rows),
                         'actual_parameter_group_lrs_json': json.dumps(last['actual_parameter_group_lrs']),
                         'rank0_peak_allocated_GiB': last['peak_GPU_allocated_bytes_rank0'] / 2**30,
                         'checkpoint_tag': tag, 'host': run.get('host', ''),
                         'notes': run['notes']})
        run_meta.append({'path': str(root), 'native_identity': ident['identity'],
                         'dataset': dataset, 'group': group,
                         'contract': {k: v for k, v in ident.items() if k != 'config'},
                         'frozen_config': ident['config'],
                         'latest_checkpoint_complete': checkpoint,
                         'metrics_prefix_bytes': len(raw), 'metrics_prefix_sha256': digest(raw),
                         'metrics_last_segment_start_byte': boundary,
                         'tail_sha256': digest(b'\n'.join(tail) + b'\n')})
        for folder in sorted(root.glob('*_scene_metrics')):
            if not (folder / 'COMPLETE.json').exists():
                continue
            c, p, s = (read(folder / name) for name in
                       ('COMPLETE.json', 'identity.json', 'SCENE_METRICS.json'))
            require(c['identity'] == p['identity'] == s['identity'], 'Scene metric identity mismatch')
            expected = 1696 if dataset == 'navsim' else 4969
            require(c['scenes'] == s['scenes'] == expected, 'Scene metric population mismatch')
            scene_fields.append({'dataset': dataset, 'group': group,
                                 'updates': int(folder.name.split('_')[1]),
                                 'population': 'dev' if dataset == 'navsim' else 'val',
                                 'source_path': str(folder), 'protocol': p, 'summary': s})

    planning, tables, evaluations = [], {}, []
    canonical_core, population = None, None
    keys = sorted((queue / 'completed').glob('*.json'))
    for path in keys:
        entry = read(path)
        r = entry['request']
        require(entry['valid'] and r['population'] == 'navtest' and
                r['seed'] == 42 and r['scenes'] == 12146 and r['logs'] == 136,
                'Noncanonical or incomplete queue entry')
        require(r['training_identity'] == identities[('navsim', r['group'])]['identity'],
                'Evaluation belongs to another native run')
        folder = Path(entry['score_complete']).parent
        c, p = read(folder / 'COMPLETE.json'), read(folder / 'identity.json')
        require(c['identity'] == p['identity'], 'Score identity mismatch')
        core = {k: v for k, v in p.items() if k not in ('identity', 'prediction_identity')}
        if canonical_core is None:
            canonical_core = core
        require(core == canonical_core, 'Canonical protocol differs across checkpoints')
        for variant in ('q0', 'q_final'):
            summary = c[variant]
            require(summary['valid'] and summary['failed'] == 0 and summary['scenes'] == 12146,
                    'Incomplete score population')
            scores_path = folder / variant / 'scenes.csv'
            raw = source(scores_path)
            require(digest(raw) == summary['scene_csv_sha256'], 'Canonical CSV hash mismatch')
            table = read_scenes(scores_path)
            if population is None:
                population = {t: row['log'] for t, row in table.items()}
            require(population == {t: row['log'] for t, row in table.items()}, 'Population changed')
            for label, key in {**METRICS, 'DDC': 'driving_direction_compliance'}.items():
                require(abs(np.mean([float(x[key]) for x in table.values()]) -
                            summary['metrics'][key]) < 1e-10, 'CSV and summary differ: ' + label)
            zeros = sum(float(x['score']) == 0 for x in table.values())
            require(abs(zeros / len(table) - summary['zero_fraction']) < 1e-12, 'Zero count differs')
            tables[(r['group'][:2], r['updates'], variant)] = table
            planning.append({'group': r['group'], 'updates': r['updates'], 'variant': variant,
                             **{k: summary['metrics'][v] * 100 for k, v in
                                {**METRICS, 'DDC': 'driving_direction_compliance'}.items()},
                             'zero_scenes': zeros, 'scenes': 12146, 'logs': 136, 'failed': 0,
                             'training_seed': 42, 'sampling_seed': 42,
                             'evaluation_identity': c['identity'], 'completed_UTC': entry['completed_utc']})
        evaluations.append({'request': r, 'completed_UTC': entry['completed_utc'],
                            'canonical_identity': p, 'score_path': str(folder)})
    require(len(keys) == spec['expected_completed_navtest_checkpoints'],
            'Queue population changed; register a new input snapshot before exporting')
    comparisons = []
    for item in spec['comparisons']:
        a, b = tuple(item['candidate']), tuple(item['baseline'])
        # Stable per-pair draws; do not overwrite earlier reports with other RNG ordering.
        stat = paired(tables[a], tables[b], item['name'], np.random.default_rng(42), 20000)
        stat['interpretation'] = item['interpretation']
        comparisons.append({'name': item['name'], 'candidate': a, 'baseline': b,
                            'paired': stat, 'failure_transfer': compare(tables[a], tables[b], item['name'])})

    nuscenes = []
    native_prefix_max_abs_difference = 0.0
    source(repo / 'tools/structured_world/summarize_nuscenes_observations.py')
    for path in spec['nuscenes_aggregations']:
        agg = read(path)
        for obs in agg['observations']:
            folder = Path(obs['planning_root'])
            c, p = read(folder / 'COMPLETE.json'), read(folder / 'identity.json')
            require(sources[str(folder / 'COMPLETE.json')]['sha256'] == obs['source_complete_sha256'] and
                    c['identity'] == obs['original_identity'] == p['identity'] and
                    obs['samples'] == c['scenes'] == 4969 and obs['scenes'] == 150,
                    'nuScenes aggregation identity/population mismatch')
            group, update = folder.parent.name, int(folder.name.split('_')[1])
            for variant in ('q0', 'q_final'):
                for mode in ('prefix', 'endpoint'):
                    values = obs[variant][mode]
                    row = {'group': group, 'updates': update, 'epochs': update / 726,
                           'variant': variant, 'time_aggregation': mode,
                           'samples': 4969, 'scenes': 150, 'evaluation_identity': c['identity']}
                    for metric in ('L2', 'box_collision', 'point_collision'):
                        scale, unit = (1, 'm') if metric == 'L2' else (100, 'percent')
                        for idx, sec in enumerate((1, 2, 3)):
                            value = values[metric]['at_1_2_3s'][idx]
                            row[f'{metric}_{sec}s_{unit}'] = value * scale
                            if mode == 'prefix' and metric in ('L2', 'box_collision'):
                                difference = abs(value - c[variant][f'{metric}_{sec}s'])
                                native_prefix_max_abs_difference = max(native_prefix_max_abs_difference, difference)
                                # Same tolerance as the locked existing observation exporter:
                                # averaging per-step records and per-sample prefixes rounds differently.
                                require(difference <= 1e-7,
                                        'Native and archived nuScenes metrics differ')
                        row[f'{metric}_average_{unit}'] = values[metric]['average'] * scale
                    nuscenes.append(row)

    historical = list(csv.DictReader(io.StringIO(source(spec['historical_best']).decode())))
    all_history = list(csv.DictReader(io.StringIO(source(spec['historical_checkpoints']).decode())))
    for arm in ('C0', 'C1', 'C4'):
        candidates = [r for r in all_history if r['series'] == 'C' and r['arm'] == arm]
        best = max(candidates, key=lambda r: float(r['PDMS']))
        historical.append({'name': 'original/' + arm, 'arm': arm, 'version': best['version'],
                           **{k: best[k] for k in ('update', 'PDMS', 'NC', 'DAC', 'TTC', 'EP',
                                                  'zero_scenes', 'scenes', 'logs', 'failed',
                                                  'checkpoint_sha256')},
                           'observed_navtest_checkpoints': len(candidates), 'score_summary': best['source']})
    for path in spec['evidence']:
        source(path)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'NAVSIM_NAVTEST.csv', planning)
    write_csv(args.output / 'TRAINING_STATUS.csv', training)
    write_csv(args.output / 'NUSCENES_PLANNING.csv', nuscenes)
    write_csv(args.output / 'HISTORICAL_BEST.csv', historical)
    dump(args.output / 'PAIRED_COMPARISONS.json', {
        'bootstrap': {'replicates': 20000, 'seed_reset_for_each_pair': 42, 'unit': 'log',
                      'scene_weighting': 'pooled sampled log sums / sampled scene counts',
                      'limits': 'Single training seed; unadjusted exploratory 95% intervals'},
        'comparisons': comparisons})
    dump(args.output / 'SCENE_SUPERVISION.json', scene_fields)
    dump(args.output / 'SOURCE_VERSIONS.json', {
        'runs': run_meta, 'canonical_navtest_protocol': canonical_core,
        'navtest_evaluations': evaluations, 'source_files': sources,
        'note': 'Weight hashes are retained receipts, not a new read/verification of large weights. '
                'Source scene identifiers are not included. Log byte prefixes bind the live snapshot.'})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    for group in sorted({r['group'] for r in planning}):
        rows = sorted([r for r in planning if r['group'] == group and r['variant'] == 'q_final'],
                      key=lambda r: r['updates'])
        q0 = {r['updates']: r['PDMS'] for r in planning if r['group'] == group and r['variant'] == 'q0'}
        x = [r['updates'] / 1000 for r in rows]
        axes[0].plot(x, [r['PDMS'] for r in rows], marker='o', label=group[:2])
        axes[1].plot(x, [r['PDMS'] - q0[r['updates']] for r in rows], marker='o', label=group[:2])
    axes[0].set(ylabel='Navtest PDMS (0–100)', title='Executed trajectory: q_final')
    axes[1].set(ylabel='PDMS points', title='Within-model refinement: q_final − q0')
    for ax in axes:
        ax.set_xlabel('Optimizer updates (thousands)')
        ax.grid(alpha=.25)
        ax.legend(ncol=2)
    fig.suptitle('12,146 scenes / 136 logs · one training seed · markers = completed observations', fontsize=10)
    fig.savefig(args.output / 'NAVSIM_LEARNING_CURVES.png', dpi=160)
    plt.close(fig)
    outputs = {p.name: {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
               for p in sorted(args.output.iterdir())}
    manifest = {'schema': 'round1_results_consolidation_v1', 'started_UTC': started,
                'finished_UTC': utc(), 'inputs_sha256': digest(args.inputs.read_bytes()),
                'exporter_sha256': digest(Path(__file__).read_bytes()),
                'report_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                'counts': {'navtest_checkpoints': len(keys), 'navtest_variants': len(planning),
                           'nuscenes_checkpoint_observations': len(nuscenes) // 4,
                           'training_runs': len(training), 'scene_field_observations': len(scene_fields)},
                'nuscenes_prefix_check': {'existing_exporter_tolerance': 1e-7,
                                          'max_abs_difference': native_prefix_max_abs_difference},
                'outputs': outputs,
                'checks': ['36 canonical CSV hashes, all means, zero counts and identical token/log membership',
                           'canonical protocol equality, registered training and checkpoint identities',
                           'bounded live log prefixes and explicit G1 resume segment',
                           'nuScenes native completion hashes and prefix aggregation consistency'],
                'limits': ['No new inference, training, model selection or performance benchmark',
                           'Latest/best observations are not a common completed training endpoint',
                           'Repeatedly observed Navtest is not an untouched final holdout',
                           'Scene bootstrap and inference sampling seeds do not replace training seeds',
                           'Training GPU-hour counters omit preprocessing, pilots, scoring and discarded branches',
                           'Current tail timings include any cohost interference and are not isolated benchmarks']}
    dump(args.output / 'MANIFEST.json', manifest)
    print(json.dumps({'output': str(args.output), 'counts': manifest['counts'], 'captured': started}))


if __name__ == '__main__':
    main()
