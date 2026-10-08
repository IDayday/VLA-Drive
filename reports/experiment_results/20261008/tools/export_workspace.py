"""Export completed experiment metrics and label data without model execution."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

METRICS = {'PDMS': 'score', 'NC': 'no_at_fault_collisions',
           'DAC': 'drivable_area_compliance', 'TTC': 'time_to_collision_within_bound',
           'EP': 'ego_progress', 'Comfort': 'comfort', 'Direction': 'driving_direction_compliance'}
FIELDS = ['evaluation_id', 'scene_id', 'log_id', *METRICS, 'status']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def score_summary(row):
    source = Path(row['source'])
    if source.name == 'summary.json':
        return source
    if source.name == 'result.json':
        candidate = source.parent / 'scores/summary.json'
        return candidate if candidate.exists() else source
    if source.parent.name in ['navtest_AV_50k_20261007_v1', 'navtest_AV_75k_20261007_v1']:
        return source.parent / row['arm'] / 'scores/summary.json'
    return (source.parent / 'jobs' / f"{row['arm']}_{int(row['update']):06d}"
            / 'evaluation' / row['arm'] / 'scores/summary.json')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    out, workspace = a.output.resolve(), a.workspace.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def remember(path):
        path = Path(path)
        key = str(path.relative_to(workspace))
        if key not in sources:
            sources[key] = {'sha256': digest(path), 'bytes': path.stat().st_size}

    def copy(path, dest):
        remember(path)
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    combined = workspace / 'experiment-results-summary/20261008_CS_AV'
    best = workspace / 'experiment-results-summary/20261008_S_AV_best'
    objective = workspace / 'trajectory-objective-training-artifacts/20261006/config_v2_validation'
    sroot = workspace / 'action-video-foresight-artifacts/20261002'
    avroot = workspace / 'planning-interface-transfer-artifacts/20261005'
    optroot = workspace / 's3-optimized-training-artifacts'
    for name in ['CHECKPOINT_HISTORY.csv', 'DEV_CHECKPOINTS.csv', 'NAVTEST_CHECKPOINTS.csv']:
        copy(combined / name, 'data/' + name.lower())
    copy(best / 'BEST_CHECKPOINTS.csv', 'data/best_checkpoints.csv')
    copy(best / 'SUMMARY.json', 'analysis/best_checkpoint_summary.json')
    for name in ['NAVTEST_BEST_CURVES.png', 'NAVTEST_BEST_CURVES.svg']:
        copy(best / name, 'figures/' + name.lower())
    for name in ['NAVTEST_CURVES.png', 'NAVTEST_CURVES.svg']:
        copy(combined / name, 'figures/all_' + name.lower())
    for name in ['LABEL_COMPARISON.json', 'OBJECTIVE_ANALYSIS.json', 'OBJECTIVE_PAIRED_50K.json',
                 'OFFROAD_FITTING_50K.json', 'TRAINING_DIAGNOSTICS.json']:
        copy(combined / name, 'analysis/historical/' + name.lower())
    copy(sroot / 'c_s_summary_20261005_1331/PAIRED_100K.json', 'analysis/c_s_paired_100k.json')
    copy(avroot / 'BASELINE_REUSE_PROOF_v2.json', 'analysis/baseline_reuse_proof.json')
    for arm in ['S0', 'S1', 'S2', 'S3', 'S4']:
        copy(avroot / f'audit_{arm}_100k_complete_v1/SUMMARY.json', f'analysis/auxiliary/{arm}.json')
    for name in ['FINAL_REPORT.json', 'DIAGNOSIS.json', 'PLAN.json', 'RESULTS.csv',
                 'RUNTIME_SUMMARY.json', 'SOURCE_VERSIONS.json', 'STATUS_SUMMARY.json',
                 'CONFIG_MISMATCHES.csv', 'candidate.json', 'control.json',
                 'EVALUATION_REPORT.json', 'DIAGNOSTIC_EVALUATION_REPORT.json']:
        copy(objective / name, 'analysis/objective_v2/' + name.lower())
    for name in ['DEV_CHECKPOINT_CURVES.png', 'GRADIENT_CALIBRATION.png']:
        copy(objective / name, 'figures/objective_v2_' + name.lower())
    copy(objective.parent / 'maintenance/20261008-objective-config-v2/PAUSE_REPORT.json',
         'analysis/objective_v2/original_training_pause.json')
    for version, root in [('r1', optroot / 'labels_v1'), ('V12', objective.parent / 'labels_v12')]:
        for name in ['labels.npz', 'identity.json', 'COMPLETE.json']:
            copy(root / name, f'data/optimized_labels/{version}/{name}')
    for family, root, pattern in [('s', sroot / 'registrations', 'formal_S*_full100k_v2*.json'),
                                  ('av', avroot / 'registrations', 'formal_*_full100k_v*.json'),
                                  ('optimized', optroot / 'registrations', '*.json'),
                                  ('objective_v2', objective / 'configs', '*.json')]:
        paths = sorted(root.glob(pattern))
        assert paths, (family, root)
        for path in paths:
            copy(path, f'configs/{family}/{path.name}')

    history_path = combined / 'CHECKPOINT_HISTORY.csv'
    rows = list(csv.DictReader(history_path.open()))
    jobs = []
    for row in rows:
        summary_path = score_summary(row)
        summary = read(summary_path)
        if 'export_identity' in summary:
            cp, protocol = summary['export_identity']['checkpoint'], summary['export_identity']['protocol']
            assert summary['valid'] and not summary['diagnostic'] and summary['failed'] == 0
            assert protocol['sampling_seed'] == 42 and protocol['steps'] == 10
            assert protocol['precision'] == 'FP32' and not protocol['tf32']
            assert protocol['candidates_per_scene'] == 1 and not protocol['future_conditioning']
            assert protocol['scorer'] is None and protocol['auxiliary_heads_removed']
            pdms, scenes_path = summary['PDMS'] * 100, summary_path.parent / 'scenes.csv'
        else:
            # One retained S0@50k export has its frozen completion receipt and
            # original complete CSV, but its scoring-directory copy was removed.
            cp = summary['checkpoint']
            assert summary['status'] == 'COMPLETE' and summary['failed'] == 0
            assert summary['sampling_seed'] == 42 and summary['optimizer_updates_in_evaluation'] == 0
            assert summary['precision'] == 'FP32 masters/FP32 compute/TF32off'
            scenes_path = summary_path.parent / 'complete.csv'
            assert digest(scenes_path) == summary['complete_csv_sha256']
            pdms = summary['PDMS_points']
        assert cp['completed'] == int(row['update']) and cp['sha256'] == row['checkpoint_sha256']
        assert abs(pdms - float(row['PDMS'])) < 1e-8
        name = ':'.join([row['split'], row['series'].replace('/', '_'), row['version'], row['arm'], row['update'], '42'])
        jobs.append({'evaluation_id': name, 'dataset': row['split'], 'series': row['series'],
                     'version': row['version'], 'arm': row['arm'], 'update': int(row['update']),
                     'sampling_seed': 42, 'training_seed': 42, 'diagnostic_only': False,
                     'checkpoint_sha256': cp['sha256'], 'expected_scenes': int(row['scenes']),
                     'expected_logs': int(row['logs']), 'PDMS': float(row['PDMS']),
                     'source': str(scenes_path)})
        remember(summary_path)
    for name, dataset in [('EVALUATION_REPORT.json', 'objective_v2_dev'),
                          ('DIAGNOSTIC_EVALUATION_REPORT.json', 'objective_v2_diagnostic')]:
        for entry in read(objective / name)['results']:
            assert entry['status'] == 'COMPLETE' and entry['reference_parity_passed'] and entry['failures'] == 0
            summary_path = Path(entry['scenes_csv']).parent / 'summary.json'
            summary = read(summary_path)
            assert summary['valid'] and summary['failed'] == 0
            job_id = f"{dataset}:{entry['variant']}:{entry['role']}:{entry['step']}:{entry['seed']}"
            jobs.append({'evaluation_id': job_id, 'dataset': dataset, 'series': 'S',
                         'version': 'config_v2_' + entry['role'], 'arm': entry['variant'],
                         'update': entry['step'], 'sampling_seed': entry['seed'], 'training_seed': 42,
                         'diagnostic_only': entry['diagnostic_only'], 'run_id': entry['run_id'],
                         'expected_scenes': entry['scenes'], 'expected_logs': entry['logs'],
                         'PDMS': summary['PDMS'] * 100, 'source': entry['scenes_csv']})
            remember(summary_path)
            remember(entry['parity'])
    assert len({j['evaluation_id'] for j in jobs}) == len(jobs) == 186
    anonymous = {}

    def anon(kind, split, value):
        key = (kind, split, value)
        if key not in anonymous:
            anonymous[key] = hashlib.sha256(f'cs-planning-results-v1:{kind}:{split}:{value}'.encode()).hexdigest()[:24]
        return anonymous[key]

    log_rows, datasets, bootstrap_order = [], {}, {}
    for dataset in ['navtest', 'dev', 'objective_v2_dev', 'objective_v2_diagnostic']:
        target = out / f'data/scenes/{dataset}.csv.gz'
        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=6) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8', newline='') as text:
                writer = csv.DictWriter(text, FIELDS)
                writer.writeheader()
                for job in [j for j in jobs if j['dataset'] == dataset]:
                    remember(job['source'])
                    scene_rows = list(csv.DictReader(open(job['source'])))
                    assert len(scene_rows) == len({r['token'] for r in scene_rows}) == job['expected_scenes']
                    assert all(r['status'] == 'ok' for r in scene_rows)
                    assert abs(math.fsum(float(r['score']) for r in scene_rows) / len(scene_rows) * 100 - job['PDMS']) < 1e-8
                    per_log = defaultdict(list)
                    split = 'navtest' if dataset == 'navtest' else 'dev'
                    order = [anon('log', split, log) for log in sorted({r['log'] for r in scene_rows})]
                    if dataset not in bootstrap_order:
                        bootstrap_order[dataset] = order
                    assert bootstrap_order[dataset] == order
                    for row in scene_rows:
                        metric = {k: float(row[v]) * 100 for k, v in METRICS.items()}
                        assert all(math.isfinite(v) and -1e-8 <= v <= 100 + 1e-8 for v in metric.values())
                        record = {'evaluation_id': job['evaluation_id'], 'scene_id': anon('scene', split, row['token']),
                                  'log_id': anon('log', split, row['log']), **metric, 'status': row['status']}
                        writer.writerow(record)
                        per_log[record['log_id']].append(metric)
                    assert len(per_log) == job['expected_logs']
                    for log, values in per_log.items():
                        log_rows.append({'evaluation_id': job['evaluation_id'], 'log_id': log,
                                         'scenes': len(values), 'zero_scenes': sum(v['PDMS'] == 0 for v in values),
                                         **{k: math.fsum(v[k] for v in values) / len(values) for k in METRICS}})
                    job['source_artifact'] = str(Path(job.pop('source')).relative_to(workspace))
                    total += len(scene_rows)
                print(dataset, total, 'scene rows', flush=True)
        datasets[dataset] = {'file': str(target.relative_to(out)), 'rows': total,
                             'evaluations': sum(j['dataset'] == dataset for j in jobs)}
    target = out / 'data/per_log_metrics.csv'
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    write(out / 'data/evaluations.json', jobs)
    write(out / 'data/bootstrap_log_order.json', bootstrap_order)
    write(out / 'provenance/source_artifacts.json', sources)
    write(out / 'provenance/snapshot.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'COMPLETE',
        'datasets': datasets, 'evaluation_count': len(jobs), 'scene_rows': sum(d['rows'] for d in datasets.values()),
        'log_rows': len(log_rows), 'new_training_updates': 0, 'new_model_inference': 0,
        'new_trajectory_objective_research': 'PAUSED_BY_USER', 'label_versions': ['r1', 'V12'],
        'scene_identifier_rule': 'SHA256 cs-planning-results-v1:scene:split:token first24hex',
        'log_identifier_rule': 'SHA256 cs-planning-results-v1:log:split:log first24hex',
        'metric_units': 'all PDMS component columns use 0-100; ADE/FDE use metres',
        'weight_storage': 'checkpoint directories and hashes indexed in data/best_checkpoints.csv'} )


if __name__ == '__main__':
    main()
