"""Read-only formal learning curves and registered task-exposure checks.

Never changes a training run, its schedule, or model selection. Auxiliary losses
are displayed separately and cannot rank different target resolutions.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time


TASKS = ('ego_fm', 'current_dino', 'future_dino', 'interaction')


def completed_rows(path):
    lines = Path(path).read_text().splitlines(keepends=True)
    # A live writer may be between JSON and newline; never accept a partial row.
    return [json.loads(line) for line in lines if line.endswith('\n')]


def check_rows(rows, future_weight, interaction_weight):
    exposure = 0
    for update, row in enumerate(rows, 1):
        if row['update'] != update:
            raise ValueError('Noncontiguous or duplicate optimizer update')
        scenes = int(row['counts']['ego_scenes'])
        exposure += scenes
        if row['exposure'] != exposure or sum(row['horizon_scene_requests']) != scenes:
            raise ValueError('Ego exposure or independent future request count changed')
        weights = dict(ego_fm=1., current_dino=1.,
                       future_dino=future_weight * min(update / 1000, 1),
                       interaction=interaction_weight * min(update / 1000, 1))
        for task in TASKS:
            raw = row['raw_losses'][task]
            weighted = row['losses'][task]
            actual_weight = row['effective_weights'][task]
            if not all(math.isfinite(v) for v in (raw, weighted, actual_weight)):
                raise ValueError(f'Nonfinite {task} at update {update}')
            if not math.isclose(weights[task], actual_weight, abs_tol=1e-12):
                raise ValueError(f'Unregistered task weight {task} at update {update}')
            if not math.isclose(raw * actual_weight, weighted, rel_tol=2e-5, abs_tol=1e-7):
                raise ValueError(f'Raw/weighted mismatch {task} at update {update}')
        for key in ('current_dino', 'future_dino', 'interaction'):
            if row['counts'][key] < 0:
                raise ValueError('Negative task denominator')
            if row['counts'][key] == 0 and row['raw_losses'][key] != 0:
                raise ValueError('Nonzero loss on an empty auxiliary set')
        if row['counts']['current_dino'] == 0:
            raise ValueError('Missing all current targets for a real RGB batch')
        for name, value in row.get('shared_parameter_observation', {}).items():
            if not all(math.isfinite(value[k]) for k in ('gradient_norm', 'update_norm', 'max_abs_update')):
                raise ValueError('Nonfinite shared observation: ' + name)
            if value['gradient_nonzero'] <= 0 or value['changed_elements'] <= 0:
                raise ValueError('Missing declared shared gradient/update: ' + name)
    return exposure


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def summarize(root, out, plots=True, registration_path=None):
    registration = json.loads((Path(registration_path) if registration_path else root / 'formal_registration_v1.json').read_text())
    calibration = json.loads((root / 'four_loss_calibration_v1.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    now = time.time()
    meters = [json.loads(p.read_text()) for p in (root / 'runs').glob('*/status.json')]
    records = {}; curves = {}; flat = []
    for run_id, spec in registration['runs'].items():
        directory = root / 'students' / run_id
        if not (directory / 'status.json').exists():
            records[run_id] = dict(status='NOT_STARTED', candidate=spec['candidate'])
            continue
        state = json.loads((directory / 'status.json').read_text())
        record = dict(candidate=spec['candidate'], status=state['status'], host=state['host'],
                      source_sha=state['source_sha'], seconds_since_status=now - state['updated_unix'])
        try:
            if state['source_sha'] != registration['training_source_sha']:
                raise ValueError('Training source differs from registration')
            rows = completed_rows(directory / 'steps.jsonl') if (directory / 'steps.jsonl').exists() else []
            exposure = check_rows(rows, calibration['lambda_fut'], calibration['lambda_int'])
            observations = [r for r in rows if 'shared_parameter_observation' in r]
            record.update(completed=len(rows), exposure=exposure, checks='PASS',
                          effective_traversals=exposure / registration['scene_count'],
                          current_requests=exposure, future_requests=sum(sum(r['horizon_scene_requests']) for r in rows),
                          horizon_exposure=[sum(r['horizon_scene_requests'][i] for r in rows) for i in range(3)],
                          valid_interaction_scene_exposures=sum(r['valid_interaction_scenes'] for r in rows),
                          empty_future_batches=sum(r['counts']['future_dino'] == 0 for r in rows),
                          empty_interaction_batches=sum(r['counts']['interaction'] == 0 for r in rows),
                          observed_updates=[r['update'] for r in observations],
                          full_weight_observed=any(r['update'] >= 1000 for r in observations),
                          full_weight_finite_updates=sum(r['update'] >= 1000 for r in rows),
                          shared_observations=observations,
                          learning_conclusion='Not a convergence or planning-effectiveness test')
            own = [m for m in meters if m.get('run_id_parent') == run_id]
            record['charged_gpu_hours'] = sum(((now if m['status'] == 'RUNNING' else m['end_unix']) - m['start_unix']) * m['gpu_count'] / 3600 for m in own)
            if rows:
                recent = rows[-100:]
                record['latest_raw_losses'] = rows[-1]['raw_losses']
                record['last100_raw_means'] = {k: statistics.mean(r['raw_losses'][k] for r in recent) for k in TASKS}
                record['last100_step_p50_s'] = statistics.median(r['seconds'] for r in recent)
                record['latest_effective_weights'] = rows[-1]['effective_weights']
                curves[spec['candidate']] = rows
                for r in rows:
                    flat.append(dict(candidate=spec['candidate'], run_id=run_id, update=r['update'],
                                     exposure=r['exposure'], **r['raw_losses'],
                                     **{k + '_weighted': r['losses'][k] for k in TASKS},
                                     seconds=r['seconds'], valid_interaction_scenes=r['valid_interaction_scenes']))
        except Exception as error:
            record.update(checks='FAILED', error=repr(error))
        records[run_id] = record
    if flat:
        path = out / 'learning_curves.csv'; temp = path.with_suffix('.tmp')
        with temp.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        temp.replace(path)
    if plots and curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for task, axis in zip(TASKS, axes.flat):
            for candidate, rows in curves.items():
                # Fixed 50-update blocks, not selected favorable checkpoints.
                blocks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
                axis.plot([b[-1]['exposure'] for b in blocks], [statistics.mean(r['raw_losses'][task] for r in b) for b in blocks], label=candidate)
            axis.set(title=task + ' (raw, 50-update blocks)', xlabel='Actual scene exposure', ylabel='Loss')
            axis.grid(alpha=.2); axis.legend()
        fig.suptitle('Formal full four-task training; auxiliary losses do not rank configurations')
        fig.tight_layout(); fig.savefig(out / 'learning_curves.png', dpi=140); plt.close(fig)
    result = dict(snapshot_unix=now, training_source=registration['training_source_sha'],
                  registration_identity=registration['identity'], runs=records,
                  failed_checks=sum(v.get('checks') == 'FAILED' for v in records.values()),
                  formal_planning_results='See separately scored complete dev populations; never inferred from these curves')
    atomic_json(out / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watch-seconds', type=int, default=0)
    p.add_argument('--interval', type=int, default=60)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--registration', type=Path)
    a = p.parse_args()
    if a.watch_seconds < 0 or a.interval < 10:
        raise ValueError('Bounded watch and interval>=10 required')
    deadline = time.monotonic() + a.watch_seconds
    while True:
        result = summarize(a.campaign_root, a.output, plots=not a.no_plots, registration_path=a.registration)
        print(json.dumps({k: {f: v.get(f) for f in ('status', 'completed', 'checks', 'full_weight_observed')} for k, v in result['runs'].items()}), flush=True)
        if result['failed_checks']:
            raise RuntimeError('Actual formal-run discrepancy; report retained, training not modified')
        if time.monotonic() >= deadline:
            return
        time.sleep(min(a.interval, deadline - time.monotonic()))


if __name__ == '__main__':
    main()
