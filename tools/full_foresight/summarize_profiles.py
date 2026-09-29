"""Measured full-method costs; incomplete runs remain visible and cannot select execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stable_steps(run):
    path = run / 'steps.jsonl'
    return [r for r in lines(path) if 20 < r['update'] <= 120] if path.exists() else []


def concurrent_rate(step_lists):
    """Count completed batches entirely inside the actually shared measured interval."""
    if len(step_lists) < 2 or any(not rows for rows in step_lists):
        return {'valid': False, 'reason': 'No common measured window'}
    left = max(rows[0]['ended_unix'] - rows[0]['seconds'] for rows in step_lists)
    right = min(rows[-1]['ended_unix'] for rows in step_lists)
    if right <= left:
        return {'valid': False, 'reason': 'Measured windows did not overlap'}
    exposures = []
    for rows in step_lists:
        # Drop partial boundary batches, never multiply one job throughput by two.
        exposures.append(sum(r['batch_scenes'] for r in rows
                             if r['ended_unix'] - r['seconds'] >= left and r['ended_unix'] <= right))
    return {'valid': all(n > 0 for n in exposures), 'seconds': right - left,
            'completed_scenes_per_job': exposures, 'samples_per_s': sum(exposures) / (right - left),
            'boundary_policy': 'only whole measured batches in common overlap; conservative'}


def summarize(run, population=None):
    identity = read(run / 'identity.json')
    if identity['schema'] != 'ddp_full_foresight_student_v1' or identity['scope'] != 'profile':
        raise ValueError('Complete-method profiles only')
    config = identity['config']['foresight']
    if not all(config[k] for k in ('enable_current_dino', 'enable_future_dino', 'enable_interaction')):
        raise ValueError('Reduced objective cannot stand in for a full-method profile')
    status = read(run / 'status.json') if (run / 'status.json').exists() else {'status': 'LOADING'}
    rows = stable_steps(run)
    result = {'run': run.name, 'candidate': config['candidate'], 'source_sha': identity['source_sha'],
              'status': status['status'], 'world_size': identity['world_size'],
              'global_batch': identity['global_batch'], 'micro_batch': identity['micro_batch'],
              'measured_steps': len(rows), 'measurement_complete': len(rows) == 100 and status['status'] == 'COMPLETE',
              'all_four_losses': True, 'profile_only_not_planning_evidence': True}
    if not rows:
        return result
    rank_values = np.asarray([r['per_rank_profile'] for r in rows], dtype=np.float64)
    seconds = rank_values[:, :, 0].max(1)
    batch = identity['global_batch']
    result.update(step_p50_s=float(np.median(seconds)), step_p95_s=float(np.percentile(seconds, 95)),
                  mean_step_s=float(seconds.mean()), samples_per_s=float(batch / seconds.mean()),
                  optimizer_gpu_hours_per_1000_updates=float(seconds.mean() * identity['world_size'] / 3.6),
                  data_wait_p50_s=float(np.median(rank_values[:, :, 1].max(1))),
                  per_rank_peak_allocated_bytes=rank_values[:, :, 2].max(0).tolist(),
                  per_rank_peak_reserved_bytes=rank_values[:, :, 3].max(0).tolist(),
                  sum_rank_peak_allocated_bytes=int(rank_values[:, :, 2].max(0).sum()),
                  per_batch_rank_max_sequence_minmax=[int(rank_values[:, :, 4].min()), int(rank_values[:, :, 4].max())],
                  effective_accumulation=batch / identity['world_size'] / identity['micro_batch'],
                  full_training_population=population)
    if population is not None:
        result['optimizer_gpu_hours_per_epoch_estimate'] = float(
            seconds.mean() * identity['world_size'] * int(np.ceil(population / batch)) / 3600)
    result['rank0_head_forward_ms'] = {
        key: {'p50': float(np.median([r['rank0_readout_forward_gpu_ms'][key] for r in rows])),
              'p95': float(np.percentile([r['rank0_readout_forward_gpu_ms'][key] for r in rows], 95))}
        for key in ('current_dino', 'future_dino', 'interaction')}
    result['head_timing_scope'] = 'forward CUDA events only; total step includes all backward and optimizer work'
    if (run / 'inference_profile.json').exists():
        inference = read(run / 'inference_profile.json')
        times = np.asarray(inference['per_rank_seconds'])
        result['deployment'] = {'batch1_p50_s': float(np.median(times)), 'batch1_p95_s': float(np.percentile(times, 95)),
                                'W_retained': inference['reasoning_retained'], 'teachers': False,
                                'readout_heads': False, 'RGB_loading_included': inference['includes_rgb_loading']}
    if (run / 'checkpoint_costs.jsonl').exists():
        result['checkpoint_costs'] = lines(run / 'checkpoint_costs.jsonl')
    result['task_counters'] = status.get('counters', {})
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--campaign-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.campaign_root)
    from starVLA.model.modules.vehicle_joint.initialization import file_sha256
    index = read(root / 'dino_index_v1' / 'identity.json')
    train_manifest = root / 'dino_index_v1' / 'train_scenes.json'
    if file_sha256(train_manifest) != index['files']['train']:
        raise ValueError('Training manifest changed')
    population = len(read(train_manifest))
    profiles, concurrent, failures = [], [], []
    for run in sorted((root / 'students').glob('*')):
        if not (run / 'identity.json').exists():
            continue
        identity = read(run / 'identity.json')
        if identity['scope'] != 'profile':
            continue
        try:
            if identity['data']['index_sha256'] != index['index_hashes']['train']:
                raise ValueError('Profile population is from another split')
            profiles.append(summarize(run, population))
        except Exception as error:
            failures.append({'run': run.name, 'error': repr(error)})
    for path in sorted((root / 'queues').glob('*.json')):
        queue = read(path)
        if queue['mode'] != 'paired4':
            continue
        jobs = queue['jobs']
        for offset in range(0, len(jobs), 2):
            pair = jobs[offset:offset+2]
            rows = []
            for job in pair:
                run = root / 'students' / job['run_id']
                batch = read(run / 'identity.json')['global_batch'] if (run / 'identity.json').exists() else 0
                rows.append([{**r, 'batch_scenes': batch} for r in stable_steps(run)])
            concurrent.append({'queue': path.name, 'jobs': [j['run_id'] for j in pair],
                               'queue_status': queue['status'], **concurrent_rate(rows)})
    atomic_json(Path(a.output), {'profiles': profiles, 'concurrent': concurrent, 'failures': failures,
        'selection': 'No configuration quality ranking from profile losses; formal paired PDMS still required',
        'cost_scope': 'optimizer-step estimates exclude loading/checkpoint/evaluation; run ledgers charge those separately'})


if __name__ == '__main__':
    main()
