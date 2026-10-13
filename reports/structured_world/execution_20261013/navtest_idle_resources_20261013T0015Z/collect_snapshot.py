"""Collect aggregate evidence only; never copy images, weights or private leases."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
A = ART / 'navtest_resources_20261013T0015Z'
Q = ART / 'navsim_navtest_dispatch_20261011T0225'


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.datetime.now(datetime.timezone.utc).isoformat()
evidence = {}
for name in (
    'G3_100K_COMPARISONS.json', 'G3_100K_VALIDATED.json',
    'G3_100K_COMPARISONS_COMMAND.json', 'G3_SCORER_STARTUP_RECOVERY_EVIDENCE.json',
    'CONTROLLER_V3_LAUNCH.json', 'SCORER_REPLACEMENT.json',
    'MAX_SPEED_EVALUATION_AUTHORIZATION.json', 'ACTIVE_SKILL_UPDATED.json',
    'FULL_DUTY_POLICY_DESCRIPTION_ALIGNED.json',
):
    p = A / name
    save(name, read(p))
    evidence[str(p)] = sha(p)

# All twelve banks have completed the dispatcher's exact population/hash check.
columns = ['key', 'group', 'updates', 'variant', 'scenes', 'failed', 'zero_scenes',
           'score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'ego_progress', 'time_to_collision_within_bound', 'comfort',
           'driving_direction_compliance']
rows = []
banks = []
for completion in sorted((Q / 'completed').glob('G*.json')):
    record = read(completion)
    if record.get('valid') is not True:
        continue
    key = completion.stem
    p = Q / 'jobs' / key / 'scores/COMPLETE.json'
    scores = read(p)
    request = record['request']
    banks.append({'key': key, 'completion': str(completion),
                  'completion_sha256': sha(completion), 'scores_sha256': sha(p),
                  'training_identity': request['training_identity'],
                  'checkpoint_tag': request['tag'], 'scores': scores})
    for variant in ('q0', 'q_final'):
        x = scores[variant]
        assert x['valid'] and x['scenes'] == 12146 and x['failed'] == 0
        rows.append(dict(key=key, group=request['group'], updates=request['updates'],
                         variant=variant, scenes=x['scenes'], failed=x['failed'],
                         zero_scenes=round(x['zero_fraction'] * x['scenes']),
                         **{k: 100 * v for k, v in x['metrics'].items()}))
with (OUT / 'LEARNING_CURVES.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r['group'], r['updates'], r['variant'])))
save('VALIDATED_BANKS.json', {'utc': now, 'metric_scale': '0 to 100', 'banks': banks})

training = []
for group, host, parent in (
    ('G0_NO_FUTURE_LABEL', 'training-rl-zt2', 'navsim_g0_rlzt2_deploy_20261010T2355/formal'),
    ('G1_FULL_UNIFORM', 'training-vlawm-zt2', 'formal_navsim_zt2_G1_G0_v3_io'),
    ('G2_EVENT_UNIFORM', 'training-vlawm-zt4', 'formal_navsim_local_G3_G2_v3_io'),
    ('G3_EVENT_LOCAL', 'completed', 'formal_navsim_local_G3_G2_v3_io'),
):
    root = ART / parent / group
    metrics = root / 'metrics.jsonl'
    with metrics.open('rb') as f:
        f.seek(max(0, metrics.stat().st_size - 500000))
        tail = [json.loads(line) for line in f.read().splitlines()[1:]][-32:]
    last = tail[-1]
    median = statistics.median(x['seconds'] for x in tail)
    training.append({'group': group, 'host': host, 'run': str(root),
                     'updates': last['updates'], 'target_updates': 100000,
                     'scene_exposure': last['scene_exposure'],
                     'reported_GPU_hours': last['GPU_hours'],
                     'window_updates': [tail[0]['updates'], last['updates']],
                     'median_update_seconds_last32': median,
                     'remaining_at_recent_rate_hours': max(0, 100000-last['updates'])*median/3600,
                     'metrics_mtime_utc': datetime.datetime.fromtimestamp(
                         metrics.stat().st_mtime, datetime.timezone.utc).isoformat(),
                     'training_identity': read(root/'identity.json')['identity'],
                     'execution_policy': read(root/'execution_policy.json')})
save('TRAINING_STATUS.json', {'utc': now, 'runs': training, 'training_changed': False,
     'ETA_limits': 'Recent wall time extrapolation only; shared load varies. No promised completion deadline.'})

state = read(Q / 'STATE.json')
job = Q / 'jobs' / state['request']['key']
reg = read(job / 'REGISTRATION.json')
reports = [read(p) for p in (job/'attempts'/state['attempt']/'shards').glob('*.json')]
save('EVALUATION_STATUS.json', {
    'utc': now, 'status': state['status'], 'key': state['request']['key'],
    'attempt': state['attempt'], 'registered_utc': reg['registered_utc'],
    'registered_unique_GPUs': len({r['uuid'] for r in reg['routes']}),
    'GPUs_by_host': dict(collections.Counter(r['host'] for r in reg['routes'])),
    'all_registered_duties_one': all(r['cohost_duty_cycle'] == 1 for r in reg['routes']),
    'worker_statuses': dict(collections.Counter(r['status'] for r in reports)),
    'scoring': read(job/'scores/PROGRESS.json'), 'CPU_budget': read(job/'CPU_BUDGET.json'),
    'scorer_launch': read(job/'SCORER_LAUNCH.json'),
    'next_prequalified_request': 'G0_NO_FUTURE_LABEL_0050000',
    'partial_progress_is_not_a_full_result': True,
})
inventory = read(Q/'resource_snapshots'/f"{state['request']['key']}.json")
reservation = read(Path('/mnt/project/server_dispatch_policy.json'))[
    'host_reservations'].get('training-vla-zt2', {})
save('RESOURCE_DECISIONS.json', {'utc': inventory['observed_utc'],
     'excluded': inventory['excluded'], 'selection': inventory['selection'],
     'policy': read(Q/'COHOST_COST_POLICY.json'),
     'task_preference': read(Path('/mnt/project/server_dispatch_policy.json'))[
         'task_authorizations']['structured_world_fgtr_round1']['evaluation_execution_preference'],
     'vla_zt2_live_reservation': {k: reservation.get(k) for k in (
         'task', 'exclude_new_other_workloads', 'utc', 'updated_utc')}})

restoration = []
for host in ('training-vlawm-tsinghua-zt', 'training-vlawm-tsinghua-zt2'):
    pressure = A/'idle_pressure'/host
    status = read(pressure/'STATUS.json')
    release = read(A/f'{host}_PRESSURE_RELEASE.json')
    restoration.append({'host': host, 'launch': read(pressure/'LAUNCH_V2.json'),
                        'policy_path': str(pressure/'POLICY_V2.json'),
                        'policy_sha256': sha(pressure/'POLICY_V2.json'),
                        'released_GPU_count': len(release['journal']['selected_gpu_uuids']),
                        'lease_state': status['lease_state'], 'pressure_alive': status['pressure_alive'],
                        'observed_utc': status['observed_utc'],
                        'idle_required_seconds': status['idle_required_seconds'],
                        'pending_requests': status['pending_requests'],
                        'no_private_capture_contents_exported': True})
save('PRESSURE_RESTORATION.json', {'utc': now, 'hosts': restoration,
     'rule': 'Restore exact captured filler after no queued/active useful work and 600 seconds idle.'})

g3 = Q/'jobs/G3_EVENT_LOCAL_0100000'
complete = read(g3/'COMPLETE.json')
parts = [read(p) for p in (g3/'attempts'/complete['final_attempt']/'shards').glob('*.json')]
assert len(parts) == 19 and all(p['status'] == 'COMPLETE' for p in parts)
save('G3_100K_INFERENCE_COST.json', {'inference': complete,
     'scoring_seconds_overlapping_inference': read(g3/'scores/COMPLETE.json')['seconds'],
     'worker_peak_allocated_GiB_max': max(p['peak_allocated_bytes'] for p in parts)/2**30,
     'worker_peak_reserved_GiB_max': max(p['peak_reserved_bytes'] for p in parts)/2**30,
     'counted_idle_seconds_sum': sum(p['scheduling_idle_seconds'] for p in parts),
     'limits': 'GPU_hours is summed worker inference wall time /3600, including sleeps/gaps; '
               'not CUDA kernel time or exclusive device billing. Model staging/qualification/export '
               'costs are additional. CPU scoring overlaps inference and must not be added as serial time.'})
for key in ('G3_EVENT_LOCAL_0100000', 'G1_FULL_UNIFORM_0070000', 'G0_NO_FUTURE_LABEL_0050000'):
    p = Q/'qualification'/key/'FROZEN_PARITY_PASSED.json'
    save(key+'_FROZEN_PARITY.json', read(p))
save('SOURCE_VERSIONS.json', {'model_training_and_inference_source': '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d',
     'dispatcher': read(A/'frozen_dispatcher_v3/SOURCE.json'),
     'pressure_monitor': read(A/'frozen_pressure_monitor/SOURCE.json'),
     'evidence_sha256': evidence, 'shared_artifact_root': str(A),
     'evaluation_model_or_metric_changed': False})
print(json.dumps({'snapshot_utc': now, 'completed_banks': len(banks),
                  'current_job': state['request']['key'], 'GPUs': len(reg['routes']),
                  'training_updates': {r['group']: r['updates'] for r in training}}, ensure_ascii=False))
