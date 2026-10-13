"""Collect public aggregate evidence after complete G2 latest-checkpoint Navtest."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
A = ART / 'navtest_g2_latest_20261013T0157Z'
Q = ART / 'navsim_navtest_dispatch_20261011T0225'
KEY = 'G2_EVENT_UNIFORM_0045000'

def read(path):
    return json.loads(path.read_text())

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

marker = read(Q / 'completed' / (KEY + '.json'))
assert marker['valid'] is True
job = Q / 'jobs' / KEY
score = read(job / 'scores/COMPLETE.json')
assert all(score[v]['valid'] and score[v]['scenes'] == 12146 and score[v]['failed'] == 0
           for v in ('q0', 'q_final'))
prediction = read(job / 'COMPLETE.json')
reg = read(job / 'REGISTRATION.json')
primary = [read(p) for p in (job / 'attempts' / prediction['final_attempt'] / 'shards').glob('*.json')]
assert len(primary) == prediction['shards'] and all(p['status'] == 'COMPLETE' for p in primary)
expansions = [read(p) for p in (job / 'expansions').glob('*/COMPLETE.json')]
assert len(expansions) == len(list((job / 'expansions').glob('*/GATE.json')))
routes = list(reg['routes'])
parts = list(primary)
for value in expansions:
    assert value['all_supplemental_launchers_exited'] and not any(value['exit_codes'])
    routes.extend(value['gate']['routes'])
    parts.extend(value['reports'])
assert len({r['uuid'] for r in routes}) == len(routes)
assert len(parts) == prediction['attempt_reports']
save(KEY + '_COMPLETE.json', {
    'request': marker['request'], 'completion': marker, 'scores': score,
    'score_sha256': digest(job / 'scores/COMPLETE.json'), 'prediction_complete': prediction,
    'GPUs_by_host': dict(collections.Counter(r['host'] for r in routes)),
    'primary_workers': len(primary), 'supplemental_workers': len(parts)-len(primary),
    'expansions': expansions,
    'CPU_budget': read(job / 'CPU_BUDGET.json'), 'registration_utc': reg['registered_utc'],
    'max_allocated_GiB': max(p['peak_allocated_bytes'] for p in parts) / 2**30,
    'max_reserved_GiB': max(p['peak_reserved_bytes'] for p in parts) / 2**30,
    'all_registered_inference_duties_one': all(r['cohost_duty_cycle'] == 1 for r in routes),
    'intentional_idle_seconds_sum': sum(p['scheduling_idle_seconds'] for p in parts),
    'cost_note': 'Worker inference GPU-hours include waits; CPU scoring overlaps. '
                 'Qualification, export, validation and resource waiting are separate.'})

names = ['REQUEST_INTENT.json', 'PIN_COMPLETE.json', 'SOURCE_VERSIONS.json',
         'CADENCE_WITH_G2_LATEST.json', 'CONTROLLER_LAUNCH.json', 'OLD_CONTROLLER_EXITED.json',
         'COORDINATOR_RELOCATION.json', 'SSH_ROUTING_SETUP.json', 'G2_45K_VALIDATED.json',
         'G2_45K_COMPARISONS.json', 'G2_45K_ANALYSIS_COMMAND.json',
         'ZT4_EXPANSION_AUTHORIZATION.json', 'EXPANSION_QUALIFIED_ROUTES.json',
         'EXPANSION_LAUNCH.json', 'PARITY_BEFORE_EXPANSION.json']
evidence = {}
for name in names:
    path = A / name
    save(name, read(path))
    evidence[str(path)] = digest(path)
save('EVIDENCE_HASHES.json', evidence)

rows = []
for path in (Q / 'completed').glob('G*.json'):
    m = read(path)
    assert m['valid']
    s = read(Q / 'jobs' / path.stem / 'scores/COMPLETE.json')
    for variant in ('q0', 'q_final'):
        x = s[variant]
        assert x['valid'] and x['scenes'] == 12146 and x['failed'] == 0
        rows.append(dict(key=path.stem, group=m['request']['group'], updates=m['request']['updates'],
                         variant=variant, scenes=x['scenes'], failed=x['failed'],
                         zero_scenes=round(x['zero_fraction'] * x['scenes']),
                         **{k: v * 100 for k, v in x['metrics'].items()}))
columns = ['key', 'group', 'updates', 'variant', 'scenes', 'failed', 'zero_scenes',
           'score', 'no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance']
with (OUT / 'LEARNING_CURVES.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r['group'], r['updates'], r['variant'])))

pin = ART / 'navsim_navtest_cadence_20261012/native_pins' / KEY
save('CHECKPOINT_PIN_RELEASE.json', read(pin / 'RELEASED.json'))
pressure = []
for host in ('training-vlawm-tsinghua-zt', 'training-vlawm-tsinghua-zt2'):
    status = read(ART / 'navtest_resources_20261013T0015Z/idle_pressure' / host / 'STATUS.json')
    pressure.append({'host': host, 'handoff': read(A / 'idle_pressure' / host / 'HANDOFF.json'),
                     'status': {k: status.get(k) for k in ('status', 'observed_utc', 'pressure_alive',
                        'idle_seconds', 'idle_required_seconds', 'pending_requests', 'deferred_reason')}})
save('PRESSURE_MONITORS.json', pressure)
print(json.dumps({'completed_banks': len(rows) // 2, 'directory': str(OUT),
                  'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
