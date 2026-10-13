"""Archive aggregate completed results and public execution evidence."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
B = ART/'navtest_g0_immediate_20261013T0120Z'
Q = ART/'navsim_navtest_dispatch_20261011T0225'


def read(p):
    return json.loads(p.read_text())


def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


evidence = {}
for name in ('G0_CONTROLLER_LAUNCH.json', 'G1_RECOVERY_COMMAND.json',
             'G1_COMPLETED_SSH_RECOVERY_EVIDENCE.json', 'G1_SCORE_FINALIZATION_LAUNCH.json',
             'G1_70K_VALIDATED.json', 'G1_70K_COMPARISONS.json', 'G1_70K_ANALYSIS_COMMAND.json',
             'G0_50K_VALIDATED.json', 'G0_50K_COMPARISONS.json', 'G0_50K_ANALYSIS_COMMAND.json',
             'CPU_VALIDATION.json', 'CONTROLLER_V4_HANDOFF.json'):
    p = B/name
    save(name, read(p))
    evidence[str(p)] = digest(p)

for key in ('G0_NO_FUTURE_LABEL_0050000', 'G1_FULL_UNIFORM_0070000'):
    marker = Q/'completed'/f'{key}.json'
    assert read(marker)['valid'] is True
    j = Q/'jobs'/key
    scores = read(j/'scores/COMPLETE.json')
    assert all(scores[v]['valid'] and scores[v]['scenes'] == 12146 and scores[v]['failed'] == 0
               for v in ('q0', 'q_final'))
    reg = read(j/'REGISTRATION.json')
    prediction = read(j/'COMPLETE.json')
    parts = [read(p) for p in (j/'attempts'/prediction['final_attempt']/'shards').glob('*.json')]
    assert len(parts) == prediction['shards'] and all(p['status'] == 'COMPLETE' for p in parts)
    save(key+'_COMPLETE.json', {'request': reg['request'], 'completion': read(marker),
         'prediction_complete': prediction, 'scores': scores,
         'score_sha256': digest(j/'scores/COMPLETE.json'),
         'GPUs_by_host': dict(collections.Counter(x['host'] for x in reg['routes'])),
         'all_registered_inference_duties_one': all(x['cohost_duty_cycle'] == 1 for x in reg['routes']),
         'max_allocated_GiB': max(p['peak_allocated_bytes'] for p in parts)/2**30,
         'max_reserved_GiB': max(p['peak_reserved_bytes'] for p in parts)/2**30,
         'intentional_idle_seconds_sum': sum(p['scheduling_idle_seconds'] for p in parts),
         'cost_note': 'GPU_hours sums worker inference wall time, including waits; CPU scoring overlaps. '
                      'Initialization/qualification/export and recovery validation are separate costs.',
         'registration_utc': reg['registered_utc'], 'CPU_budget': read(j/'CPU_BUDGET.json')})

columns = ['key', 'group', 'updates', 'variant', 'scenes', 'failed', 'zero_scenes',
           'score', 'no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance']
rows = []
for marker in (Q/'completed').glob('G*.json'):
    m = read(marker)
    if m.get('valid') is not True:
        continue
    request = m['request']
    scores = read(Q/'jobs'/marker.stem/'scores/COMPLETE.json')
    for variant in ('q0', 'q_final'):
        x = scores[variant]
        assert x['valid'] and x['scenes'] == 12146 and x['failed'] == 0
        rows.append(dict(key=marker.stem, group=request['group'], updates=request['updates'],
                         variant=variant, scenes=x['scenes'], failed=x['failed'],
                         zero_scenes=round(x['zero_fraction']*x['scenes']),
                         **{k: v*100 for k, v in x['metrics'].items()}))
with (OUT/'LEARNING_CURVES.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r['group'], r['updates'], r['variant'])))

restoration = []
for h in ('training-vlawm-tsinghua-zt', 'training-vlawm-tsinghua-zt2'):
    p = B/'idle_pressure'/h
    status = read(ART/'navtest_resources_20261013T0015Z/idle_pressure'/h/'STATUS.json')
    restoration.append({'host': h, 'launch': read(p/'LAUNCH_V3_VERIFIED.json'),
                        'policy_path': str(p/'POLICY_V3.json'),
                        'policy_sha256': digest(p/'POLICY_V3.json'),
                        'status': {k: status.get(k) for k in ('status', 'controller_pid', 'observed_utc',
                                   'idle_seconds', 'idle_required_seconds', 'pressure_alive', 'pending_requests')},
                        'earlier_policy_identity_rejection_preserved_at': str(p/'monitor.log')})
save('PRESSURE_MONITORS.json', restoration)
save('SOURCE_VERSIONS.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'model_and_canonical_inference_source': '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d',
     'G0_G1_GPU_dispatcher_source': '11eee58068e5cc6a20ae30083b3b63fa4779a830',
     'manual_G1_recovery_source': read(B/'frozen_recovery_v1/SOURCE.json'),
     'new_controller_source': read(B/'frozen_dispatcher_v4/SOURCE.json'),
     'evidence_sha256': evidence, 'completed_banks': len(rows)//2,
     'training_recipe_or_processes_changed': False})
print(json.dumps({'completed_banks': len(rows)//2, 'directory': str(OUT)}))
