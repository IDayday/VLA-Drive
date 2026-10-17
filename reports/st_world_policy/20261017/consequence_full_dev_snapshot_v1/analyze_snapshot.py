"""Bounded read-only snapshot using the registered, frozen analysis functions."""
import csv
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
signal.alarm(120)
START = time.monotonic()
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
sys.path.insert(0, str(RUNTIME))
import numpy as np
from tools.st_world_policy.evaluate_campaign import (
    load_population, load_run, paired_statistics, read_json, pinned_json,
    file_sha256, METRICS)

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016/formal/evaluation')
POLICY_PATH = Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_execution_v1/POLICY.json')
policy = read_json(POLICY_PATH)
plan = pinned_json(policy['analysis']['plan'])
population_path = policy['population_manifest']['path']
assert file_sha256(population_path) == policy['population_manifest']['sha256']
selected = load_population(plan, population_path, 'FORMAL')
assert len(selected['scenes']) == 1696 and len({r['log'] for r in selected['scenes']}) == 16
pins = [{'path': str(POLICY_PATH), 'sha256': file_sha256(POLICY_PATH)},
        policy['analysis']['plan'], policy['population_manifest']]
for entry in policy['execution_runtime']['source_files']:
    assert file_sha256(entry['path']) == entry['sha256']
pins.extend(policy['execution_runtime']['source_files'])
c2_policy_path = POLICY_PATH.parent.parent / 'full_dev_C2_C3_execution_v1/POLICY.json'
c2_policy = read_json(c2_policy_path)
c2_plan = pinned_json(c2_policy['execution_plan'])
assert pinned_json(c2_policy['parent_policy']) == policy
assert c2_plan['statistics'] == plan['statistics']
assert c2_policy['population_manifest'] == policy['population_manifest']
assert c2_policy['score'] == policy['score']
pins.extend([{'path': str(c2_policy_path), 'sha256': file_sha256(c2_policy_path)},
             c2_policy['execution_plan']])
jobs = [('C0', 512, 'DIAGNOSTIC_512'), ('C0', 1024, 'DIAGNOSTIC_1024'),
        ('C0', 4096, 'COMMON_ENDPOINT'), ('C1', 512, 'DIAGNOSTIC_512'),
        ('C2', 512, 'DIAGNOSTIC_512')]
runs, aggregate, by_log = {}, [], []
for arm, step, stage in jobs:
    job_id = f'{arm}_step{step:06d}_dev1696_seed42_v1'
    job = ROOT / 'jobs' / job_id
    registration = read_json(ROOT / 'registrations' / job_id / 'REGISTRATION.json')
    assert registration['job']['arm'] == arm and registration['job']['expected_completed'] == step
    assert registration['requested_scenes'] == 1696 and registration['navtest'] is False
    complete_path = job / 'official_score/COMPLETE.json'
    assert complete_path.is_file(), 'Snapshot only accepts finished results'
    run = load_run(job / 'official_score', job / 'export', selected,
                   population_path, c2_plan if arm == 'C2' else plan, 'FORMAL', arm, stage)
    assert run['export']['checkpoint']['completed'] == step
    complete = read_json(complete_path)
    tokens = sorted(run['rows'])
    values = {k: np.array([run['rows'][s][k] for s in tokens]) for k in METRICS}
    logs = [run['rows'][s]['source_log'] for s in tokens]
    assert len(tokens) == 1696 and len(set(logs)) == 16
    zero = int((values['score'] <= plan['statistics']['zero_epsilon']).sum())
    assert zero == complete['aggregate']['zero_count']
    for k in METRICS:
        assert abs(float(values[k].mean()) - complete['aggregate']['metrics'][k]) < 1e-14
    row = {'arm': arm, 'step': step, 'stage': stage, 'scenes': len(tokens), 'logs': len(set(logs)),
           **{k: float(v.mean()) for k, v in values.items()}, 'zero_count': zero,
           'zero_rate': zero / len(tokens), 'invalid': complete['invalid'], 'unresolved': complete['unresolved']}
    aggregate.append(row)
    for log in sorted(set(logs)):
        ids = np.asarray(logs) == log
        by_log.append({'arm': arm, 'step': step, 'source_log': log, 'scenes': int(ids.sum()),
            **{k: float(v[ids].mean()) for k, v in values.items()},
            'zero_count': int((values['score'][ids] <= plan['statistics']['zero_epsilon']).sum())})
    runs[(arm, step)] = {'values': values, 'logs': logs, 'tokens': tokens,
        'reference_inputs': {s: run['records'][s]['reference_contract']['input'] for s in tokens},
        'observations': run['export']['observations'], 'environment': run['export']['environment'],
        'failed_attempts': run['failed_attempts'], 'checkpoint': run['export']['checkpoint']}
    for suffix in ['official_score/IDENTITY.json', 'official_score/COMPLETE.json',
                   'official_score/SCENES.csv', 'export/IDENTITY.json', 'export/COMPLETE_rank0.json']:
        path = job / suffix
        pins.append({'path': str(path), 'sha256': file_sha256(path)})
    path = ROOT / 'registrations' / job_id / 'REGISTRATION.json'
    pins.append({'path': str(path), 'sha256': file_sha256(path)})
    del run
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024 * 1024, 'RAM2GiB ceiling'
    print(json.dumps({'verified': job_id, 'score': row['score'], 'zero_count': zero}), flush=True)

paired_deltas = {}
for treatment in ('C1', 'C2'):
    a, b = runs[('C0', 512)], runs[(treatment, 512)]
    assert a['tokens'] == b['tokens'] and a['logs'] == b['logs']
    assert a['observations'] == b['observations'] and a['environment'] == b['environment']
    assert a['reference_inputs'] == b['reference_inputs']
    statistics, flags = paired_statistics(a['values'], b['values'], a['logs'], plan['statistics'])
    statistics['diagnostic_only'] = True
    statistics['comparison'] = f'{treatment}_512_minus_C0_512'
    statistics['statistics_definition'] = plan['statistics']
    statistics['safety_transitions'] = {}
    for k in ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound'):
        va, vb = a['values'][k], b['values'][k]
        failed_a, failed_b = va < 1 - 1e-8, vb < 1 - 1e-8
        statistics['safety_transitions'][k] = {'control_failure_count': int(failed_a.sum()),
            'treatment_failure_count': int(failed_b.sum()), 'new_failures': int((~failed_a & failed_b).sum()),
            'repaired_failures': int((failed_a & ~failed_b).sum()), 'denominator': 1696}
    delta = b['values']['score'] - a['values']['score']
    statistics['scene_direction'] = {'improved': int((delta > 1e-8).sum()),
        'degraded': int((delta < -1e-8).sum()), 'tie': int((np.abs(delta) <= 1e-8).sum())}
    filename = 'PAIRED_512_STATISTICS.json' if treatment == 'C1' else 'PAIRED_C2_C0_512_STATISTICS.json'
    (OUT / filename).write_text(json.dumps(statistics, indent=2, sort_keys=True) + '\n')
    paired_deltas[treatment] = statistics['metrics']['score']['delta']
for name, rows in [('METRICS.csv', aggregate), ('BY_LOG_AGGREGATES.csv', by_log)]:
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
evidence = {'status': 'VERIFIED_READ_ONLY_SNAPSHOT', 'source_commit': 'da11402a449e724416fad02fd53a655d6eb4efa0',
    'analysis_runtime': str(RUNTIME), 'analysis_source_commit': policy['execution_runtime']['source_sha'],
    'script_sha256': file_sha256(Path(__file__)), 'pins': pins,
    'score_protocol_sha256': plan['score_protocol_sha256'],
    'checkpoints': {f'{a}_{s}': r['checkpoint'] for (a, s), r in runs.items()},
    'failed_attempts': {f'{a}_{s}': r['failed_attempts'] for (a, s), r in runs.items()},
    'coverage': 'Full1696/16logs each; CSV/exports/prediction arrays/admissions/ledger result hashes/reference v2 checked by frozen load_run',
    'boundary': 'Native weight and metric-cache payload bytes not reopened; their registered receipts are reused.',
    'C2': 'Read only after COMPLETE and explicit Lead addendum; matched to C0_512 using additive registered execution plan',
    'training_calls': 0, 'scoring_calls': 0, 'GPU_calls': 0,
    'wall_seconds': time.monotonic() - START, 'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    'cpu_affinity_count': len(os.sched_getaffinity(0))}
(OUT / 'EVIDENCE_PINS.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
print(json.dumps({'paired_deltas': paired_deltas,
                  'wall_seconds': evidence['wall_seconds'], 'peak_RSS_MiB': evidence['peak_RSS_MiB']}), flush=True)
