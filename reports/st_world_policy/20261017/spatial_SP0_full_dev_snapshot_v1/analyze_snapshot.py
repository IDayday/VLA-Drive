"""Read-only SP0 endpoint validation through the registered frozen evaluator."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
signal.alarm(120)
START = time.monotonic()
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
sys.path.insert(0, str(RUNTIME))
import numpy as np
from tools.st_world_policy.evaluate_campaign import (
    METRICS, file_sha256, load_population, load_run, pinned_json, read_json)

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016/formal/evaluation')
JOB_ID = 'SP0_step005000_dev1696_seed42_v1'
job = ROOT / 'jobs' / JOB_ID
reg_path = ROOT / 'registrations' / JOB_ID / 'REGISTRATION.json'
registration = read_json(reg_path)
policy = pinned_json(registration['job']['evaluation_policy'])
plan = pinned_json(policy['analysis']['plan'])
population_path = policy['population_manifest']['path']
assert file_sha256(population_path) == policy['population_manifest']['sha256']
selected = load_population(plan, population_path, 'FORMAL')
route = pinned_json(registration['route'])
assert registration['route']['sha256'] == 'aae92a88866f70c0efbac1230cd511a86097601511d90bcda11fc28a5c775bc0'
assert route['checkpoint_tag'] == 'step_005000_1792217229597178910'
assert route['source_sha'] == policy['execution_runtime']['source_sha'] == '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
assert route['expected_completed'] == registration['job']['expected_completed'] == 5000
assert registration['navtest'] is False and registration['ACTIVE_update'] is False
for item in policy['execution_runtime']['source_files']:
    assert file_sha256(item['path']) == item['sha256']
config = pinned_json(registration['job']['training_config'])
native_receipt = pinned_json(registration['complete_receipt'])
assert native_receipt['schema'] == 'st_native_checkpoint_v1'
assert native_receipt['files'] == registration['native_files']
assert native_receipt['receipt_identity'] == registration['receipt_identity'] == route['receipt_identity']
assert native_receipt['state']['completed'] == 5000
assert native_receipt['state']['tag'] == route['checkpoint_tag']
assert native_receipt['state']['identity'] == route['runtime_identity']
assert config['space_time']['cell'] == 'S-P0'
assert config['execution']['stop_after'] == 5000
assert config['execution']['max_updates'] == 20000
run = load_run(job / 'official_score', job / 'export', selected,
               population_path, plan, 'FORMAL', 'SP0', 'SCREENING')
assert run['export']['checkpoint']['completed'] == 5000
assert run['export']['checkpoint']['tag'] == route['checkpoint_tag']
assert run['export']['checkpoint']['native_receipt_sha256'] == registration['complete_receipt']['sha256']
assert run['export']['checkpoint']['receipt_identity'] == route['receipt_identity']
assert run['export']['checkpoint']['run_identity'] == route['runtime_identity']
complete = read_json(job / 'official_score/COMPLETE.json')
tokens = sorted(run['rows'])
logs = np.asarray([run['rows'][s]['source_log'] for s in tokens])
values = {k: np.asarray([run['rows'][s][k] for s in tokens]) for k in METRICS}
assert len(tokens) == 1696 and len(set(logs)) == 16
epsilon = plan['statistics']['zero_epsilon']
zero = int((values['score'] <= epsilon).sum())
assert zero == complete['aggregate']['zero_count']
for k in METRICS:
    assert abs(float(values[k].mean()) - complete['aggregate']['metrics'][k]) < 1e-14
aggregate = {'arm': 'SP0', 'step': 5000, 'stage': 'SCREENING', 'scenes': len(tokens),
    'logs': len(set(logs)), **{k: float(v.mean()) for k, v in values.items()},
    'zero_count': zero, 'zero_rate': zero / len(tokens), 'invalid': complete['invalid'],
    'unresolved': complete['unresolved']}
by_log = []
for log in sorted(set(logs)):
    ids = logs == log
    row = {'source_log': log, 'scenes': int(ids.sum()),
        **{k: float(v[ids].mean()) for k, v in values.items()},
        'zero_count': int((values['score'][ids] <= epsilon).sum())}
    assert row['scenes'] == complete['by_log'][log]['scenes']
    assert row['zero_count'] == complete['by_log'][log]['zero_count']
    for k in METRICS:
        assert abs(row[k] - complete['by_log'][log]['metrics'][k]) < 1e-14
    by_log.append(row)
for name, rows in [('METRICS.csv', [aggregate]), ('BY_LOG_AGGREGATES.csv', by_log)]:
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
pins = [registration['job']['evaluation_policy'], policy['analysis']['plan'],
    policy['population_manifest'], registration['route'], registration['complete_receipt'],
    registration['job']['training_config'], registration['w_launch_supplement'],
    *policy['execution_runtime']['source_files']]
for suffix in ['official_score/IDENTITY.json', 'official_score/COMPLETE.json',
               'official_score/SCENES.csv', 'export/IDENTITY.json', 'export/COMPLETE_rank0.json']:
    path = job / suffix
    pins.append({'path': str(path), 'sha256': file_sha256(path)})
pins.append({'path': str(reg_path), 'sha256': file_sha256(reg_path)})
invocation = ROOT / 'registrations' / JOB_ID / 'score_invocations/1792221238235922913'
launch = read_json(invocation / 'LAUNCH.json')
host = subprocess.check_output(['hostname'], text=True).strip()
assert host == launch['host'] and launch['pid'] == 443185
ps = subprocess.run(['ps', '-p', '443185', '-o', 'pid,lstart,stat,args'], capture_output=True, text=True)
last_line = (invocation / 'SCORE.log').read_text().strip().splitlines()[-1]
assert json.loads(last_line)['status'] == 'COMPLETE'
for name in ('LAUNCH.json', 'ADMISSION.json', 'SCORE.log'):
    path = invocation / name
    pins.append({'path': str(path), 'sha256': file_sha256(path)})
missing_comparisons = {arm: (ROOT / 'jobs' / f'{arm}_step005000_dev1696_seed42_v1' /
    'official_score/COMPLETE.json').exists() for arm in ('SP1', 'SP2', 'SP3')}
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024 * 1024
evidence = {'status': 'VERIFIED_READ_ONLY_SNAPSHOT', 'checked_utc': datetime.now(timezone.utc).isoformat(),
    'source_commit': '04570f26235b8264fb21b538e2e193cc6d8a699b',
    'analysis_runtime': str(RUNTIME), 'analysis_source_commit': route['source_sha'],
    'script_sha256': file_sha256(__file__), 'pins': pins,
    'score_protocol_sha256': plan['score_protocol_sha256'],
    'checkpoint': run['export']['checkpoint'], 'failed_attempts': run['failed_attempts'],
    'arm_definition': plan['arms']['SP0'], 'training_config': config,
    'component_failure_counts': {k: int((v < 1 - epsilon).sum()) for k, v in values.items() if k != 'score'},
    'other_spatial_complete_receipt_present': missing_comparisons,
    'other_spatial_result_files_read': False, 'bootstrap_performed': False,
    'process_inspection': {'host': host, 'pid': launch['pid'], 'ps_exit_code': ps.returncode,
        'process_present': ps.returncode == 0, 'log_final_status': 'COMPLETE',
        'native_shell_exit_code': None, 'boundary': 'No persisted exit-code receipt; no inferred exit code'},
    'coverage': 'All1696/16logs; registered formal lineage, CSV, predictions, ledger results and reference v2 arrays/hash checked by frozen load_run',
    'boundary': 'Large native weights and metric-cache payload bytes not reopened; original pinned receipts reused. No cross-arm effect estimate.',
    'training_calls': 0, 'scoring_calls': 0, 'GPU_calls': 0, 'dispatch_calls': 0,
    'wall_seconds': time.monotonic() - START,
    'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    'cpu_affinity_count': len(os.sched_getaffinity(0))}
(OUT / 'EVIDENCE_PINS.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': evidence['status'], 'metrics': aggregate,
    'failures': evidence['component_failure_counts'], 'other_complete': missing_comparisons,
    'process_inspection': evidence['process_inspection'],
    'wall_seconds': evidence['wall_seconds'], 'peak_RSS_MiB': evidence['peak_RSS_MiB']}), flush=True)
