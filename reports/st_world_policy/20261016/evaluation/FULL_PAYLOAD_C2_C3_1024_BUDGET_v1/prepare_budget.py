"""Fixed C0/C2/C3@1024 stat preflight and at most one validated result record.

No analysis, scoring, models, GPU, images, teacher caches or native-weight reads.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
START = time.monotonic()
ROOT = Path('/mnt/project/st-world-policy-20261016')
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
OUT = Path(__file__).resolve().parent
assert not (OUT / 'BUDGET.json').exists(), 'Sealed preparation must not be re-read or overwritten'
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
PLAN = OUT.parent / 'full_dev_C4_execution_v1/EXECUTION_PLAN.json'
SOURCE_HASH = '3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3'
COMPLETE_PINS = {'C0': '3a06edfabcb2f0fab62b055c290097abaae8d54d8dfba85551d2e37e4782738f',
    'C2': '3eeaa46c74d0ab6c9fbe4493848896250ea4811ba267aa2e92f3e24b043f3f03',
    'C3': '8076856d4c88d80bb711285bbf00cb5ebba36e9af95a6699affe0fa95afb5b2c'}
METADATA = {}
META_BYTES = 0
PAYLOAD_BYTES = 0
SAMPLE_PATH = None
SAMPLE_OPENED = False


def guard(event, args):
    global SAMPLE_OPENED
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0]))
    if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
        return
    if path.suffix.lower() in ('.pt', '.pth', '.npz', '.npy', '.pkl', '.gz', '.safetensors', '.jpg', '.jpeg', '.png'):
        assert SAMPLE_PATH is not None and path.resolve() == SAMPLE_PATH and not SAMPLE_OPENED, 'Unapproved payload read'
        SAMPLE_OPENED = True


sys.addaudithook(guard)


def raw(path):
    global META_BYTES
    path = Path(path).resolve()
    assert path.suffix in ('.json', '.py', '.csv') and path.stat().st_size <= 32 * 2**20
    if path not in METADATA:
        assert META_BYTES + path.stat().st_size <= 128 * 2**20
        METADATA[path] = path.read_bytes()
        META_BYTES += len(METADATA[path])
    return METADATA[path]


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def read(path):
    return json.loads(raw(path))


def pin(path):
    return {'path': str(path), 'sha256': sha(path)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stats(paths):
    rows = []
    for path in paths:
        s = path.stat()
        rows.append({'path': str(path), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512})
    assert rows
    return {'files': len(rows), 'logical_bytes': sum(r['bytes'] for r in rows),
            'allocated_bytes': sum(r['allocated_bytes'] for r in rows),
            'minimum_file_bytes': min(r['bytes'] for r in rows),
            'maximum_file_bytes': max(r['bytes'] for r in rows),
            'stat_inventory_sha256': fingerprint(rows), 'payload_hashes_recomputed': False}


assert sha(RUNTIME / 'tools/st_world_policy/evaluate_campaign.py') == SOURCE_HASH
plan = read(PLAN)
assert sha(PLAN) == 'dddf0077c272e1310b873321006d469fcc261f82620ef6804135067bc40fca6c'
assert plan['statistics']['bootstrap_replicates'] == 2000 and plan['statistics']['bootstrap_seed'] == 20261017
population_ref = plan['populations']['dev']['manifest']
assert sha(population_ref['path']) == population_ref['sha256']
selected = read(population_ref['path'])
expected = {r['token']: r['log'] for r in selected['scenes']}
assert len(expected) == 1696 and len(set(expected.values())) == 16 and selected['split'] == 'dev'
csv_result = OUT.parent / 'FIXED_PAIRS_C2_C3_1024_v1/RESULTS.json'
assert sha(csv_result) == '730b387d46220990398c310854e345ae6e02010c4d42854c02504ea568d21695'

runs = {}
first_sample = None
for arm in ('C0', 'C2', 'C3'):
    job_id = f'{arm}_step001024_dev1696_seed42_v1'
    job = ROOT / 'formal/evaluation/jobs' / job_id
    score = job / 'official_score'
    complete = read(score / 'COMPLETE.json')
    assert sha(score / 'COMPLETE.json') == COMPLETE_PINS[arm]
    assert complete['status'] == 'COMPLETE' and complete['valid_scored'] == 1696 and complete['invalid'] == complete['unresolved'] == 0
    identity = read(score / 'IDENTITY.json')
    assert identity['identity'] == complete['identity'] == fingerprint({k: v for k, v in identity.items() if k != 'identity'})
    requests = {r['scene_id']: r for r in identity['request_identities']}
    assert set(requests) == set(expected) and identity['retry_attempts'] == 1
    ledger_identity = read(score / 'ledger/IDENTITY.json')
    assert ledger_identity['identity'] == identity['identity']
    result_paths, represented, admitted_calls, complete_calls = [], {}, 0, 0
    groups = sorted(p for p in (score / 'ledger').iterdir() if p.is_dir())
    sizes = []
    for folder in groups:
        batch = read(folder / 'REQUESTS.json')
        items = batch['requests']
        assert fingerprint(items) == folder.name
        sizes.append(len(items))
        for req in items:
            assert req == requests[req['scene_id']] and req['source_log'] == expected[req['scene_id']]
            assert req['scene_id'] not in represented
            represented[req['scene_id']] = req['source_log']
        for attempt in range(2):
            directory = folder / f'attempt_{attempt:02d}'
            admission_path = directory / 'ADMITTED.json'
            if not admission_path.exists():
                continue
            admission = read(admission_path)
            assert admission['request_hash'] == folder.name and admission['calls'] == len(admission['indices'])
            admitted_calls += admission['calls']
            assert (directory / 'COMPLETE.json').exists(), 'Unresolved metadata needs a separate budget review'
            receipt = read(directory / 'COMPLETE.json')
            path = directory / 'RESULTS.pt'
            assert receipt['admission_sha256'] == sha(admission_path)
            assert Path(receipt['result']['path']).resolve() == path.resolve()
            result_paths.append(path)
            complete_calls += admission['calls']
            if first_sample is None:
                first_sample = (path, receipt, admission, items)
    assert represented == expected and admitted_calls == complete_calls == 1696
    predictions = [job / 'export/predictions' / (token + '.npz') for token in sorted(expected)]
    prediction_receipts = [p.with_suffix('.json') for p in predictions]
    runs[arm] = {'job_id': job_id, 'score_complete': pin(score / 'COMPLETE.json'), 'score_identity': pin(score / 'IDENTITY.json'),
                 'export_identity': pin(job / 'export/IDENTITY.json'), 'ledger_identity': pin(score / 'ledger/IDENTITY.json'),
                 'ledger': stats(result_paths), 'predictions': stats(predictions), 'prediction_receipts_stat_only': stats(prediction_receipts),
                 'ledger_groups': len(groups), 'requests_per_group': sorted(set(sizes)),
                 'represented_scenes_from_metadata': len(represented), 'logs': len(set(represented.values())),
                 'admitted_calls': admitted_calls, 'completed_calls_from_metadata': complete_calls,
                 'result_records_not_yet_fully_loaded_or_verified': True}

gt_root = Path(plan['populations']['dev']['current_root'])
assert sha(gt_root / 'ego_identity.json') == selected['ego_identity_sha256']
ego_identity = read(gt_root / 'ego_identity.json')
assert ego_identity['schema'] == 'foresight_ego_labels_v1' and ego_identity['split'] == 'dev'
assert ego_identity['index_sha256'] == selected['source_index_hash']
gt = stats([gt_root / 'ego' / (token + '.pt') for token in sorted(expected)])
gt['identity'] = pin(gt_root / 'ego_identity.json')

# One physical bundle contains four original records. Validate only the first;
# disclose the other three materialized records without inspecting their arrays.
path, receipt, admission, items = first_sample
assert path.stat().st_size <= 16 * 2**20
SAMPLE_PATH = path.resolve()
payload = path.read_bytes()
PAYLOAD_BYTES += len(payload)
assert hashlib.sha256(payload).hexdigest() == receipt['result']['sha256']
sys.path.insert(0, str(RUNTIME))
import numpy as np
import torch
from starVLA.model.modules.st_world.consequence import ConsequenceRequest, STATE_COLUMNS, trajectory_sha256, validated_reference_input
torch.set_num_threads(2)
sample_start = time.monotonic()
bundle = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
assert len(bundle) == len(admission['indices']) == 4
record = bundle[0]
request_identity = items[admission['indices'][0]]
assert record['identity'] == request_identity
assert trajectory_sha256(record['requested_trajectory']) == request_identity['trajectory_sha256']
request = ConsequenceRequest(**request_identity, trajectory=record['requested_trajectory'])
validated_reference_input(record, request)
states = record['executed_states']
assert states.shape == (41, 11) and states.dtype.str == '<f8' and np.isfinite(states).all()
assert tuple(record['state_columns']) == STATE_COLUMNS
assert np.array_equal(record['executed_state_times_seconds'], np.arange(41) * .1)
assert np.isfinite([record['progress']['raw_meters'], record['progress']['reference_raw_meters']]).all()
assert np.isfinite(np.hypot(states[1:, 3], states[1:, 4]).mean())
sample = {'status': 'ONE_RECORD_SCHEMA_AND_REFERENCE_V2_VALIDATED_NOT_A_POPULATION_ESTIMATE',
    'physical_bundle': {'path': str(path), 'sha256': receipt['result']['sha256'], 'bytes': len(payload)},
    'receipt': pin(path.parent / 'COMPLETE.json'), 'records_materialized_by_original_bundle_format': len(bundle),
    'records_validated': 1, 'other_records_inspected': 0, 'record_index_in_bundle': 0,
    'selection_rule': 'First lexicographic C0 ledger group, first admitted result; no outcome-based selection',
    'scene_id': request_identity['scene_id'], 'source_log': request_identity['source_log'],
    'schema': record['schema'], 'reference_contract_schema': record['reference_contract']['schema'],
    'executed_states_shape': list(states.shape), 'executed_states_dtype': states.dtype.str,
    'state_columns': list(STATE_COLUMNS), 'time_samples': 41, 'time_step_seconds': .1,
    'progress_fields': sorted(record['progress']), 'raw_and_reference_progress_finite': True,
    'metric_values_and_stop_classification_not_reported': True, 'deserialization_and_single_validation_seconds': time.monotonic() - sample_start}

def payload_bound(arm):
    # Hash plus two np.load openings; ledger hash plus torch.load; GT hash+load.
    return 3 * runs[arm]['predictions']['logical_bytes'] + 2 * runs[arm]['ledger']['logical_bytes'] + 2 * gt['logical_bytes']

per_run = {arm: payload_bound(arm) for arm in runs}
pair_bytes = {f'{arm}_minus_C0': per_run['C0'] + per_run[arm] for arm in ('C2', 'C3')}
unique_bytes = sum(r['predictions']['logical_bytes'] + r['ledger']['logical_bytes'] for r in runs.values()) + gt['logical_bytes']
budget = {'schema': 'st_fixed_full_payload_budget_preparation_v1', 'status': 'PREPARED_NOT_AUTHORIZED_NOT_EXECUTED',
    'source_sha': SOURCE, 'runtime': str(RUNTIME), 'analysis_source': pin(RUNTIME / 'tools/st_world_policy/evaluate_campaign.py'),
    'source_files': {name: pin(RUNTIME / name) for name in ['tools/st_world_policy/export_world.py', 'tools/st_world_policy/score_world.py', 'starVLA/model/modules/st_world/consequence.py']},
    'plan': pin(PLAN), 'prior_csv_audit': pin(csv_result), 'population': population_ref,
    'comparisons': ['C2_minus_C0', 'C3_minus_C0'], 'stage': 'DIAGNOSTIC_1024', 'scenes_per_pair': 1696, 'logs': 16,
    'statistics_unchanged': plan['statistics'], 'runs': runs, 'gt_unique_payload_stat_only': gt,
    'payload_read_estimate': {'unique_existing_content_bytes': unique_bytes,
        'full_file_read_open_multipliers': {'prediction_npz': 3, 'ledger_results_pt': 2, 'ego_gt_pt_per_run': 2},
        'per_run_load_plus_trajectory_diagnostics_bytes': per_run, 'two_original_cli_pair_bytes': pair_bytes,
        'two_original_cli_total_bytes': sum(pair_bytes.values()),
        'three_unique_run_function_calls_no_extra_gt_cache_bytes': sum(per_run.values()),
        'scope': 'Conservative full file size per source read-open; not measured physical I/O. Original CLI reruns C0 and GT for the second pair. Compressed container seek/read behavior may differ.'},
    'sample': sample, 'proposal': {'launch_authorized': False, 'execution': 'Two unchanged frozen CLI comparisons, sequential',
        'cpu_threads': 2, 'rss_cap_bytes': 2 * 2**30, 'wall_cap_seconds_per_pair': 1800,
        'wall_cap_seconds_total': 3600, 'payload_read_cap_bytes_total': 512 * 2**20,
        'metadata_source_read_cap_bytes_total': 128 * 2**20, 'output_cap_bytes_per_pair': 32 * 2**20,
        'full_run_cpu_wall_and_peak_rss': 'NOT_MEASURED; proposed admission ceilings, not a benchmark',
        'no_new_cache': True, 'gpu_count': 0, 'scoring_calls': 0, 'requires_lead_budget_registration': True},
    'preparation_cost': {'metadata_source_unique_bytes_read': META_BYTES, 'payload_file_read_bytes': PAYLOAD_BYTES,
        'payload_file_read_opens': int(SAMPLE_OPENED), 'payload_record_validations': 1,
        'cpu_affinity_count': len(os.sched_getaffinity(0)), 'peak_rss_bytes': int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:'))) * 1024,
        'wall_seconds': time.monotonic() - START},
    'no_population_payload_analysis_performed': True, 'no_new_threshold_or_comparison': True, 'navtest': False}
assert META_BYTES <= 128 * 2**20 and PAYLOAD_BYTES <= 16 * 2**20
assert budget['preparation_cost']['peak_rss_bytes'] < 2 * 2**30
assert sum(pair_bytes.values()) < budget['proposal']['payload_read_cap_bytes_total']
with (OUT / 'BUDGET.json').open('x') as stream:
    stream.write(json.dumps(budget, sort_keys=True, indent=2) + '\n')
print(json.dumps({'payload_estimate': budget['payload_read_estimate'], 'preparation_cost': budget['preparation_cost']}))
