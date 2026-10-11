"""Bounded native A/AA/B captures: continuous 500 updates, original V1 points.

Default is CPU-only planning. --execute runs sequential full8 pilots on the
qualified host after source/resource/storage admission. No formal deployment,
checkpoint deletion, archival, automatic retry or numeric acceptance is provided.
Reuse the benchmark's physical GPU lease and native process monitor.
--cases selects a dispatch phase from the unchanged four-case V1 manifest;
unselected cases remain pending and no phase is formal acceptance.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import benchmark_engineering_execution as bench
from tools.structured_world.native_capture import GiB, observation_plan, storage_admission

SCRIPT = Path(__file__).resolve()
POINTS = (1, 32, 100, 250, 500)
REQUIRED = {('G1', 'early'), ('G3', 'early'), ('G1', 'mid'), ('G3', 'mid')}


class BudgetRefused(ValueError):
    """Boundary refusal only; never terminate a running job."""


def validate_gpu_budget(manifest, jobs):
    """Explicit dispatch snapshot; no guessed external spending or defaults."""
    budget = manifest.get('budget')
    bench.require(isinstance(budget, dict), 'Explicit known GPU-hour budget required at --execute')
    bench.require(budget.get('limit_kind') == 'internal_finite_test_registration',
                  'Explicit internal finite test registration limit_kind required')
    bench.require('user_GPU_hours_limit' in budget, 'Explicit user_GPU_hours_limit disclosure required; null means no user cap')
    user_limit = budget['user_GPU_hours_limit']
    bench.require(user_limit is None or (type(user_limit) in (int, float) and
                  math.isfinite(user_limit) and user_limit > 0), 'User GPU-hour limit must be null or finite positive')
    required = ('phase_limit', 'already_spent_external', 'total_internal_limit', 'reserved_elsewhere')
    for name in required:
        value = budget.get(name)
        bench.require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                      'Explicit finite GPU-hour budget field required: '+name)
    bench.require(0 < budget['phase_limit'] <= budget['total_internal_limit'],
                  'Positive phase budget within explicit finite internal total required')
    bench.require(user_limit is None or budget['total_internal_limit'] <= user_limit,
                  'Internal total must not exceed the explicit user GPU-hour limit')
    reserves = budget.get('per_job_reserves')
    bench.require(isinstance(reserves, dict), 'Explicit per-job GPU-hour reserves required')
    for stage in {j['stage'] for j in jobs}:
        value = reserves.get(stage)
        bench.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                      'Finite positive job reserve required: '+stage)
    for source in budget.get('accounting_sources', []):
        bench.require(bench.digest(source['path']) == source['sha256'], 'External budget accounting source changed')
    return budget


def actual_process_charges(jobs):
    """Charge actual exits, including failures; partial receipt costs are not zero."""
    entries, paths = [], set()
    for job in jobs:
        root = Path(job.get('process_run', job['run']))
        path = root/'PROCESS_EXIT.json'
        bench.require(str(path.resolve()) not in paths, 'Process cost path reused between stages')
        paths.add(str(path.resolve()))
        if not path.exists():
            bench.require(not (root/'PROCESS.json').exists(), 'Started process lacks exit cost; budget unknown, no next launch')
            continue
        record = bench.read(path)
        wall = record.get('wall_seconds')
        bench.require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0 and
                      type(record.get('returncode')) is int, 'Invalid real process exit cost')
        entries.append({'key': job['key'], 'path': str(path), 'sha256': bench.digest(path),
            'returncode': record['returncode'], 'monitor_error': record.get('monitor_error'),
            'wall_seconds': wall, 'GPU_hours': wall*8/3600,
            'charged_despite_failure': record['returncode'] != 0 or record.get('monitor_error') is not None})
    return {'entries': entries, 'actual_phase_GPU_hours': math.fsum(x['GPU_hours'] for x in entries),
            'basis': 'full8 process wall including startup/IO/logging/save; failed exits charged; no native numeric acceptance implied'}


def gpu_admission(budget, jobs, charges, *, whole_plan=False, next_job=None):
    actual = charges['actual_phase_GPU_hours']
    bench.require(type(actual) in (int, float) and math.isfinite(actual) and actual >= 0, 'Unknown actual phase cost')
    completed = {x['key'] for x in charges.get('entries', [])}
    remaining = [j for j in jobs if j['key'] not in completed]
    if next_job is not None:
        bench.require(remaining and remaining[0]['key'] == next_job['key'], 'Only next unstarted job admitted; no retry or skipped stage')
    reserve = math.fsum(budget['per_job_reserves'][j['stage']] for j in remaining) if whole_plan or next_job else 0.
    phase = actual+reserve
    joint = budget['already_spent_external']+budget['reserved_elsewhere']+phase
    if phase > budget['phase_limit'] or joint > budget['total_internal_limit']:
        raise BudgetRefused('GPU-hour boundary refusal: actual '+str(actual)+' + reserve '+str(reserve)+
                            '; phase '+str(phase)+'/'+str(budget['phase_limit'])+
                            '; joint '+str(joint)+'/'+str(budget['total_internal_limit']))
    return {'admitted': True, 'whole_plan': whole_plan, 'next_job': next_job['key'] if next_job else None,
        'actual_phase_GPU_hours': actual, 'requested_reserve_GPU_hours': reserve,
        'remaining_jobs': [j['key'] for j in remaining],
        'next_job_reserve_GPU_hours': budget['per_job_reserves'][next_job['stage']] if next_job else None,
        'projected_phase_GPU_hours': phase, 'projected_joint_GPU_hours': joint,
        'phase_limit': budget['phase_limit'], 'total_internal_limit': budget['total_internal_limit'],
        'limit_kind': budget['limit_kind'], 'user_GPU_hours_limit': budget['user_GPU_hours_limit'],
        'already_spent_external': budget['already_spent_external'], 'reserved_elsewhere': budget['reserved_elsewhere']}


def run_budgeted_job(job, locks, budget, jobs, output):
    """Wait for real owned workers, then reconcile even when the monitor raises."""
    before = actual_process_charges(jobs)
    gate = gpu_admission(budget, jobs, before, next_job=job)
    bench.write(output/(job['key']+'_GPU_BUDGET_BEFORE.json'), gate)
    owned_process = {**job, 'run': job.get('process_run', job['run'])}
    try:
        return bench.run_job(owned_process, locks)
    finally:
        # run_job retains inherited leases and waits for actual child exit on
        # its failures. Every written PROCESS_EXIT contributes, not just results.
        after = actual_process_charges(jobs)
        bench.write(output/(job['key']+'_GPU_COST_AFTER.json'), after)


def native_rng_sources(manifest, selected_cases=None):
    sources = []
    for case in manifest['cases']:
        if selected_cases is not None and case['name'] not in selected_cases:
            continue
        native = bench.origin(case)
        sources.append({'case': case['name'], 'run': native['run'], 'tag': native['tag'],
            'COMPLETE_sha256': native['complete_sha256'], 'origin_identity': native['contract']['identity'],
            'rng_payload_sha256_from_COMPLETE': {n: h for n, h in native['complete']['file_sha256'].items() if n.startswith('rng_rank')},
            'verification': 'Original native trainer independently hashes payloads and checks all-rank recovery boundary; budget provenance is recorded separately'})
    return sources


def select_cases(manifest, requested=None):
    """Validate the overall V1 matrix before selecting host-local resources."""
    cases = manifest['cases']
    bench.require(len(cases) == 4 and {(c['group'], c['stage']) for c in cases} == REQUIRED,
                  'Original V1 requires early and mid native triplets for both G1/G3')
    names = [c['name'] for c in cases]
    bench.require(all(isinstance(n, str) and re.fullmatch('[A-Za-z0-9_.-]+', n) for n in names) and
                  len(set(names)) == len(names), 'Unique safe case name required')
    # Overall metadata must remain a valid full-population matrix, even where
    # an unselected origin/storage file is local to another qualified host.
    # Normalize lexically here; do not resolve/stat those host-local resources.
    outputs = []
    for case in cases:
        args = case['train_args']
        bench.require(len(args) % 2 == 0 and all(isinstance(x, str) for x in args), 'Scientific option/value argv pairs required')
        values = dict(zip(args[::2], args[1::2]))
        bench.require(len(values) == len(args)//2 and set(values) == bench.BASE_FLAGS and values['--micro-batch'] in ('1', '2', '4'),
                      'Exact explicit scientific asset flags; no schedule/subset/stop overrides')
        bench.require(Path(case['pilot_origin_run']).is_absolute() and Path(case['output_root']).is_absolute(), 'Absolute origin/output roots required')
        budget = case['storage_budget']
        bench.require(Path(budget['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}', budget['sha256']), 'Bound frozen storage budget required')
        outputs.append(Path(os.path.normpath(case['output_root'])))
    bench.require(len(set(outputs)) == len(outputs) and
                  all(not a.is_relative_to(b) for a in outputs for b in outputs if a != b),
                  'Overall four-case output roots must be unique and nonoverlapping')
    if requested is None:
        wanted = names
    else:
        bench.require(isinstance(requested, str), '--cases must be a comma-separated case-name string')
        wanted = [n.strip() for n in requested.split(',')]
        bench.require(all(wanted) and len(set(wanted)) == len(wanted) and set(wanted) <= set(names),
                      '--cases requires nonempty unique names present in the full V1 manifest')
    selected = [c for c in cases if c['name'] in wanted]
    stage_counts = lambda subset: {stage: sum(c['stage'] == stage for c in subset) for stage in ('early', 'mid')}
    scope = {'overall_cases': names, 'selected_cases': [c['name'] for c in selected],
        'pending_unselected_cases': [n for n in names if n not in wanted],
        'overall_case_stage_counts': stage_counts(cases), 'selected_case_stage_counts': stage_counts(selected),
        'outstanding_case_stage_counts': stage_counts([c for c in cases if c['name'] not in wanted]),
        'selected_jobs': 3*len(selected), 'outstanding_unselected_jobs': 3*(len(cases)-len(selected)),
        'unselected_output_validation': 'lexical metadata only; no host-local asset stat/resolve',
        'dispatch_scope': 'full_matrix' if len(selected) == len(cases) else 'selected_phase',
        'case_order': 'original manifest order', 'overall_V1_acceptance': 'PENDING',
        'budget_scope': 'selected jobs only; external spending and reservations remain explicit dispatch inputs',
        'unselected_receipts_imported': False, 'passed': False, 'formal_authorized': False}
    return selected, scope


def case_capture_progress(plan, results):
    """Capture receipts cannot certify the undispatched matrix or numeric gates."""
    scope = plan['case_selection']
    received = {r['key'] for r in results}
    complete = [name for name in scope['selected_cases'] if
                {j['key'] for j in plan['jobs'] if j['case'] == name} <= received]
    pending = [n for n in scope['overall_cases'] if n not in complete]
    return {'completed_selected_capture_cases': complete, 'pending_overall_capture_cases': pending,
        'selected_phase_captures_complete': len(complete) == len(scope['selected_cases']),
        'overall_capture_matrix_complete': not pending, 'overall_V1_acceptance': 'PENDING',
        'passed': False, 'formal_authorized': False}


def build_plan(manifest, output, selected_cases=None):
    bench.require(manifest['schema'] == 'native_execution_continuation_manifest_v1', 'Native manifest schema required')
    root = Path(manifest['source_root']).resolve()
    candidate = manifest['candidate_mode']
    bench.require(candidate in bench.CANDIDATES, 'One separate candidate per registered matrix')
    bench.require(len(manifest['gpu_uuids']) == len(set(manifest['gpu_uuids'])) == 8 and
                  all(u.startswith('GPU-') for u in manifest['gpu_uuids']), 'Full8 physical UUIDs required')
    for key in ('source_root', 'python', 'artifact_root'):
        bench.require(Path(manifest[key]).is_absolute(), 'Absolute '+key+' required')
    bench.require(manifest['minimum_free_MiB'] > 0, 'Explicit GPU memory reserve required')
    port = manifest.get('master_port', 29590)
    bench.require(type(port) is int and 1024 <= port <= 65535, 'Valid torchrun port required')
    cases, selection = select_cases(manifest, selected_cases)
    runs, jobs = set(), []
    if 'original_V1_criteria' in manifest:
        binding = manifest['original_V1_criteria']
        bench.require(bench.digest(binding['path']) == binding['sha256'] and binding['changed'] is False, 'Original V1 registration changed')
    for case in cases:
        name = case['name']
        args = case['train_args']
        budget = case['storage_budget']
        bench.require(bench.digest(budget['path']) == budget['sha256'], 'Storage budget changed')
        native = bench.origin(case)  # metadata/inventory only; no large tensor load
        steps = 1 if case['stage'] == 'early' else 500
        start = native['complete']['completed']
        bench.require(start+steps <= native['contract']['schedule_horizon'], 'Continuation exceeds unchanged native schedule')
        offsets = [1] if steps == 1 else list(POINTS[:-1])
        capture = observation_plan(','.join(map(str, offsets)), start, steps)
        plan_budget = bench.read(budget['path'])
        bench.require(plan_budget['host'] == manifest['host'], 'Storage budget host differs')
        for role in ('A', 'AA', 'B'):
            mode = candidate if role == 'B' else 'io_preserving_v1'
            run = (Path(case['output_root'])/role).resolve()
            bench.require(str(run) not in runs and not run.exists(), 'Fresh unique OWN role output required; no overwrite/retry')
            runs.add(str(run))
            owner = next((o for o in plan_budget['owned_runs'] if Path(o['path']).resolve() == run), None)
            banks = 1 if steps == 1 else len(POINTS)
            bench.require(owner is not None and owner['kind'] == 'owned_numerical_pilot' and
                          owner['cap_bytes'] >= banks*plan_budget['checkpoint_upper_bytes'], 'Storage plan must cover every retained native bank')
            bench.require(not run.is_relative_to(Path(case['pilot_origin_run']).resolve()) and
                          not Path(case['pilot_origin_run']).resolve().is_relative_to(run), 'OWN output must not overlap its immutable origin')
            original = args+['--scope', 'profile', '--debug-updates', str(steps),
                '--pilot-origin-run', case['pilot_origin_run'], '--pilot-origin-tag', case['pilot_origin_tag'],
                '--execution-mode', mode, '--capture-updates', ','.join(map(str, offsets)),
                '--capture-storage-budget', budget['path'], '--save-every', '2147483647',
                '--logging-every', '20', '--evidence-every', '100', '--output', str(run)]
            command = [manifest['python'], '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=8',
                '--max_restarts=0', '--master_port='+str(port), '--', str(root/'tools/structured_world/train_vla.py')]+original
            jobs.append({'key': name+'_'+role, 'case': name, 'group': case['group'], 'stage': case['stage'],
                'role': role, 'mode': mode, 'run': str(run), 'trainer': str(root/'tools/structured_world/train_vla.py'),
                'gpu_uuids': manifest['gpu_uuids'], 'command': command, 'capture_plan': capture,
                'retained_global_updates': [start+x for x in ([1] if steps == 1 else POINTS)],
                'steps': steps, 'origin': {k: native[k] for k in ('run', 'tag', 'complete_sha256')},
                'storage_budget': budget, 'training_source_sha': manifest['source_sha']})
    paths = [Path(j['run']) for j in jobs]
    bench.require(all(not a.is_relative_to(b) for a in paths for b in paths if a != b), 'Role run trees overlap')
    return {'schema': 'native_execution_continuation_plan_v1', 'passed': False, 'formal_authorized': False,
        'manifest': manifest, 'case_selection': selection, 'driver_sha256': bench.digest(SCRIPT), 'source_root': str(root),
        'source_sha': manifest['source_sha'], 'maximum_total_updates': sum(j['steps'] for j in jobs),
        'expected_full_native_banks': sum(len(j['retained_global_updates']) for j in jobs), 'jobs': jobs,
        'continuous_mid_updates': 500, 'original_V1_capture_points': list(POINTS),
        'original_V1_numeric_thresholds_and_OLS': 'unchanged; capture success is not acceptance',
        'forced_restarts_at_observations': 0, 'quota_visibility': 'reported, not an invented proof prerequisite'}


def capture_receipt(job, native, process, *, origin_boundary=None, save_updates=None):
    """Small metadata/JSON audit; leave real 42 GiB tensor comparisons to CPU tools."""
    run = Path(job['run'])
    identity, done = bench.read(run/'identity.json'), bench.read(run/'TRAINING_COMPLETE.json')
    boundary = origin_boundary if origin_boundary is not None else bench.read(run/'RESUME_BOUNDARY_VERIFIED.json')
    start, end = native['complete']['completed'], job['retained_global_updates'][-1]
    bench.require(identity['training_source_sha'] == job['training_source_sha'], 'Actual training source differs from registered source')
    bench.require(done['updates'] == end and done['identity'] == identity['identity'], 'Native endpoint not completed')
    bench.require(boundary['passed'] and boundary['completed'] == start and boundary['all_task_RNG'] == 'exact all ranks', 'Origin boundary proof missing')
    bench.require(identity['execution_mode'] == job['mode'] and identity['benchmark_no_end_checkpoint'] is False, 'Wrong mode or capture suppression')
    ancestry = identity['engineering_continue_origin']
    bench.require((ancestry['identity'], ancestry['tag'], ancestry['complete_sha256']) ==
                  (native['contract']['identity'], native['tag'], native['complete_sha256']), 'Native ancestry changed')
    plan = identity['engineering_native_capture']
    bench.require(all(plan[k] == v for k, v in job['capture_plan'].items()) and plan['storage_budget_sha256'] == job['storage_budget']['sha256'], 'Capture identity differs from plan')
    bench.require(bench.digest(run/'optimizer_groups.json') == native['optimizer_groups_sha256'], 'Native optimizer inventory/groups changed')
    for key in ('config', 'cache_identity', 'current_and_GT_identity', 'DINO_identity', 'shared_geometry_identity',
                'training_seed', 'world_size', 'effective_global_batch', 'FM_repeat', 'microbatch',
                'dataset', 'scenes_per_epoch', 'updates_per_epoch', 'determinism', 'precision',
                'schedule_horizon', 'schedule_warmup'):
        bench.require(identity[key] == native['contract'][key], 'Scientific origin field changed: '+key)
    inventory = set(native['complete']['file_sha256'])
    checkpoints = []
    for completed in job['retained_global_updates']:
        tag = f'milestone_{completed:07d}'
        folder = run/'checkpoints'/tag
        complete = bench.read(folder/'COMPLETE.json')
        bench.require(complete['identity'] == identity['identity'] and complete['completed'] == completed and
                      complete['tag'] == tag and complete['ranks'] == 8 and set(complete['file_sha256']) == inventory,
                      'Incomplete/foreign native capture')
        bench.require(all(re.fullmatch('[0-9a-f]{64}', h) and (folder/n).is_file() and (folder/n).stat().st_size > 0
                          for n, h in complete['file_sha256'].items()), 'Native file inventory missing or malformed')
        bench.require(complete['status'] == ('COMPLETE' if completed == end else 'RUNNING'), 'Unexpected pause or interrupted observation')
        checkpoints.append({'path': str(folder), 'COMPLETE_sha256': bench.digest(folder/'COMPLETE.json'),
                            'completed': completed, 'save_seconds': complete['save_seconds'],
                            'checkpoint_GPU_hours': complete['checkpoint_GPU_hours']})
    bench.require((run/'checkpoints'/'latest').read_text().strip() == f'milestone_{end:07d}', 'Latest is not retained endpoint')
    metrics = bench.json_lines(run/'metrics.jsonl')
    bench.require([x['updates'] for x in metrics] == list(range(start+1, end+1)), 'Incomplete metric update population')
    references, logging = [], []
    for rank in range(8):
        path = run/f'capture_denominators_rank{rank}.jsonl'
        rows = bench.json_lines(path)
        bench.require([x['updates'] for x in rows] == list(range(start+1, end+1)), 'Missing/duplicate all-rank count evidence')
        for row, metric in zip(rows, metrics):
            count = row['global_counts']
            bench.require(count == metric['global_counts'] and count.get('ego_scenes') == len(row['global_scene_indices']), 'Global denominators differ or scene coverage wrong')
            bench.require(all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in count.values()), 'Invalid global denominator')
            bench.require(row['rank'] == rank and row['local_scene_indices'] == row['global_scene_indices'][rank::8]
                          and row['source_population_scenes'] == 101592 and row['schedule_horizon'] == 100000
                          and row['schedule_warmup'] == 5000 and row['continuation_endpoint'] == end,
                          'Rank scene/schedule evidence differs')
            bench.require(len(row['global_scene_indices']) == len(set(row['global_scene_indices'])) and
                          all(type(v) is int and 0 <= v < 101592 for v in row['global_scene_indices']), 'Scene population invalid')
            expected_horizons = ({'dino_horizon'} if 'future_dino' in count else set()) | ({'visual_horizon'} if 'visual' in count else set())
            bench.require(set(row['loss_horizons']) == expected_horizons and
                          all(len(v) == len(row['local_scene_indices']) and all(type(h) is int and 0 <= h < 3 for h in v)
                              for v in row['loss_horizons'].values()), 'Actual selected loss-horizon evidence incomplete')
        references.append({'rank': rank, 'path': str(path), 'sha256': bench.digest(path), 'updates': len(rows)})
        cost = bench.read(run/f'capture_logging_rank{rank}.json')
        bench.require(cost['completed'] == end and cost['rank'] == rank and math.isfinite(cost['logging_wall_seconds']) and
                      cost['logging_wall_seconds'] >= 0, 'Pilot logging cost missing')
        logging.append(cost)
    admissions = bench.json_lines(run/'capture_storage_admissions.jsonl')
    bench.require([x['completed'] for x in admissions] == (job['retained_global_updates'] if save_updates is None else save_updates) and
                  all(x['ok'] and x['budget_sha256'] == job['storage_budget']['sha256'] for x in admissions), 'Save admissions incomplete')
    return {'schema': 'native_execution_capture_receipt_v1', 'key': job['key'], 'captures_complete': True,
        'passed': False, 'qualification': 'PENDING actual native tensor/RNG comparisons, triplet count/horizon equality, canonical dev and recovery',
        'role': job['role'], 'mode': job['mode'], 'run_identity': identity['identity'],
        'origin': job['origin'], 'training_source_sha': identity['training_source_sha'],
        'effective_training_precision': identity['effective_training_precision'], 'checkpoint_payload_hashes_verified_here': False,
        'hash_verification': 'Native save bound every payload; comparator must independently verify full payload hashes before numerical acceptance',
        'checkpoints': checkpoints, 'count_evidence': references, 'logging_cost_by_rank': logging,
        'storage_admission_wall_seconds_rank0': sum(x['admission_wall_seconds'] for x in admissions),
        'checkpoint_save_seconds': sum(x['save_seconds'] for x in checkpoints),
        'process_wall_seconds': process['wall_seconds'], 'process_GPU_hours': process['wall_seconds']*8/3600,
        'process_wall_includes': 'restore/hash/startup, all pilot count logs, storage checks, native save/hash and training; not a throughput benchmark'}


def count_triplets(results):
    """Discrete E1 evidence only; never promote this to numeric acceptance."""
    fields = ('updates', 'epoch', 'offset_after_update', 'global_counts', 'schedule_horizon',
              'schedule_warmup', 'continuation_endpoint', 'source_population_scenes',
              'rank', 'global_scene_indices', 'local_scene_indices', 'loss_horizons')
    cases = {}
    for result in results:
        case = result['key'].rsplit('_', 1)[0]
        cases.setdefault(case, {})[result['role']] = result
    evidence = []
    for case, roles in cases.items():
        bench.require(set(roles) == {'A', 'AA', 'B'}, 'Incomplete discrete-evidence triplet')
        for rank in range(8):
            rows = {role: bench.json_lines(value['count_evidence'][rank]['path']) for role, value in roles.items()}
            wanted = [{k: r[k] for k in fields} for r in rows['A']]
            bench.require(all([{k: r[k] for k in fields} for r in rows[role]] == wanted for role in ('AA', 'B')),
                          'E1 global denominator/scene/loss-horizon drift: '+case+' rank'+str(rank))
        evidence.append({'case': case, 'all8_count_scene_horizon_exact': True,
                         'numeric_checkpoint_RNG_equality': 'PENDING independent native comparator'})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', help='Comma-separated case names from the full V1 manifest; default: all four')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    bench.require(not output.exists(), 'Fresh controller directory required; no retry/overwrite')
    manifest = bench.read(args.manifest)
    plan = build_plan(manifest, output, args.cases)
    plan['manifest_sha256'] = bench.digest(args.manifest)
    bench.immutable_source(plan['source_root'], plan['source_sha'])
    bench.require("'--capture-updates'" in Path(plan['jobs'][0]['trainer']).read_text(), 'Freeze a new trainer with pilot capture support')
    output.mkdir(parents=True, exist_ok=False)
    bench.write(output/'PLAN.json', plan)
    if not args.execute:
        print(json.dumps({'status': 'PLANNED_ONLY', 'jobs': len(plan['jobs']), 'native_banks': plan['expected_full_native_banks'],
                          'maximum_total_updates': plan['maximum_total_updates'], 'case_selection': plan['case_selection'],
                          'plan': str(output/'PLAN.json')})); return 0
    complete_status = ('CAPTURES_COMPLETE_ACCEPTANCE_PENDING' if plan['case_selection']['dispatch_scope'] == 'full_matrix'
                       else 'PHASE_CAPTURES_COMPLETE_OVERALL_V1_PENDING')
    results, status, error, stop, discrete = [], complete_status, None, [False], []
    def stop_after_current(*unused): stop[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop_after_current)
    gpu_budget = None
    try:
        gpu_budget = validate_gpu_budget(manifest, plan['jobs'])
        initial = gpu_admission(gpu_budget, plan['jobs'], actual_process_charges(plan['jobs']), whole_plan=True)
        bench.write(output/'GPU_BUDGET_REGISTRATION.json', {'budget': gpu_budget, 'admission': initial,
            'case_selection': plan['case_selection'],
            'manifest_sha256': bench.digest(args.manifest), 'budget_block_sha256': hashlib.sha256(json.dumps(gpu_budget, sort_keys=True).encode()).hexdigest(),
            'external_counters': 'Explicit manifest dispatch snapshot; not inferred from RNG or successful job receipts'})
        bench.write(output/'NATIVE_RNG_SOURCES.json', {'sources': native_rng_sources(manifest, plan['case_selection']['selected_cases']),
            'case_selection': plan['case_selection']})
        driver_root = SCRIPT.parents[2]
        driver_sha = bench.checked(['git', 'rev-parse', 'HEAD'], driver_root)
        bench.immutable_source(driver_root, driver_sha)
        q = manifest['qualification']
        bench.require(bench.digest(q['path']) == q['sha256'], 'Qualification record changed')
        qualification = bench.read(q['path'])
        bench.require(qualification['passed'] and qualification['host'] == manifest['host'] and
                      set(qualification['physical_gpu_UUIDs']) == set(manifest['gpu_uuids']), 'Full8 qualification mismatch')
        env = bench.runtime_snapshot(manifest['python'])
        for name, value in qualification.get('environment', {}).get('packages', {}).items():
            bench.require(env['packages'].get(name) == value, 'Qualified private environment changed: '+name)
        bench.write(output/'REGISTRATION.json', {'driver_source_sha': driver_sha, 'driver_sha256': bench.digest(SCRIPT),
            'training_source_sha': plan['source_sha'], 'manifest_sha256': bench.digest(args.manifest),
            'qualification_sha256': q['sha256'], 'environment': env, 'passed': False,
            'case_selection': plan['case_selection'],
            'numeric_criteria': 'original V1 unchanged; no new tensor format, no formal acceptance'})
        with ExitStack() as stack:
            root = Path(manifest['artifact_root'])
            physical = hashlib.sha256(json.dumps(sorted(manifest['gpu_uuids'])).encode()).hexdigest()
            locks = [bench.acquire(stack, root/'physical_gpu_locks'/(physical+'.lock')),
                     bench.acquire(stack, root/'native_capture_locks'/'storage_io.lock')]
            for job in plan['jobs']:
                if stop[0] or (output/'STOP_BEFORE_NEXT_JOB').exists(): status = 'STOPPED_BEFORE_NEXT_JOB'; break
                bench.immutable_source(plan['source_root'], plan['source_sha'])
                bench.require(bench.digest(SCRIPT) == plan['driver_sha256'] and bench.digest(q['path']) == q['sha256'], 'Source/qualification changed')
                case = next(c for c in manifest['cases'] if c['name'] == job['case'])
                native = bench.origin(case)
                gate = bench.readiness(manifest)
                bench.write(output/(job['key']+'_RESOURCE_GATE.json'), gate)
                storage = storage_admission(job['storage_budget']['path'], job['run'], expected_digest=job['storage_budget']['sha256'])
                bench.write(output/(job['key']+'_STORAGE_GATE.json'), storage)
                process = run_budgeted_job(job, locks, gpu_budget, plan['jobs'], output)
                bench.require(bench.origin(case) == native, 'Read-only native origin changed during pilot')
                receipt = capture_receipt(job, native, process)
                results.append(receipt)
                bench.write(output/(job['key']+'_CAPTURE.json'), receipt)
                print(json.dumps({'job': job['key'], 'captures_complete': True, 'acceptance': 'PENDING'}), flush=True)
            if len(results) == len(plan['jobs']):
                discrete = count_triplets(results)
                bench.write(output/'COUNT_SCENE_HORIZON_TRIPLETS.json', {'discrete_evidence': discrete, 'passed': False,
                    'case_selection': plan['case_selection'],
                    'numeric_acceptance': 'PENDING; exact data/count evidence alone is insufficient'})
    except BaseException as exc:
        status, error = ('BUDGET_REFUSED_BEFORE_NEXT_JOB' if isinstance(exc, BudgetRefused) else 'FAILED'), repr(exc)
        bench.write(output/'FAILURE.json', {'error': error, 'completed_jobs': len(results),
            'preceding_checkpoints_preserved': True, 'no_cleanup_or_retry': True})
    try:
        costs = actual_process_charges(plan['jobs'])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        costs = {'actual_phase_GPU_hours': None, 'cost_unknown': True, 'error': repr(exc)}
    bench.write(output/'RESULTS.json', {'status': status, 'passed': False, 'formal_authorized': False,
        'error': error, 'jobs': results, 'planned_jobs': len(plan['jobs']),
        'case_selection': plan['case_selection'], 'case_capture_progress': case_capture_progress(plan, results),
        'gpu_budget': gpu_budget, 'actual_process_costs_including_failed': costs,
        'final_budget_within_limits': (costs['actual_phase_GPU_hours'] is not None and gpu_budget is not None and
            costs['actual_phase_GPU_hours'] <= gpu_budget['phase_limit'] and
            costs['actual_phase_GPU_hours']+gpu_budget['already_spent_external']+gpu_budget['reserved_elsewhere'] <= gpu_budget['total_internal_limit']),
        'discrete_evidence': discrete,
        'capture_only_not_numerical_acceptance': True})
    print(json.dumps({'status': status, 'completed_jobs': len(results), 'error': error}), flush=True)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
