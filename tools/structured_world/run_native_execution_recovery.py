"""One native-origin recovery triplet: A32 / AA32 / B16 + B resume to32.

CPU planning by default. Only finite owned profiles are paused. All role
contracts bind capture=32 and endpoint=origin+32; B resume restores its own
native latest16 checkpoint. Training source can remain immutable be5 unchanged.
No numerical acceptance, automatic retry or formal deployment is provided.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import run_native_execution_continuation as native
from tools.structured_world.native_capture import observation_plan, storage_admission
bench = native.bench
SCRIPT = Path(__file__).resolve()


def build_plan(manifest, output):
    bench.require(manifest['schema'] == 'native_execution_recovery_manifest_v1' and len(manifest['cases']) == 1,
                  'Exactly one immutable-origin recovery triplet required')
    case = manifest['cases'][0]
    bench.require(case['group'] in ('G1', 'G3') and manifest['candidate_mode'] in bench.CANDIDATES, 'Registered group/candidate required')
    control_mode = manifest.get('control_mode', 'io_preserving_v1')
    bench.require(control_mode in ('io_preserving_v1', 'engineering_fast') and
                  (control_mode != 'engineering_fast' or manifest['candidate_mode'] == 'engineering_fast'),
                  'Controls must be IO, or engineering_fast paired with an engineering_fast candidate')
    bench.require(Path(case['pilot_origin_run']).is_absolute() and Path(manifest['source_root']).is_absolute() and Path(manifest['python']).is_absolute() and
                  Path(manifest['artifact_root']).is_absolute() and Path(case['output_root']).is_absolute(), 'Absolute source/runtime/artifact/output paths required')
    bench.require(len(manifest['gpu_uuids']) == len(set(manifest['gpu_uuids'])) == 8 and
                  all(u.startswith('GPU-') for u in manifest['gpu_uuids']) and manifest['minimum_free_MiB'] > 0, 'Qualified full8 UUIDs/reserve required')
    args = case['train_args']
    values = dict(zip(args[::2], args[1::2]))
    bench.require(all(isinstance(x, str) for x in args) and len(args) % 2 == 0 and len(values) == len(args)//2 and
                  set(values) == bench.BASE_FLAGS and values['--micro-batch'] in ('1', '2', '4'), 'Exact scientific asset args; no overrides')
    original = bench.origin(case)
    start = original['complete']['completed']
    bench.require(start+32 <= original['contract']['schedule_horizon'], 'Recovery continuation exceeds fixed native schedule')
    budget = case['storage_budget']
    bench.require(Path(budget['path']).is_absolute() and bench.digest(budget['path']) == budget['sha256'], 'Bound storage budget changed')
    storage = bench.read(budget['path'])
    bench.require(storage['host'] == manifest['host'], 'Storage pool belongs to another host')
    port = manifest.get('master_port', 29591)
    bench.require(type(port) is int and 1024 <= port <= 65535, 'Valid torchrun port required')
    root, output = Path(manifest['source_root']).resolve(), Path(output).resolve()
    trainer = root/'tools/structured_world/train_vla.py'
    jobs, roles = [], {}
    capture = observation_plan('32', start, 32)
    for role in ('A', 'AA', 'B'):
        run = (Path(case['output_root'])/role).resolve()
        bench.require(not run.exists() and not run.is_relative_to(Path(case['pilot_origin_run']).resolve()) and
                      not Path(case['pilot_origin_run']).resolve().is_relative_to(run), 'Fresh OWN role cannot overlap origin')
        owner = next((x for x in storage['owned_runs'] if Path(x['path']).resolve() == run), None)
        banks = 2 if role == 'B' else 1
        bench.require(owner and owner['kind'] == 'owned_numerical_pilot' and
                      owner['cap_bytes'] >= banks*storage['checkpoint_upper_bytes'], 'Storage must cover A1/AA1/B2 native banks')
        mode = manifest['candidate_mode'] if role == 'B' else control_mode
        flags = args+['--scope', 'profile', '--debug-updates', '32',
            '--pilot-origin-run', case['pilot_origin_run'], '--pilot-origin-tag', case['pilot_origin_tag'],
            '--execution-mode', mode, '--capture-updates', '32', '--capture-storage-budget', budget['path'],
            '--save-every', '2147483647', '--logging-every', '20', '--evidence-every', '100', '--output', str(run)]
        roles[role] = {'run': str(run), 'flags': flags, 'mode': mode}
    for key, role, extra in (('A32', 'A', []), ('AA32', 'AA', []),
                             ('B16', 'B', ['--stop-after', str(start+16)]), ('B_resume32', 'B', ['--resume'])):
        entry = roles[role]
        command = [manifest['python'], '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=8',
            '--max_restarts=0', '--master_port='+str(port), '--', str(trainer)]+entry['flags']+extra
        jobs.append({'key': key, 'case': case['name'], 'role': role, 'stage': 'recovery', 'mode': entry['mode'],
            'run': entry['run'], 'process_run': str(output/'processes'/key), 'command': command,
            'trainer': str(trainer), 'gpu_uuids': manifest['gpu_uuids'], 'training_source_sha': manifest['source_sha'],
            'capture_plan': capture, 'retained_global_updates': [start+32], 'storage_budget': budget,
            'origin': {k: original[k] for k in ('run', 'tag', 'complete_sha256')},
            'expected_resume_completed': start+16 if extra == ['--resume'] else start,
            'expected_exit_completed': start+16 if key == 'B16' else start+32})
    return {'schema': 'native_execution_recovery_plan_v1', 'manifest': manifest, 'jobs': jobs,
        'source_root': str(root), 'source_sha': manifest['source_sha'], 'driver_sha256': bench.digest(SCRIPT),
        'fixed_relative_endpoint': 32, 'maximum_total_optimizer_updates': 96, 'full_native_banks': 4,
        'storage_cap_example_GiB': {'A': 45, 'AA': 45, 'B': 90, 'total': 180},
        'passed': False, 'formal_authorized': False, 'numeric_acceptance': 'PENDING actual native receipts'}


def stage_snapshot(job, original):
    """Preserve small stage evidence before B resume overwrites boundary/cost JSON."""
    run = Path(job['run'])
    identity = bench.read(run/'identity.json')
    boundary = bench.read(run/'RESUME_BOUNDARY_VERIFIED.json')
    completed = job['expected_exit_completed']
    tag = ('latest_' if job['key'] == 'B16' else 'milestone_')+f'{completed:07d}'
    folder = run/'checkpoints'/tag
    complete = bench.read(folder/'COMPLETE.json')
    bench.require(identity['training_source_sha'] == job['training_source_sha'] and identity['execution_mode'] == job['mode'], 'Unexpected stage source/mode')
    plan = identity['engineering_native_capture']
    bench.require(all(plan[k] == v for k, v in job['capture_plan'].items()) and
                  plan['storage_budget_sha256'] == job['storage_budget']['sha256'], 'Stage capture identity/end changed')
    ancestry = identity['engineering_continue_origin']
    bench.require((ancestry['identity'], ancestry['tag'], ancestry['complete_sha256']) ==
                  (original['contract']['identity'], original['tag'], original['complete_sha256']), 'Stage origin ancestry changed')
    bench.require(boundary['passed'] and boundary['completed'] == job['expected_resume_completed'] and
                  boundary['identity'] == identity['identity'] and boundary['all_task_RNG'] == 'exact all ranks', 'Stage native resume boundary missing')
    inventory = original['complete']['file_sha256']
    bench.require(complete['completed'] == completed and complete['identity'] == identity['identity'] and complete['ranks'] == 8 and
                  complete['tag'] == tag and set(complete['file_sha256']) == set(inventory) and
                  complete['status'] == ('PAUSED' if job['key'] == 'B16' else 'COMPLETE'), 'Stage native exit missing')
    bench.require(all((folder/name).is_file() and (folder/name).stat().st_size > 0 for name in inventory), 'Stage native payload missing')
    bench.require((run/'checkpoints/latest').read_text().strip() == tag, 'Stage latest pointer differs')
    if job['key'] == 'B16': bench.require(not (run/'TRAINING_COMPLETE.json').exists(), 'Split16 falsely claims endpoint32')
    logging = [bench.read(run/f'capture_logging_rank{r}.json') for r in range(8)]
    bench.require(all(x['rank'] == r and x['completed'] == completed for r, x in enumerate(logging)), 'Stage logging receipt missing')
    return {'key': job['key'], 'run_identity': identity['identity'], 'identity_sha256': bench.digest(run/'identity.json'),
        'origin_boundary': boundary, 'checkpoint': {'path': str(folder), 'COMPLETE_sha256': bench.digest(folder/'COMPLETE.json'), 'complete': complete},
        'logging_cost_by_rank': logging, 'passed': False, 'native_numeric_acceptance': 'PENDING'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    bench.require(not output.exists(), 'Fresh recovery controller required; no implicit retry')
    manifest = bench.read(args.manifest)
    plan = build_plan(manifest, output)
    bench.immutable_source(plan['source_root'], plan['source_sha'])
    output.mkdir(parents=True, exist_ok=False)
    bench.write(output/'PLAN.json', plan)
    if not args.execute:
        print(json.dumps({'status': 'PLANNED_ONLY', 'stages': 4, 'native_banks': 4, 'fixed_endpoint': 'origin+32'})); return 0
    results, stages, stop, error, budget = [], [], [False], None, None
    status = 'RECOVERY_CAPTURES_COMPLETE_ACCEPTANCE_PENDING'
    def stop_after_current(*unused): stop[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop_after_current)
    try:
        budget = native.validate_gpu_budget(manifest, plan['jobs'])
        gate = native.gpu_admission(budget, plan['jobs'], native.actual_process_charges(plan['jobs']), whole_plan=True)
        bench.write(output/'GPU_BUDGET_REGISTRATION.json', {'budget': budget, 'admission': gate,
            'manifest_sha256': bench.digest(args.manifest), 'budget_block_sha256': hashlib.sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest()})
        bench.write(output/'NATIVE_RNG_SOURCES.json', {'sources': native.native_rng_sources(manifest)})
        driver_sha = bench.checked(['git', 'rev-parse', 'HEAD'], ROOT)
        bench.immutable_source(ROOT, driver_sha)
        qualification = manifest['qualification']
        bench.require(bench.digest(qualification['path']) == qualification['sha256'], 'Qualified environment record changed')
        q = bench.read(qualification['path'])
        bench.require(q['passed'] and q['host'] == manifest['host'] and set(q['physical_gpu_UUIDs']) == set(manifest['gpu_uuids']), 'Full8 qualification mismatch')
        env = native.qualified_runtime_snapshot(manifest['python'], q.get('environment', {}).get('packages', {}))
        bench.require(all(env['packages'].get(k) == v for k, v in q.get('environment', {}).get('packages', {}).items()), 'Runtime differs from qualification')
        bench.write(output/'REGISTRATION.json', {'driver_source_sha': driver_sha, 'driver_sha256': bench.digest(SCRIPT),
            'budget_driver_sha256': bench.digest(native.SCRIPT), 'training_source_sha': plan['source_sha'], 'environment': env,
            'same_B_identity_before_after_resume': True, 'numeric_acceptance': 'PENDING', 'passed': False})
        original = bench.origin(manifest['cases'][0])
        with ExitStack() as stack:
            root = Path(manifest['artifact_root'])
            key = hashlib.sha256(json.dumps(sorted(manifest['gpu_uuids'])).encode()).hexdigest()
            locks = [bench.acquire(stack, root/'physical_gpu_locks'/(key+'.lock')),
                     bench.acquire(stack, root/'native_capture_locks'/'storage_io.lock')]
            for job in plan['jobs']:
                if stop[0] or (output/'STOP_BEFORE_NEXT_JOB').exists(): status = 'STOPPED_BEFORE_NEXT_STAGE'; break
                bench.immutable_source(plan['source_root'], plan['source_sha'])
                bench.require(bench.digest(SCRIPT) == plan['driver_sha256'] and bench.origin(manifest['cases'][0]) == original, 'Source/origin changed')
                if job['key'] == 'B_resume32':
                    first = stages[-1]
                    folder = Path(first['checkpoint']['path'])
                    bench.require(first['key'] == 'B16' and bench.digest(folder/'COMPLETE.json') == first['checkpoint']['COMPLETE_sha256'] and
                                  bench.digest(Path(job['run'])/'identity.json') == first['identity_sha256'], 'Own paused B checkpoint/identity changed')
                bench.write(output/(job['key']+'_RESOURCE_GATE.json'), bench.readiness(manifest))
                bench.write(output/(job['key']+'_STORAGE_GATE.json'), storage_admission(job['storage_budget']['path'], job['run'], expected_digest=job['storage_budget']['sha256']))
                process = native.run_budgeted_job(job, locks, budget, plan['jobs'], output)
                snapshot = stage_snapshot(job, original)
                bench.write(output/(job['key']+'_STAGE.json'), snapshot)
                stages.append(snapshot)
                if job['key'] != 'B16':
                    first = stages[-2] if job['key'] == 'B_resume32' else None
                    if first: bench.require(snapshot['run_identity'] == first['run_identity'] and snapshot['identity_sha256'] == first['identity_sha256'], 'B resume changed its bound role contract')
                    receipt = native.capture_receipt(job, original, process,
                        origin_boundary=first['origin_boundary'] if first else None,
                        save_updates=[original['complete']['completed']+16, original['complete']['completed']+32] if first else None)
                    receipt['key'] = manifest['cases'][0]['name']+'_'+job['role']
                    if first:
                        receipt['split16_checkpoint'] = first['checkpoint']
                        receipt['first16_logging_cost_by_rank'] = first['logging_cost_by_rank']
                        receipt['process_cost_scope'] = 'resume stage only here; total B/full phase includes both actual PROCESS_EXIT files in cost ledger'
                    results.append(receipt)
                    bench.write(output/(job['key']+'_CAPTURE.json'), receipt)
            if len(stages) == 4:
                bench.write(output/'COUNT_SCENE_HORIZON_TRIPLETS.json', {'discrete_evidence': native.count_triplets(results), 'passed': False,
                    'numeric_acceptance': 'PENDING real native model/master/moments/BN/RNG comparisons; old strict results preserved'})
    except BaseException as exc:
        error = repr(exc)
        status = 'BUDGET_REFUSED_BEFORE_NEXT_STAGE' if isinstance(exc, native.BudgetRefused) else 'FAILED'
        bench.write(output/'FAILURE.json', {'error': error, 'completed_stages': len(stages), 'no_retry_or_checkpoint_cleanup': True})
    try: costs = native.actual_process_charges(plan['jobs'])
    except (ValueError, OSError, KeyError, TypeError) as exc: costs = {'actual_phase_GPU_hours': None, 'cost_unknown': True, 'error': repr(exc)}
    bench.write(output/'RESULTS.json', {'status': status, 'error': error, 'stages': stages, 'jobs': results,
        'gpu_budget': budget, 'actual_process_costs_including_failed': costs, 'passed': False, 'formal_authorized': False,
        'final_budget_within_limits': (costs['actual_phase_GPU_hours'] is not None and budget is not None and
            costs['actual_phase_GPU_hours'] <= budget['phase_limit'] and
            costs['actual_phase_GPU_hours']+budget['already_spent_external']+budget['reserved_elsewhere'] <= budget['total_internal_limit'])})
    print(json.dumps({'status': status, 'completed_stages': len(stages), 'error': error}), flush=True)
    return 1 if error else 0


if __name__ == '__main__': raise SystemExit(main())
