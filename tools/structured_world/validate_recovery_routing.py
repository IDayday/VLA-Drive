"""Reproducible CPU/mock controller checks; never dispatch worker commands.

Run with the original local qwen Python and --output NEW_DIRECTORY.
Scientific records are read only; mock files live in temporary directories
inside the requested output and are removed. Results describe mocked evidence.
"""
import argparse
import copy
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
CONTROL = None
CODE = Path(__file__).resolve().parent
ARTIFACTS = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')


def load(kind):
    path = CODE/f'resume_{kind}_group.py'
    compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('routing_'+kind, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(kind, module, original, namespace, scenario):
    """Exercise actual main() with subprocess/locks/readiness mocked."""
    initial = module.latest(original.output/'G1_FULL_UNIFORM')
    with tempfile.TemporaryDirectory(prefix='mock_routing_', dir=CONTROL) as folder:
        temp = Path(folder)
        args = copy.copy(original)
        args.output = temp/'historical_queue'
        run = args.output/'G1_FULL_UNIFORM'
        run.mkdir(parents=True)
        old_run = original.output/'G1_FULL_UNIFORM'
        (run/'checkpoints').mkdir()
        (run/'checkpoints/latest').symlink_to(old_run/'checkpoints/latest')
        for target in module.TARGETS:
            for stage, cmd, gpu, complete in module.commands(original, namespace, target):
                if complete is not None and complete.exists():
                    link = run/complete.relative_to(old_run)
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(complete)
        if kind == 'nuscenes':
            (run/'metrics.jsonl').symlink_to(old_run/'metrics.jsonl')
            args.gpu_lock_root = temp/'locks'
            args.gpu_lock_root.mkdir()
        private_artifacts = temp/'artifacts'
        (private_artifacts/'physical_gpu_locks').mkdir(parents=True)
        output = temp/'control'
        fake_prepared = temp/'IN_MEMORY_MOCK_ONLY_NOT_A_QUALIFICATION.json'
        destination = ARTIFACTS/'nuscenes_full8_deployment_20261009/G3_EVENT_LOCAL_qualification/QUALIFICATION_REVIEWED.json'
        qualification = json.loads(destination.read_text())
        snapshot = [{'uuid': u, 'used_MiB': 0, 'free_MiB': 81153, 'utilization': 0}
                    for u in qualification['physical_gpu_UUIDs']]
        progress = [copy.deepcopy(initial)]
        calls, sleeps, locks = [], [], []
        real_read, real_digest = module.read, module.digest
        digests = {}

        def translate(path):
            path = Path(path)
            if path.is_relative_to(args.output):
                return original.output/path.relative_to(args.output)
            if path == private_artifacts/'FORMAL_RESOURCE_REGISTRATION_v1.json':
                return ARTIFACTS/path.name
            return path

        def read(path):
            path = Path(path)
            if path == fake_prepared:
                manifest = json.loads((output/'REGISTRATION.json').read_text())
                if scenario == 'preparation_stale_controller_rejected':
                    manifest['controller_code_sha256'] = 'stale_mock_controller'
                return {'passed': True, 'manifest': manifest, 'report': {
                    'native_checkpoint_files_sha256': real_read(Path(initial['complete_file']))['file_sha256'],
                    'image_payload_verification': {'all_SHA256_passed': True, 'images': 304776},
                    'CPU_import_readiness': {'passed': scenario != 'preparation_missing_import_gate_rejected',
                                             'mock_only': True}}}
            if path == run/'RESUME_BOUNDARY_VERIFIED.json':
                completed = initial['completed'] if scenario != 'resume_stale_report_rejected' else (728 if kind == 'navsim' else 4356)
                return {'passed': True, 'identity': module.IDENTITY, 'completed': completed, 'mock_only': True}
            if path.parent.parent == run/'checkpoints' and path.name == 'COMPLETE.json':
                translated = translate(path)
                if not translated.exists():
                    return {'identity': module.IDENTITY, 'completed': int(path.parent.name.split('_')[1])}
            return real_read(translate(path))

        def digest(path):
            path = translate(path)
            if path == old_run/'RESUME_BOUNDARY_VERIFIED.json':
                return hashlib.sha256(json.dumps(read(run/'RESUME_BOUNDARY_VERIFIED.json')).encode()).hexdigest()
            if path not in digests:
                digests[path] = real_digest(path)
            return digests[path]

        def worker(command, **kwargs):
            calls.append(command)
            if scenario.startswith('resume_') and len(calls) == 1:
                progress[0] = {**initial, 'completed': 50000 if kind == 'navsim' else 8712}
                return subprocess.CompletedProcess(command, 0)
            return subprocess.CompletedProcess(command, 7)

        def flock(stream, flags):
            locks.append({'path': stream.name, 'flags': flags})
            assert Path(stream.name).is_relative_to(temp), 'Mock must never open real GPU locks'
            if scenario == 'lock_wait_then_stage_failure' and len(locks) == 2:
                raise BlockingIOError('simulated occupied physical lock')

        def path(value='.'):
            value = Path(value)
            if str(value) == '/proc/987654321/cmdline':
                return temp/'mock_watch_cmdline'
            return value

        if scenario == 'active_original_watcher_blocks':
            watcher = private_artifacts/'recovery_dispatch_20261010/navsim_G1/LAUNCHED_WATCHER.json'
            watcher.parent.mkdir(parents=True)
            watcher.write_text(json.dumps({'pid': 987654321}))
            (temp/'mock_watch_cmdline').write_bytes(b'python\0wait_and_resume.py\0')
        controller = (ARTIFACTS/'formal_navsim_zt2_G1_G0_v3_io/EXECUTED_CONTROLLER.json' if kind == 'navsim'
                      else ARTIFACTS/'nuscenes_full8_deployment_20261009/G1_FULL_UNIFORM_FORMAL_CONTROLLER.json')
        argv = [str(CODE/f'resume_{kind}_group.py'), '--original-controller', str(controller),
                '--destination-host', 'training-vlawm-zt', '--destination-qualification', str(destination),
                '--output', str(output), '--wait-for-resource']
        if scenario == 'inspect_no_dispatch':
            argv.append('--inspect-only')
        elif kind == 'navsim':
            argv.extend(['--execute', '--prepared', str(fake_prepared)])
        with ExitStack() as stack:
            for name, value in [('original_plan', lambda _: (args, namespace)), ('latest', lambda _: copy.deepcopy(progress[0])),
                                ('read', read), ('digest', digest), ('authorize', lambda _: None), ('Path', path)]:
                stack.enter_context(patch.object(module, name, value))
            if kind == 'navsim':
                stack.enter_context(patch.object(module, 'ARTIFACTS', private_artifacts))
                stack.enter_context(patch.object(module, 'clean_source', lambda: None))
                stack.enter_context(patch.object(module, 'probe', lambda *a, **k: {'GPU_snapshot': snapshot}))
            else:
                stack.enter_context(patch.object(module, 'readiness', lambda *a, **k: ({'mock_only': True}, snapshot)))
            stack.enter_context(patch.object(module.subprocess, 'run', worker))
            stack.enter_context(patch.object(module.fcntl, 'flock', flock))
            stack.enter_context(patch.object(module.time, 'sleep', lambda seconds: sleeps.append(seconds)))
            stack.enter_context(patch.object(sys, 'argv', argv))
            error = None
            try:
                with redirect_stdout(io.StringIO()): module.main()
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
        if scenario == 'inspect_no_dispatch':
            assert error is None and not output.exists() and not calls and not locks, repr(error)
            return {'passed': True, 'mode': 'mock', 'worker_calls': 0, 'control_directory_created': False}
        state = json.loads((output/'STATE.json').read_text())
        assert state['status'] == 'FAILED' and state['checkpoints_preserved'] and not state['unrelated_processes_stopped']
        if scenario in ('preparation_missing_import_gate_rejected', 'preparation_stale_controller_rejected'):
            assert not calls and not locks and 'Preparation is stale/foreign' in error
        elif scenario == 'active_original_watcher_blocks':
            assert not calls and not locks and 'watcher is active' in error
        elif scenario == 'resume_pass_records_reference':
            records = [json.loads(line) for line in (output/'executed_commands.jsonl').read_text().splitlines()]
            passed = [r for r in records if r.get('status') == 'PASSED']
            assert len(calls) == 2 and len(passed) == 1
            assert passed[0]['resume_boundary']['completed_reference'] == initial['completed']
        elif scenario == 'resume_stale_report_rejected':
            assert len(calls) == 1 and 'boundary/progress verification failed' in error
        else:
            assert len(calls) == 1 and 'failed with exit 7' in error
        if scenario == 'lock_wait_then_stage_failure':
            assert sleeps == [30] and len(locks) == 4
        return {'passed': True, 'mode': 'mock', 'worker_calls': len(calls), 'real_workers_dispatched': 0,
                'sleep_requests_seconds': sleeps, 'persisted_state': state, 'expected_error': error}


def check_sharing(module):
    rows = [{'free_MiB': 73066, 'used_MiB': 8087, 'utilization': 0} for _ in range(8)]
    results = []
    policy = {'task_authorizations': {'structured_world_fgtr_round1':
                                     {'shared_GPU_deployment_authorized': True}}}
    with patch.object(module, 'read', return_value=policy):
        checks = [('default_full_idle_rejects_existing_memory',
                   not module.resource_ready(rows, 55*1024)),
                  ('explicit_authorized_sharing_accepts_low_util_capacity',
                   module.resource_ready(rows, 55*1024, True)),
                  ('sharing_rejects_active_GPU_load', not module.resource_ready(
                      [{**r, 'utilization': 6} for r in rows], 55*1024, True)),
                  ('sharing_rejects_insufficient_memory_reserve', not module.resource_ready(
                      [{**r, 'free_MiB': 55*1024-1} for r in rows], 55*1024, True))]
        for name, passed in checks:
            assert passed, name
            results.append({'helper': 'navsim', 'case': name, 'passed': True, 'mode': 'policy_mock'})
    policy['task_authorizations']['structured_world_fgtr_round1']['shared_GPU_deployment_authorized'] = False
    with patch.object(module, 'read', return_value=policy):
        try:
            module.resource_ready(rows, 55*1024, True)
            raise AssertionError('Revoked sharing policy was accepted')
        except ValueError:
            pass
    results.append({'helper': 'navsim', 'case': 'sharing_stops_after_policy_revocation',
                    'passed': True, 'mode': 'policy_mock'})
    return results


def main():
    global CONTROL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    CONTROL = args.output.resolve()
    protected = (CODE.parents[1], ARTIFACTS)
    if any(CONTROL.is_relative_to(p.resolve()) or p.resolve().is_relative_to(CONTROL) for p in protected):
        raise ValueError('Validation output overlaps code or original artifacts')
    CONTROL.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'routing_controller_CPU_and_mock_validation_v1', 'GPU_training_dispatched': False,
              'qualification': False, 'started_unix': time.time(), 'cases': [], 'helper_sha256': {}}
    for kind in ('navsim', 'nuscenes'):
        module = load(kind)
        controller = (ARTIFACTS/'formal_navsim_zt2_G1_G0_v3_io/EXECUTED_CONTROLLER.json' if kind == 'navsim'
                      else ARTIFACTS/'nuscenes_full8_deployment_20261009/G1_FULL_UNIFORM_FORMAL_CONTROLLER.json')
        args, namespace = module.original_plan(controller)
        plan = [{'stage': s, 'command': c} for t in module.TARGETS for s, c, gpu, p in module.commands(args, namespace, t)]
        train_commands = [p for p in plan if 'train_to_' in p['stage']]
        assert len(plan) == 16 and len(train_commands) == 4
        assert all('--resume' in p['command'] for p in train_commands)
        if kind == 'navsim':
            original = json.loads((ARTIFACTS/'recovery_dispatch_20261010/navsim_G1_vlawm_preparation_v1/PREPARATION_COMPLETE.json').read_text())
            assert plan == [{k: p[k] for k in ('stage', 'command')} for p in original['manifest']['plan']]
        result['cases'].append({'helper': kind, 'case': 'frozen_source_and_command_planning', 'passed': True, 'mode': 'read_only', 'commands': plan})
        scenarios = ['inspect_no_dispatch', 'stage_failure_stops', 'lock_wait_then_stage_failure',
                     'resume_pass_records_reference', 'resume_stale_report_rejected']
        if kind == 'navsim':
            scenarios.extend(['active_original_watcher_blocks', 'preparation_missing_import_gate_rejected',
                              'preparation_stale_controller_rejected'])
        for scenario in scenarios:
            result['cases'].append({'helper': kind, 'case': scenario, **check(kind, module, args, namespace, scenario)})
        if kind == 'navsim':
            result['cases'].extend(check_sharing(module))
        result['helper_sha256'][kind] = sha(CODE/f'resume_{kind}_group.py')
    result.update(passed=True, ended_unix=time.time(), validation_script_sha256=sha(__file__))
    destination = CONTROL/'VALIDATION_RESULTS.json'
    with destination.open('x') as stream: stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'cases': len(result['cases']), 'result': str(destination),
                      'helper_sha256': result['helper_sha256'], 'GPU_training_dispatched': False}, indent=2))


if __name__ == '__main__': main()
