"""CPU-only recovery routing controls and exact frozen-command reproduction.

Writes fresh original-schema manifests with NO dispatch GPU budget. No worker,
GPU query, native tensor load/copy, or old evidence modification is performed.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import run_native_execution_recovery as recovery

CLARIFICATION = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/perf_native_continuation_20261011/recovery_only_v2_20261011/frozen_v1/CLARIFICATION.json')
BASELINE = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-native-driver-v8-frozen-20261011/tools/structured_world/run_native_execution_recovery.py')


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': recovery.bench.digest(path)}


def command_digest(command):
    return hashlib.sha256(json.dumps(command, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def original_builder():
    # Compare the original frozen function in the same globals, including the
    # NEW driver's file identity. This checks behavior, not an old driver claim.
    node = next(n for n in ast.parse(BASELINE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'build_plan')
    context = dict(recovery.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BASELINE), 'exec'), context)
    return context['build_plan']


def actual_manifests(clarification_path):
    c = recovery.bench.read(clarification_path)
    recovery.bench.require(c['schema'] == 'execution_recovery_only_clarification_v2' and
                          (c['load_relative_update'], c['endpoint_relative_update']) == (16, 32), 'Frozen recovery V2 required')
    manifests, frozen = {}, {}
    for group, bound in c['plans'].items():
        recovery.bench.require(recovery.bench.digest(bound['path']) == bound['sha256'], 'Frozen exact plan changed')
        plan = recovery.bench.read(bound['path']); jobs = plan['jobs']; command = jobs[0]['command']
        args = command[command.index('--')+2:command.index('--scope')]
        port = int(next(x.split('=', 1)[1] for x in command if x.startswith('--master_port=')))
        origin = plan['origin']
        manifest = {k: copy.deepcopy(plan[k]) for k in ('host', 'python', 'gpu_uuids', 'qualification', 'artifact_root', 'minimum_free_MiB', 'source_root', 'source_sha')}
        manifest.update(schema='native_execution_recovery_manifest_v1', candidate_mode='engineering_fast',
            control_mode='engineering_fast', master_port=port,
            cases=[{'name': jobs[0]['case'], 'group': group, 'stage': 'mid',
                'pilot_origin_run': origin['run'], 'pilot_origin_tag': origin['tag'],
                'origin_identity': origin['identity'], 'origin_complete_sha256': origin['COMPLETE_sha256'],
                'train_args': args, 'output_root': str(Path(jobs[0]['run']).parent),
                'storage_pool': 'LOCAL', 'storage_budget': copy.deepcopy(jobs[0]['storage_budget'])}],
            recommended_controller_output=str(Path(jobs[0]['process_run']).parents[1]),
            frozen_exact_commands=copy.deepcopy(bound), recovery_only_V2_clarification=reference(clarification_path),
            original_V1_criteria=copy.deepcopy(plan['original_V1_criteria']),
            prospective_acceptance_bindings=copy.deepcopy(plan['prospective_acceptance_bindings']),
            prior_frozen_driver=copy.deepcopy(plan['driver_binding']),
            new_operator_binding={'status': 'PENDING_PARENT_REVIEW_AND_COLD_FREEZE', 'mutable_script': reference(recovery.SCRIPT)},
            budget_binding_state='ABSENT; parent must bind fresh known GPU-hour budget before --execute; no numeric placeholders',
            dispatch_ready=False, passed=False, formal_authorized=False, GPU_jobs_started=0)
        manifests[group], frozen[group] = manifest, plan
    recovery.bench.require(set(manifests) == {'G1', 'G3'}, 'Both original recovery groups required')
    return manifests, frozen


class ControlModeTests(unittest.TestCase):
    clarification_path = CLARIFICATION

    def setUp(self):
        self.manifests, self.frozen = actual_manifests(self.clarification_path)

    def build(self, manifest):
        return recovery.build_plan(manifest, manifest['recommended_controller_output'])

    def test_eight_actual_commands_and_process_paths_exact(self):
        for group, manifest in self.manifests.items():
            with self.subTest(group=group):
                generated = self.build(manifest)
                self.assertEqual([j['command'] for j in generated['jobs']], [j['command'] for j in self.frozen[group]['jobs']])
                self.assertEqual([j['process_run'] for j in generated['jobs']], [j['process_run'] for j in self.frozen[group]['jobs']])
                self.assertEqual([j['mode'] for j in generated['jobs']], ['engineering_fast']*4)

    def test_default_behavior_exact_original_for_every_candidate(self):
        old = original_builder()
        for group, manifest in self.manifests.items():
            for candidate in recovery.bench.CANDIDATES:
                with self.subTest(group=group, candidate=candidate):
                    m = copy.deepcopy(manifest); m.pop('control_mode'); m['candidate_mode'] = candidate
                    before = copy.deepcopy(m)
                    self.assertEqual(self.build(m), old(m, m['recommended_controller_output']))
                    self.assertEqual(m, before)

    def test_explicit_IO_matches_original_for_every_candidate(self):
        old = original_builder()
        for candidate in recovery.bench.CANDIDATES:
            m = copy.deepcopy(self.manifests['G1']); m.update(control_mode='io_preserving_v1', candidate_mode=candidate)
            self.assertEqual(self.build(m), old(m, m['recommended_controller_output']))

    def test_invalid_controls_rejected_before_origin_IO(self):
        for mode in ('reference', 'engineering_timevec', 'precision_fast', 'loss_preserving_v1', '', None):
            m = copy.deepcopy(self.manifests['G1']); m['control_mode'] = mode
            with self.subTest(mode=mode), patch.object(recovery.bench, 'origin') as origin, self.assertRaisesRegex(ValueError, 'Controls must be IO'):
                self.build(m)
            origin.assert_not_called()

    def test_fast_controls_require_fast_candidate(self):
        for candidate in ('engineering_timevec', 'precision_fast'):
            m = copy.deepcopy(self.manifests['G1']); m['candidate_mode'] = candidate
            with self.subTest(candidate=candidate), patch.object(recovery.bench, 'origin') as origin, self.assertRaisesRegex(ValueError, 'Controls must be IO'):
                self.build(m)
            origin.assert_not_called()

    def test_no_dispatch_budget_or_fake_numeric_placeholder(self):
        for manifest in self.manifests.values():
            self.assertNotIn('budget', manifest)
            with self.assertRaisesRegex(ValueError, 'Explicit known GPU-hour budget'):
                recovery.native.validate_gpu_budget(manifest, self.build(manifest)['jobs'])

    def test_actual_split_boundary_and_endpoint_stay_16_32(self):
        for group, manifest in self.manifests.items():
            plan = self.build(manifest); start = self.frozen[group]['origin']['completed']
            self.assertEqual([j['expected_resume_completed'] for j in plan['jobs']], [start, start, start, start+16])
            self.assertEqual([j['expected_exit_completed'] for j in plan['jobs']], [start+32, start+32, start+16, start+32])
            self.assertTrue(all(j['capture_plan']['end_update'] == start+32 for j in plan['jobs']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clarification', type=Path, default=CLARIFICATION)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recovery.bench.require(not args.output.exists(), 'Fresh CPU evidence directory required')
    ControlModeTests.clarification_path = args.clarification
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlModeTests))
    args.output.mkdir(parents=True, exist_ok=False)
    recovery.bench.write(args.output/'CPU_CONTROLS.json', {'schema': 'recovery_control_mode_CPU_controls_v1',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'CPU_controls_passed': result.wasSuccessful(), 'driver': reference(recovery.SCRIPT), 'tests_source': reference(__file__),
        'original_default_function': reference(BASELINE), 'clarification': reference(args.clarification),
        'scope': 'CPU planning and exact argv reproduction only; no actual GPU recovery proof',
        'passed': False, 'formal_authorized': False, 'GPU_jobs_started': 0,
        'unittest_failures': [{'test': str(t), 'traceback': e} for t, e in result.failures+result.errors]})
    if not result.wasSuccessful(): return 1
    manifests, frozen = actual_manifests(args.clarification)
    rows, files = [], []
    for group, manifest in manifests.items():
        plan = recovery.build_plan(manifest, manifest['recommended_controller_output'])
        for a, b in zip(plan['jobs'], frozen[group]['jobs']):
            recovery.bench.require(a['command'] == b['command'] and a['process_run'] == b['process_run'], 'Frozen command/process mismatch; do not publish replacement registration')
            rows.append({'group': group, 'key': a['key'], 'argv_exact': True, 'process_directory_exact': True,
                'generated_command_sha256': command_digest(a['command']), 'frozen_command_sha256': command_digest(b['command'])})
        for name, value in ((group+'_RECOVERY_MANIFEST_BUDGET_PENDING.json', manifest), (group+'_GENERATED_PLAN.json', plan)):
            path = args.output/name; recovery.bench.write(path, value); files.append(reference(path))
    recovery.bench.write(args.output/'EXACT_EIGHT_COMMANDS.json', {'schema': 'frozen_recovery_argv_exact_comparison', 'jobs': rows,
        'command_count': 8, 'all_argv_exact': True, 'all_process_directories_exact': True, 'GPU_jobs_started': 0})
    recovery.bench.write(args.output/'HANDOFF.json', {'status': 'STABLE_CPU_ROUTING_READY_FOR_PARENT_COLD_FREEZE',
        'driver': reference(recovery.SCRIPT), 'tests_source': reference(__file__), 'CPU_controls': reference(args.output/'CPU_CONTROLS.json'),
        'exact_commands': reference(args.output/'EXACT_EIGHT_COMMANDS.json'), 'files': files,
        'frozen_recovery_V2': reference(args.clarification), 'training_source_sha': manifests['G1']['source_sha'],
        'default_control_mode': 'io_preserving_v1', 'new_control_mode': 'engineering_fast only with engineering_fast candidate',
        'operator_CPU_or_execute_command': ['QUALIFIED_PRIVATE_PYTHON', '-B', 'NEW_PARENT_FROZEN_DRIVER/tools/structured_world/run_native_execution_recovery.py', '--manifest', 'FRESH_PARENT_BUDGET_BOUND_MANIFEST', '--output', 'manifest.recommended_controller_output', 'OPTIONAL_EXPLICIT_EXECUTE'],
        'remaining': 'Parent review/cold freeze, fresh budget binding and actual host/resource/lock gates. No scientific argv or V2 criteria change.',
        'B16_metadata_note': 'Existing retained_global_updates denotes planned capture32; expected_exit_completed and preserved B16 stage prove actual16. Existing runner metadata remains unchanged.',
        'old_artifacts_overwritten': False, 'validator_or_old_bindings_edited': False, 'passed': False,
        'actual_GPU_recovery_evidence': 'PENDING', 'formal_authorized': False, 'GPU_jobs_started': 0})
    print(json.dumps({'status': 'CPU_CONTROLS_AND_8_EXACT_COMMANDS_VERIFIED', 'output': str(args.output), 'tests': result.testsRun, 'GPU_jobs_started': 0}))
    return 0


if __name__ == '__main__': sys.exit(main())
