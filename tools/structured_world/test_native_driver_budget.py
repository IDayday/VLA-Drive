"""CPU tests for dispatch phases, budgets and fixed32 native recovery commands.

An actual failed CPU subprocess exercises the existing process monitor/ledger;
no GPU worker or native numerical acceptance is exercised.
"""
import argparse
import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import shlex
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import run_native_execution_continuation as driver
from tools.structured_world import run_native_execution_recovery as recovery
from tools.structured_world import test_native_capture as old_tests


def budget():
    return {'limit_kind': 'internal_finite_test_registration', 'user_GPU_hours_limit': None,
            'phase_limit': 36., 'already_spent_external': 50., 'total_internal_limit': 120.,
            'reserved_elsewhere': 34., 'per_job_reserves': {'early': .5, 'mid': 5.5, 'recovery': .5},
            'accounting_sources': []}


def job(root, key, stage='early', process=None):
    value = {'key': key, 'stage': stage, 'run': str(root/key)}
    if process: value['process_run'] = str(root/process)
    return value


def exit_receipt(value, hours, returncode=0):
    root = Path(value.get('process_run', value['run'])); root.mkdir(parents=True)
    (root/'PROCESS_EXIT.json').write_text(json.dumps({'wall_seconds': hours*3600/8, 'returncode': returncode}))


class QualifiedRuntimeTests(unittest.TestCase):
    runtime_manifest = None
    real_runtime_receipt = None

    def metadata_query(self, expected):
        names = []
        def version(name):
            names.append(name)
            return '2.7.4.post1' if name == 'flash_attn' else 'CPU_control_version'
        def checked(command, *, env):
            self.assertEqual(command[:3], [sys.executable, '-B', '-c'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            output = io.StringIO()
            with redirect_stdout(output), patch('importlib.metadata.version', side_effect=version):
                exec(compile(command[3], '<actual qualified metadata query>', 'exec'), {})
            return output.getvalue().strip()
        with patch.object(driver.bench, 'checked', side_effect=checked):
            result = driver.qualified_runtime_snapshot(sys.executable, expected)
        return result, names

    def test_extra_flash_attn_queried_with_every_default(self):
        expected = {'flash_attn': '2.7.4.post1', 'numpy': 'CPU_control_version'}
        before = copy.deepcopy(expected)
        result, names = self.metadata_query(expected)
        self.assertEqual(set(names), set(driver.bench.PACKAGES) | {'flash_attn'})
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(result['packages']['flash_attn'], '2.7.4.post1')
        self.assertEqual(expected, before)

    def test_empty_qualification_still_queries_all_defaults(self):
        result, names = self.metadata_query({})
        self.assertEqual(set(names), set(driver.bench.PACKAGES))
        self.assertEqual(set(result['packages']), set(driver.bench.PACKAGES))

    def test_invalid_package_name_blocks_query(self):
        with patch.object(driver.bench, 'checked') as query:
            for name in ('', 'flash_attn;echo bad', 'flash attn', '../flash_attn', 'flash_attn\n', "flash_attn'bad"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    driver.qualified_runtime_snapshot(sys.executable, {name: 'version'})
            query.assert_not_called()

    def test_missing_metadata_actual_CPU_query_raises_without_partial_snapshot(self):
        missing = 'structured_world_control_missing_metadata_20261011'
        def checked(command, *, env):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            return subprocess.check_output(command, env=env, text=True, stderr=subprocess.PIPE, timeout=10).strip()
        # Isolate the deliberately missing distribution from local defaults;
        # run the real generated query in a CPU-only subprocess.
        with patch.object(driver.bench, 'PACKAGES', ()), patch.object(driver.bench, 'checked', side_effect=checked):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                driver.qualified_runtime_snapshot(sys.executable, {missing: 'unavailable'})
        self.assertIn('PackageNotFoundError', error.exception.stderr)
        self.assertIn(missing, error.exception.stderr)

    def test_real_zt3_CPU_metadata_matches_bound_qualification(self):
        if self.runtime_manifest is None:
            self.skipTest('Actual remote CPU query requires explicit --runtime-manifest')
        manifest_path = self.runtime_manifest
        manifest = driver.bench.read(manifest_path)
        self.assertEqual(manifest['host'], 'training-vlawm-zt3')
        self.assertTrue(Path(manifest['python']).is_absolute())
        binding = manifest['qualification']
        self.assertEqual(driver.bench.digest(binding['path']), binding['sha256'])
        qualification = driver.bench.read(binding['path'])
        self.assertEqual(qualification['host'], manifest['host'])
        expected = qualification['environment']['packages']
        self.assertEqual(expected['flash_attn'], '2.7.4.post1')
        def remote_checked(command, *, env):
            self.assertEqual(command[:3], [manifest['python'], '-B', '-c'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            code = "assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES') == '';"+command[3]
            remote = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', command[0], '-B', '-c', code])
            return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                manifest['host'], remote], text=True, stderr=subprocess.PIPE, timeout=25).strip()
        started = time.monotonic()
        with patch.object(driver.bench, 'checked', side_effect=remote_checked):
            actual = driver.qualified_runtime_snapshot(manifest['python'], expected)
        self.assertEqual(set(actual['packages']), set(driver.bench.PACKAGES) | set(expected))
        for name, version in expected.items():
            self.assertEqual(actual['packages'][name], version, name)
        self.assertEqual(actual['packages']['flash_attn'], '2.7.4.post1')
        self.assertEqual(driver.bench.digest(binding['path']), binding['sha256'])
        type(self).real_runtime_receipt = {'host': manifest['host'], 'python': manifest['python'],
            'manifest_path': str(manifest_path), 'manifest_sha256': driver.bench.digest(manifest_path),
            'qualification': binding, 'actual': actual, 'all_pinned_versions_exact': True,
            'CUDA_VISIBLE_DEVICES': '', 'query_seconds': time.monotonic()-started,
            'GPU_probe_or_model_import': False, 'qualification_passed_here': False}


class BudgetTests(unittest.TestCase):
    def test_exact_whole_phase_and_joint_boundaries(self):
        jobs = [{'key': str(i), 'stage': 'early' if i < 6 else 'mid'} for i in range(12)]
        b = driver.validate_gpu_budget({'budget': budget()}, jobs)
        gate = driver.gpu_admission(b, jobs, {'actual_phase_GPU_hours': 0.}, whole_plan=True)
        self.assertEqual(gate['projected_phase_GPU_hours'], 36.)
        self.assertEqual(gate['projected_joint_GPU_hours'], 120.)
        for mutation in ({'phase_limit': 35.999}, {'already_spent_external': 50.001}):
            with self.subTest(mutation=mutation), self.assertRaises(driver.BudgetRefused):
                driver.gpu_admission({**b, **mutation}, jobs, {'actual_phase_GPU_hours': 0.}, whole_plan=True)

    def test_no_default_for_unknown_nonfinite_or_negative_budget(self):
        jobs = [{'stage': 'mid'}]
        for missing in budget():
            if missing == 'accounting_sources': continue
            b = budget(); b.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError): driver.validate_gpu_budget({'budget': b}, jobs)
        for change in ({'phase_limit': None}, {'reserved_elsewhere': -1}, {'already_spent_external': float('nan')},
                       {'total_internal_limit': float('inf')}, {'total_internal_limit': -1}, {'phase_limit': True},
                       {'limit_kind': 'user_cap'}, {'limit_kind': None},
                       {'user_GPU_hours_limit': 0}, {'user_GPU_hours_limit': -1}, {'user_GPU_hours_limit': True},
                       {'user_GPU_hours_limit': float('nan')}, {'user_GPU_hours_limit': float('inf')},
                       {'per_job_reserves': {'mid': 0}}, {'per_job_reserves': {'mid': float('nan')}}):
            with self.subTest(change=change), self.assertRaises(ValueError): driver.validate_gpu_budget({'budget': {**budget(), **change}}, jobs)
        with self.assertRaises(ValueError): driver.validate_gpu_budget({}, jobs)

    def test_explicit_internal_140_no_user_cap_and_no_automatic_budget_increase(self):
        jobs = [{'key': str(i), 'stage': 'early' if i < 6 else 'mid'} for i in range(12)]
        registered = {**budget(), 'total_internal_limit': 140., 'already_spent_external': 60., 'reserved_elsewhere': 44.}
        original = copy.deepcopy(registered)
        admitted = driver.validate_gpu_budget({'budget': registered}, jobs)
        gate = driver.gpu_admission(admitted, jobs, {'actual_phase_GPU_hours': 0.}, whole_plan=True)
        self.assertEqual(gate['projected_joint_GPU_hours'], 140.)
        self.assertEqual(gate['limit_kind'], 'internal_finite_test_registration')
        self.assertIsNone(gate['user_GPU_hours_limit'])
        with self.assertRaises(driver.BudgetRefused):
            driver.gpu_admission(admitted, jobs, {'actual_phase_GPU_hours': .001}, whole_plan=True)
        self.assertEqual(registered, original)
        # Existing registrations are still enforced as declared; neither
        # failed costs nor selected phases automatically revise their limits.
        old = driver.validate_gpu_budget({'budget': budget()}, jobs)
        with self.assertRaises(driver.BudgetRefused):
            driver.gpu_admission(old, jobs, {'actual_phase_GPU_hours': .001}, whole_plan=True)
        self.assertEqual(old['total_internal_limit'], 120.)

    def test_actual_positive_user_cap_is_enforced_separately(self):
        jobs = [{'key': 'mid', 'stage': 'mid'}]
        for cap in (119.999, 100.):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'explicit user GPU-hour limit'):
                driver.validate_gpu_budget({'budget': {**budget(), 'user_GPU_hours_limit': cap}}, jobs)
        for cap in (120., 140.):
            b = driver.validate_gpu_budget({'budget': {**budget(), 'user_GPU_hours_limit': cap}}, jobs)
            gate = driver.gpu_admission(b, jobs, {'actual_phase_GPU_hours': 0.}, whole_plan=True)
            self.assertEqual(gate['user_GPU_hours_limit'], cap)
        revised = {**budget(), 'total_internal_limit': 140., 'user_GPU_hours_limit': 130.}
        with self.assertRaises(ValueError): driver.validate_gpu_budget({'budget': revised}, jobs)

    def test_failed_exit_and_success_are_both_charged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); jobs = [job(root, 'A'), job(root, 'B'), job(root, 'C')]
            exit_receipt(jobs[0], .25); exit_receipt(jobs[1], .75, returncode=7)
            charges = driver.actual_process_charges(jobs)
            self.assertEqual(charges['actual_phase_GPU_hours'], 1.)
            self.assertTrue(charges['entries'][1]['charged_despite_failure'])
            gate = driver.gpu_admission({**budget(), 'phase_limit': 1.5}, jobs, charges, next_job=jobs[2])
            self.assertEqual(gate['projected_phase_GPU_hours'], 1.5)
            with self.assertRaises(driver.BudgetRefused):
                driver.gpu_admission({**budget(), 'phase_limit': 1.499}, jobs, charges, next_job=jobs[2])

    def test_missing_exit_or_bad_cost_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); j = job(root, 'A'); Path(j['run']).mkdir()
            (Path(j['run'])/'PROCESS.json').write_text('{}')
            with self.assertRaises(ValueError): driver.actual_process_charges([j])
            for wall in (None, -1, float('nan')):
                (Path(j['run'])/'PROCESS_EXIT.json').write_text(json.dumps({'wall_seconds': wall, 'returncode': 0}))
                with self.subTest(wall=wall), self.assertRaises(ValueError): driver.actual_process_charges([j])

    def test_same_B_run_two_stages_have_separate_cost_records(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); jobs = [job(root, 'B', 'recovery', 'B16_process'), job(root, 'B', 'recovery', 'Bresume_process')]
            jobs[1]['key'] = 'B_resume32'
            exit_receipt(jobs[0], .25); exit_receipt(jobs[1], .5)
            self.assertEqual(driver.actual_process_charges(jobs)['actual_phase_GPU_hours'], .75)
            with self.assertRaises(ValueError): driver.actual_process_charges([jobs[0], jobs[0]])

    def test_next_job_refusal_does_not_invoke_process_monitor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); current, future = job(root, 'A'), job(root, 'B', 'mid')
            exit_receipt(current, 1.1)
            with patch.object(driver.bench, 'run_job') as run, self.assertRaises(driver.BudgetRefused):
                driver.run_budgeted_job(future, [], {**budget(), 'phase_limit': 6.5}, [current, future], root)
            run.assert_not_called(); self.assertFalse(Path(future['run']).exists())

    def test_remaining_matrix_reserved_even_if_next_job_alone_fits(self):
        jobs = [{'key': 'next', 'stage': 'early'}, {'key': 'later', 'stage': 'mid'}]
        with self.assertRaises(driver.BudgetRefused):
            driver.gpu_admission({**budget(), 'phase_limit': 6.}, jobs,
                {'actual_phase_GPU_hours': .1, 'entries': []}, next_job=jobs[0])

    def test_optional_external_accounting_source_hash_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'external.json'; path.write_text('{"actual_GPU_hours":50}')
            b = {**budget(), 'accounting_sources': [{'path': str(path), 'sha256': driver.bench.digest(path)}]}
            self.assertEqual(driver.validate_gpu_budget({'budget': b}, [{'stage': 'mid'}])['already_spent_external'], 50.)
            path.write_text('{"actual_GPU_hours":51}')
            with self.assertRaises(ValueError): driver.validate_gpu_budget({'budget': b}, [{'stage': 'mid'}])

    def test_actual_failed_CPU_process_monitor_cost_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); actual = {**job(root, 'owned_cpu_failure'),
                'trainer': str(ROOT/'tools/structured_world/train_vla.py'), 'gpu_uuids': [],
                'command': [sys.executable, '-B', '-c', 'import sys; sys.exit(7)']}
            with self.assertRaises(ValueError):
                driver.run_budgeted_job(actual, [], {**budget(), 'already_spent_external': 0., 'reserved_elsewhere': 0.}, [actual], root)
            charges = driver.actual_process_charges([actual])
            self.assertGreater(charges['actual_phase_GPU_hours'], 0.)
            self.assertEqual(charges['entries'][0]['returncode'], 7)
            self.assertTrue((root/'owned_cpu_failure_GPU_COST_AFTER.json').exists())

    def test_execute_unknown_budget_never_reaches_gpu_launch(self):
        for module in (driver, recovery):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); m = root/'m.json'; m.write_text('{}')
                plan = {'jobs': [{**job(root, 'never_started'), 'case': 'pending', 'trainer': str(ROOT/'tools/structured_world/train_vla.py')}],
                    'case_selection': {'overall_cases': ['pending'], 'selected_cases': ['pending'], 'dispatch_scope': 'full_matrix'},
                    'source_root': str(ROOT), 'source_sha': 'a'*40}
                args = [str(module.SCRIPT), '--manifest', str(m), '--output', str(root/'controller'), '--execute']
                with patch.object(sys, 'argv', args), patch.object(module, 'build_plan', return_value=plan), \
                     patch.object(driver.bench, 'immutable_source'), patch.object(driver.bench, 'readiness') as readiness, \
                     patch.object(driver.bench, 'run_job') as run:
                    self.assertEqual(module.main(), 1)
                    readiness.assert_not_called(); run.assert_not_called()
                self.assertFalse(Path(plan['jobs'][0]['run']).exists())


class PhaseSelectionTests(unittest.TestCase):
    def fixture(self, root):
        manifest, natives = old_tests.PlanTests().fixture(root)
        for case in manifest['cases']:
            path = Path(case['storage_budget']['path']); storage = driver.bench.read(path)
            for owner in storage['owned_runs']:
                owner['cap_bytes'] += driver.GiB  # Metadata/logs count against OWN caps too.
            path.write_text(json.dumps(storage)); case['storage_budget']['sha256'] = driver.bench.digest(path)
        for name, native in natives.items():
            native['contract']['identity'] = name+'_origin_identity'
            native['complete']['file_sha256'] = {'rng_rank'+str(i)+'.pt': 'c'*64 for i in range(8)}
        return manifest, natives

    def test_default_full_matrix_unchanged_and_explicit_all_equivalent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, natives = self.fixture(root); original = copy.deepcopy(manifest)
            with patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]):
                default = driver.build_plan(manifest, root/'default')
                explicit = driver.build_plan(manifest, root/'explicit', ','.join(c['name'] for c in manifest['cases']))
            self.assertEqual(manifest, original); self.assertEqual(default, explicit)
            self.assertEqual((len(default['jobs']), default['maximum_total_updates'], default['expected_full_native_banks']), (12, 3006, 36))
            self.assertEqual(default['case_selection']['dispatch_scope'], 'full_matrix')
            self.assertEqual(default['case_selection']['pending_unselected_cases'], [])
            self.assertEqual(default['original_V1_capture_points'], [1, 32, 100, 250, 500])

    def test_selected_mid_resources_only_and_original_manifest_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, natives = self.fixture(root); original = copy.deepcopy(manifest)
            early_paths = set()
            for case in manifest['cases']:
                if case['stage'] == 'early':
                    path = Path(case['storage_budget']['path']); early_paths.add(str(path)); path.unlink()
                    (Path(case['output_root'])/'A').mkdir(parents=True)
            real_digest = driver.bench.digest
            def selected_origin(case):
                self.assertEqual(case['stage'], 'mid')
                return natives[case['name']]
            def selected_digest(path):
                self.assertNotIn(str(path), early_paths)
                return real_digest(path)
            with patch.object(driver.bench, 'origin', side_effect=selected_origin) as origin, \
                 patch.object(driver.bench, 'digest', side_effect=selected_digest):
                plan = driver.build_plan(manifest, root/'controller', ' G3_mid, G1_mid ')
                rng = driver.native_rng_sources(manifest, plan['case_selection']['selected_cases'])
            self.assertEqual(manifest, original); self.assertEqual(plan['manifest'], original)
            self.assertEqual((len(plan['jobs']), plan['maximum_total_updates'], plan['expected_full_native_banks']), (6, 3000, 30))
            self.assertEqual([s['case'] for s in rng], ['G1_mid', 'G3_mid'])
            self.assertEqual(origin.call_count, 4)
            self.assertEqual(plan['case_selection']['selected_cases'], ['G1_mid', 'G3_mid'])
            self.assertEqual(plan['case_selection']['pending_unselected_cases'], ['G1_early', 'G3_early'])
            self.assertEqual(plan['case_selection']['overall_case_stage_counts'], {'early': 2, 'mid': 2})
            self.assertEqual(plan['case_selection']['selected_case_stage_counts'], {'early': 0, 'mid': 2})
            self.assertEqual(plan['case_selection']['outstanding_case_stage_counts'], {'early': 2, 'mid': 0})
            self.assertEqual(plan['case_selection']['outstanding_unselected_jobs'], 6)
            for value in plan['jobs']:
                storage = driver.bench.read(value['storage_budget']['path'])
                owner = next(o for o in storage['owned_runs'] if o['path'] == value['run'])
                self.assertGreaterEqual(owner['cap_bytes'], 5*storage['checkpoint_upper_bytes']+driver.GiB)
                self.assertEqual(value['steps'], 500)
                self.assertEqual(value['capture_plan']['relative_updates'], [1, 32, 100, 250])
                start = natives[value['case']]['complete']['completed']
                self.assertEqual(value['retained_global_updates'], [start+x for x in (1, 32, 100, 250, 500)])
                options = dict(zip(value['command'][value['command'].index('--')+2:][::2], value['command'][value['command'].index('--')+2:][1::2]))
                self.assertEqual(options['--debug-updates'], '500'); self.assertEqual(options['--capture-updates'], '1,32,100,250')
                self.assertNotIn('--stop-after', options); self.assertNotIn('--benchmark-no-end-save', options)
                self.assertEqual(options['--execution-mode'], manifest['candidate_mode'] if value['role'] == 'B' else 'io_preserving_v1')
            for name in ('G1_mid', 'G3_mid'):
                self.assertEqual([j['role'] for j in plan['jobs'] if j['case'] == name], ['A', 'AA', 'B'])

    def test_single_mid_triplet_and_phase_budget_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, natives = self.fixture(root)
            with patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]):
                both = driver.build_plan(manifest, root/'both', 'G1_mid,G3_mid')
                single = driver.build_plan(manifest, root/'single', 'G3_mid')
            self.assertEqual((len(single['jobs']), single['maximum_total_updates'], single['expected_full_native_banks']), (3, 1500, 15))
            b = {**budget(), 'phase_limit': 33., 'reserved_elsewhere': 37.}
            gate = driver.gpu_admission(driver.validate_gpu_budget({'budget': b}, both['jobs']), both['jobs'],
                {'actual_phase_GPU_hours': 0.}, whole_plan=True)
            self.assertEqual(gate['projected_phase_GPU_hours'], 33.)
            self.assertEqual(gate['projected_joint_GPU_hours'], 120.)
            self.assertEqual(len(gate['remaining_jobs']), 6)
            with self.assertRaises(driver.BudgetRefused):
                driver.gpu_admission({**b, 'phase_limit': 32.999}, both['jobs'], {'actual_phase_GPU_hours': 0.}, whole_plan=True)
            with self.assertRaises(ValueError): driver.validate_gpu_budget({}, single['jobs'])

    def test_single_bank_metadata_needs_headroom_under_unchanged_storage_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, _ = self.fixture(root)
            case = next(c for c in manifest['cases'] if c['name'] == 'G1_early')
            path = Path(case['storage_budget']['path']); storage = driver.bench.read(path)
            storage.update(schema='native_capture_storage_budget_v1', approved_for_pilot=True,
                reserve_bytes=100*driver.GiB, pool_cap_bytes=138*driver.GiB, filesystem_probe_path=str(root))
            owner = storage['owned_runs'][0]; output = Path(owner['path']); output.mkdir(parents=True)
            (output/'identity.json').write_text('{"owned_pilot_metadata":true}\n')
            capture = old_tests.capture
            with patch.object(capture.socket, 'gethostname', return_value='qualified-test-worker-0'), \
                 patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=973*driver.GiB)), \
                 patch.object(capture.shutil, 'which', return_value=None):
                owner['cap_bytes'] = 45*driver.GiB; path.write_text(json.dumps(storage))
                with self.assertRaisesRegex(ValueError, 'Owned run checkpoint cap exceeded'):
                    capture.storage_admission(path, output, expected_digest=capture.digest(path))
                owner['cap_bytes'] = 46*driver.GiB; path.write_text(json.dumps(storage))
                result = capture.storage_admission(path, output, expected_digest=capture.digest(path))
                self.assertTrue(result['ok']); self.assertGreater(result['owned_usage'][0]['allocated_bytes'], 0)
                self.assertEqual(result['next_checkpoint_upper_bytes'], 45*driver.GiB)
                self.assertEqual(result['reserve_bytes'], 100*driver.GiB)

    def test_invalid_selection_or_partial_overall_manifest_rejected_before_origin_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, _ = self.fixture(root)
            with patch.object(driver.bench, 'origin') as origin:
                for selected in ('', ' ', 'G1_mid,', ',G1_mid', 'G1_mid,G1_mid', 'unknown', ['G1_mid']):
                    with self.subTest(selected=selected), self.assertRaises(ValueError):
                        driver.build_plan(manifest, root/'controller', selected)
                partial = copy.deepcopy(manifest); partial['cases'] = [c for c in partial['cases'] if c['stage'] == 'mid']
                with self.assertRaises(ValueError): driver.build_plan(partial, root/'controller', 'G1_mid,G3_mid')
                duplicate = copy.deepcopy(manifest); duplicate['cases'][0]['name'] = 'G1_mid'
                with self.assertRaises(ValueError): driver.build_plan(duplicate, root/'controller', 'G1_mid')
                origin.assert_not_called()

    def test_unselected_metadata_population_and_output_overlap_still_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, _ = self.fixture(root)
            mid_root = next(c['output_root'] for c in manifest['cases'] if c['name'] == 'G1_mid')
            mutations = [{'output_root': mid_root}, {'output_root': mid_root+'/child'},
                         {'output_root': mid_root+'/child/..'},
                         {'train_args': manifest['cases'][0]['train_args']+['--debug-scenes', '32']}]
            with patch.object(driver.bench, 'origin') as origin:
                for mutation in mutations:
                    changed = copy.deepcopy(manifest); changed['cases'][0].update(mutation)
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        driver.build_plan(changed, root/'controller', 'G1_mid,G3_mid')
                origin.assert_not_called()

    def test_phase_receipts_leave_early_pending_and_partial_triplet_pending(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, natives = self.fixture(root)
            with patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]):
                plan = driver.build_plan(manifest, root/'controller', 'G1_mid,G3_mid')
            receipts = [{'key': j['key'], 'role': j['role']} for j in plan['jobs']]
            complete = driver.case_capture_progress(plan, receipts)
            self.assertTrue(complete['selected_phase_captures_complete']); self.assertFalse(complete['overall_capture_matrix_complete'])
            self.assertEqual(complete['pending_overall_capture_cases'], ['G1_early', 'G3_early'])
            self.assertEqual(complete['overall_V1_acceptance'], 'PENDING')
            self.assertFalse(complete['passed']); self.assertFalse(complete['formal_authorized'])
            partial = driver.case_capture_progress(plan, receipts[:-1])
            self.assertFalse(partial['selected_phase_captures_complete'])
            self.assertIn('G3_mid', partial['pending_overall_capture_cases'])

    def test_cli_selected_phase_plan_and_execute_reports_cannot_claim_full_acceptance(self):
        # Synthetic CPU control flow only: all GPU/storage/receipt producers are
        # mocked; these files are not native execution or numerical receipts.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, natives = self.fixture(root)
            q = root/'qualification.json'
            q.write_text(json.dumps({'passed': True, 'host': manifest['host'], 'physical_gpu_UUIDs': manifest['gpu_uuids']}))
            manifest['qualification'] = {'path': str(q), 'sha256': driver.bench.digest(q)}
            manifest['budget'] = {**budget(), 'phase_limit': 33., 'reserved_elsewhere': 37.}
            m = root/'overall.json'; m.write_text(json.dumps(manifest)); before = driver.bench.digest(m)
            for execute in (False, True):
                output = root/('synthetic_execute' if execute else 'plan')
                args = [str(driver.SCRIPT), '--manifest', str(m), '--output', str(output), '--cases', 'G1_mid,G3_mid']
                if execute: args += ['--execute']
                def fake_receipt(value, *unused): return {'key': value['key'], 'role': value['role']}
                with patch.object(sys, 'argv', args), patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]), \
                     patch.object(driver.bench, 'immutable_source'), patch.object(driver.bench, 'checked', return_value='a'*40), \
                     patch.object(driver, 'qualified_runtime_snapshot', return_value={'packages': {}}), \
                     patch.object(driver.bench, 'acquire', return_value=None), patch.object(driver.bench, 'readiness', return_value={'CPU_mock': True}), \
                     patch.object(driver, 'storage_admission', return_value={'CPU_mock': True}), \
                     patch.object(driver, 'run_budgeted_job', return_value={'wall_seconds': 0.}) as run, \
                     patch.object(driver, 'capture_receipt', side_effect=fake_receipt), patch.object(driver, 'count_triplets', return_value=[]):
                    self.assertEqual(driver.main(), 0)
                plan = driver.bench.read(output/'PLAN.json')
                self.assertEqual(plan['manifest'], manifest); self.assertEqual(plan['manifest_sha256'], before)
                self.assertEqual(len(plan['jobs']), 6); self.assertFalse(plan['formal_authorized'])
                if execute:
                    self.assertEqual(run.call_count, 6)
                    result = driver.bench.read(output/'RESULTS.json')
                    self.assertEqual(result['status'], 'PHASE_CAPTURES_COMPLETE_OVERALL_V1_PENDING')
                    self.assertEqual(result['case_capture_progress']['pending_overall_capture_cases'], ['G1_early', 'G3_early'])
                    self.assertFalse(result['passed']); self.assertFalse(result['formal_authorized'])
                    rng = driver.bench.read(output/'NATIVE_RNG_SOURCES.json')
                    self.assertEqual([s['case'] for s in rng['sources']], ['G1_mid', 'G3_mid'])
                    gate = driver.bench.read(output/'GPU_BUDGET_REGISTRATION.json')['admission']
                    self.assertEqual(gate['projected_phase_GPU_hours'], 33.)
                else:
                    run.assert_not_called(); self.assertFalse((output/'RESULTS.json').exists())
            self.assertEqual(driver.bench.digest(m), before)


class RecoveryTests(unittest.TestCase):
    def test_fresh_recovery_template_headroom_retains_old_budget_and_unknown_execution_budget(self):
        reports = ROOT/'reports/structured_world/execution_20261011'
        old = driver.bench.read(reports/'native_driver_budget/RECOVERY_G1_STORAGE_BUDGET_V1.json')
        path = reports/'native_phase_selection/RECOVERY_G1_STORAGE_BUDGET_V2.json'
        new = driver.bench.read(path)
        manifest = driver.bench.read(reports/'native_phase_selection/RECOVERY_G1_MANIFEST_V2_PENDING_BUDGET.json')
        self.assertEqual([o['cap_bytes']//driver.GiB for o in old['owned_runs']], [45, 45, 90])
        self.assertEqual([o['cap_bytes']//driver.GiB for o in new['owned_runs']], [46, 46, 91])
        self.assertEqual(new['pool_cap_bytes'], 183*driver.GiB)
        self.assertEqual(new['reserve_bytes'], old['reserve_bytes'])
        self.assertEqual(new['checkpoint_upper_bytes'], old['checkpoint_upper_bytes'])
        self.assertTrue(set(o['path'] for o in old['owned_runs']).isdisjoint(o['path'] for o in new['owned_runs']))
        self.assertEqual(manifest['cases'][0]['storage_budget']['sha256'], driver.bench.digest(path))
        self.assertEqual(manifest['source_sha'], 'be5e72248a46afd43e1bf279c21134c391c5edad')
        self.assertIsNone(manifest['budget']['already_spent_external'])
        self.assertIsNone(manifest['budget']['reserved_elsewhere'])
        self.assertFalse(manifest['acceptance_passed'])
        with self.assertRaises(ValueError): driver.validate_gpu_budget(manifest, [{'stage': 'recovery'}])

    def fixture(self, root):
        m, natives = old_tests.PlanTests().fixture(root)
        case = next(c for c in m['cases'] if c['name'] == 'G1_mid')
        path = Path(case['storage_budget']['path'])
        storage = driver.bench.read(path)
        for o in storage['owned_runs']: o['cap_bytes'] = (90 if Path(o['path']).name == 'B' else 45)*(1 << 30)
        path.write_text(json.dumps(storage)); case['storage_budget']['sha256'] = driver.bench.digest(path)
        m.update(schema='native_execution_recovery_manifest_v1', cases=[case], budget={**budget(), 'phase_limit': 2., 'already_spent_external': 0., 'reserved_elsewhere': 0.})
        return m, natives

    def test_actual_trainer_parser_fixed_identity_and_only_B_resume(self):
        from torch.distributed.run import get_args_parser
        tree = ast.parse((ROOT/'tools/structured_world/train_vla.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
            isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and
            n.value.func.value.id == 'parser' and n.value.func.attr == 'add_argument']
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest, origins = self.fixture(root)
            with patch.object(driver.bench, 'origin', side_effect=lambda c: origins[c['name']]):
                plan = recovery.build_plan(manifest, root/'controller')
            parsed = []
            for j in plan['jobs']:
                torch_args = get_args_parser().parse_args(j['command'][3:])
                parser = argparse.ArgumentParser()
                exec(compile(ast.Module(body=calls, type_ignores=[]), '<actual trainer parser>', 'exec'), {'parser': parser, 'Path': Path})
                args = parser.parse_args(torch_args.training_script_args)
                driver.observation_plan(args.capture_updates, 25000, args.debug_updates)
                self.assertEqual(args.scope, 'profile'); self.assertEqual(args.debug_updates, 32)
                self.assertEqual(args.capture_updates, '32'); self.assertEqual(args.pilot_origin_tag, manifest['cases'][0]['pilot_origin_tag'])
                self.assertFalse(args.benchmark_no_end_save); self.assertEqual(args.debug_scenes, 0)
                self.assertEqual(torch_args.nproc_per_node, '8'); parsed.append(args)
            self.assertEqual([a.resume for a in parsed], [False, False, False, True])
            self.assertEqual([a.stop_after for a in parsed], [0, 0, 25016, 0])
            self.assertEqual(parsed[2].output, parsed[3].output)
            self.assertEqual([j['expected_exit_completed'] for j in plan['jobs']], [25032, 25032, 25016, 25032])
            self.assertEqual(len({j['process_run'] for j in plan['jobs']}), 4)
            self.assertEqual(plan['maximum_total_optimizer_updates'], 96); self.assertEqual(plan['full_native_banks'], 4)
            for j in plan['jobs']: self.assertEqual(j['capture_plan']['end_update'], 25032)
            b = driver.validate_gpu_budget(manifest, plan['jobs'])
            self.assertEqual(driver.gpu_admission(b, plan['jobs'], {'actual_phase_GPU_hours': 0.}, whole_plan=True)['projected_phase_GPU_hours'], 2.)

    def test_B_needs_two_banks_before_recovery_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m, natives = self.fixture(root)
            case = m['cases'][0]; path = Path(case['storage_budget']['path'])
            storage = driver.bench.read(path)
            storage['owned_runs'][-1]['cap_bytes'] = 45*(1 << 30)
            path.write_text(json.dumps(storage)); case['storage_budget']['sha256'] = driver.bench.digest(path)
            with patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]), self.assertRaises(ValueError):
                recovery.build_plan(m, root/'controller')

    def test_recovery_command_endpoint_with_actual_trainer_resume_expression(self):
        code = old_tests.trainer_expression('end_update')
        self.assertEqual(eval(code, {}, {'pilot_origin': {'completed': 25000},
            'args': argparse.Namespace(debug_updates=32, use_formal_schedule=False), 'horizon': 100000}), 25032)
        # Resume cursor is 25016; debug-updates remains32, not an extra32 or16.
        self.assertEqual(25032-25016, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    parser.add_argument('--runtime-manifest', type=Path,
                        help='Explicit bound zt3 manifest for one read-only CPU package metadata query')
    args = parser.parse_args()
    QualifiedRuntimeTests.runtime_manifest = args.runtime_manifest
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = (Path(__file__), driver.SCRIPT, recovery.SCRIPT)
    report = {'schema': 'native_driver_budget_CPU_receipt_v1', 'CPU_tests_passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'seconds': time.monotonic()-started, 'GPU_jobs_launched': 0, 'qualification_passed': False,
        'scope': 'CPU qualified package union/default coverage, invalid-name and actual missing-metadata controls; optional explicit real zt3 CPU metadata receipt. Selected-phase/budget/recovery controls retained; GPU numerical/recovery acceptance PENDING',
        'actual_CPU_runtime_query': QualifiedRuntimeTests.real_runtime_receipt,
        'source_sha256': {str(p.relative_to(ROOT)): driver.bench.digest(p) for p in sources},
        'training_source_sha256': driver.bench.digest(ROOT/'tools/structured_world/train_vla.py')}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
