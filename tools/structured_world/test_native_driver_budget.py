"""CPU tests for dispatch budgets and the fixed32 native recovery commands.

An actual failed CPU subprocess exercises the existing process monitor/ledger;
no GPU worker or native numerical acceptance is exercised.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import run_native_execution_continuation as driver
from tools.structured_world import run_native_execution_recovery as recovery
from tools.structured_world import test_native_capture as old_tests


def budget():
    return {'phase_limit': 36., 'already_spent_external': 50., 'total_internal_limit': 120.,
            'reserved_elsewhere': 34., 'per_job_reserves': {'early': .5, 'mid': 5.5, 'recovery': .5},
            'accounting_sources': []}


def job(root, key, stage='early', process=None):
    value = {'key': key, 'stage': stage, 'run': str(root/key)}
    if process: value['process_run'] = str(root/process)
    return value


def exit_receipt(value, hours, returncode=0):
    root = Path(value.get('process_run', value['run'])); root.mkdir(parents=True)
    (root/'PROCESS_EXIT.json').write_text(json.dumps({'wall_seconds': hours*3600/8, 'returncode': returncode}))


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
                       {'total_internal_limit': float('inf')}, {'total_internal_limit': 121}, {'phase_limit': True},
                       {'per_job_reserves': {'mid': 0}}, {'per_job_reserves': {'mid': float('nan')}}):
            with self.subTest(change=change), self.assertRaises(ValueError): driver.validate_gpu_budget({'budget': {**budget(), **change}}, jobs)
        with self.assertRaises(ValueError): driver.validate_gpu_budget({}, jobs)

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
                plan = {'jobs': [{**job(root, 'never_started'), 'trainer': str(ROOT/'tools/structured_world/train_vla.py')}], 'source_root': str(ROOT), 'source_sha': 'a'*40}
                args = [str(module.SCRIPT), '--manifest', str(m), '--output', str(root/'controller'), '--execute']
                with patch.object(sys, 'argv', args), patch.object(module, 'build_plan', return_value=plan), \
                     patch.object(driver.bench, 'immutable_source'), patch.object(driver.bench, 'readiness') as readiness, \
                     patch.object(driver.bench, 'run_job') as run:
                    self.assertEqual(module.main(), 1)
                    readiness.assert_not_called(); run.assert_not_called()
                self.assertFalse(Path(plan['jobs'][0]['run']).exists())


class RecoveryTests(unittest.TestCase):
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
    args = parser.parse_args()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = (Path(__file__), driver.SCRIPT, recovery.SCRIPT)
    report = {'schema': 'native_driver_budget_CPU_receipt_v1', 'CPU_tests_passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'seconds': time.monotonic()-started, 'GPU_jobs_launched': 0, 'qualification_passed': False,
        'scope': 'CPU budget boundaries, actual CPU failed process receipt, real torchrun/trainer parser and recovery fixed endpoint; GPU numerical/recovery acceptance PENDING',
        'source_sha256': {str(p.relative_to(ROOT)): driver.bench.digest(p) for p in sources},
        'training_source_sha256': driver.bench.digest(ROOT/'tools/structured_world/train_vla.py')}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
