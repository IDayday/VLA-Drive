"""CPU receipts: observation does not change optimizer/RNG; exact capture retention.

No CUDA, native DeepSpeed numerical equivalence or formal acceptance is claimed.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import native_capture as capture
from tools.structured_world import run_native_execution_continuation as driver


def equality(test, a, b):
    import torch
    if torch.is_tensor(a): test.assertTrue(torch.equal(a, b)); return
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a: equality(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b): equality(test, x, y)
    else: test.assertEqual(a, b)


def trainer_expression(name):
    tree = ast.parse((ROOT/'tools/structured_world/train_vla.py').read_text())
    if name == 'tag':
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'save')
    values = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(values) != 1: raise ValueError('Expected unique real trainer expression: '+name)
    return compile(ast.Expression(values[0]), '<actual trainer '+name+'>', 'eval')


class CaptureTests(unittest.TestCase):
    def args(self, **kw):
        values = dict(capture_updates='1,32,100,250', capture_storage_budget=Path('/own/budget.json'),
                      scope='profile', pilot_origin_run=Path('/origin'), pilot_origin_tag='milestone_0025000',
                      debug_scenes=0, benchmark_no_end_save=False, resume_origin_run=None)
        return SimpleNamespace(**{**values, **kw})

    def test_profile_only_guard_and_default(self):
        capture.validate_args(self.args())
        capture.validate_args(self.args(capture_updates=None, capture_storage_budget=None))
        self.assertIsNone(capture.observation_plan(None, 0, 0))
        for kw in ({'scope': 'formal'}, {'pilot_origin_run': None}, {'pilot_origin_tag': None},
                   {'debug_scenes': 32}, {'benchmark_no_end_save': True}, {'resume_origin_run': '/parent'},
                   {'capture_storage_budget': None}, {'capture_updates': None}):
            with self.subTest(kw=kw), self.assertRaises(ValueError): capture.validate_args(self.args(**kw))

    def test_offsets_are_relative_to_fixed_origin_not_resume_cursor(self):
        plan = capture.observation_plan('1,32,100,250', 25000, 500)
        self.assertEqual(plan['absolute_updates'], [25001, 25032, 25100, 25250])
        end = trainer_expression('end_update')
        cursor = 25032
        self.assertEqual(eval(end, {}, {'pilot_origin': {'completed': 25000}, 'args': SimpleNamespace(debug_updates=500, use_formal_schedule=False), 'horizon': 100000}), 25500)
        self.assertEqual([u for u in plan['absolute_updates'] if u > cursor], [25100, 25250])
        self.assertEqual(capture.observation_plan('1', 2, 1)['absolute_updates'], [3])
        for text in ('', '0', '1,1', '32,1', '-1', '501', '1,x'):
            with self.subTest(text=text), self.assertRaises(ValueError): capture.observation_plan(text, 25000, 500)

    def test_real_trainer_save_conditions_and_retention(self):
        milestones_expr, due, tag = map(trainer_expression, ('milestones', 'save_due', 'tag'))
        plan = capture.observation_plan('1,32,100,250', 25000, 500)
        args = SimpleNamespace(scope='profile', save_every=2147483647)
        milestones = eval(milestones_expr, {}, {'args': args, 'dataset': 'navsim', 'end_update': 25500})
        milestones.update(plan['absolute_updates'])
        tags = []
        for completed in range(25001, 25501):
            env = dict(stop=False, completed=completed, end_update=25500, milestones=milestones, args=args)
            io_due = completed in milestones or completed % args.save_every == 0
            self.assertEqual(eval(due, {}, env), io_due)
            if io_due: tags.append(eval(tag, {}, env))
        self.assertEqual(tags, ['milestone_'+str(n).zfill(7) for n in (25001, 25032, 25100, 25250, 25500)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in tags+['latest_0025010', 'latest_0025020', 'latest_0025030']:
                (root/name).mkdir()
            # Actual retention glob excludes every native observation.
            rolling = sorted(root.glob('latest_*'))
            for old in rolling[:-2]: old.rmdir()
            self.assertTrue(all((root/name).is_dir() for name in tags))
            self.assertEqual(len(list(root.glob('latest_*'))), 2)
        # End overlap is one set member/save, not two callbacks.
        self.assertEqual(set(capture.observation_plan('1', 2, 1)['absolute_updates']) | {3}, {3})

    def test_metadata_base_precision_unchanged_and_e3_explicit(self):
        tree = ast.parse((ROOT/'tools/structured_world/train_vla.py').read_text())
        names = {'base_precision', 'base_precision_scope', 'effective_training_precision'}
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1 and
            isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Name) and
            n.targets[0].value.id == 'contract' and isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value in names]
        code = compile(ast.Module(body=assignments, type_ignores=[]), '<precision metadata>', 'exec')
        for mode in ('reference', 'io_preserving_v1', 'engineering_fast', 'engineering_timevec', 'precision_fast'):
            env = {'contract': {'precision': 'original locked scientific precision'}, 'args': SimpleNamespace(execution_mode=mode)}
            exec(code, env)
            c = env['contract']
            self.assertEqual(c['precision'], c['base_precision'])
            if mode != 'precision_fast': self.assertEqual(c['effective_training_precision'], c['precision'])
            else:
                for text in ('BF16 Qwen', 'BF16 Geo R50/FPN/BEV', 'FP32 DepthNet', 'gradient communication', 'TF32 off', 'canonical inference FP32'):
                    self.assertIn(text, c['effective_training_precision'])

    def test_actual_resume_first_and_capture_boundary_evidence(self):
        code = trainer_expression('evidence_due')
        def due(completed, loaded=16, fast=True, capture_plan=True):
            return eval(code, {}, {'completed': completed, 'loaded_boundary_completed': loaded, 'fast': fast,
                'capture_plan': capture_plan, 'milestones': {32, 100, 250, 500}, 'args': SimpleNamespace(evidence_every=100)})
        self.assertTrue(due(16)); self.assertFalse(due(17)); self.assertTrue(due(31))
        self.assertTrue(due(249)); self.assertTrue(due(499)); self.assertTrue(due(100))
        self.assertFalse(due(31, capture_plan=None)); self.assertTrue(due(17, fast=False))
        self.assertFalse(due(16, loaded=None, capture_plan=None))
        tree = ast.parse((ROOT/'tools/structured_world/train_vla.py').read_text())
        activation = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == 'record_activation_metrics' for t in n.targets)]
        self.assertEqual(len(activation), 1)
        self.assertTrue(isinstance(activation[0].value, ast.Name) and activation[0].value.id == 'evidence_due')

    def test_continuous_500_cpu_adam_bn_dropout_and_restore(self):
        import torch
        torch.set_num_threads(2)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(4, 4)
                self.bn = torch.nn.BatchNorm1d(4)
                self.drop = torch.nn.Dropout(.2)
            def forward(self, x): return self.drop(self.bn(self.dense(x)))
        def run(folder, observations, resume=None):
            torch.manual_seed(42)
            model = Tiny(); optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
            generator = torch.Generator().manual_seed(43)
            start = 0
            if resume:
                state = torch.load(resume, weights_only=False)
                model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                generator.set_state(state['generator']); torch.set_rng_state(state['rng']); start = state['step']
            losses = []
            milestones = set(observations) | {500}
            for step in range(start+1, 501):
                x = torch.randn(8, 4, generator=generator)
                optimizer.zero_grad(); loss = model(x).square().mean(); loss.backward(); optimizer.step()
                losses.append(loss.detach().clone())
                if observations and step in milestones:
                    # Serialize actual model/master/moments/buffers/task+dropout
                    # RNG at an observation without pausing/restarting the loop.
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                        'generator': generator.get_state(), 'rng': torch.get_rng_state()}, folder/f'milestone_{step:07d}.pt')
            return {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                'rng': torch.get_rng_state().clone(), 'generator': generator.get_state().clone()}, losses
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            reference, loss_a = run(folder, [])
            observed, loss_b = run(folder, [1, 32, 100, 250])
            equality(self, reference, observed); equality(self, loss_a, loss_b)
            restored, loss_c = run(folder, [], folder/'milestone_0000032.pt')
            equality(self, observed, restored); equality(self, loss_b[32:], loss_c)
            self.assertEqual([int(p.stem.rsplit('_', 1)[1]) for p in sorted(folder.glob('milestone_*.pt'))], list(driver.POINTS))

    def test_actual_denominator_observation_no_extra_rng_or_collective(self):
        import torch
        from tools.foresight.student_state import optimizer_batch_counts
        targets = {'ego': torch.zeros(4, 6, 3), 'future_dino': torch.ones(4, 3, 3, 2, 2, 2),
            'future_dino_valid': torch.ones(4, 3, 3, 2, 2, dtype=torch.bool),
            'future_clip': torch.zeros(4, 8, 1, 1, 1, 1), 'future_clip_valid': torch.ones(4, 8, 1, 1, 1, dtype=torch.bool)}
        generator = torch.Generator().manual_seed(42)
        with patch('tools.foresight.student_state.dist.is_initialized', return_value=True), \
             patch('tools.foresight.student_state.dist.all_reduce', side_effect=lambda values: values.mul_(8)) as reduce:
            counts = optimizer_batch_counts(targets, generator, torch.device('cpu'))
            before = generator.get_state().clone()
            row = {'global_counts': dict(counts), 'loss_horizons': {'dino_horizon': targets['dino_horizon'].detach().tolist()}}
            result = json.loads(json.dumps(row))
            self.assertEqual(result['global_counts'], {'ego_scenes': 32., 'future_clip': 32., 'future_dino': 768.})
            self.assertEqual(len(result['loss_horizons']['dino_horizon']), 4)
            self.assertTrue(torch.equal(before, generator.get_state()))
            self.assertEqual(reduce.call_count, 1)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.out = self.root/'owned'/'G1_mid'/'A'
        self.path = self.root/'budget.json'
        self.budget = {'schema': 'native_capture_storage_budget_v1', 'approved_for_pilot': True,
            'host': 'qualified-test', 'reserve_bytes': 100*capture.GiB, 'checkpoint_upper_bytes': 45*capture.GiB,
            'owned_runs': [{'path': str(self.out), 'kind': 'owned_numerical_pilot', 'cap_bytes': 225*capture.GiB}],
            'pool_cap_bytes': 225*capture.GiB, 'filesystem_probe_path': str(self.root)}
        self.path.write_text(json.dumps(self.budget))
        self.patchers = [patch.object(capture.socket, 'gethostname', return_value='qualified-test-worker-0'),
            patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=973*capture.GiB)),
            patch.object(capture.shutil, 'which', return_value=None)]
        for p in self.patchers: p.start()
    def tearDown(self):
        for p in reversed(self.patchers): p.stop()
        self.tmp.cleanup()
    def update(self, **kw):
        self.budget.update(kw); self.path.write_text(json.dumps(self.budget))
    def test_unknown_quota_is_disclosed_but_allows_bounded_work(self):
        result = capture.storage_admission(self.path, self.out, expected_digest=capture.digest(self.path))
        self.assertTrue(result['ok']); self.assertEqual(result['quota']['status'], 'quota_not_exposed')
        self.assertFalse(result['exclusive_filesystem_reservation']); self.assertFalse(self.out.exists())
    def test_actual_free_space_reserve_and_next_full_checkpoint(self):
        with patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=145*capture.GiB)):
            self.assertTrue(capture.storage_admission(self.path, self.out)['ok'])
        for free in (100*capture.GiB, 145*capture.GiB-1):
            with patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), self.assertRaises(ValueError):
                capture.storage_admission(self.path, self.out)
    def test_explicit_caps_and_foreign_roots(self):
        with self.assertRaises(ValueError): capture.storage_admission(self.path, self.root/'formal')
        self.update(owned_runs=[{**self.budget['owned_runs'][0], 'cap_bytes': 1}])
        with self.assertRaises(ValueError): capture.storage_admission(self.path, self.out)
        self.update(owned_runs=[{**self.budget['owned_runs'][0], 'cap_bytes': 225*capture.GiB}], pool_cap_bytes=224*capture.GiB)
        with self.assertRaises(ValueError): capture.storage_admission(self.path, self.out)
    def test_source_bound_budget_mutation_and_symlink(self):
        before = capture.digest(self.path)
        self.update(pool_cap_bytes=226*capture.GiB)
        with self.assertRaises(ValueError): capture.storage_admission(self.path, self.out, expected_digest=before)
        self.out.mkdir(parents=True)
        (self.out/'bad').symlink_to(self.path)
        with self.assertRaises(ValueError): capture.storage_admission(self.path, self.out)
    def test_observed_quota_error_does_not_claim_unlimited(self):
        with patch.object(capture.shutil, 'which', side_effect=lambda name: '/usr/bin/lfs' if name == 'lfs' else None), \
             patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='not exposed')):
            q = capture.storage_admission(self.path, self.out)['quota']
            self.assertEqual(q['status'], 'quota_not_exposed'); self.assertFalse(q['unlimited_claimed'])


class PlanTests(unittest.TestCase):
    def test_all_rank_triplet_detects_single_step_horizon_drift(self):
        with tempfile.TemporaryDirectory() as d:
            folder, results = Path(d), []
            for role in ('A', 'AA', 'B'):
                evidence = []
                for rank in range(8):
                    path = folder/(role+'_rank'+str(rank)+'.jsonl')
                    with path.open('w') as stream:
                        for u in range(25001, 25501):
                            row = {'updates': u, 'epoch': 8, 'offset_after_update': u-25000,
                                'global_counts': {'ego_scenes': 32., 'future_dino': 768.},
                                'schedule_horizon': 100000, 'schedule_warmup': 5000, 'continuation_endpoint': 25500,
                                'source_population_scenes': 101592, 'rank': rank,
                                'global_scene_indices': list(range(32)), 'local_scene_indices': list(range(rank, 32, 8)),
                                'loss_horizons': {'dino_horizon': [(u+rank)%3]*4}, 'logging_seconds_prior': u*(1 if role == 'A' else 2)}
                            stream.write(json.dumps(row)+'\n')
                    evidence.append({'path': str(path), 'rank': rank})
                results.append({'key': 'G1_mid_'+role, 'role': role, 'count_evidence': evidence})
            self.assertTrue(driver.count_triplets(results)[0]['all8_count_scene_horizon_exact'])
            path = Path(results[-1]['count_evidence'][7]['path'])
            rows = driver.bench.json_lines(path)
            rows[249]['loss_horizons']['dino_horizon'][0] = (rows[249]['loss_horizons']['dino_horizon'][0]+1)%3
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): driver.count_triplets(results)

    def fixture(self, root):
        manifest = {'schema': 'native_execution_continuation_manifest_v1', 'source_root': str(ROOT), 'source_sha': 'a'*40,
            'python': sys.executable, 'host': 'qualified-test', 'artifact_root': str(root/'locks'),
            'gpu_uuids': ['GPU-test'+str(i) for i in range(8)], 'minimum_free_MiB': 79000,
            'candidate_mode': 'engineering_fast', 'cases': []}
        natives = {}
        for group, stage in sorted(driver.REQUIRED):
            name = group+'_'+stage; output = root/name
            budget_path = root/(name+'_budget.json')
            banks = 1 if stage == 'early' else 5
            budget_path.write_text(json.dumps({'host': manifest['host'], 'checkpoint_upper_bytes': 45*capture.GiB,
                'owned_runs': [{'path': str(output/r), 'kind': 'owned_numerical_pilot', 'cap_bytes': banks*45*capture.GiB} for r in ('A', 'AA', 'B')]}))
            case = {'name': name, 'group': group, 'stage': stage, 'output_root': str(output),
                'pilot_origin_run': str(root/('origin_'+group)), 'pilot_origin_tag': 'milestone_0000002',
                'storage_budget': {'path': str(budget_path), 'sha256': capture.digest(budget_path)},
                'train_args': sum(([flag, '2' if flag == '--micro-batch' else '/frozen/'+flag[2:]] for flag in sorted(driver.bench.BASE_FLAGS)), [])}
            manifest['cases'].append(case)
            start = 2 if stage == 'early' else 25000 if group == 'G1' else 50000
            natives[name] = {'complete': {'completed': start}, 'contract': {'schedule_horizon': 100000},
                'run': case['pilot_origin_run'], 'tag': case['pilot_origin_tag'], 'complete_sha256': 'b'*64}
        return manifest, natives
    def test_actual_torchrun_parser_fixed_continuous_matrix(self):
        from torch.distributed.run import get_args_parser
        with tempfile.TemporaryDirectory() as d:
            manifest, natives = self.fixture(Path(d))
            with patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]):
                plan = driver.build_plan(manifest, Path(d)/'controller')
            self.assertEqual((len(plan['jobs']), plan['maximum_total_updates'], plan['expected_full_native_banks']), (12, 3006, 36))
            for job in plan['jobs']:
                command = job['command']; argv = command[3:]
                parsed = get_args_parser().parse_args(argv)
                self.assertEqual(parsed.nproc_per_node, '8'); self.assertEqual(parsed.max_restarts, 0)
                args = dict(zip(parsed.training_script_args[::2], parsed.training_script_args[1::2]))
                self.assertEqual(args['--debug-updates'], '1' if job['stage'] == 'early' else '500')
                self.assertEqual(args['--capture-updates'], '1' if job['stage'] == 'early' else '1,32,100,250')
                self.assertEqual(args['--execution-mode'], manifest['candidate_mode'] if job['role'] == 'B' else 'io_preserving_v1')
                self.assertNotIn('--stop-after', args); self.assertNotIn('--benchmark-no-end-save', args)
                self.assertEqual(args['--save-every'], '2147483647')
            self.assertFalse(plan['passed']); self.assertFalse(plan['formal_authorized'])
    def test_partial_population_or_capture_budget_missing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, natives = self.fixture(Path(d))
            manifest['cases'][0]['train_args'] += ['--debug-scenes', '32']
            with patch.object(driver.bench, 'origin', side_effect=lambda c: natives[c['name']]), self.assertRaises(ValueError):
                driver.build_plan(manifest, Path(d)/'controller')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'schema': 'native_capture_CPU_receipt_v1', 'CPU_tests_passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'seconds': time.monotonic()-start, 'qualification_passed': False, 'GPU_jobs_launched': 0,
        'scope': 'CPU callback/optimizer/BN/dropout/RNG restore and real trainer expressions; not native GPU acceptance',
        'source_sha256': {str(p.relative_to(ROOT)): capture.digest(p) for p in
            (Path(__file__), ROOT/'tools/structured_world/native_capture.py', ROOT/'tools/structured_world/train_vla.py',
             ROOT/'tools/structured_world/run_native_execution_continuation.py')}}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
