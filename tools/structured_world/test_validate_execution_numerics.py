"""CPU controls: NEW advisory checks never relabel OLD strict failure."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tools.structured_world import validate_execution_numerics as producer

torch.set_num_threads(1)


class NumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, cls.bounds = producer.registration()

    def test_original_strict_fail_and_new_forward_advisory_pass_are_separate(self):
        a, b = torch.ones(10), torch.ones(10)+2e-5
        self.assertFalse(torch.allclose(a, b, rtol=1e-6, atol=1e-7))
        stats = producer.tensor_stats(a, b, self.bounds)
        self.assertTrue(stats['torch_allclose_passed'])
        self.assertTrue(stats['advisory_passed'])
        self.assertEqual(stats['violating_elements'], 0)
        self.assertLessEqual(stats['relative_L2'], 1e-4)

    def test_maximum_alone_cannot_prove_elementwise_and_relative_checks(self):
        a = torch.tensor([1., 0.])
        b = a+1.9e-5
        stats = producer.tensor_stats(a, b, self.bounds)
        self.assertLessEqual(stats['relative_L2'], 1e-4)
        self.assertFalse(stats['torch_allclose_passed'])
        self.assertEqual(stats['violating_elements'], 1)
        self.assertFalse(stats['advisory_passed'])

    def test_nonfinite_and_zero_reference_do_not_get_epsilon_acceptance(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            stats = producer.tensor_stats(torch.tensor([value]), torch.tensor([value]), self.bounds)
            self.assertEqual(stats['nonfinite_elements'], 1)
            self.assertFalse(stats['advisory_passed'])
            json.dumps(stats, allow_nan=False)
        zero = torch.zeros(2)
        tiny = torch.tensor([1e-8, 0.])
        stats = producer.tensor_stats(zero, tiny, self.bounds)
        self.assertTrue(stats['torch_allclose_passed'])
        self.assertTrue(stats['zero_reference_norm'])
        self.assertIsNone(stats['relative_L2'])
        self.assertFalse(stats['advisory_passed'])
        self.assertEqual(stats['violating_elements'], 1)
        self.assertTrue(producer.tensor_stats(zero, zero.clone(), self.bounds)['advisory_passed'])
        self.assertFalse(producer.bitwise(torch.tensor([0.]), torch.tensor([-0.])))

    def test_meter_euclidean_error_and_circular_heading(self):
        a = torch.tensor([[[0., 0., math.pi-1e-6]]], dtype=torch.float64)
        b = torch.tensor([[[8e-5, 8e-5, -math.pi+1e-6]]], dtype=torch.float64)
        stats = producer.physical_stats(a, b)
        self.assertGreater(stats['max_position_error_m'], 1e-4)
        self.assertAlmostEqual(stats['max_heading_error_rad'], 2e-6, places=10)
        with self.assertRaisesRegex(ValueError, 'physical'):
            producer.physical_stats(torch.ones(1, 8, 4), torch.ones(1, 8, 4))

    def test_hash_catches_query_order_time_type_shape_without_changing_rng(self):
        value = torch.tensor([[1., 2.], [3., 4.]])
        rng = torch.get_rng_state().clone()
        baseline = producer.typed_tensor_hash(value)
        self.assertEqual(baseline, producer.typed_tensor_hash(value.clone()))
        for changed in (value.flip(0), value.flip(1), value.double(), value.reshape(1, 4)):
            self.assertNotEqual(baseline['sha256'], producer.typed_tensor_hash(changed)['sha256'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, 'detached'):
            producer.typed_tensor_hash(value.requires_grad_())

    def test_real_query_and_global_count_observers_preserve_returns_and_rng(self):
        from tools.structured_world.test_query_execution import labels, GRID
        from starVLA.model.modules.structured_world.queries import sample_queries
        from tools.foresight.student_state import optimizer_batch_counts
        collector = producer.Collector(self.bounds)
        target = labels(); target['ego'] = torch.zeros(3, 8, 4)
        generator = torch.Generator().manual_seed(77)
        observed_counts = producer.observe_counts(optimizer_batch_counts, collector)
        original_counts = optimizer_batch_counts(target, generator, torch.device('cpu'))
        actual_counts = observed_counts(target, generator, torch.device('cpu'))
        self.assertEqual(actual_counts, original_counts)
        self.assertEqual(collector.global_counts['A'], {'ego_scenes': 3.})
        collector.begin()
        proposal = torch.zeros(3, 8, 3)
        start = generator.get_state().clone()
        global_rng = torch.get_rng_state().clone()
        expected = sample_queries(proposal, target, mode='local', count=32, grid=GRID, generator=generator)
        expected_rng = generator.get_state().clone()
        generator.set_state(start)
        returned = []
        def original(*a, **kw):
            result = sample_queries(*a, **kw); returned.append(result); return result
        actual = producer.observe_queries(original, collector)(proposal, target, mode='local', count=32, grid=GRID, generator=generator)
        self.assertIs(actual, returned[0])
        self.assertTrue(producer.bitwise(expected, actual))
        self.assertTrue(torch.equal(expected_rng, generator.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertEqual(collector.queries['A']['poses'], producer.typed_tensor_hash(actual['poses']))
        self.assertEqual(collector.queries['A']['time_index'], producer.typed_tensor_hash(actual['time_index']))

    def test_rank_norms_cannot_hide_zero_or_small_reference_rank(self):
        rows = [producer.tensor_stats(torch.ones(100), torch.ones(100), self.bounds),
                producer.tensor_stats(torch.tensor([1e-8]), torch.tensor([1.01e-8]), self.bounds)]
        pooled = producer.aggregate_stat(rows, self.bounds)
        self.assertLess(pooled['relative_L2'], 1e-4)
        self.assertFalse(pooled['all_rank_bounds_passed'])
        self.assertFalse(pooled['advisory_passed'])
        rows[1] = producer.tensor_stats(torch.zeros(1), torch.tensor([1e-8]), self.bounds)
        self.assertEqual(producer.aggregate_stat(rows, self.bounds)['violating_elements'], 1)

    def test_registration_hash_and_protected_historical_sources(self):
        self.assertEqual(producer.digest(producer.REGISTRATION), producer.REGISTRATION_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'altered.json'
            changed = copy.deepcopy(self.policy)
            changed['tiers']['E2']['forward_by_mode']['engineering_timevec']['module_tolerances']['rtol'] = .01
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'): producer.registration(path)

    def test_prospective_binding_allows_only_exact_declared_capture_trainer_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in producer.BINDING_FILES:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT/name).read_bytes())
            for old in self.policy['protected_files']:
                raw = subprocess.check_output(['git', 'show', self.policy['frozen_comparator_commit']+':'+old['path']], cwd=ROOT)
                (root/old['path']).write_bytes(raw)
            trainer = root/'tools/structured_world/train_vla.py'
            trainer.write_bytes(trainer.read_bytes()+b'\n# Synthetic owned capture-only metadata fixture.\n')
            old = next(row for row in self.policy['protected_files'] if row['path'].endswith('/train_vla.py'))
            binding = {'schema': 'execution_forward_producer_source_binding_v1', 'state': 'frozen_before_candidate',
                       'candidate_mode': 'engineering_timevec',
                       'roles': {'A':'io_preserving_v1','AA':'io_preserving_v1','B':'engineering_timevec'},
                       'registration_sha256': producer.REGISTRATION_SHA, 'frozen_UTC': '2026-01-01T00:00:00+00:00',
                       'source_git_sha': 'a'*40,
                       'files_sha256': {name: producer.digest(root/name) for name in producer.BINDING_FILES},
                       'protected_source_changes': [{'path': old['path'], 'historical_sha256': old['sha256'],
                            'producer_sha256': producer.digest(trainer), 'reason': producer.CAPTURE_REASON, 'algorithm_changed': False}]}
            path = root/'binding.json'
            def write(): path.write_text(json.dumps(binding)); return producer.digest(path)
            with mock.patch.object(producer, 'ROOT', root), mock.patch.object(producer.subprocess, 'check_output',
                    side_effect=lambda argv, **kw: 'a'*40+'\n' if argv[1] == 'rev-parse' else ''):
                proof = producer.source_binding(path, write(), self.policy)
                self.assertEqual(len(proof['protected_source_changes']), 1)
                binding['protected_source_changes'][0]['algorithm_changed'] = True
                with self.assertRaisesRegex(ValueError, 'capture-only'): producer.source_binding(path, write(), self.policy)
                binding['protected_source_changes'][0]['algorithm_changed'] = False
                binding['state'] = 'template_not_frozen'
                with self.assertRaisesRegex(ValueError, 'frozen'): producer.source_binding(path, write(), self.policy)
                binding['state'] = 'frozen_before_candidate'
                write()
                trainer.write_bytes(trainer.read_bytes()+b'# changed after freeze\n')
                with self.assertRaisesRegex(ValueError, 'source changed'): producer.source_binding(path, producer.digest(path), self.policy)

    def test_legacy_failure_exit_and_exact_original_receipt_survive_new_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'new_output'
            framework = SimpleNamespace(sample_queries=lambda q0: {'poses': q0.clone(), 'time_index': torch.arange(8)[None]})
            class NativeCPUModel(nn.Module):
                def __init__(self): super().__init__(); self.p = nn.Parameter(torch.ones(1)); self.calls = 0
                def forward(self, ignored=None):
                    q0 = torch.zeros(1, 8, 3)
                    framework.sample_queries(q0)
                    delta = 2e-5 if self.calls == 2 else 0.
                    self.calls += 1
                    return {'loss': self.p.sum(), 'losses': {'task': self.p.sum()},
                            'diagnostics': {'q0': q0, 'q_final': q0.clone(), 'H_A': torch.ones(1), 'B0': torch.ones(1), 'Bt': torch.ones(1)+delta},
                            'metrics': {'raw': {'task': self.p.detach().sum()}, 'effective_counts': {'task': 1.},
                                        'query_strata': [{'count': 8}], 'proposal_out_of_range': torch.zeros(1,8,dtype=torch.bool),
                                        'refine_effective_weight': 1., 'activation_gradient_samples': {name: torch.ones(8) for name in ('H_A','B0','Bt')}}}
            class NativeCPUDist:
                def all_gather_object(self, rows, value):
                    for rank in range(len(rows)):
                        rows[rank] = copy.deepcopy(value)
                        rows[rank]['rank'] = rank
                        rows[rank]['tokens'] = ['owned_CPU_fixture_'+str(rank)]
            legacy = SimpleNamespace(VLAStructuredFGTR=NativeCPUModel, capture_rng=lambda generators: {'CPU': torch.zeros(2,dtype=torch.uint8)},
                                     optimizer_batch_counts=lambda *a: {'ego_scenes': 32.}, dist=NativeCPUDist())
            strict = {'passed': False, 'strict_gradient_passed': False, 'old_thresholds': {'atol': 1e-7,'rtol': 1e-6}}
            def native_main():
                model = legacy.VLAStructuredFGTR()
                for _ in range(3):
                    model.zero_grad(set_to_none=True)
                    legacy.optimizer_batch_counts(None)
                    result = model(None); result['loss'].backward(); legacy.capture_rng(None)
                legacy.dist.all_gather_object([None]*8, {'rank': 0, 'tokens': ['fixture'], 'checks': []})
                producer.write_new(output/'strict'/'COMPLETE.json', strict)
                raise AssertionError('Real model execution equivalence failed; formal rollout prohibited')
            legacy.main = native_main
            imports = {'tools.structured_world.validate_engineering_execution': legacy,
                       'starVLA.model.framework.vla_structured_fgtr': framework}
            args = ['test', '--run', tmp, '--tag', 'fixed', '--cache', tmp, '--dino-root', tmp, '--output', str(output),
                    '--source-binding', str(Path(tmp)/'binding.json'), '--source-binding-sha256', 'b'*64]
            with mock.patch.object(sys, 'argv', args), mock.patch.object(producer.importlib, 'import_module', side_effect=imports.__getitem__), \
                    mock.patch.object(producer, 'source_binding', return_value={'sha256': 'b'*64}):
                with self.assertRaisesRegex(AssertionError, 'formal rollout prohibited'): producer.main()
            self.assertEqual(json.loads((output/'strict'/'COMPLETE.json').read_text()), strict)
            report = json.loads((output/'FORWARD_COMPARISON.json').read_text())
            self.assertFalse(report['original_strict']['passed'])
            self.assertTrue(report['advisory_forward_passed'])
            self.assertFalse(report['passed']); self.assertFalse(report['trainer_consumable']); self.assertFalse(report['formal_authorized'])
            self.assertEqual(report['original_strict']['sha256'], producer.digest(output/'strict'/'COMPLETE.json'))
            from tools.structured_world import evaluate_execution_acceptance as reviewer
            payload = {'forward_exact':report['forward_exact'],
                       'forward_comparison':{'path':str(output/'FORWARD_COMPARISON.json'),
                                             'sha256':producer.digest(output/'FORWARD_COMPARISON.json')}}
            case = {'forward_module_inventory':report['forward_module_inventory'],
                    'science':{'physical_conversion_sha256':report['physical_q_final']['conversion_source_sha256']}}
            self.assertEqual(reviewer.forward_checks(self.policy, 'engineering_timevec', payload, output, case), [])
            corrupted = copy.deepcopy(report)
            corrupted['modules']['rank0/Bt']['A_B']['relative_L2'] = 1.01e-4
            producer.write_new(output/'rank_failure.json',corrupted)
            payload['forward_comparison'] = {'path':str(output/'rank_failure.json'),'sha256':producer.digest(output/'rank_failure.json')}
            self.assertTrue(reviewer.forward_checks(self.policy, 'engineering_timevec', payload, output, case))
            self.assertEqual(report['per_rank'][0]['roles']['A']['query_and_count_metadata']['optimizer_batch_counts'], {'ego_scenes':32.})
            self.assertIs(legacy.VLAStructuredFGTR, NativeCPUModel)
            with self.assertRaises(FileExistsError): producer.write_new(output/'FORWARD_COMPARISON.json', {})
            with self.assertRaisesRegex(ValueError, 'rank coverage'):
                producer.combine(report['per_rank'][:-1], self.bounds, report['original_strict'])

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists(): raise FileExistsError(args.output)
    sources = {str(ROOT/name): producer.digest(ROOT/name) for name in ('tools/structured_world/validate_execution_numerics.py',
                'tools/structured_world/test_validate_execution_numerics.py', 'tools/structured_world/validate_engineering_execution.py')}
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericsTests))
    stable = all(producer.digest(path) == sha for path,sha in sources.items())
    report = {'schema':'execution_numerics_independent_CPU_controls_v1','passed':result.wasSuccessful() and stable,
              'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
              'registration_sha256':producer.REGISTRATION_SHA,'source_sha256':sources,'sources_unchanged':stable,
              'CUDA_initialized':torch.cuda.is_initialized(),'limits':['Synthetic CPU controls, not trained GPU acceptance','Original strict failure and guard remain valid']}
    if args.output: producer.write_new(args.output,report)
    print(json.dumps({'passed':report['passed'],'tests_run':result.testsRun,'receipt':str(args.output) if args.output else None}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
