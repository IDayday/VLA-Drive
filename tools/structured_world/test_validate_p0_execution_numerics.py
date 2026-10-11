"""Owned CPU controls for external P0 observation; no GPU/model training probe."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_p0_execution_numerics as producer
shared = producer.shared_collector('/mnt/project/VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011')
import torch
from torch import nn
torch.set_num_threads(1)

MODEL_ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011')
ORIGINAL = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-v3-frozen-20261011')
ACCEPTANCE = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/training_efficiency_20261011T0546/P0_V1_PROSPECTIVE_BINDINGS/BINDINGS.json')


class P0Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, change=None):
        output = self.root/'owned_CPU_probe'
        args = SimpleNamespace(run=self.root, tag='OWNED_SYNTHETIC', cache=self.root, dino_root=self.root,
            output=output, microbatch=4, model_state_file=None, model_source_root=MODEL_ROOT,
            source_binding=self.root/'owned_binding.json', source_binding_sha256='a'*64)
        queries = []
        def native_queries(q0):
            value = {'poses': q0[:, :4].clone(), 'time_index': torch.arange(4).repeat(4, 1)}
            queries.append(value)
            if change:
                change('query', len(queries)-1, value)
            return value
        framework = SimpleNamespace(sample_queries=native_queries)
        class NativeModel(nn.Module):
            def __init__(self):
                super().__init__(); self.p = nn.Parameter(torch.ones(1)); self.calls = 0
                self.register_buffer('running_mean', torch.zeros(1)); self.register_buffer('num_batches_tracked', torch.zeros((), dtype=torch.long))
            def forward(self):
                call = self.calls; self.calls += 1
                q0 = torch.zeros(4, 8, 3); framework.sample_queries(q0)
                losses = {name: self.p.sum()*(n+1) for n, name in enumerate(
                    ('original_FM', 'current_DINO', 'current_geometry', 'local_relations', 'future_semantics', 'refine'))}
                value = {'loss': sum(losses.values()), 'losses': losses,
                    'diagnostics': {'q0': q0, 'q_final': q0.clone(), 'H_A': torch.zeros(1), 'B0': torch.ones(1), 'Bt': torch.ones(1)},
                    'metrics': {'raw': {'depth': self.p.detach().sum(), 'road': torch.zeros(())},
                    'effective_counts': {'task': 4.}, 'query_strata': [{'count': 4}], 'refine_effective_weight': 1.,
                    'proposal_out_of_range': torch.zeros(4, 8, dtype=torch.bool),
                    'activation_gradient_samples': {name: torch.ones(8) for name in ('H_A', 'B0', 'Bt')}}}
                if change:
                    change('forward', call, value)
                return value
        class NativeDist:
            def all_gather_object(self, rows, value):
                for rank in range(len(rows)):
                    rows[rank] = copy.deepcopy(value); rows[rank]['rank'] = rank
                    rows[rank]['tokens'] = [f'OWNED_{rank}_{s}' for s in range(4)]
        counts, rechecks = [], []
        def native_counts(*a):
            value = {'ego_scenes': 32., 'dino_scalars': 4096., 'horizon_counts': [32., 32., 32.]}
            if change:
                change('counts', len(counts), value)
            counts.append(value); return value
        legacy = SimpleNamespace(VLAStructuredFGTR=NativeModel, capture_rng=lambda *a: {'CPU': torch.get_rng_state().clone()},
                                 dist=NativeDist(), optimizer_batch_counts=native_counts)
        strict = {'passed': False, 'strict_gradient_passed': False, 'gradient_tolerance': {'rtol': 1e-4, 'atol': 1e-6},
                  'forward_loss_tolerance': {'rtol': 1e-6, 'atol': 1e-7}, 'fixture': 'OWNED_SYNTHETIC_CPU_ONLY'}
        start = torch.get_rng_state().clone()
        returned_losses = []
        def native_main():
            model = legacy.VLAStructuredFGTR()
            for _ in range(3):
                model.zero_grad(set_to_none=True); torch.set_rng_state(start)
                legacy.optimizer_batch_counts(None)
                result = model(); returned_losses.append(result['loss'].item())
                result['loss'].backward(); legacy.capture_rng(None)
            legacy.dist.all_gather_object([None]*8, {'rank': 0, 'tokens': [], 'checks': []})
            producer.write_new(output/'strict/COMPLETE.json', strict)
            raise AssertionError('Original legacy strict failure retained')
        legacy.main = native_main
        binding = {'model': {'git_sha': producer.MODEL_SHA}, 'original_acceptance_bindings': {'path': 'OWNED', 'sha256': 'b'*64},
                   'verification_relationship': 'OWNED optional future verification, not historical native binding'}
        with self.assertRaisesRegex(AssertionError, 'Original legacy strict failure retained'):
            producer.run_observed(args, shared, legacy, framework, binding, lambda: rechecks.append('B'))
        self.assertEqual(producer.load(output/'strict/COMPLETE.json'), strict)
        self.assertIs(legacy.VLAStructuredFGTR, NativeModel); self.assertIs(legacy.optimizer_batch_counts, native_counts)
        self.assertIs(framework.sample_queries, native_queries); self.assertEqual(rechecks, ['B'])
        self.assertTrue(torch.equal(start, torch.get_rng_state()))
        return producer.load(output/'P0_FORWARD_COMPARISON.json'), returned_losses, queries, counts

    def test_completed_strict_failure_preserved_full_loss_three_pairs_and_no_authorization(self):
        report, losses, queries, counts = self.run_fixture()
        self.assertTrue(report['advisory_forward_passed']); self.assertFalse(report['original_strict']['passed'])
        self.assertEqual(report['scene_count'], 32); self.assertEqual(report['ranks'], 8)
        self.assertEqual(report['preset'], producer.PRESET); self.assertEqual(set(report['modules']['loss']), set(shared.PAIRS))
        self.assertEqual(losses, [21., 21., 21.]); self.assertEqual(len(queries), 3)
        for role in ('A', 'AA', 'B'):
            facts = report['per_rank'][0]['roles'][role]
            self.assertEqual(facts['scalar_values']['loss'], 21.)
            self.assertEqual(facts['scalar_values']['task_loss/refine'], 6.)
            self.assertEqual(facts['scalar_values']['raw_loss/depth'], 1.)
            self.assertEqual(facts['query_and_count_metadata']['optimizer_batch_counts'], counts[0])
        for name in ('passed', 'formal_authorized', 'trainer_consumable', 'full_parameter_gradient_statistics_captured'):
            self.assertFalse(report[name])
        self.assertEqual(report['original_strict']['sha256'], producer.digest(self.root/'owned_CPU_probe/strict/COMPLETE.json'))
        text = (self.root/'owned_CPU_probe/P0_FORWARD_COMPARISON.json').read_text()
        self.assertNotIn('raw_images', text); self.assertIn('representative_gradients_are_diagnostic_only', text)

    def test_tiny_loss_difference_cannot_use_old_tolerance(self):
        def change(kind, call, value):
            if kind == 'forward' and call == 2:
                value['loss'] = value['loss']+1e-5
        report, *_ = self.run_fixture(change)
        self.assertFalse(report['advisory_forward_passed']); self.assertFalse(report['forward_exact']['loss'])
        self.assertTrue(report['forward_exact']['Bt'])
        self.assertFalse(report['modules']['loss']['AA_B']['bitwise_equal'])

    def test_signed_zero_bitwise_required_even_when_allclose_zero_passes(self):
        def change(kind, call, value):
            if kind == 'forward' and call == 2:
                value['diagnostics']['H_A'] = torch.tensor([-0.])
        report, *_ = self.run_fixture(change)
        self.assertTrue(report['modules']['H_A']['A_B']['torch_allclose_passed'])
        self.assertFalse(report['modules']['H_A']['A_B']['bitwise_equal'])
        self.assertFalse(report['advisory_forward_passed'])

    def test_nonfinite_and_query_time_and_global_denominator_drift_reject(self):
        changes = ('nan', 'query', 'counts')
        for mode in changes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                self.root = Path(tmp)
                def change(kind, call, value):
                    if call != 2:
                        return
                    if mode == 'nan' and kind == 'forward':
                        value['metrics']['raw']['depth'] = torch.tensor(float('nan'))
                    if mode == 'query' and kind == 'query':
                        value['time_index'] = value['time_index'].flip(1)
                    if mode == 'counts' and kind == 'counts':
                        value['horizon_counts'][1] = 31.
                report, *_ = self.run_fixture(change)
                self.assertFalse(report['advisory_forward_passed'])
                if mode != 'nan':
                    self.assertFalse(report['forward_exact_by_pair']['AA_B']['query_and_counts'])

    def test_missing_rank_and_no_overwrite(self):
        report, *_ = self.run_fixture()
        with self.assertRaisesRegex(ValueError, 'rank coverage'):
            producer.combine(shared, report['per_rank'][:-1], report['original_strict'])
        with self.assertRaises(FileExistsError):
            producer.write_new(self.root/'owned_CPU_probe/P0_FORWARD_COMPARISON.json', {})

    def test_separate_roots_prospective_binding_and_original_bindings_unchanged(self):
        original_hash = producer.digest(ACCEPTANCE)
        path = self.root/'OWNED_SYNTHETIC_SOURCE_BINDING.json'
        old_call = producer.subprocess.check_output
        def git_show(cmd, **kw):
            if cmd[:2] == ['git', 'show'] and cmd[2] == 'a'*40+':'+producer.RELATIVE:
                return Path(producer.__file__).read_bytes()
            return old_call(cmd, **kw)
        old_frozen = producer.frozen_root
        def owned_frozen(root, expected=None):
            if Path(root) == producer.ROOT:
                producer.require(expected in (None, 'a'*40), 'Immutable root commit mismatch')
                return 'a'*40  # Explicitly mocked OWNED producer commit, never an executed fact.
            return old_frozen(root, expected)
        with patch.object(producer, 'frozen_root', side_effect=owned_frozen), patch.object(producer.subprocess, 'check_output', side_effect=git_show):
            ref = producer.register(MODEL_ROOT, ACCEPTANCE, original_hash, ORIGINAL, path)
            b = producer.check_binding(path, ref['sha256'], MODEL_ROOT, shared)
            self.assertEqual(b['model']['git_sha'], producer.MODEL_SHA)
            self.assertEqual(b['producer']['git_sha'], 'a'*40)
            self.assertFalse(b['native_binding_or_prior_reports_modified'])
            self.assertEqual(b['preset']['atol'], 0.)
            for mutate in (lambda v: v['preset'].update(atol=1e-6),
                           lambda v: v['model']['files_sha256'].update({'tools/structured_world/train_vla.py': '0'*64}),
                           lambda v: v.update(frozen_UTC=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()),
                           lambda v: v['producer'].update(git_sha='c'*40),
                           lambda v: v['protected_source_changes'][0].update(reason='unreviewed'),
                           lambda v: v['cases']['G1/mid']['origin'].update(completed=1)):
                changed = copy.deepcopy(b); mutate(changed)
                p = self.root/(str(len(list(self.root.iterdir())))+'_OWNED_BAD.json'); producer.write_new(p, changed)
                with self.assertRaises(ValueError):
                    producer.check_binding(p, producer.digest(p), MODEL_ROOT, shared)
        self.assertEqual(original_hash, producer.digest(ACCEPTANCE))

    def test_actual_P0_registration_no_timevec_preset_reinterpretation(self):
        policy = producer.policy_and_preset(shared, ORIGINAL)
        self.assertIsNone(policy['tiers']['E2']['forward_by_mode']['engineering_fast']['module_tolerances'])
        self.assertEqual(producer.PRESET['atol'], 0.)
        self.assertEqual(producer.PRESET['rtol'], 0.)
        self.assertEqual(producer.PRESET['relative_L2_max'], 0.)
        self.assertEqual(producer.digest(MODEL_ROOT/'tools/structured_world/validate_execution_numerics.py'), producer.COLLECTOR_SHA)

    def test_origin_preflight_config_science_ancestry_and_safe_tag(self):
        config = {'OWNED_SYNTHETIC_CONFIG': True}
        science = {'config_sha256': __import__('hashlib').sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'current_and_GT_identity': {'identity': 'owned-current'}, 'DINO_identity': 'owned-dino',
            'training_seed': 42, 'global_batch': 32, 'FM_repeat': 8, 'schedule_horizon': 100000, 'warmup': 5000}
        identity = {'identity': 'owned-origin', 'config': config, 'current_and_GT_identity': science['current_and_GT_identity'],
            'DINO_identity': 'owned-dino', 'training_seed': 42, 'effective_global_batch': 32, 'FM_repeat': 8,
            'schedule_horizon': 100000, 'schedule_warmup': 5000, 'world_size': 8, 'microbatch': 4, 'dataset': 'navsim'}
        tag = 'OWNED_TAG'; complete = self.root/'checkpoints'/tag/'COMPLETE.json'
        producer.write_new(self.root/'identity.json', identity)
        producer.write_new(complete, {'identity': 'owned-origin', 'completed': 25000})
        binding = {'cases': {'G1/mid': {'origin': {'identity': 'owned-origin', 'completed': 25000,
                   'COMPLETE_sha256': producer.digest(complete)}, 'science': science}}}
        self.assertEqual(producer.selected_origin(binding, self.root, tag), 'G1/mid')
        for mutation in (lambda x: x.update(FM_repeat=4), lambda x: x['config'].update(changed=True), lambda x: x.update(identity='foreign')):
            changed = copy.deepcopy(identity); mutation(changed)
            (self.root/'identity.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                producer.selected_origin(binding, self.root, tag)
        with self.assertRaisesRegex(ValueError, 'safe tag'):
            producer.selected_origin(binding, self.root, '../latest')

    def test_real_uniform_and_local_query_and_count_observation_preserve_rng(self):
        from tools.structured_world.test_query_execution import labels, GRID
        from starVLA.model.modules.structured_world.queries import sample_queries
        from tools.foresight.student_state import optimizer_batch_counts
        for mode in ('uniform', 'local'):
            collector = producer.collector_class(shared)(producer.PRESET)
            target = labels(); target['ego'] = torch.zeros(3, 8, 4)
            generator = torch.Generator().manual_seed(77)
            expected_counts = optimizer_batch_counts(target, generator, torch.device('cpu'))
            actual_counts = shared.observe_counts(optimizer_batch_counts, collector)(target, generator, torch.device('cpu'))
            self.assertEqual(expected_counts, actual_counts); collector.begin()
            proposal = torch.zeros(3, 8, 3); start = generator.get_state().clone(); rng = torch.get_rng_state().clone()
            expected = sample_queries(proposal, target, mode=mode, count=32, grid=GRID, generator=generator)
            expected_rng = generator.get_state().clone(); generator.set_state(start)
            returned = []
            def native(*a, **kw):
                value = sample_queries(*a, **kw); returned.append(value); return value
            actual = shared.observe_queries(native, collector)(proposal, target, mode=mode, count=32, grid=GRID, generator=generator)
            self.assertIs(actual, returned[0]); self.assertTrue(shared.bitwise(expected, actual))
            self.assertTrue(torch.equal(generator.get_state(), expected_rng)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(collector.queries['A']['poses'], shared.typed_tensor_hash(actual['poses']))
            self.assertEqual(collector.queries['A']['time_index'], shared.typed_tensor_hash(actual['time_index']))

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    paths = [Path(producer.__file__), Path(__file__), MODEL_ROOT/'tools/structured_world/validate_execution_numerics.py',
             MODEL_ROOT/'tools/structured_world/validate_engineering_execution.py', ACCEPTANCE]
    before = {str(p): producer.digest(p) for p in paths}; start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(P0Tests))
    after = {str(p): producer.digest(p) for p in paths}
    producer.write_new(args.output, {'schema': 'P0_external_forward_CPU_controls_v1', 'passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'failures': [(str(t), e) for t, e in result.failures], 'errors': [(str(t), e) for t, e in result.errors],
        'source_sha256': after, 'sources_unchanged': before == after, 'CUDA_initialized': torch.cuda.is_initialized(),
        'wall_seconds': time.monotonic()-start, 'preset': producer.PRESET,
        'scope': 'OWNED synthetic CPU model/8-rank records only; no real GPU proof; producer commit mocked in binding test only'})
    raise SystemExit(0 if result.wasSuccessful() and before == after else 1)


if __name__ == '__main__':
    main()
