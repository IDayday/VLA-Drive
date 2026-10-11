"""Owned synthetic CPU controls; no real GPU/model training probe."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
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
import validate_p0_fullgrad_numerics as full
import torch
from torch import nn
torch.set_num_threads(1)

MODEL_ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011')
FORWARD_ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-p0-forward-v1-frozen-20261011')
FORWARD_BINDING = Path('/mnt/project/structured-world-fgtr-execution-artifacts/20261011/p0_forward_v1_prospective/SOURCE_BINDING.json')
p0 = full.upstream(FORWARD_ROOT)
shared = p0.shared_collector(MODEL_ROOT)
cmp = full.comparator(MODEL_ROOT)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 1, bias=False)
        self.head = nn.Linear(2, 1, bias=False)
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)


GROUPS = [{'names': ['encoder.weight'], 'elements': 3}, {'names': ['head.weight'], 'elements': 2}]


def capture(values, none=False, chunk=2):
    model = Tiny()
    banks = full.FullGradientBanks(cmp, GROUPS, chunk)
    start = torch.get_rng_state().clone()
    for role, vector in zip(('A', 'AA', 'B'), values):
        model.encoder.weight.grad = torch.tensor(vector[:3]).reshape(1, 3)
        model.head.weight.grad = None if none else torch.tensor(vector[3:]).reshape(1, 2)
        identities = {n: (id(p), id(p.grad), p.detach().clone(), None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
        banks.capture(role, model)
        for name, p in model.named_parameters():
            pid, gid, before, gradient = identities[name]
            assert id(p) == pid and id(p.grad) == gid and torch.equal(before, p)
            assert gradient is None or torch.equal(gradient, p.grad) or torch.isnan(gradient).any()
    assert torch.equal(start, torch.get_rng_state())
    return banks


class FullGradientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_actual_three_pairs_AA_B_not_triangle_or_A_pair_copy(self):
        vectors = [[1., 2., 3., 4., 5.], [1., 2., 4., 4., 5.], [1., 2., 2., 4., 5.]]
        bank = capture(vectors); stats = bank.report()
        report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
        self.assertTrue(report['full_parameter_rank_coverage']); self.assertTrue(report['AA_B_directly_measured'])
        self.assertEqual(report['comparisons']['A_A']['gradient']['all']['difference_norm'], 1.)
        self.assertEqual(report['comparisons']['A_B']['gradient']['all']['difference_norm'], 1.)
        self.assertEqual(report['comparisons']['AA_B']['gradient']['all']['difference_norm'], 2.)
        for pair, (left, right) in full.PAIRS.items():
            a, b = (torch.tensor(vectors[('A', 'AA', 'B').index(r)], dtype=torch.float64) for r in (left, right))
            row = report['comparisons'][pair]['gradient']['all']
            self.assertAlmostEqual(row['dot'], float((a*b).sum()))
            self.assertAlmostEqual(row['reference_norm'], float(a.norm()))
            self.assertAlmostEqual(row['candidate_norm'], float(b.norm()))
            self.assertAlmostEqual(row['difference_norm'], float((a-b).norm()))
            self.assertEqual(row['elements'], 5)
            self.assertEqual(sum(v['elements'] for v in report['comparisons'][pair]['gradient']['by_component'].values()), 5)
            self.assertEqual(sum(v['elements'] for v in report['comparisons'][pair]['gradient']['by_optimizer_group'].values()), 5)
        self.assertIsNone({row[0]: row[-1] for row in stats['inventory']['A']}['frozen'])
        self.assertFalse(bank.banks)
        self.assertFalse(report['passed']); self.assertFalse(report['formal_authorized'])

    def test_chunk_FP64_matches_unpartitioned_and_all_inventory(self):
        values = [[1e8, 1e-8, -3., 0., 9.], [1e8, -1e-8, -3.1, 0., 8.], [1e8, 0., -3.2, 0., 7.]]
        one, many = capture(values, chunk=1).report(), capture(values, chunk=100).report()
        self.assertEqual(one['gradient_inventory_sha256'], many['gradient_inventory_sha256'])
        for name in one['tensors']:
            for pair in full.PAIRS:
                for key, value in one['tensors'][name]['pairs'][pair].items():
                    self.assertAlmostEqual(value, many['tensors'][name]['pairs'][pair][key], delta=max(abs(value)*1e-14, 1e-15))
        self.assertEqual(len(one['inventory']['A']), 3)
        self.assertEqual(one['trainable_parameter_count'], 2); self.assertEqual(one['trainable_elements'], 5)

    def test_None_is_not_fabricated_zero_or_full_coverage(self):
        stats = capture([[0.]*5]*3, none=True).report()
        report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
        self.assertFalse(report['full_parameter_rank_coverage'])
        self.assertNotIn('head.weight', stats['tensors'])
        self.assertIsNone(stats['inventory']['A'][2][-1])
        self.assertEqual(report['comparisons']['AA_B']['gradient']['all']['elements'], 3)

    def test_zero_exact_nonzero_and_signed_zero(self):
        for values, zero in (([[0.]*5]*3, True), ([[0.]*5, [0.]*5, [1e-8, 0., 0., 0., 0.]], False)):
            stats = capture(values).report()
            report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
            row = report['comparisons']['AA_B']['gradient']['all']
            self.assertEqual(row['zero_reference_norm'], True)
            self.assertEqual(row['relative_L2'], 0. if zero else None)
            self.assertIsNone(row['cosine'])
        stats = capture([[0.]*5, [0.]*5, [-0., 0., 0., 0., 0.]]).report()
        report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
        self.assertEqual(report['comparisons']['AA_B']['gradient']['all']['difference_norm'], 0.)
        self.assertFalse(report['comparisons']['AA_B']['gradient']['all']['bitwise_passed'])

    def test_NaN_Inf_never_complete_finite_norms(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            stats = capture([[1.]*5, [1.]*5, [invalid, 1., 1., 1., 1.]]).report()
            report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
            row = report['comparisons']['AA_B']['gradient']['all']
            self.assertEqual(row['nonfinite_elements'], 1); self.assertIsNone(row['reference_norm'])
            self.assertFalse(row['fixed_numeric_passed']); json.dumps(report, allow_nan=False)

    def test_missing_duplicate_rank_wrong_source_inventory_and_pair_reject(self):
        stats = capture([[1.]*5]*3).report(); record = {'rank': 0, 'fullgrad': stats}
        for records, sha in (([], stats['gradient_inventory_sha256']), ([record, record], stats['gradient_inventory_sha256']),
                             ([record], 'f'*64)):
            with self.assertRaises(ValueError): full.combine_gradients(records, sha, 1)
        invalid = copy.deepcopy(record); del invalid['fullgrad']['tensors']['encoder.weight']['pairs']['AA_B']
        with self.assertRaisesRegex(ValueError, 'AA_B'):
            full.combine_gradients([invalid], stats['gradient_inventory_sha256'], 1)

    def test_changed_None_or_dtype_roles_order_and_duplicate_capture_reject(self):
        model = Tiny(); bank = full.FullGradientBanks(cmp, GROUPS, 2)
        with self.assertRaises(ValueError): bank.capture('AA', model)
        model.encoder.weight.grad = torch.ones_like(model.encoder.weight); model.head.weight.grad = torch.ones_like(model.head.weight)
        bank.capture('A', model)
        with self.assertRaises(ValueError): bank.capture('A', model)
        model.head.weight.grad = None
        with self.assertRaisesRegex(ValueError, 'None-gradient'): bank.capture('AA', model)
        with self.assertRaises(ValueError): bank.report()
        with self.assertRaisesRegex(ValueError, 'FP32 parameter'):
            full.FullGradientBanks(cmp, GROUPS).capture('A', Tiny().double())

    def test_optimizer_group_names_counts_and_no_overwrite(self):
        model = Tiny()
        for groups in ([{'names': ['missing'], 'elements': 1}], [{'names': ['encoder.weight', 'head.weight'], 'elements': 4}]):
            with self.assertRaises(ValueError): full.FullGradientBanks(cmp, groups).capture('A', model)
        with self.assertRaises(ValueError): full.FullGradientBanks(cmp, GROUPS+GROUPS)
        with self.assertRaises(ValueError): full.FullGradientBanks(cmp, GROUPS, 0)
        path = self.root/'owned.json'; full.write_new(path, {'owned': True})
        with self.assertRaises(FileExistsError): full.write_new(path, {})

    def test_eight_rank_pool_direct_norm_and_named_scope_coverage(self):
        records = []
        for rank in range(8):
            stats = capture([[float(rank+1)]*5, [float(rank+1)+.5]*5, [float(rank+1)-.5]*5]).report()
            records.append({'rank': rank, 'fullgrad': stats})
        report = full.combine_gradients(records, stats['gradient_inventory_sha256'])
        self.assertEqual(report['comparisons']['AA_B']['gradient']['all']['elements'], 40)
        self.assertAlmostEqual(report['comparisons']['AA_B']['gradient']['all']['difference_norm'], 40**.5)
        self.assertTrue(report['full_parameter_rank_coverage'])

    def test_fresh_prospective_binding_preserves_44c_native819_and_sources(self):
        old_hash = full.digest(FORWARD_BINDING)
        original, _ = full.checked_forward(p0, MODEL_ROOT, FORWARD_BINDING, old_hash)
        native_hash = full.digest(original['original_acceptance_bindings']['path'])
        args = SimpleNamespace(model_source_root=MODEL_ROOT, forward_source_root=FORWARD_ROOT,
            forward_source_binding=FORWARD_BINDING, forward_source_binding_sha256=old_hash,
            output=self.root/'OWNED_MOCK_PROSPECTIVE.json', chunk_elements=2, memory_reserve_bytes=0)
        real_frozen, real_git = p0.frozen_root, full.subprocess.check_output
        def owned_frozen(root, expected=None):
            if Path(root) == full.ROOT:
                full.require(expected in (None, 'a'*40), 'Immutable root commit mismatch'); return 'a'*40
            return real_frozen(root, expected)
        def owned_git(cmd, **kw):
            if cmd[:2] == ['git', 'show'] and cmd[2] == 'a'*40+':'+full.RELATIVE: return Path(full.__file__).read_bytes()
            return real_git(cmd, **kw)
        with patch.object(full, 'upstream', return_value=p0), patch.object(p0, 'frozen_root', side_effect=owned_frozen), patch.object(full.subprocess, 'check_output', side_effect=owned_git):
            ref = full.register(args)
            args.source_binding, args.source_binding_sha256 = args.output, ref['sha256']
            binding, _ = full.check_binding(args, p0)
            self.assertFalse(binding['native_binding_or_prior_reports_modified'])
            self.assertEqual(binding['producer']['git_sha'], 'a'*40)
            self.assertEqual(binding['forward_producer']['git_sha'], full.FORWARD_COMMIT)
            for mutate in (lambda v: v['contract'].update(original_gradient_rtol=1.),
                           lambda v: v['producer']['files_sha256'].update({full.RELATIVE: '0'*64}),
                           lambda v: v['forward_source_binding'].update(sha256='1'*64),
                           lambda v: v['cases']['G1/mid']['origin'].update(completed=1),
                           lambda v: v.update(frozen_UTC=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())):
                wrong = copy.deepcopy(binding); mutate(wrong)
                target = self.root/(str(len(list(self.root.iterdir())))+'_BAD.json'); full.write_new(target, wrong)
                args.source_binding, args.source_binding_sha256 = target, full.digest(target)
                with self.assertRaises(ValueError): full.check_binding(args, p0)
        self.assertEqual(full.digest(FORWARD_BINDING), old_hash)
        self.assertEqual(full.digest(original['original_acceptance_bindings']['path']), native_hash)

    def test_actual_original_sources_thresholds_unchanged_and_memory_accounting(self):
        self.assertEqual(full.digest(MODEL_ROOT/'tools/structured_world/validate_engineering_execution.py'), p0.LEGACY_SHA)
        self.assertEqual((cmp.RTOL, cmp.ATOL), (1e-4, 1e-6))
        self.assertEqual(p0.PRESET['atol'], 0.); self.assertEqual(p0.PRESET['rtol'], 0.)
        estimate = full.memory_estimate(2630000000)
        self.assertEqual(estimate['new_three_banks_bytes_all_ranks'], 252480000000)
        self.assertEqual(estimate['legacy_A_bank_bytes_all_ranks'], 84160000000)
        self.assertGreater(estimate['minimum_host_available_bytes_before_A'], 336640000000)
        with_legacy_scratch = full.memory_estimate(2630000000, largest_parameter_elements=100000000)
        self.assertEqual(with_legacy_scratch['minimum_host_available_bytes_before_A']-estimate['minimum_host_available_bytes_before_A'], 32000000000)
        self.assertGreater(full.host_available_bytes(), 0)

    def test_existing_V1_consumer_accepts_complete_shape_and_reads_AA_B(self):
        path = MODEL_ROOT/'tools/structured_world/evaluate_execution_acceptance.py'
        gate = full.source_module(path, full.digest(path), '_owned_readonly_original_V1')
        policy = p0.policy_and_preset(shared, '/mnt/project/VLA-Drive-structured-world-fgtr-perf-v3-frozen-20261011')
        stats = capture([[1.]*5]*3).report()
        report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
        failures, _ = gate.numerical_checks(policy, 'E2', report, native=False)
        self.assertEqual(failures, [])
        values = [[1.]*5, [1.]*5, [-1.]*5]
        stats = capture(values).report()
        report = full.combine_gradients([{'rank': 0, 'fullgrad': stats}], stats['gradient_inventory_sha256'], 1)
        failures, _ = gate.numerical_checks(policy, 'E2', report, native=False)
        self.assertTrue(any('/AA_B/' in row for row in failures))
        del report['comparisons']['AA_B']
        with self.assertRaisesRegex(ValueError, 'All AA/AB/AA_B'):
            gate.numerical_checks(policy, 'E2', report, native=False)

    def test_wrong_upstream_source_rejected_before_import(self):
        path = self.root/'tools/structured_world/validate_p0_execution_numerics.py'
        path.parent.mkdir(parents=True); path.write_text('OWNED incorrect source, must not execute')
        with self.assertRaisesRegex(ValueError, 'Pinned source hash'):
            full.upstream(self.root)

    def run_observer_fixture(self, incomplete=False, low_memory=False):
        output = self.root/'owned_probe'; model_holder, returned = {}, []
        start = torch.get_rng_state().clone()
        framework = SimpleNamespace(sample_queries=lambda q0: {'poses': q0.clone(), 'time_index': torch.zeros(4, 8, dtype=torch.long)})
        class Native(nn.Module):
            def __init__(self):
                super().__init__(); self.encoder = nn.Linear(3, 1, bias=False); self.head = nn.Linear(2, 1, bias=False)
                with torch.no_grad(): self.encoder.weight.zero_(); self.head.weight.zero_()
                self.calls = 0; model_holder['model'] = self
                self.register_buffer('running_mean', torch.zeros(1)); self.register_buffer('num_batches_tracked', torch.zeros((), dtype=torch.long))
            def forward(self):
                index = self.calls; self.calls += 1
                q0 = torch.zeros(4, 8, 3); framework.sample_queries(q0)
                loss = self.encoder.weight.sum()*(1., 2., 0.)[index]+self.head.weight.sum()
                value = {'loss': loss, 'losses': {'original_FM': loss},
                    'diagnostics': {'q0': q0, 'q_final': q0.clone(), 'H_A': torch.zeros(1), 'B0': torch.zeros(1), 'Bt': torch.zeros(1)},
                    'metrics': {'raw': {'depth': torch.zeros(())}, 'effective_counts': {'task': 4.}, 'query_strata': [{'count': 4}],
                    'refine_effective_weight': 1., 'proposal_out_of_range': torch.zeros(4, 8, dtype=torch.bool),
                    'activation_gradient_samples': {n: torch.ones(8) for n in ('H_A', 'B0', 'Bt')}}}
                returned.append(value); return value
        class Dist:
            def all_gather_object(self, rows, row):
                for rank in range(len(rows)):
                    rows[rank] = copy.deepcopy(row); rows[rank]['rank'] = rank
                    rows[rank]['tokens'] = [f'OWNED_{rank}_{i}' for i in range(4)]
        capture_calls = []
        def native_rng(*a):
            value = {'CPU': torch.get_rng_state().clone()}; capture_calls.append(value); return value
        native_counts = lambda *a: {'ego_scenes': 32., 'horizon_counts': [32.]*8, 'dino_scalars': 1024.}
        legacy = SimpleNamespace(VLAStructuredFGTR=Native, capture_rng=native_rng, dist=Dist(), optimizer_batch_counts=native_counts)
        args = SimpleNamespace(run=self.root, tag='OWNED', cache=self.root, dino_root=self.root, output=output,
            microbatch=4, model_state_file=None, model_source_root=MODEL_ROOT, source_binding=self.root/'OWNED_binding', source_binding_sha256='a'*64)
        expected = [['encoder.weight', [1, 3], 'torch.float32'], ['head.weight', [1, 2], 'torch.float32']]
        binding = {'model': {'git_sha': p0.MODEL_SHA}, 'original_acceptance_bindings': {'path': 'OWNED', 'sha256': 'b'*64},
            'forward_source_binding': {'path': str(FORWARD_BINDING), 'sha256': full.FORWARD_BINDING_SHA},
            'verification_relationship': 'OWNED synthetic future supplement; not historical evidence', 'chunk_elements': 2,
            'memory_reserve_bytes': 0, 'cases': {'G1/mid': {'gradient_inventory_sha256': full.metadata_hash(expected)}}}
        strict = {'passed': False, 'gradient_tolerance': {'rtol': 1e-4, 'atol': 1e-6}, 'fixture': 'OWNED_CPU_ONLY'}
        def native_main():
            model = legacy.VLAStructuredFGTR()
            for role in range(3):
                torch.set_rng_state(start); model.zero_grad(set_to_none=True); legacy.optimizer_batch_counts(None)
                result = model(); self.assertIs(result, returned[-1]); result['loss'].backward()
                rng = legacy.capture_rng(None); self.assertIs(rng, capture_calls[-1])
                self.assertTrue(torch.equal(model.encoder.weight.grad, torch.full_like(model.encoder.weight, (1., 2., 0.)[role])))
                if incomplete and role == 0: raise RuntimeError('owned incomplete failure')
            legacy.dist.all_gather_object([None]*8, {'rank': 0, 'tokens': [], 'checks': []})
            p0.write_new(output/'strict/COMPLETE.json', strict)
            raise AssertionError('original strict failure retained')
        legacy.main = native_main
        old_collector, old_combine, old_queries = p0.collector_class, p0.combine, framework.sample_queries
        rechecks = []
        pattern = 'Insufficient actual host RAM' if low_memory else 'owned incomplete failure' if incomplete else 'original strict failure retained'
        error = ValueError if low_memory else RuntimeError if incomplete else AssertionError
        with patch.object(full, 'host_available_bytes', return_value=0 if low_memory else 10**15):
            with self.assertRaisesRegex(error, pattern):
                full.run_observed(args, p0, shared, cmp, legacy, framework, binding, lambda: rechecks.append('B'), GROUPS, 'G1/mid')
        self.assertIs(p0.collector_class, old_collector); self.assertIs(p0.combine, old_combine)
        self.assertIs(legacy.VLAStructuredFGTR, Native); self.assertIs(legacy.capture_rng, native_rng)
        self.assertIs(legacy.optimizer_batch_counts, native_counts); self.assertIs(framework.sample_queries, old_queries)
        if incomplete or low_memory:
            self.assertFalse((output/'P0_FULL_GRADIENT_COMPARISON.json').exists()); return
        self.assertEqual(rechecks, ['B']); self.assertEqual(p0.load(output/'strict/COMPLETE.json'), strict)
        report = full.load(output/'P0_FULL_GRADIENT_COMPARISON.json')
        forward = p0.load(output/'P0_FORWARD_COMPARISON.json')
        self.assertTrue(forward['advisory_forward_passed']); self.assertFalse(forward['full_parameter_gradient_statistics_captured'])
        self.assertTrue(report['full_parameter_gradient_statistics_captured']); self.assertTrue(report['AA_B_directly_measured'])
        self.assertFalse(report['original_strict']['passed']); self.assertFalse(report['passed']); self.assertFalse(report['trainer_consumable'])
        self.assertEqual(report['comparisons']['AA_B']['gradient']['all']['elements'], 40)
        self.assertAlmostEqual(report['comparisons']['AA_B']['gradient']['all']['difference_norm'], (8*3*4)**.5)
        self.assertNotIn('gradient_bank', json.dumps(report))
        self.assertTrue(torch.equal(start, torch.get_rng_state()))

    def test_original_strict_failure_full_capture_return_RNG_grads_preserved(self):
        self.run_observer_fixture()

    def test_incomplete_and_exception_restore_all_wrappers_no_completion(self):
        self.run_observer_fixture(incomplete=True)

    def test_actual_low_RAM_stops_observation_not_fabricated_full_capture(self):
        self.run_observer_fixture(low_memory=True)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    paths = [Path(full.__file__), Path(__file__), FORWARD_ROOT/'tools/structured_world/validate_p0_execution_numerics.py',
             MODEL_ROOT/'tools/structured_world/validate_engineering_execution.py', MODEL_ROOT/'tools/structured_world/compare_execution_checkpoints.py',
             MODEL_ROOT/'tools/structured_world/validate_execution_numerics.py', FORWARD_BINDING]
    before = {str(p): full.digest(p) for p in paths}; begin = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FullGradientTests))
    after = {str(p): full.digest(p) for p in paths}
    full.write_new(args.output, {'schema': 'p0_fullgrad_supplement_CPU_controls_v1', 'tests_run': result.testsRun,
        'passed': result.wasSuccessful(), 'failures': [(str(t), e) for t, e in result.failures], 'errors': [(str(t), e) for t, e in result.errors],
        'source_sha256': after, 'sources_unchanged': before == after, 'CUDA_initialized': torch.cuda.is_initialized(),
        'wall_seconds': time.monotonic()-begin, 'scope': 'OWNED tiny CPU models and synthetic rank records only; producer commit/RAM mocked in indicated tests; no actual GPU proof'})
    raise SystemExit(0 if result.wasSuccessful() and before == after else 1)


if __name__ == '__main__': main()
