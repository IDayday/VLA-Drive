"""CPU synthetic native-format receipts, including cross-rank slices/padding.

No real model, GPU, distributed job or training task is launched. The fixtures
exercise observed native ZeRO-2 structure with independently calculable norms.
"""
import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'tools/structured_world/compare_execution_checkpoints.py'
spec = importlib.util.spec_from_file_location('compare_execution_checkpoints', SOURCE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
RESULTS = []


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def native_fixture(root, label, *, origin_binding=None, completed=2, epoch=0, offset=2):
    run = root/label
    folder = run/'checkpoints'/f'milestone_{completed:07d}'
    folder.mkdir(parents=True)
    identity = {key: None for key in comparison.SCIENCE_FIELDS}
    identity.update(identity=hashlib.sha256(label.encode()).hexdigest(), execution_mode='engineering_fast' if label == 'B' else 'io_preserving_v1',
                    config={'fixed': True}, dataset='navsim', world_size=2, training_seed=42,
                    schedule_horizon=100000, schedule_warmup=5000, effective_global_batch=32, FM_repeat=8,
                    microbatch=4, scenes_per_epoch=101592, updates_per_epoch=3175,
                    cache_identity='same-cache', current_and_GT_identity='same-current-GT',
                    shared_geometry_identity='same-geometry', training_source_sha='synthetic-source')
    if origin_binding:
        identity['engineering_continue_origin'] = origin_binding
    write_json(run/'identity.json', identity)
    groups = [dict(base_lr=1e-5, elements=10, names=['future_space.weight', 'refiner.weight']),
              dict(base_lr=1e-4, elements=6, names=['geometry.backbone.weight'])]
    write_json(run/'optimizer_groups.json', groups)
    shapes = [OrderedDict([('future_space.weight', torch.Size([2, 3])), ('refiner.weight', torch.Size([2, 2]))]),
              OrderedDict([('geometry.backbone.weight', torch.Size([3, 2]))])]
    delta = .01 if origin_binding else 0.
    module = OrderedDict([
        ('future_space.weight', torch.arange(6, dtype=torch.float32).reshape(2, 3)/10.+.2+delta),
        ('refiner.weight', torch.arange(4, dtype=torch.float32).reshape(2, 2)/10.+.4+delta),
        ('geometry.backbone.weight', torch.arange(6, dtype=torch.float32).reshape(3, 2)/10.+.6+delta),
        ('qwen.visual.frozen', torch.tensor([.2, .3])),
        ('qwen.alias', torch.tensor([.2, .3])),
        ('geometry.bn.running_mean', torch.tensor([.1, .2])+delta),
        ('geometry.bn.running_var', torch.tensor([.8, .9])+delta),
        ('geometry.bn.num_batches_tracked', torch.tensor(completed)),
        ('future_space.times_s', torch.tensor([.5, 1.]))])
    model = dict(module=module, buffer_names=list(module)[5:], optimizer=None,
                 param_shapes=shapes, frozen_param_shapes=OrderedDict([('qwen.visual.frozen', torch.Size([2]))]),
                 frozen_param_fragments={'qwen.visual.frozen': module['qwen.visual.frozen']},
                 shared_params={'qwen.alias': 'qwen.visual.frozen'},
                 identity=identity['identity'], completed=completed, epoch=epoch, offset=offset,
                 exposure=32*completed, elapsed=1.+len(label), scheduler_horizon=100000,
                 shared_geometry_identity='same-geometry', global_steps=completed,
                 global_samples=32*completed, skipped_steps=0, dp_world_size=2, mp_world_size=1,
                 lr_scheduler={'last_epoch': completed, 'base_lrs': [1e-5, 1e-4]},
                 data_sampler=None, random_ltd=None, ds_config={'zero_optimization': {'stage': 2}}, ds_version='0.16.9')
    torch.save(model, folder/'mp_rank_00_model_states.pt')
    flats = [torch.cat([module[name].reshape(-1) for name in shape]) for shape in shapes]
    for rank in range(2):
        # Group 0 splits a logical parameter between ranks. Group 1 has two
        # alignment elements only in rank 1's Adam state, excluded from masters.
        mapping = [OrderedDict([('future_space.weight', {'start': 0, 'numel': 5})]) if rank == 0 else
                   OrderedDict([('future_space.weight', {'start': 0, 'numel': 1}), ('refiner.weight', {'start': 1, 'numel': 4})]),
                   OrderedDict([('geometry.backbone.weight', {'start': 0, 'numel': 4 if rank == 0 else 2})])]
        masters = [flats[0][rank*5:rank*5+5].clone(), flats[1][rank*4:rank*4+4].clone()]
        paddings = [0, 0 if rank == 0 else 2]
        adam = {}
        for group, master in enumerate(masters):
            size = master.numel()+paddings[group]
            adam[group] = {'step': completed, 'exp_avg': torch.full((size,), .03+delta),
                           'exp_avg_sq': torch.full((size,), .01+delta)}
        optimizer = dict(single_partition_of_fp32_groups=masters, param_slice_mappings=mapping,
                         base_optimizer_state={'state': adam, 'param_groups': [
                             dict(params=[0], lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.001),
                             dict(params=[1], lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=.001)]},
                         zero_stage=2, group_paddings=paddings, partition_count=[2, 2],
                         loss_scaler=None, dynamic_loss_scale=False, overflow=False, clip_grad=1., ds_version='0.16.9')
        torch.save({'optimizer_state_dict': optimizer}, folder/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')
        rng = {'python': random.Random(42).getstate(), 'numpy': np.random.RandomState(42).get_state(),
               'torch': torch.Generator().manual_seed(42).get_state(),
               'cuda': torch.Generator().manual_seed(43).get_state(),
               'generators': {name: torch.Generator().manual_seed(i+44).get_state() for i, name in enumerate(('noise', 'time', 'proposal', 'queries', 'counts'))}}
        torch.save(rng, folder/f'rng_rank{rank}.pt')
    complete = dict(identity=identity['identity'], tag=folder.name, completed=completed, ranks=2,
                    status='RUNNING', save_seconds=.1, file_sha256={})
    write_json(folder/'COMPLETE.json', complete)
    rehash(folder)
    return folder


def rehash(folder):
    p = folder/'COMPLETE.json'
    complete = json.loads(p.read_text())
    complete['file_sha256'] = {x.name: comparison.digest(x) for x in folder.glob('*.pt')}
    write_json(p, complete)


class NativeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.origin = native_fixture(self.root, 'origin')
        binding = dict(run=str(self.origin.parent.parent), tag=self.origin.name,
                       identity=json.loads((self.origin/'COMPLETE.json').read_text())['identity'],
                       completed=2, complete_sha256=comparison.digest(self.origin/'COMPLETE.json'))
        self.paths = [native_fixture(self.root, label, origin_binding=binding, completed=3, offset=3) for label in ('A', 'AA', 'B')]

    def tearDown(self):
        self.temp.cleanup()

    def run_comparison(self):
        r = comparison.compare_checkpoints(*self.paths, chunk_elements=2)
        RESULTS.append({'test': self._testMethodName, 'comparison_performed': r['comparison_performed'],
                        'qualification_passed': r['qualification_passed'],
                        'E1': r.get('E1'), 'original_strict': r.get('original_strict'), 'error': r.get('error')})
        return r

    def modify(self, bank, name, update):
        path = self.paths[bank]/name
        # Mutate only these small synthetic files, with independent CPU storage.
        value = torch.load(path, map_location='cpu', weights_only=False)
        update(value)
        torch.save(value, path)
        rehash(self.paths[bank])

    def test_identical_full_native_state_and_independent_update_norm(self):
        r = self.run_comparison()
        self.assertTrue(r['qualification_passed'], r.get('error'))
        self.assertEqual(r['coverage']['actual_parameter_elements'], 16)
        self.assertEqual(r['coverage']['trainable_parameters'], 3)
        masters = r['comparisons']['A_B']['master_FP32']['all']
        self.assertEqual(masters['logical_tensor_count'], 3)
        expected = float(torch.linalg.vector_norm(torch.full((16,), .01)))
        self.assertAlmostEqual(masters['origin_relative_updates']['update_norms']['A'], expected, places=6)
        self.assertEqual(masters['bit_different_elements'], 0)
        self.assertEqual(masters['logical_tensor_max_error_quantiles'], {'p50': 0., 'p90': 0., 'p99': 0.})

    def test_changed_optimizer_moment_detected_when_model_identical(self):
        def change(value):
            value['optimizer_state_dict']['base_optimizer_state']['state'][0]['exp_avg'][2] += .002
        self.modify(2, 'zero_pp_rank_0_mp_rank_00_optim_states.pt', change)
        r = self.run_comparison()
        self.assertTrue(r['E1']['A_B']['passed'])
        self.assertFalse(r['original_strict']['A_B']['passed'])
        self.assertFalse(r['fixed_numeric_results']['A_B']['passed'])
        self.assertTrue(r['comparisons']['A_B']['model_FP32']['all']['bitwise_passed'])
        self.assertGreater(r['comparisons']['A_B']['Adam/exp_avg']['all']['max_absolute'], .001)

    def test_master_fragment_norms_cross_rank_quantiles_and_padding(self):
        def change(value):
            # rank1 hp offset0 is the final element of future_space.weight.
            value['optimizer_state_dict']['single_partition_of_fp32_groups'][0][0] += .005
        self.modify(2, 'zero_pp_rank_1_mp_rank_00_optim_states.pt', change)
        r = self.run_comparison()
        s = r['comparisons']['A_B']['master_FP32']['all']
        self.assertAlmostEqual(s['difference_norm'], .005, places=6)
        self.assertEqual(s['logical_tensor_count'], 3)
        self.assertAlmostEqual(s['logical_tensor_max_error_quantiles']['p90'], .004, places=6)
        self.assertAlmostEqual(s['origin_relative_updates']['candidate_minus_reference_over_reference_update_norm'], .125, places=5)
        self.assertFalse(r['qualification_passed'])

    def test_unused_alignment_padding_is_excluded_but_no_parameter_is_dropped(self):
        def change(value):
            value['optimizer_state_dict']['base_optimizer_state']['state'][1]['exp_avg'][-2:] = 12345.
        self.modify(2, 'zero_pp_rank_1_mp_rank_00_optim_states.pt', change)
        r = self.run_comparison()
        self.assertTrue(r['qualification_passed'], r.get('error'))
        self.assertEqual(r['comparisons']['A_B']['Adam/exp_avg']['all']['elements'], 16)

    def test_AA_background_never_waives_original_strict_failure(self):
        for bank, difference in ((1, .004), (2, .002)):
            def change(value, difference=difference):
                value['module']['refiner.weight'][0, 0] += difference
            self.modify(bank, 'mp_rank_00_model_states.pt', change)
        r = self.run_comparison()
        self.assertFalse(r['original_strict']['A_A']['passed'])
        self.assertFalse(r['original_strict']['A_B']['passed'])
        self.assertFalse(r['qualification_passed'])
        self.assertFalse(r['A_A_context']['thresholds_changed'])
        self.assertTrue(r['A_A_context']['A_B_strict_failure_preserved'])
        self.assertEqual(r['fixed_numeric_gate'], {'atol': 1e-6, 'rtol': 1e-4, 'configurable': False})

    def test_E1_step_scheduler_data_rng_and_optimizer_group_changes(self):
        cases = [
            ('mp_rank_00_model_states.pt', lambda v: v.__setitem__('epoch', 1), 'client_step_scheduler_data/epoch'),
            ('mp_rank_00_model_states.pt', lambda v: v['lr_scheduler'].__setitem__('last_epoch', 4), 'lr_scheduler/last_epoch'),
            ('rng_rank1.pt', lambda v: v['generators']['proposal'].__setitem__(0, 12), 'RNG/rank1'),
            ('zero_pp_rank_0_mp_rank_00_optim_states.pt', lambda v: v['optimizer_state_dict']['base_optimizer_state']['state'][0].__setitem__('step', 4), 'scalar_state/0/step'),
            ('zero_pp_rank_0_mp_rank_00_optim_states.pt', lambda v: v['optimizer_state_dict']['base_optimizer_state']['param_groups'][0].__setitem__('lr', 1e-4), 'param_groups/0/lr'),
        ]
        for filename, update, expected in cases:
            with self.subTest(expected=expected):
                original = torch.load(self.paths[2]/filename, map_location='cpu', weights_only=False)
                self.modify(2, filename, update)
                r = self.run_comparison()
                self.assertFalse(r['E1']['A_B']['passed'])
                self.assertTrue(any(expected in path for path in r['E1']['A_B']['failure_paths']))
                self.assertFalse(r['qualification_passed'])
                torch.save(original, self.paths[2]/filename); rehash(self.paths[2])

    def test_BN_float_discrete_counter_and_nonfinite(self):
        filename = 'mp_rank_00_model_states.pt'
        original = torch.load(self.paths[2]/filename, map_location='cpu', weights_only=False)
        self.modify(2, filename, lambda v: v['module']['geometry.bn.running_mean'].__setitem__(0, .4))
        r = self.run_comparison()
        self.assertTrue(r['E1']['A_B']['passed'])
        self.assertFalse(r['comparisons']['A_B']['BN_float']['all']['bitwise_passed'])
        torch.save(original, self.paths[2]/filename); rehash(self.paths[2])
        self.modify(2, filename, lambda v: v['module']['geometry.bn.num_batches_tracked'].add_(1))
        self.assertFalse(self.run_comparison()['E1']['A_B']['passed'])
        torch.save(original, self.paths[2]/filename); rehash(self.paths[2])
        self.modify(2, filename, lambda v: v['module']['geometry.bn.running_mean'].__setitem__(0, float('nan')))
        r = self.run_comparison()
        self.assertGreater(r['comparisons']['A_B']['BN_float']['all']['nonfinite_elements'], 0)
        self.assertFalse(r['qualification_passed'])
        json.dumps(r, allow_nan=False)

    def test_missing_shard_corruption_and_shape_fail_closed(self):
        path = self.paths[2]/'zero_pp_rank_1_mp_rank_00_optim_states.pt'
        data = path.read_bytes(); path.unlink()
        self.assertFalse(self.run_comparison()['valid_native_inputs'])
        path.write_bytes(data+b'corruption')
        self.assertIn('hash changed', self.run_comparison()['error']['message'])
        path.write_bytes(data)
        self.modify(2, 'mp_rank_00_model_states.pt', lambda v: v['module'].__setitem__('refiner.weight', torch.zeros(2, 3)))
        self.assertFalse(self.run_comparison()['valid_native_inputs'])

    def test_wrong_origin_binding_and_changed_scientific_population_fail_closed(self):
        p = self.paths[2].parent.parent/'identity.json'
        original = json.loads(p.read_text())
        changed = copy.deepcopy(original)
        changed['engineering_continue_origin']['complete_sha256'] = '0'*64
        write_json(p, changed)
        self.assertIn('origins differ', self.run_comparison()['error']['message'])
        for bank in self.paths:
            q = bank.parent.parent/'identity.json'
            value = json.loads(q.read_text())
            value['engineering_continue_origin'] = original['engineering_continue_origin']
            value['cache_identity'] = 'changed-but-common-cache'
            write_json(q, value)
        self.assertIn('changed from origin', self.run_comparison()['error']['message'])

    def test_bitwise_failure_remains_when_fixed_numeric_gate_passes(self):
        def change(value):
            value['module']['refiner.weight'][0, 0] = torch.nextafter(value['module']['refiner.weight'][0, 0], torch.tensor(1.))
        self.modify(2, 'mp_rank_00_model_states.pt', change)
        r = self.run_comparison()
        self.assertTrue(r['fixed_numeric_results']['A_B']['passed'])
        self.assertFalse(r['original_strict']['A_B']['passed'])
        self.assertFalse(r['qualification_passed'])
        skipped = comparison.compare_checkpoints(*self.paths, verify_hashes=False)
        self.assertFalse(skipped['qualification_passed'])

    def test_origin_missing_moments_zero_initialization_explicit(self):
        for rank in range(2):
            path = self.origin/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
            value = torch.load(path, map_location='cpu', weights_only=False)
            value['optimizer_state_dict']['base_optimizer_state']['state'] = {}
            torch.save(value, path)
        rehash(self.origin)
        for path in self.paths:
            p = path.parent.parent/'identity.json'; value = json.loads(p.read_text())
            value['engineering_continue_origin']['complete_sha256'] = comparison.digest(self.origin/'COMPLETE.json')
            write_json(p, value)
        r = self.run_comparison()
        self.assertTrue(r['qualification_passed'], r.get('error'))
        self.assertTrue(r['comparisons']['A_B']['Adam/exp_avg']['all']['origin_relative_updates']['missing_origin_moments_initialized_as_zero'])

    def test_CLI_writes_exact_failure_JSON_and_metadata_has_no_payload_comparison(self):
        output = self.root/'receipt.json'
        cmd = [sys.executable, '-B', str(SOURCE), '--reference', str(self.paths[0]), '--reference-replay', str(self.paths[1]),
               '--candidate', str(self.paths[2]), '--output', str(output), '--chunk-elements', '2']
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(output.read_text())['qualification_passed'])
        self.modify(2, 'mp_rank_00_model_states.pt', lambda v: v['module']['refiner.weight'].add_(.001))
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(json.loads(output.read_text())['original_strict']['A_B']['passed'])
        result = subprocess.run(cmd+['--metadata-only'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(output.read_text())
        self.assertFalse(receipt['comparison_performed'])
        self.assertFalse(receipt['qualification_passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeComparisonTests))
    report = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'controls': RESULTS,
              'wall_seconds': time.monotonic()-started, 'torch': torch.__version__, 'device': 'cpu',
              'source_sha256': {str(path.relative_to(ROOT)): comparison.digest(path) for path in (SOURCE, Path(__file__))},
              'scope': 'synthetic observed-native ZeRO-2 structure, 2 ranks, cross-rank parameter and alignment padding; no real optimizer training or GPU',
              'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures+result.errors]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('controls', 'failures')}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
