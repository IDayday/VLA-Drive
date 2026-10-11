"""P1 controls against the independent pinned Future/FGTR implementation.

CPU is the default. CUDA probes require a separately admitted parent slot.
No trainer, optimizer, experiment recipe or default execution flag is changed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.structured_world.future import SingleFrameFuture, can_batch_independent
from starVLA.model.modules.structured_world.fgtr import DDPProposalFGTR
from starVLA.model.modules.structured_world.grid import GridSpec

BASE = '267701f3b2c6cf36a97ff3a5e6bfb535e5aab232'
DEVICE = torch.device('cpu')
RESULTS = []
# Fixed numeric gate; A/A, metadata, RNG and buffers are checked bitwise.
ATOL, RTOL = 1e-6, 1e-4


def original_classes():
    def load(name, path, package=None):
        module = types.ModuleType(name)
        module.__package__ = package
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE+':'+path], text=True)
        exec(compile(source, BASE+':'+path, 'exec'), module.__dict__)
        return module
    token = load('original_timevec_token', 'third_party/resworld/ported/tokenlearner.py')
    future = load('original_timevec_future', 'starVLA/model/modules/structured_world/future.py',
                  'starVLA.model.modules.structured_world')
    future.TokenFuser = token.TokenFuser
    fgtr = load('original_timevec_fgtr', 'starVLA/model/modules/structured_world/fgtr.py',
                'starVLA.model.modules.structured_world')
    return future.SingleFrameFuture, fgtr.DDPProposalFGTR


def snapshot(model, inputs):
    inputs = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
    model.zero_grad(set_to_none=True)
    calls, handles = {}, []
    for name, module in model.named_modules():
        if name in ('res_latent_decoder', 'action_mln', 'tokenfuser.mlp', 'col_attn', 'coordinate_decoder'):
            calls[name] = []
            def count(_module, args, kwargs, name=name):
                value = kwargs.get('query', args[0] if args else None)
                calls[name].append(list(value.shape))
            handles.append(module.register_forward_pre_hook(count, with_kwargs=True))
    torch.manual_seed(81011)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    try:
        output = model(*inputs)
        numeric = output['q_final_encoded'] if isinstance(output, dict) else output
        weight = torch.linspace(.2, 1.2, numeric.numel(), device=DEVICE).reshape_as(numeric)
        loss = (numeric*weight).mean()+.1*numeric.square().mean()
        loss.backward()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize(DEVICE)
    finally:
        for handle in handles:
            handle.remove()
    return {
        'output': ({k: v.detach().clone() for k, v in output.items()} if isinstance(output, dict)
                   else {'future': output.detach().clone()}),
        'input_gradients': [None if x.grad is None else x.grad.detach().clone() for x in inputs],
        'parameter_gradients': {k: None if p.grad is None else p.grad.detach().clone()
                                for k, p in model.named_parameters()},
        'buffers': {k: v.detach().clone() for k, v in model.named_buffers()},
        'CPU_RNG': torch.get_rng_state().clone(),
        'device_RNG': torch.cuda.get_rng_state(DEVICE) if DEVICE.type == 'cuda' else None,
        'calls': calls, 'seconds': time.perf_counter()-started,
    }


def compare(actual, expected, *, exact=False):
    atol, rtol = (0, 0) if exact else (ATOL, RTOL)
    maxima = {'output_max_abs_error': 0., 'gradient_max_abs_error': 0.}
    bitwise = {'outputs_bitwise': True, 'gradients_bitwise': True}
    assert list(actual['output']) == list(expected['output'])
    for key, other in expected['output'].items():
        value = actual['output'][key]
        metadata = key in ('out_of_range', 'reference_points')
        torch.testing.assert_close(value, other, atol=0 if metadata else atol, rtol=0 if metadata else rtol)
        error = 0. if value.dtype == torch.bool else float((value-other).abs().max())
        maxima['output_max_abs_error'] = max(maxima['output_max_abs_error'], error)
        bitwise['outputs_bitwise'] &= torch.equal(value, other)
    for key in ('CPU_RNG', 'device_RNG'):
        if expected[key] is None:
            assert actual[key] is None
        else:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert list(actual['buffers']) == list(expected['buffers'])
    for key, value in expected['buffers'].items():
        torch.testing.assert_close(actual['buffers'][key], value, rtol=0, atol=0)
    assert list(actual['parameter_gradients']) == list(expected['parameter_gradients'])
    gradients = [(f'input/{i}', a, b) for i, (a, b) in enumerate(zip(actual['input_gradients'], expected['input_gradients']))]
    gradients += [(k, actual['parameter_gradients'][k], v) for k, v in expected['parameter_gradients'].items()]
    elements, worst = 0, None
    for name, value, other in gradients:
        if other is None:
            assert value is None, name
            continue
        assert value is not None and torch.isfinite(value).all(), name
        error = float((value-other).abs().max())
        if error > maxima['gradient_max_abs_error']:
            maxima['gradient_max_abs_error'], worst = error, name
        bitwise['gradients_bitwise'] &= torch.equal(value, other)
        elements += value.numel()
        torch.testing.assert_close(value, other, rtol=rtol, atol=atol, msg=lambda text: name+': '+text)
    return {**maxima, **bitwise, 'worst_gradient': worst, 'gradient_elements': elements,
            'RNG_buffers_metadata_bitwise': True}


class TimeVectorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OldFuture, cls.OldFGTR = original_classes()

    def future(self, steps=8, *, train=True, batch=2, channels=32, scene_tokens=4):
        torch.manual_seed(91011)
        model = self.OldFuture(24, steps, channels=channels, scene_tokens=scene_tokens,
                              grid=GridSpec(xmin=-2, xmax=3, ymin=-2, ymax=2, dx=1, dy=1))
        with torch.no_grad():
            for layer in (model.action_mln.gamma, model.action_mln.beta):
                layer.weight.normal_(std=.08)
                layer.bias.normal_(std=.08)
        model = model.to(DEVICE).train(train)
        values = (torch.randn(batch, channels, 4, 5, device=DEVICE, requires_grad=True),
                  torch.randn(batch, 7, 24, device=DEVICE, requires_grad=True),
                  torch.randn(batch, steps, 3, device=DEVICE))
        return model, values

    def fgtr(self, steps=8, *, train=True, batch=2):
        torch.manual_seed(111011)
        model = self.OldFGTR(24, steps, channels=32,
                            grid=GridSpec(xmin=-2, xmax=3, ymin=-2, ymax=2, dx=1, dy=1))
        # A cold zero head hides all attention and geometry gradient errors.
        with torch.no_grad():
            model.coordinate_decoder[-1].weight.normal_(std=.08)
            model.coordinate_decoder[-1].bias.normal_(std=.08)
        model = model.to(DEVICE).train(train)
        geometry = torch.randn(batch, steps, 32, 4, 5, device=DEVICE, requires_grad=True)
        # Scene/time-specific field offsets detect accidentally mixed B/T axes.
        with torch.no_grad():
            geometry.add_(torch.arange(batch*steps, device=DEVICE).reshape(batch, steps, 1, 1, 1)/10.)
        physical = torch.randn(batch, steps, 3, device=DEVICE)
        physical[:, 0, :2] = physical.new_tensor([-2., -2.])  # lower boundary is legal
        physical[:, 1, :2] = physical.new_tensor([3., 2.])    # upper boundary is out of range
        physical[:, 2, :2] = physical.new_tensor([3.1, 2.1])  # finite beyond the grid; never clamp
        values = (torch.randn(batch, 7, 24, device=DEVICE, requires_grad=True), geometry,
                  torch.randn(batch, steps, 4, device=DEVICE), physical)
        return model, values

    def candidate(self, original, *, vectorized=True, chunk=0, fast=True):
        is_future = isinstance(original, self.OldFuture)
        if is_future:
            model = SingleFrameFuture(24, original.steps, channels=original.channels,
                                      scene_tokens=original.tokenlearner.num_tokens, grid=original.grid)
        else:
            model = DDPProposalFGTR(24, original.steps, channels=original.channels, grid=original.grid)
        # Copy real dropout fixtures without replacing native attention operators.
        originals = dict(original.named_modules())
        for name, child in model.named_modules():
            other = originals.get(name)
            if isinstance(child, nn.modules.dropout._DropoutNd):
                child.p = other.p
            if isinstance(child, nn.MultiheadAttention):
                child.dropout = other.dropout
        if not is_future and len(original.coordinate_decoder) != len(model.coordinate_decoder):
            model.coordinate_decoder = copy.deepcopy(original.coordinate_decoder)
        self.assertEqual(list(model.state_dict()), list(original.state_dict()))
        self.assertEqual(list(dict(model.named_parameters())), list(dict(original.named_parameters())))
        model.load_state_dict(original.state_dict(), strict=True)
        model.to(DEVICE).train(original.training)
        if is_future:
            self.assertFalse(model.execution_fast)
            self.assertFalse(model.future_time_vectorized)
            model.execution_fast, model.future_time_vectorized = fast, vectorized
            model.future_time_chunk_size = chunk
        else:
            self.assertFalse(model.fgtr_time_vectorized)
            model.fgtr_time_vectorized, model.fgtr_time_chunk_size = vectorized, chunk
        return model

    def control(self, label, original, values, candidate, *, exact_output=False):
        reference = snapshot(copy.deepcopy(original), values)
        replay = snapshot(copy.deepcopy(original), values)
        aa = compare(replay, reference, exact=DEVICE.type == 'cpu')
        actual = snapshot(candidate, values)
        ab = compare(actual, reference)
        if exact_output:
            for key, other in reference['output'].items():
                torch.testing.assert_close(actual['output'][key], other, rtol=0, atol=0)
        # Both live inputs and key trained parameters must receive real gradients.
        for value in actual['input_gradients'][:2]:
            self.assertIsNotNone(value)
            self.assertGreater(float(value.abs().sum()), 0)
        self.assertIsNone(actual['input_gradients'][-1])
        RESULTS.append({'case': label, 'A_A': aa, 'A_B': ab, 'reference_calls': reference['calls'],
                        'candidate_calls': actual['calls'], 'reference_seconds': reference['seconds'],
                        'candidate_seconds': actual['seconds']})
        return actual

    def test_default_paths_remain_pinned(self):
        for kind in ('future', 'fgtr'):
            for steps in (6, 8):
                original, values = getattr(self, kind)(steps)
                candidate = self.candidate(original, vectorized=False, fast=False)
                self.control(f'default_{kind}_{steps}', original, values, candidate, exact_output=True)

    def test_future_full_and_chunked_independent_batches(self):
        for steps in (6, 8):
            for train in (False, True):
                for chunk in (0, 1, 2, 4):
                    with self.subTest(steps=steps, train=train, chunk=chunk):
                        original, values = self.future(steps, train=train, batch=3)
                        candidate = self.candidate(original, chunk=chunk)
                        actual = self.control(f'future_{steps}_train{train}_chunk{chunk}', original, values, candidate)
                        calls = actual['calls']['res_latent_decoder']
                        size = chunk or steps
                        self.assertEqual(len(calls), (steps+size-1)//size)
                        for i, shape in enumerate(calls):
                            self.assertEqual(shape, [4, 3*min(size, steps-i*size), 32])
                        self.assertEqual(len(actual['calls']['action_mln']), 1)

    def test_future_flag_is_independent_and_real_channel_width(self):
        original, values = self.future(channels=256, scene_tokens=16)
        actual = self.control('future_C256_N16_P0_disabled', original, values,
                              self.candidate(original, fast=False))
        self.assertEqual(actual['calls']['res_latent_decoder'], [[16, 16, 256]])
        self.assertEqual(len(actual['calls']['tokenfuser.mlp']), 1)

    def test_future_dropout_fallback_preserves_masks(self):
        for native_dropout in (False, True):
            original, values = self.future()
            if native_dropout:
                # Native MHA.float dropout must be detected even without a child Dropout.
                for child in original.res_latent_decoder.modules():
                    if isinstance(child, nn.MultiheadAttention):
                        child.dropout = .2
            else:
                for child in original.res_latent_decoder.modules():
                    if isinstance(child, nn.Dropout):
                        child.p = .2
            candidate = self.candidate(original)
            self.assertFalse(can_batch_independent(candidate.res_latent_decoder))
            actual = self.control(f'future_dropout_fallback_native{native_dropout}', original, values, candidate,
                                  exact_output=True)
            self.assertEqual(len(actual['calls']['res_latent_decoder']), 8)
        original, values = self.future()
        original.tokenfuser.dropout.p = .2
        candidate = self.candidate(original, chunk=4)
        actual = self.control('future_zero_decoder_live_fuser_dropout', original, values, candidate)
        self.assertEqual(len(actual['calls']['res_latent_decoder']), 2)

    def test_fgtr_training_native_dropout_remains_per_time(self):
        for steps in (6, 8):
            original, values = self.fgtr(steps)
            candidate = self.candidate(original)
            self.assertEqual(candidate.col_attn.dropout.p, .1)
            self.assertFalse(can_batch_independent(candidate.col_attn))
            actual = self.control(f'fgtr_train_{steps}_native_dropout', original, values, candidate)
            self.assertEqual(len(actual['calls']['col_attn']), steps)
            self.assertEqual(actual['calls']['coordinate_decoder'], [[2, steps, 32]])
            for key in ('col_attn.value_proj.weight', 'col_attn.sampling_offsets.weight',
                        'coordinate_decoder.4.weight'):
                self.assertGreater(float(actual['parameter_gradients'][key].abs().sum()), 0)

    def test_fgtr_eval_full_and_chunked_native_operator(self):
        for steps in (6, 8):
            for chunk in (0, 1, 2, 4):
                original, values = self.fgtr(steps, train=False, batch=3)
                candidate = self.candidate(original, chunk=chunk)
                actual = self.control(f'fgtr_eval_{steps}_chunk{chunk}', original, values, candidate)
                size = chunk or steps
                calls = actual['calls']['col_attn']
                self.assertEqual(len(calls), (steps+size-1)//size)
                for i, shape in enumerate(calls):
                    self.assertEqual(shape, [1, 3*min(size, steps-i*size), 32])
                self.assertFalse(bool(actual['output']['out_of_range'][:, 0].any()))
                self.assertTrue(bool(actual['output']['out_of_range'][:, 1:3].all()))
                self.assertTrue(bool((actual['output']['reference_points'][:, 2] > 1).all()))

    def test_fgtr_coordinate_dropout_falls_back_without_rng_reorder(self):
        original, values = self.fgtr()
        original.coordinate_decoder.add_module('fixture_dropout', nn.Dropout(.2))
        candidate = self.candidate(original)
        actual = self.control('fgtr_live_coordinate_dropout_fallback', original, values, candidate, exact_output=True)
        self.assertEqual(len(actual['calls']['coordinate_decoder']), 8)
        self.assertEqual(len(actual['calls']['col_attn']), 8)

    def test_batch_normalization_guard_uses_real_running_state(self):
        # Native heads have no BN. A real BN stress fixture must preserve both
        # batch statistics and the original B,C layout when running stats freeze.
        for train in (True, False):
            original, values = self.fgtr(train=train)
            original.coordinate_decoder.add_module('fixture_bn', nn.BatchNorm1d(4).to(DEVICE))
            original.train(train)
            candidate = self.candidate(original)
            self.assertFalse(can_batch_independent(candidate.coordinate_decoder))
            actual = self.control(f'fgtr_real_BN_train{train}_fallback', original, values, candidate,
                                  exact_output=True)
            self.assertEqual(len(actual['calls']['coordinate_decoder']), 8)
            self.assertEqual(actual['buffers']['coordinate_decoder.fixture_bn.num_batches_tracked'].item(),
                             8 if train else 0)
        self.assertFalse(can_batch_independent(nn.BatchNorm1d(4, track_running_stats=False).eval()))

    def test_local_reuse_second_forward_axes_and_detached_guard(self):
        original, values = self.future()
        candidate = self.candidate(original, chunk=4)
        self.control('future_first_forward', original, values, candidate)
        changed = (values[0]*.7+.2, values[1]*.8, values[2]*1.3)
        self.control('future_second_forward', original, changed, candidate)
        original, values = self.fgtr(train=False)
        candidate = self.candidate(original, chunk=4)
        with torch.no_grad():
            before = candidate(*values)['q_final_encoded']
            changed = [value.detach().clone() for value in values]
            changed[1][:, 3] += .7
            after = candidate(*changed)['q_final_encoded']
            kept = [t for t in range(8) if t != 3]
            torch.testing.assert_close(after[:, kept], before[:, kept], rtol=0, atol=0)
            self.assertGreater(float((after[:, 3]-before[:, 3]).abs().max()), 0)
        for kind in ('future', 'fgtr'):
            original, values = getattr(self, kind)()
            candidate = self.candidate(original)
            illegal = list(values)
            illegal[-1] = illegal[-1].detach().clone().requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'must be detached'):
                candidate(*illegal)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimeVectorizationTests))
    paths = ['starVLA/model/modules/structured_world/future.py', 'starVLA/model/modules/structured_world/fgtr.py',
             'third_party/resworld/ported/tokenlearner.py', 'tools/structured_world/test_time_vectorization.py']
    report = {'base_source': BASE, 'device': str(DEVICE), 'torch': torch.__version__,
              'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'controls': RESULTS,
              'numeric_tolerance': {'atol': ATOL, 'rtol': RTOL},
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              'scope': 'P1 module FP32 output/input/parameter gradients; exact A/A CPU, metadata, RNG, buffers; no GPU peak or training-speed qualification',
              'FGTR_training': 'native attention dropout 0.1 remains per-time; deterministic coordinate MLP is batched',
              'defaults': 'Future/FGTR vectorization False; engineering_fast does not enable P1'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'controls'}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
