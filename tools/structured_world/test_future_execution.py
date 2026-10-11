"""P0 future execution controls against the pinned pre-optimization source.

CPU example (does not admit or occupy a GPU):
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -B \
    tools/structured_world/test_future_execution.py --device cpu --report REPORT.json

An explicitly selected CUDA device is also supported for a separately scheduled
parent probe. This is a module control, not a full-model throughput qualification.
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
from starVLA.model.modules.structured_world.future import SingleFrameFuture
from starVLA.model.modules.structured_world.grid import GridSpec
from third_party.resworld.ported.tokenlearner import TokenFuser

BASE = '267701f3b2c6cf36a97ff3a5e6bfb535e5aab232'
DEVICE = torch.device('cpu')
RESULTS = []
# The existing full-model execution gradient gate uses these fixed tolerances.
GRAD_ATOL, GRAD_RTOL = 1e-6, 1e-4


def original_classes():
    def source(path):
        return subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':' + path], text=True)
    token = types.ModuleType('original_tokenlearner')
    exec(compile(source('third_party/resworld/ported/tokenlearner.py'), BASE + ':tokenlearner.py', 'exec'), token.__dict__)
    future = types.ModuleType('original_future')
    future.__package__ = 'starVLA.model.modules.structured_world'
    exec(compile(source('starVLA/model/modules/structured_world/future.py'), BASE + ':future.py', 'exec'), future.__dict__)
    # The baseline must use the old fuser too, not its newly split helpers.
    future.TokenFuser = token.TokenFuser
    return future.SingleFrameFuture, token.TokenFuser


def snapshot(model, values, run=None):
    values = [x.detach().clone().requires_grad_(x.requires_grad) for x in values]
    model.zero_grad(set_to_none=True)
    counts, handles = {}, []
    names = ('action_mln', 'res_latent_decoder', 'tokenfuser.norm', 'tokenfuser.mlp',
             'tokenfuser.dense', 'tokenfuser.dropout')
    for name, module in model.named_modules():
        if name in names:
            counts[name] = 0
            def count(_module, _args, _result, name=name):
                counts[name] += 1
            handles.append(module.register_forward_hook(count))
    torch.manual_seed(10911)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)
    start = time.perf_counter()
    try:
        output = (run or model)(*values)
        # Distinct time/spatial weights exercise every output and gradient path.
        weight = torch.linspace(.2, 1.2, output.numel(), device=output.device, dtype=output.dtype).view_as(output)
        loss = (output * weight).mean() + .1 * output.square().mean()
        loss.backward()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize(DEVICE)
    finally:
        for handle in handles:
            handle.remove()
    return {
        'output': output.detach().clone(),
        'input_gradients': [None if x.grad is None else x.grad.detach().clone() for x in values],
        'parameter_gradients': {name: None if p.grad is None else p.grad.detach().clone()
                                for name, p in model.named_parameters()},
        'buffers': {name: b.detach().clone() for name, b in model.named_buffers()},
        'CPU_RNG': torch.get_rng_state().clone(),
        'device_RNG': torch.cuda.get_rng_state(DEVICE) if DEVICE.type == 'cuda' else None,
        'counts': counts, 'seconds': time.perf_counter() - start,
    }


def compare(actual, expected, *, exact_gradients=False):
    torch.testing.assert_close(actual['output'], expected['output'], rtol=0, atol=0)
    for name in ('CPU_RNG', 'device_RNG'):
        if expected[name] is None:
            assert actual[name] is None
        else:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert list(actual['buffers']) == list(expected['buffers'])
    for name in expected['buffers']:
        torch.testing.assert_close(actual['buffers'][name], expected['buffers'][name], rtol=0, atol=0)
    assert list(actual['parameter_gradients']) == list(expected['parameter_gradients'])
    maximum, worst, elements, bitwise = 0., None, 0, True
    gradients = [(f'input/{i}', a, b) for i, (a, b) in enumerate(zip(actual['input_gradients'], expected['input_gradients']))]
    gradients += [(name, actual['parameter_gradients'][name], value)
                  for name, value in expected['parameter_gradients'].items()]
    for name, a, b in gradients:
        if b is None:
            assert a is None, name
            continue
        assert a is not None and torch.isfinite(a).all(), name
        error = float((a - b).abs().max())
        if error > maximum:
            maximum, worst = error, name
        elements += a.numel()
        bitwise = bitwise and torch.equal(a, b)
        torch.testing.assert_close(a, b, rtol=0 if exact_gradients else GRAD_RTOL,
                                   atol=0 if exact_gradients else GRAD_ATOL, msg=lambda msg: name + ': ' + msg)
    return {'output_RNG_buffers_bitwise': True, 'gradient_max_abs_error': maximum,
            'worst_gradient': worst, 'gradient_elements': elements, 'gradients_bitwise': bitwise}


class FutureExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OldFuture, cls.OldFuser = original_classes()

    def future(self, steps=8, *, train=True, stochastic_spatial=False, batch_norm=False):
        torch.manual_seed(11011)
        model = self.OldFuture(24, steps, channels=32, scene_tokens=4,
                               grid=GridSpec(xmin=-2, xmax=3, ymin=-2, ymax=2, dx=1, dy=1))
        # Avoid cold MLN zero-gamma/beta weights hiding a broken condition gradient.
        with torch.no_grad():
            for layer in (model.action_mln.gamma, model.action_mln.beta):
                layer.weight.normal_(std=.08)
                layer.bias.normal_(std=.08)
        # Use real registered dropout modules in decoder and final fusion.
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and ('decoder' in name or name == 'tokenfuser.dropout'):
                module.p = .2
        if stochastic_spatial:
            model.tokenfuser.mlp.dropout.p = .25
            model.tokenfuser.mlp.dropout2.p = .25
        if batch_norm:
            model.tokenfuser.norm = nn.BatchNorm1d(32)
        model = model.to(DEVICE).train(train)
        current = torch.randn(2, 32, 4, 5, device=DEVICE, requires_grad=True)
        hidden = torch.randn(2, 7, 24, device=DEVICE, requires_grad=True)
        proposal = torch.randn(2, steps, 3, device=DEVICE)
        return model, (current, hidden, proposal)

    def candidate(self, original, *, fast):
        model = SingleFrameFuture(24, original.steps, channels=32, scene_tokens=4, grid=original.grid)
        model.to(DEVICE).train(original.training)
        for name, module in model.named_modules():
            other = dict(original.named_modules()).get(name)
            if isinstance(module, nn.Dropout):
                module.p = other.p
        if isinstance(original.tokenfuser.norm, nn.BatchNorm1d):
            model.tokenfuser.norm = copy.deepcopy(original.tokenfuser.norm)
        self.assertEqual(list(model.state_dict()), list(original.state_dict()))
        self.assertEqual(list(dict(model.named_parameters())), list(dict(original.named_parameters())))
        model.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(model.execution_fast)
        model.execution_fast = fast
        return model

    def control(self, label, original, values, candidate, run=None):
        reference = snapshot(copy.deepcopy(original), values)
        replay = snapshot(copy.deepcopy(original), values)
        aa = compare(replay, reference, exact_gradients=DEVICE.type == 'cpu')
        actual = snapshot(candidate, values, run=run)
        ab = compare(actual, reference)
        RESULTS.append({'case': label, 'A_A': aa, 'A_B': ab, 'reference_calls': reference['counts'],
                        'candidate_calls': actual['counts'], 'reference_seconds': reference['seconds'],
                        'candidate_seconds': actual['seconds']})
        return actual

    def test_default_matches_pinned_source(self):
        for steps in (6, 8):
            for train in (False, True):
                with self.subTest(steps=steps, train=train):
                    original, values = self.future(steps, train=train)
                    self.control(f'default_{steps}_train{train}', original, values, self.candidate(original, fast=False))

    def test_fast_hoists_only_common_branches(self):
        for steps in (6, 8):
            for train in (False, True):
                with self.subTest(steps=steps, train=train):
                    original, values = self.future(steps, train=train)
                    actual = self.control(f'P0_{steps}_train{train}', original, values, self.candidate(original, fast=True))
                    calls = actual['counts']
                    for name in ('action_mln', 'tokenfuser.norm', 'tokenfuser.mlp'):
                        self.assertEqual(calls[name], 1)
                    for name in ('res_latent_decoder', 'tokenfuser.dense', 'tokenfuser.dropout'):
                        self.assertEqual(calls[name], steps)

    def test_training_spatial_dropout_keeps_step_masks(self):
        original, values = self.future(stochastic_spatial=True)
        candidate = self.candidate(original, fast=True)
        self.assertFalse(candidate.tokenfuser.can_reuse_spatial_mix())
        actual = self.control('spatial_dropout_train_fallback', original, values, candidate)
        self.assertEqual(actual['counts']['tokenfuser.mlp'], 8)
        self.assertEqual(actual['counts']['action_mln'], 1)
        original, values = self.future(stochastic_spatial=True, train=False)
        candidate = self.candidate(original, fast=True)
        self.assertTrue(candidate.tokenfuser.can_reuse_spatial_mix())
        actual = self.control('spatial_dropout_eval_reuse', original, values, candidate)
        self.assertEqual(actual['counts']['tokenfuser.mlp'], 1)

    def test_real_batchnorm_running_state_is_not_reused(self):
        # Native future has GN/LN and no BN. This real BN stress fixture guards
        # against a future spatial-normalization substitution changing counters.
        original, values = self.future(batch_norm=True)
        candidate = self.candidate(original, fast=True)
        self.assertFalse(candidate.tokenfuser.can_reuse_spatial_mix())
        actual = self.control('BatchNorm_train_fallback', original, values, candidate)
        self.assertEqual(actual['counts']['tokenfuser.norm'], 8)
        self.assertEqual(actual['buffers']['tokenfuser.norm.num_batches_tracked'].item(), 8)

    def test_split_fuser_preserves_shapes_norm_and_dropout(self):
        for shape_4d in (False, True):
            for normalized in (False, True):
                for train in (False, True):
                    with self.subTest(shape_4d=shape_4d, normalized=normalized, train=train):
                        torch.manual_seed(1011)
                        original = self.OldFuser(4, 32, use_normalization=normalized, dropout_rate=.25).to(DEVICE).train(train)
                        candidate = TokenFuser(4, 32, use_normalization=normalized, dropout_rate=.25).to(DEVICE).train(train)
                        self.assertEqual(list(original.state_dict()), list(candidate.state_dict()))
                        candidate.load_state_dict(original.state_dict(), strict=True)
                        tokens = torch.randn(2, 4, 32, device=DEVICE, requires_grad=True)
                        shape = (2, 32, 4, 5) if shape_4d else (2, 20, 32)
                        spatial = torch.randn(*shape, device=DEVICE, requires_grad=True)
                        def split(tokens, spatial):
                            # Preserve original call order when dropout is live.
                            token_values = candidate.prepare_token_values(tokens)
                            mix = candidate.prepare_spatial_mix(spatial)
                            self.assertTrue(mix.requires_grad)
                            return candidate.fuse(token_values, mix, spatial)
                        self.control(f'fuser_4d{shape_4d}_norm{normalized}_train{train}', original,
                                     (tokens, spatial), candidate, run=split)

    def test_reuse_is_local_to_one_forward_and_input_guard_remains(self):
        original, values = self.future()
        candidate = self.candidate(original, fast=True)
        self.control('first_forward', original, values, candidate)
        second = (values[0] * .7 + .2, values[1] * .8, values[2] * 1.3)
        self.control('second_forward_new_inputs', original, second, candidate)
        illegal = values[2].detach().clone().requires_grad_(True)
        for fast in (False, True):
            candidate.execution_fast = fast
            with self.assertRaisesRegex(ValueError, 'must be detached'):
                candidate(values[0], values[1], illegal)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='CPU by default; parent must separately admit any CUDA probe')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FutureExecutionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'base_source': BASE, 'device': str(DEVICE), 'torch': torch.__version__,
              'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'controls': RESULTS,
              'gradient_tolerance': {'atol': GRAD_ATOL, 'rtol': GRAD_RTOL},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT/'starVLA/model/modules/structured_world/future.py',
                                ROOT/'third_party/resworld/ported/tokenlearner.py', Path(__file__).resolve()]},
              'scope': 'P0 module outputs/input and parameter gradients/RNG/buffers; no optimizer or full-training speed claim',
              'BN_scope': 'native Future has no BatchNorm; a real BatchNorm1d stress fixture checks non-reuse of running state'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'controls'}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
