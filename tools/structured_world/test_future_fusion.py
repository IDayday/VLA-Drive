"""P1 full time fusion: native GN/dense and one shared spatial mix.

CPU is the default. Parent must separately admit any CUDA probe. Independent
pinned forwards provide outputs and all input/parameter gradient controls.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import test_time_vectorization as controls
from tools.structured_world import test_future_execution as p0_controls
from starVLA.model.modules.structured_world.future import SingleFrameFuture
from third_party.resworld.ported.tokenlearner import TokenFuser

RESULTS = []


class FuserSequence(nn.Module):
    """Test adapter for the real fuser; parameters/operations remain native."""
    def __init__(self, fuser, *, vectorized):
        super().__init__()
        self.fuser, self.vectorized = fuser, vectorized

    def forward(self, tokens, current):
        if not self.vectorized:
            return torch.stack([self.fuser(tokens[:, t], current) for t in range(tokens.shape[1])], 1)
        batch, times, count, channels = tokens.shape
        assert self.fuser.can_batch_time_fusion()
        values = self.fuser.prepare_token_values(tokens.reshape(batch*times, count, channels))
        mix = self.fuser.prepare_spatial_mix(current)
        assert values.requires_grad and mix.requires_grad
        values = values.reshape(batch, times, channels, -1)
        return self.fuser.fuse_time(values, mix, current)


class FutureFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OldFuture, _ = controls.original_classes()
        _, cls.OldFuser = p0_controls.original_classes()

    def pair(self, steps, *, train=True, chunk=0, fast=True, final_dropout=0., spatial_dropout=0., token_bn=False):
        torch.manual_seed(11111)
        from starVLA.model.modules.structured_world.grid import GridSpec
        original = self.OldFuture(24, steps, channels=32, scene_tokens=4,
                                  grid=GridSpec(xmin=-2, xmax=3, ymin=-2, ymax=2, dx=1, dy=1))
        with torch.no_grad():
            for layer in (original.action_mln.gamma, original.action_mln.beta):
                layer.weight.normal_(std=.08)
                layer.bias.normal_(std=.08)
        original.tokenfuser.dropout.p = final_dropout
        original.tokenfuser.mlp.dropout.p = spatial_dropout
        original.tokenfuser.mlp.dropout2.p = spatial_dropout
        if token_bn:
            original.tokenfuser.norm1 = nn.BatchNorm1d(32)
        candidate = SingleFrameFuture(24, steps, channels=32, scene_tokens=4, grid=original.grid)
        for name, child in candidate.named_modules():
            other = dict(original.named_modules()).get(name)
            if isinstance(child, nn.modules.dropout._DropoutNd):
                child.p = other.p
        if token_bn:
            candidate.tokenfuser.norm1 = copy.deepcopy(original.tokenfuser.norm1)
        self.assertEqual(list(candidate.state_dict()), list(original.state_dict()))
        self.assertEqual(list(dict(candidate.named_parameters())), list(dict(original.named_parameters())))
        candidate.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(candidate.future_time_vectorized)
        candidate.execution_fast, candidate.future_time_vectorized = fast, True
        candidate.future_time_chunk_size = chunk
        original.to(controls.DEVICE).train(train)
        candidate.to(controls.DEVICE).train(train)
        values = (torch.randn(3, 32, 4, 5, device=controls.DEVICE, requires_grad=True),
                  torch.randn(3, 7, 24, device=controls.DEVICE, requires_grad=True),
                  torch.randn(3, steps, 3, device=controls.DEVICE))
        return original, candidate, values

    def control(self, label, original, candidate, values):
        reference = controls.snapshot(copy.deepcopy(original), values)
        replay = controls.snapshot(copy.deepcopy(original), values)
        aa = controls.compare(replay, reference, exact=controls.DEVICE.type == 'cpu')
        calls, handles = {}, []
        fuser = candidate.tokenfuser if isinstance(candidate, SingleFrameFuture) else candidate.fuser
        for name, module in fuser.named_modules():
            if name in ('norm', 'norm1', 'norm2', 'dense', 'mlp', 'dropout'):
                calls[name] = []
                def observe(_module, args, name=name):
                    calls[name].append(list(args[0].shape))
                handles.append(module.register_forward_pre_hook(observe))
        try:
            actual = controls.snapshot(candidate, values)
        finally:
            for handle in handles:
                handle.remove()
        ab = controls.compare(actual, reference)
        self.assertTrue(bool(actual['input_gradients'][0].abs().sum() > 0))
        self.assertTrue(bool(actual['input_gradients'][1].abs().sum() > 0))
        RESULTS.append({'case': label, 'A_A': aa, 'A_B': ab, 'fuser_calls': calls,
                        'reference_seconds': reference['seconds'], 'candidate_seconds': actual['seconds']})
        return actual, calls

    def test_future_T6_T8_full_and_chunked_fusion(self):
        for steps in (6, 8):
            for train in (False, True):
                for chunk in (0, 4):
                    original, candidate, values = self.pair(steps, train=train, chunk=chunk)
                    actual, calls = self.control(f'future_T{steps}_train{train}_chunk{chunk}', original, candidate, values)
                    for name in ('norm', 'norm1', 'norm2', 'dense', 'mlp', 'dropout'):
                        self.assertEqual(len(calls[name]), 1)
                    self.assertEqual(calls['norm1'][0], [3*steps, 32, 4])
                    self.assertEqual(calls['norm2'][0], [3*steps, 32, 64])
                    self.assertEqual(calls['dense'][0], [3*steps, 32, 4])
                    self.assertEqual(calls['norm'][0], [3, 32, 20])
                    self.assertEqual(calls['dropout'][0], [3, steps, 20, 32])
                    self.assertEqual(tuple(actual['output']['future'].shape), (3, steps, 32, 4, 5))

    def test_time_flag_alone_admits_shared_mix(self):
        for steps in (6, 8):
            original, candidate, values = self.pair(steps, fast=False)
            self.assertFalse(candidate.execution_fast)
            _, calls = self.control(f'future_T{steps}_without_P0_flag', original, candidate, values)
            self.assertEqual(len(calls['mlp']), 1)
            self.assertEqual(len(calls['dense']), 1)

    def test_native_fuser_shapes_GN_toggle_and_spatial_gradients(self):
        for steps in (6, 8):
            for spatial_4d in (False, True):
                for normalized in (False, True):
                    torch.manual_seed(21111)
                    old_fuser = self.OldFuser(4, 32, use_normalization=normalized)
                    new_fuser = TokenFuser(4, 32, use_normalization=normalized)
                    self.assertEqual(list(old_fuser.state_dict()), list(new_fuser.state_dict()))
                    new_fuser.load_state_dict(old_fuser.state_dict(), strict=True)
                    original = FuserSequence(old_fuser, vectorized=False).to(controls.DEVICE)
                    candidate = FuserSequence(new_fuser, vectorized=True).to(controls.DEVICE)
                    spatial_shape = (3, 32, 4, 5) if spatial_4d else (3, 20, 32)
                    values = (torch.randn(3, steps, 4, 32, device=controls.DEVICE, requires_grad=True),
                              torch.randn(*spatial_shape, device=controls.DEVICE, requires_grad=True))
                    actual, calls = self.control(f'fuser_T{steps}_4D{spatial_4d}_GN{normalized}', original, candidate, values)
                    for key in ('fuser.dense.weight', 'fuser.mlp.fc1.weight', 'fuser.mlp.fc2.weight'):
                        self.assertGreater(float(actual['parameter_gradients'][key].abs().sum()), 0.)
                    self.assertEqual(len(calls['dense']), 1)

    def test_live_dropout_and_real_BN_keep_per_time_fusion(self):
        for steps in (6, 8):
            for fixture in ('final_dropout', 'spatial_dropout', 'token_bn'):
                kwargs = {fixture: True if fixture == 'token_bn' else .2}
                original, candidate, values = self.pair(steps, **kwargs)
                self.assertFalse(candidate.tokenfuser.can_batch_time_fusion())
                actual, calls = self.control(f'future_T{steps}_{fixture}_fallback', original, candidate, values)
                self.assertEqual(len(calls['dense']), steps)
                self.assertEqual(len(calls['dropout']), steps)
                if fixture == 'spatial_dropout':
                    self.assertEqual(len(calls['mlp']), steps)
                if fixture == 'token_bn':
                    self.assertEqual(actual['buffers']['tokenfuser.norm1.num_batches_tracked'].item(), steps)
        original, candidate, values = self.pair(8, train=False, final_dropout=.2, spatial_dropout=.2)
        self.assertTrue(candidate.tokenfuser.can_batch_time_fusion())
        _, calls = self.control('eval_inactive_dropout_time_fusion', original, candidate, values)
        self.assertEqual(len(calls['dropout']), 1)

    def test_fused_spatial_reshape_is_a_view_and_preserves_axes(self):
        torch.manual_seed(31111)
        fuser = TokenFuser(4, 32).to(controls.DEVICE)
        values = torch.randn(3, 8, 32, 64, device=controls.DEVICE, requires_grad=True)
        mix = torch.randn(3, 20, 64, device=controls.DEVICE, requires_grad=True)
        current = torch.randn(3, 32, 4, 5, device=controls.DEVICE)
        observed = []
        handle = fuser.dropout.register_forward_hook(lambda _m, _a, output: observed.append(output))
        try:
            spatial = fuser.fuse_time(values, mix, current)
        finally:
            handle.remove()
        self.assertEqual(spatial.untyped_storage().data_ptr(), observed[0].untyped_storage().data_ptr())
        self.assertEqual(spatial.numel(), observed[0].numel())
        for b in range(3):
            for t in range(8):
                expected = fuser.fuse(values[b:b+1, t], mix[b:b+1], current[b:b+1])
                torch.testing.assert_close(spatial[b:b+1, t], expected, rtol=controls.RTOL, atol=controls.ATOL)
        spatial.square().mean().backward()
        self.assertGreater(float(values.grad.abs().sum()), 0.)
        self.assertGreater(float(mix.grad.abs().sum()), 0.)
        RESULTS.append({'case': 'fuse_time_spatial_reshape_storage_alias_and_axes',
                        'reshape_shares_einsum_storage': True, 'value_and_shared_mix_gradients_live': True,
                        'scope': 'no repeated spatial bank or reshape copy; output allocation remains necessary'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    controls.DEVICE = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FutureFusionTests))
    paths = ['starVLA/model/modules/structured_world/future.py', 'third_party/resworld/ported/tokenlearner.py',
             'tools/structured_world/test_future_fusion.py', 'tools/structured_world/test_time_vectorization.py']
    report = {'base_source': controls.BASE, 'device': str(controls.DEVICE), 'torch': torch.__version__,
              'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'controls': RESULTS,
              'numeric_tolerance': {'atol': controls.ATOL, 'rtol': controls.RTOL},
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              'flag': 'future_time_vectorized remains False by default; P0 alone does not enable full fusion',
              'scope': 'FP32 module CPU outputs/all input+parameter gradients/exact RNG+buffers/storage alias; no full-model speed or GPU peak qualification'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'controls'}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
