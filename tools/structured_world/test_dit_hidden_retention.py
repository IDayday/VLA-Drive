"""CPU controls for removing unused Python retention of DiT hidden outputs.

Tests native forwards, gradients, dropout, checkpoint admission and two AdamW
updates against an independently loaded pinned implementation. Weak references
measure intermediate tensor lifetime in no_grad inference, not GPU peak memory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
import weakref

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.action_model.flow_matching_head import cross_attention_dit as candidate_module

BASE = '267701f3b2c6cf36a97ff3a5e6bfb535e5aab232'
PATH = 'starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py'
RESULTS = []


def original_module():
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE+':'+PATH], text=True)
    module = types.ModuleType('original_hidden_retention_dit')
    exec(compile(source, BASE+':'+PATH, 'exec'), module.__dict__)
    return module


def assert_exact(value, other):
    if isinstance(value, torch.Tensor):
        torch.testing.assert_close(value, other, rtol=0, atol=0, equal_nan=True)
    elif isinstance(value, dict):
        assert list(value) == list(other)
        for key in value:
            assert_exact(value[key], other[key])
    elif isinstance(value, (list, tuple)):
        assert type(value) == type(other) and len(value) == len(other)
        for a, b in zip(value, other):
            assert_exact(a, b)
    else:
        assert value == other


def tensor_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_copy(v) for v in value)
    return value


class DiTHiddenRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = original_module()

    def pair(self, *, kind='DiT', train=True, checkpoint=False):
        torch.manual_seed(12011)
        args = dict(num_attention_heads=4, attention_head_dim=8, output_dim=4,
                    num_layers=4, dropout=.2, final_dropout=True,
                    max_num_positional_embeddings=20)
        if kind == 'DiT':
            args.update(cross_attention_dim=24, interleave_self_attention=True)
        original = getattr(self.old, kind)(**args).train(train)
        candidate = getattr(candidate_module, kind)(**args).train(train)
        self.assertEqual(list(original.state_dict()), list(candidate.state_dict()))
        self.assertEqual(list(dict(original.named_parameters())), list(dict(candidate.named_parameters())))
        candidate.load_state_dict(original.state_dict(), strict=True)
        original.gradient_checkpointing = candidate.gradient_checkpointing = checkpoint
        hidden = torch.randn(2, 8, 32, requires_grad=True)
        if kind == 'DiT':
            values = (hidden, torch.randn(2, 7, 24, requires_grad=True), torch.tensor([43, 817]))
        else:
            values = (hidden,)
        return original, candidate, values

    def run_steps(self, model, values, *, collect=False, masks=False):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.001)
        records = []
        for step in range(2):
            inputs = [v.detach().clone().requires_grad_(v.requires_grad) for v in values]
            kwargs = {'return_all_hidden_states': collect}
            if masks:
                state = torch.ones(2, 8, dtype=torch.bool)
                state[:, -2:] = False
                condition = torch.ones(2, 7, dtype=torch.bool)
                condition[:, -1] = False
                with torch.no_grad():
                    inputs[0][~state] = float('nan')
                    inputs[1][~condition] = float('nan')
                kwargs.update(active_token_mask=state, condition_token_mask=condition)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(13011+step)
            output = model(*inputs, **kwargs)
            if collect:
                value, all_hidden = output
                self.assertEqual(len(all_hidden), len(model.transformer_blocks)+1)
                self.assertTrue(all(v.requires_grad for v in all_hidden))
            else:
                value = output
                all_hidden = []
            weights = torch.linspace(.2, 1.2, value.numel()).reshape_as(value)
            loss = (value*weights).mean()+.1*value.square().mean()
            # The requested hidden outputs remain live and differentiable.
            for hidden in all_hidden:
                loss = loss+.01*hidden.square().mean()
            loss.backward()
            gradient = {name: None if p.grad is None else p.grad.detach().clone()
                        for name, p in model.named_parameters()}
            input_grad = [None if v.grad is None else v.grad.detach().clone() for v in inputs]
            for item in input_grad[:2 if len(inputs) > 1 else 1]:
                self.assertIsNotNone(item)
                self.assertTrue(bool(torch.isfinite(item).all()))
                self.assertGreater(float(item.abs().sum()), 0.)
            optimizer.step()
            records.append({'output': tensor_copy(output), 'input_gradients': input_grad,
                            'parameter_gradients': gradient, 'RNG': torch.get_rng_state().clone(),
                            'parameters': copy.deepcopy(model.state_dict()),
                            'optimizer': copy.deepcopy(optimizer.state_dict())})
        return records

    def test_outputs_gradients_optimizer_and_requested_hidden_states(self):
        for kind in ('DiT', 'SelfAttentionTransformer'):
            for train in (False, True):
                for collect in (False, True):
                    checkpoints = (False, True) if kind == 'DiT' else (False,)
                    for checkpoint in checkpoints:
                        with self.subTest(kind=kind, train=train, collect=collect, checkpoint=checkpoint):
                            original, candidate, values = self.pair(kind=kind, train=train, checkpoint=checkpoint)
                            reference = self.run_steps(copy.deepcopy(original), values, collect=collect)
                            replay = self.run_steps(copy.deepcopy(original), values, collect=collect)
                            actual = self.run_steps(candidate, values, collect=collect)
                            assert_exact(replay, reference)
                            assert_exact(actual, reference)
                            RESULTS.append({'case': f'{kind}_train{train}_collect{collect}_checkpoint{checkpoint}',
                                            'steps': 2, 'A_A_bitwise': True, 'output_input_parameter_gradients_optimizer_RNG_bitwise': True})

    def test_joint_actor_masks_keep_cleaning_and_gradients(self):
        original, candidate, values = self.pair(checkpoint=True)
        reference = self.run_steps(original, values, masks=True)
        actual = self.run_steps(candidate, values, masks=True)
        assert_exact(actual, reference)
        RESULTS.append({'case': 'joint_actor_inactive_NaN_masks', 'steps': 2,
                        'output_input_parameter_gradients_optimizer_RNG_bitwise': True})

    def test_checkpoint_admission_guard_unchanged(self):
        expected = candidate_module.checkpoint
        try:
            for train in (False, True):
                for grad_enabled in (False, True):
                    _, model, values = self.pair(train=train, checkpoint=True)
                    calls = []
                    def observe(*args, **kwargs):
                        calls.append(kwargs['use_reentrant'])
                        return expected(*args, **kwargs)
                    candidate_module.checkpoint = observe
                    with torch.set_grad_enabled(grad_enabled):
                        model(*values)
                    self.assertEqual(calls, [False]*4 if train and grad_enabled else [])
        finally:
            candidate_module.checkpoint = expected
        RESULTS.append({'case': 'checkpoint_train_grad_admission_4_combinations', 'passed': True})

    def lifetime(self, model, values, *, collect):
        refs, live, handles = [], [], []
        for block in model.transformer_blocks:
            def observe(_module, _inputs, output):
                refs.append(weakref.ref(output))
                live.append(sum(ref() is not None for ref in refs))
            handles.append(block.register_forward_hook(observe))
        try:
            with torch.no_grad():
                output = model(*values, return_all_hidden_states=collect)
            retained = sum(ref() is not None for ref in refs)
        finally:
            for handle in handles:
                handle.remove()
        return live, retained, tensor_copy(output)

    def test_unrequested_outputs_are_actually_released(self):
        for kind in ('DiT', 'SelfAttentionTransformer'):
            original, candidate, values = self.pair(kind=kind, train=False)
            old_live, old_retained, old_output = self.lifetime(original, values, collect=False)
            new_live, new_retained, new_output = self.lifetime(candidate, values, collect=False)
            assert_exact(new_output, old_output)
            self.assertEqual(old_live, [1, 2, 3, 4])
            # The previous result is still an input during the next block's hook.
            self.assertLessEqual(max(new_live), 2)
            self.assertEqual(old_retained, new_retained)
            requested_live, requested_retained, _ = self.lifetime(candidate, values, collect=True)
            self.assertEqual(requested_live, [1, 2, 3, 4])
            self.assertEqual(requested_retained, 4)
            RESULTS.append({'case': f'{kind}_native_no_grad_weakref_lifetime',
                            'reference_live_intermediate_outputs': old_live,
                            'candidate_live_intermediate_outputs': new_live,
                            'requested_hidden_outputs_retained': requested_retained,
                            'scope': 'tensor object lifetime, no GPU allocator/peak claim'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DiTHiddenRetentionTests))
    report = {'base_source': BASE, 'device': 'cpu', 'torch': torch.__version__,
              'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'controls': RESULTS,
              'source_sha256': {PATH: hashlib.sha256((ROOT/PATH).read_bytes()).hexdigest(),
                                str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'scope': 'exact CPU outputs/input+parameter gradients/two AdamW steps/dropout RNG/checkpoint guard and native tensor lifetime; no GPU peak claim'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'controls'}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
