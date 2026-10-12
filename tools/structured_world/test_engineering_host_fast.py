"""CPU controls for the independent host-only pilot; no GPU or dispatch.

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/test_engineering_host_fast.py

Uses the local real processor and tiny Qwen/future modules. CPU pinning is
mocked; real CUDA transfer, full-model numerics and throughput remain untested.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import ast
import copy
from contextlib import nullcontext
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from starVLA.model.modules.structured_world.execution import configure_execution
from starVLA.model.modules.structured_world.future import SingleFrameFuture
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.dataloader.structured_world.prepared_current import transfer_optimizer_batch
from tools.structured_world import benchmark_engineering_execution as benchmark
from tools.structured_world.test_qwen_execution import _vision, repository
from tools.structured_world import test_qwen_current_processor as current

torch.set_num_threads(1)


def model_shell():
    language = SimpleNamespace(gradient_checkpointing_enable=lambda: None,
                               gradient_checkpointing_disable=lambda: None)
    return SimpleNamespace(qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(
        model=SimpleNamespace(language_model=language, visual=_vision(repository)))),
        action_model=SimpleNamespace(model=SimpleNamespace()),
        geometry=SimpleNamespace(), future_space=SimpleNamespace(), refiner=SimpleNamespace())


def parser_prefix(path, argv, *, trainer=False):
    """Execute the actual CLI/guard prefix, stopping BEFORE policy/GPU setup."""
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'policy' for t in n.targets))
    import argparse
    from tools.structured_world.native_capture import validate_args
    from tools.structured_world.world_scaling_profile import validate_args as validate_scaling_args
    namespace = dict(argparse=argparse, Path=Path, validate_capture_args=validate_args,
                     validate_scaling_args=validate_scaling_args)
    with patch.object(sys, 'argv', [str(path), *argv]):
        exec(compile(ast.Module(body=main.body[:stop], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class HostExecutionTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_mode_split_and_switching_from_every_old_mode(self):
        model = model_shell()
        for old in ('reference', 'io_preserving_v1', 'loss_preserving_v1',
                    'engineering_fast', 'engineering_timevec', 'precision_fast'):
            old_record = configure_execution(model, old)
            old_fast = old in ('engineering_fast', 'engineering_timevec', 'precision_fast')
            self.assertEqual(model.execution_fast, old_fast)
            self.assertEqual(model.execution_prepare_current, old_fast)
            self.assertEqual(old_record['query_and_body_execution_reuse'], old_fast)
            record = configure_execution(model, 'engineering_host_fast')
            self.assertTrue(model.execution_prepare_current)
            self.assertFalse(model.execution_fast)
            self.assertFalse(model.future_space.execution_fast)
            self.assertFalse(model.future_space.future_time_vectorized)
            self.assertFalse(model.refiner.fgtr_time_vectorized)
            self.assertFalse(model.geometry.conv_autocast_bf16)
            self.assertTrue(record['typed_current_CPU_processor_and_pinned_batch_transfer'])
            self.assertTrue(record['formal_acceptance_required'])
            self.assertFalse(record['future_shared_current_spatial_mix_and_action_MLN'])
            self.assertFalse(record['query_and_body_execution_reuse'])
            self.assertFalse(record['time_vectorization'])
            self.assertTrue(model.action_model.model.gradient_checkpointing)
        with self.assertRaises(ValueError): configure_execution(model, 'unregistered_host_mode')

    def test_configuration_preserves_state_and_RNG(self):
        model = model_shell()
        visual = model.qwen_vl_interface.model.model.visual
        before = {k: v.clone() for k, v in visual.state_dict().items()}
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        configure_execution(model, 'engineering_host_fast')
        self.assertEqual(python, random.getstate())
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertEqual(numpy[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(before.keys(), visual.state_dict().keys())
        for k, v in visual.state_dict().items(): self.assertTrue(torch.equal(before[k], v))

    def test_actual_Qwen_adapter_forward_gradients_and_RNG(self):
        original = _vision(repository).train()
        shell = model_shell()
        shell.qwen_vl_interface.model.model.visual = copy.deepcopy(original)
        configure_execution(shell, 'engineering_host_fast')
        grid = torch.tensor([[1, 4, 4]]*3)
        pixels = torch.randn(48, 12)
        runs = []
        for visual in (original, shell.qwen_vl_interface.model.model.visual):
            torch.manual_seed(42)
            output = visual(pixels, grid)
            (output[0].square().sum()+sum(v.square().sum() for v in output[1])).backward()
            runs.append((output, [p.grad.clone() for p in visual.parameters()], torch.get_rng_state()))
        for a, b in zip(runs[0][0][1], runs[1][0][1]): self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(runs[0][0][0], runs[1][0][0]))
        for a, b in zip(runs[0][1], runs[1][1]): self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(runs[0][2], runs[1][2]))

    def test_actual_future_reference_forward_all_gradients_and_RNG(self):
        for steps in (6, 8):
            original = SingleFrameFuture(24, steps, channels=32, scene_tokens=4,
                grid=GridSpec(xmin=-2, xmax=3, ymin=-2, ymax=2, dx=1, dy=1)).train()
            shell = model_shell(); shell.future_space = copy.deepcopy(original)
            configure_execution(shell, 'engineering_host_fast')
            tensors = (torch.randn(2, 32, 4, 5), torch.randn(2, 7, 24), torch.randn(2, steps, 3))
            runs = []
            for future in (original, shell.future_space):
                inputs = [t.clone().requires_grad_(i < 2) for i, t in enumerate(tensors)]
                torch.manual_seed(87)
                output = future(*inputs); output.square().sum().backward()
                runs.append((output, [p.grad.clone() if p.grad is not None else None for p in future.parameters()],
                    [t.grad for t in inputs], torch.get_rng_state()))
            self.assertTrue(torch.equal(runs[0][0], runs[1][0]))
            for left, right in zip(runs[0][1:3], runs[1][1:3]):
                for a, b in zip(left, right):
                    if a is None: self.assertIsNone(b)
                    else: self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(runs[0][3], runs[1][3]))

    def test_wrapper_query_loss_calls_ignore_prepared_candidates_and_keep_all_gradients(self):
        from starVLA.model.modules.structured_world.queries import sample_queries
        from starVLA.model.modules.structured_world.losses import compute_structured_losses
        from starVLA.model.modules.structured_world.semantics import StructuredSemantics
        from tools.structured_world.test_query_execution import labels, GRID, STATS, BODY
        tree = ast.parse((ROOT/'starVLA/model/framework/vla_structured_fgtr.py').read_text())
        calls = {n.func.id: compile(ast.Expression(n), '<actual wrapper call>', 'eval')
                 for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in ('sample_queries', 'compute_structured_losses')}
        for semantic_mode in ('none', 'full', 'event'):
            targets = labels(steps=6, batch=2)
            targets['ego_bodies'] = [BODY]*2
            initial = StructuredSemantics(semantic_mode, channels=2)
            current_field = torch.randn(2, 2, GRID.height, GRID.width)
            future_field = torch.randn(2, 6, 2, GRID.height, GRID.width)
            runs = []
            for mode in ('io_preserving_v1', 'engineering_host_fast'):
                shell = model_shell(); configure_execution(shell, mode)
                shell.semantic_mode, shell.query_mode, shell.grid = semantic_mode, 'local', GRID
                shell.structured_options = {'queries': 16, 'class_statistics': STATS}
                target = dict(targets)
                if mode == 'engineering_host_fast':
                    # A reference sampler must never validate or consume this.
                    target['query_candidates'] = [object()]*2
                generator = torch.Generator().manual_seed(17)
                rng = torch.get_rng_state().clone()
                model = copy.deepcopy(initial)
                namespace = dict(self=shell, targets=target, planned={'q0': torch.zeros(2, 6, 3)},
                    query_generator=generator, sample_queries=sample_queries,
                    compute_structured_losses=compute_structured_losses, scenes=2)
                query = eval(calls['sample_queries'], namespace)
                prediction = model(current_field, future_field)
                namespace.update(query=query, semantic=prediction)
                losses, counts = eval(calls['compute_structured_losses'], namespace)
                (sum(losses.values())+prediction['future_logits'].sum()*0.).backward()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                runs.append((query, losses, counts, [p.grad for p in model.parameters()], generator.get_state()))
            for a, b in zip(runs[0][:3], runs[1][:3]):
                for key in a:
                    if torch.is_tensor(a[key]): self.assertTrue(torch.equal(a[key], b[key]))
                    else: self.assertEqual(a[key], b[key])
            for a, b in zip(runs[0][3], runs[1][3]):
                if a is None: self.assertIsNone(b)
                else: self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(runs[0][4], runs[1][4]))

    def test_trainer_actual_parser_defaults_and_formal_rejection(self):
        path = ROOT/'tools/structured_world/train_vla.py'
        base = ['--config', '/unused/config', '--cache', '/unused/cache', '--output', '/unused/output',
                '--scope', 'profile', '--debug-updates', '32']
        parsed = parser_prefix(path, base+['--execution-mode', 'engineering_host_fast'], trainer=True)
        self.assertTrue(parsed['fast'])
        self.assertEqual((parsed['args'].logging_every, parsed['args'].evidence_every), (20, 100))
        for mode in ('engineering_host_fast', 'engineering_fast', 'engineering_timevec', 'precision_fast'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                parser_prefix(path, base+['--scope', 'formal', '--execution-mode', mode], trainer=True)
        self.assertIn('engineering_host_fast', benchmark.CANDIDATES)

    def test_numerical_probe_actual_parser_and_prepared_route(self):
        path = ROOT/'tools/structured_world/validate_engineering_execution.py'
        argv = ['--run', '/unused/run', '--cache', '/unused/cache', '--dino-root', '/unused/dino',
                '--output', '/unused/output', '--tag', 'milestone_0025000',
                '--optimized-mode', 'engineering_host_fast']
        self.assertEqual(parser_prefix(path, argv)['args'].optimized_mode, 'engineering_host_fast')
        tree = ast.parse(path.read_text())
        route = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Tuple) and [v.id for v in t.elts] == ['obs', 'tgt'] for t in n.targets))
        for mode, expected in [('engineering_host_fast', ('prepared', 'prepared_targets')),
                               ('io_preserving_v1', ('original', 'original_targets'))]:
            ns = dict(mode=mode, prepared_observations='prepared', prepared_targets='prepared_targets',
                      observations='original', targets='original_targets')
            exec(compile(ast.Module(body=[route], type_ignores=[]), str(path), 'exec'), ns)
            self.assertEqual((ns['obs'], ns['tgt']), expected)


class PreparedHostRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): current.CurrentProcessorTests.setUpClass()

    @classmethod
    def tearDownClass(cls): current.CurrentProcessorTests.tearDownClass()

    def setUp(self):
        self.fixture = current.CurrentProcessorTests()
        self.fixture.setUp()

    def tearDown(self): self.assertFalse(torch.cuda.is_initialized())

    def test_actual_collator_transfer_Qwen_route_validation_and_legacy_fallback(self):
        rows = self.fixture.rows(3, 3)
        collator, observations, targets = self.fixture.prepare(rows, 2)
        _, moved = transfer_optimizer_batch(observations, targets, torch.device('cpu'))
        for key in targets:
            if torch.is_tensor(targets[key]): self.assertTrue(torch.equal(targets[key], moved[key]))
        model = VLAStructuredFGTR.__new__(VLAStructuredFGTR); nn.Module.__init__(model)
        model.execution_fast = False; model.execution_prepare_current = True
        model.current_processor_identity = collator.identity
        model.qwen_vl_interface = self.fixture.interface
        model.amp = lambda: nullcontext()
        self.fixture.core.get_image_features = lambda pixels, grid: ([pixels], [])
        micro = observations[:2]; payload = micro[0]['_prepared_current']
        instructions = self.fixture.instructions(micro)
        def encode(owner, examples): return owner._build_qwen_batch(examples, instructions)
        with patch.object(DDPForesight, 'encode_current', encode):
            result = model.encode_current(micro)
            self.assertTrue(torch.equal(result[0], payload.inputs['input_ids']))
            self.assertTrue(torch.equal(result[2], payload.position_ids))
            self.assertIsNone(model._active_prepared_current)
            with self.assertRaises(ValueError): model.encode_current([micro[0], observations[2]])
            changed = [dict(row) for row in micro]; changed[0]['image'] = list(reversed(changed[0]['image']))
            with self.assertRaises(ValueError): model.encode_current(changed)
            self.assertIsNone(model._active_prepared_current)
        # Existing callers that set only execution_fast still consume payloads.
        del model.execution_prepare_current; model.execution_fast = True
        with patch.object(DDPForesight, 'encode_current', lambda owner, examples: owner._active_prepared_current):
            self.assertIs(model.encode_current(micro), payload)
            model.execution_fast = False
            self.assertIsNone(model.encode_current(micro))

    def test_geometry_prepared_route_matches_native_values_and_all_gradients(self):
        _, observations, _ = self.fixture.prepare(self.fixture.rows(3, 2), 2)
        transfer_optimizer_batch(observations, {}, torch.device('cpu'))
        class Geometry(nn.Module):
            def __init__(self): super().__init__(); self.scale = nn.Parameter(torch.tensor(1.25))
            def forward(self, images, calibration, valid, *, canonical_fp32):
                return self.scale*(images.sum()+sum(t.sum() for t in calibration.values())+valid.sum())
        model = VLAStructuredFGTR.__new__(VLAStructuredFGTR); nn.Module.__init__(model)
        model.geometry = Geometry(); model.execution_fast = False
        values = []
        for prepared in (False, True):
            model.execution_prepare_current = prepared; model.zero_grad()
            value = model.geometric_current(observations); value.backward()
            values.append((value.detach().clone(), model.geometry.scale.grad.clone()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*values)))


if __name__ == '__main__': unittest.main(verbosity=2)
