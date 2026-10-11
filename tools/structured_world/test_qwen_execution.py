"""Owned CPU-only tests of the actual pinned Qwen implementations.

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/test_qwen_execution.py
No pretrained weights, network requests, CUDA initialization or GPU kernels.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import io
from pathlib import Path
import random
import sys
from types import MethodType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl as upstream
from starVLA.model.modules.vlm.qwen3_vl import modeling_qwen3_vl as repository
from starVLA.model.modules.vlm.QWen3 import Qwen3VLForConditionalGeneration
from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from starVLA.model.modules.structured_world import qwen_execution as adapter

torch.set_num_threads(1)


def _vision(module, backend="sdpa"):
    cfg = module.Qwen3VLVisionConfig(
        depth=3, hidden_size=16, intermediate_size=32, num_heads=2,
        out_hidden_size=16, patch_size=2, spatial_merge_size=2,
        temporal_patch_size=1, num_position_embeddings=16,
        deepstack_visual_indexes=[0, 2],
    )
    cfg._attn_implementation = backend
    return module.Qwen3VLVisionModel(cfg).eval()


def _inputs(rows):
    grid = torch.tensor(rows, dtype=torch.int64)
    pixels = torch.randn(int(grid.prod(-1).sum()), 12, generator=torch.Generator().manual_seed(707))
    return pixels, grid


def _inventory(model):
    return {
        "parameters": [(k, id(v), tuple(v.shape), v.dtype, v.requires_grad) for k, v in model.named_parameters()],
        "buffers": [(k, id(v), tuple(v.shape), v.dtype) for k, v in model.named_buffers()],
        "state": [(k, v.clone()) for k, v in model.state_dict().items()],
        "modes": [(k, v.training) for k, v in model.named_modules()],
    }


def _full_model():
    cfg = Qwen3VLConfig(
        image_token_id=60, vision_start_token_id=61, vision_end_token_id=62,
        text_config={
            "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 3, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "out_hidden_size": 16, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 1,
            "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 2],
        },
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(cfg).eval()


class QwenExecutionTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
            self.assertEqual(left.dtype, right.dtype)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left: self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_runtime_classes_provenance_and_install_rng(self):
        model = _full_model()
        vision = model.model.visual
        self.assertEqual(transformers.__version__, "4.57.0")
        self.assertEqual(type(vision).__module__, repository.__name__)
        before = _inventory(model)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        record = adapter.install_qwen_vision_lengths(vision)
        self.assertEqual(record["attention_layers"], 3)
        self.assertEqual(record["upstream"][0]["source_file_sha256"], adapter._MODULE_HASHES[repository.__name__])
        self.assertEqual(record["adapter_sha256"], hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
        self.assertIn("_structured_world_lengths_cpu", record["upstream"][0]["patch"])
        self.assert_nested_equal(before, _inventory(model))
        self.assertEqual(rng[0], random.getstate())
        self.assertTrue(np.array_equal(rng[1][1], np.random.get_state()[1]))
        self.assertEqual(rng[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(rng[2], torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_three_six_cameras_heterogeneous_grids_and_temporal_chunks(self):
        layouts = [
            [[1, 4, 4]] * 3,
            [[1, 4, 4]] * 6,
            [[1, 4, 4], [1, 4, 6], [1, 6, 4]],
            [[2, 4, 6], [1, 6, 4], [3, 4, 4], [1, 4, 6], [1, 6, 6], [2, 6, 4]],
        ]
        for module in (repository, upstream):
            original = _vision(module)
            changed = copy.deepcopy(original)
            adapter.install_qwen_vision_lengths(changed)
            for rows in layouts:
                with self.subTest(module=module.__name__, grid=rows), torch.no_grad():
                    pixels, grid = _inputs(rows)
                    self.assert_nested_equal(original(pixels, grid), changed(pixels, grid))

    def test_one_tuple_and_same_object_for_every_layer_and_qkv(self):
        model = _vision(repository)
        pixels, grid = _inputs([[2, 4, 6], [1, 4, 4], [1, 6, 4]])
        tolist = torch.Tensor.tolist
        calls = []
        def observe_tolist(value):
            if value.ndim == 1 and value.dtype == torch.int32 and value.numel() == 4:
                calls.append(tolist(value))
            return tolist(value)
        with mock.patch.object(torch.Tensor, "tolist", observe_tolist), torch.no_grad():
            model(pixels, grid)
        self.assertEqual(calls, [[24, 24, 16, 24]] * (3 * len(model.blocks)))
        adapter.install_qwen_vision_lengths(model)
        split = torch.split
        objects = []
        def observe_split(value, lengths, *args, **kwargs):
            if value.ndim == 4 and kwargs.get("dim") == 2:
                objects.append(lengths)
            return split(value, lengths, *args, **kwargs)
        calls.clear()
        with mock.patch.object(torch.Tensor, "tolist", observe_tolist), \
                mock.patch.object(torch, "split", observe_split), torch.no_grad():
            model(pixels, grid)
        self.assertEqual(calls, [[24, 24, 16, 24]])
        self.assertEqual(len(objects), 3 * len(model.blocks))
        self.assertTrue(all(value is objects[0] for value in objects))
        self.assertEqual(objects[0], (24, 24, 16, 24))

    def test_fp32_and_bf16_cpu_forward_gradients_adam_rng_and_checkpoint(self):
        pixels, grid = _inputs([[1, 4, 4], [1, 4, 6], [1, 6, 4]])
        for module in (repository, upstream):
            for bf16, checkpointed in [(False, None), (True, None), (False, False), (True, False), (False, True), (True, True)]:
                with self.subTest(module=module.__name__, bf16=bf16, checkpointed=checkpointed):
                    original = _vision(module).train()
                    for block in original.blocks: block.attn.attention_dropout = .25
                    if checkpointed is not None:
                        original.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpointed})
                    changed = copy.deepcopy(original)
                    adapter.install_qwen_vision_lengths(changed)
                    runs = []
                    for model in (original, changed):
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                        torch.manual_seed(909)
                        context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
                        with context:
                            output = model(pixels, grid)
                            loss = output[0].square().sum() + sum(v.square().sum() for v in output[1])
                        loss.backward()
                        gradients = {k: p.grad.clone() if p.grad is not None else None for k, p in model.named_parameters()}
                        optimizer.step()
                        runs.append((output, gradients, model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
                    self.assert_nested_equal(runs[0], runs[1])

    def test_state_dict_optimizer_rng_restore_preserves_next_update(self):
        for module in (repository, upstream):
            with self.subTest(module=module.__name__):
                original = _vision(module).train()
                for block in original.blocks: block.attn.attention_dropout = .25
                optimizer = torch.optim.AdamW(original.parameters(), lr=1e-4)
                adapter.install_qwen_vision_lengths(original)
                pixels, grid = _inputs([[1, 4, 4], [1, 4, 6], [1, 6, 4]])

                def update(model, opt):
                    opt.zero_grad(set_to_none=True)
                    output = model(pixels, grid)
                    loss = output[0].square().sum() + sum(v.square().sum() for v in output[1])
                    loss.backward()
                    gradients = {k: p.grad.clone() if p.grad is not None else None for k, p in model.named_parameters()}
                    opt.step()
                    return output, gradients, copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), torch.get_rng_state().clone()

                torch.manual_seed(919)
                update(original, optimizer)
                stream = io.BytesIO()
                torch.save({"model": original.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, stream)
                stream.seek(0)
                checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
                self.assertFalse(any(adapter._STATE in key for key in checkpoint["model"]))

                # A checkpoint made with the adapter loads through the native
                # state contract, both with and without subsequent installation.
                restored = _vision(module).train()
                for block in restored.blocks: block.attn.attention_dropout = .25
                restored.load_state_dict(checkpoint["model"], strict=True)
                restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
                restored_optimizer.load_state_dict(checkpoint["optimizer"])
                adapter.install_qwen_vision_lengths(restored)
                adapter.uninstall_qwen_vision_lengths(original)
                torch.set_rng_state(checkpoint["rng"])
                expected = update(original, optimizer)
                torch.set_rng_state(checkpoint["rng"])
                actual = update(restored, restored_optimizer)
                self.assert_nested_equal(expected, actual)

    def test_actual_current_language_deepstack_and_mrope_unchanged(self):
        for cameras in (3, 6):
            with self.subTest(cameras=cameras), torch.no_grad():
                original = _full_model()
                changed = copy.deepcopy(original)
                adapter.install_qwen_vision_lengths(changed.model.visual)
                pixels, grid = _inputs(([[1, 4, 4], [1, 4, 6], [1, 6, 4]] * 2)[:cameras])
                tokens = [1]
                for row in grid:
                    tokens += [61] + [60] * (int(row.prod()) // 4) + [62, 2]
                ids = torch.tensor([tokens + [3]])
                mask = torch.ones_like(ids)
                self.assert_nested_equal(
                    original.model.get_rope_index(ids, grid, attention_mask=mask),
                    changed.model.get_rope_index(ids, grid, attention_mask=mask),
                )
                args = dict(input_ids=ids, attention_mask=mask, pixel_values=pixels, image_grid_thw=grid, use_cache=False)
                a, b = original.model(**args), changed.model(**args)
                self.assert_nested_equal(a.last_hidden_state, b.last_hidden_state)

    def test_backend_operator_kwargs_and_call_order_unchanged(self):
        for module in (repository, upstream):
            base = module.ALL_ATTENTION_FUNCTIONS["sdpa"]
            records = []
            def observe(owner, query, key, value, **kwargs):
                self.assertNotIn(adapter._METADATA, kwargs)
                records.append((query.shape, key.shape, value.shape, copy.deepcopy(kwargs)))
                return base(owner, query, key, value, **kwargs)
            with mock.patch.object(module, "ALL_ATTENTION_FUNCTIONS", {"sdpa": observe}):
                original = _vision(module)
                changed = copy.deepcopy(original)
                adapter.install_qwen_vision_lengths(changed)
                pixels, grid = _inputs([[1, 4, 4], [1, 4, 6], [1, 6, 4]])
                with torch.no_grad():
                    a = original(pixels, grid)
                    reference_records = records[:]
                    records.clear()
                    b = changed(pixels, grid)
                self.assert_nested_equal(a, b)
                self.assert_nested_equal(reference_records, records)
                self.assertEqual(len(records), 3 * len(original.blocks))

    def test_flash_branch_kept_native_without_metadata_transfer(self):
        for module in (repository, upstream):
            calls = []
            def fake_flash(owner, query, key, value, **kwargs):
                self.assertNotIn(adapter._METADATA, kwargs)
                calls.append((query.shape, kwargs["max_length_q"].item(), sorted(kwargs)))
                lengths = (kwargs["cu_seq_lens_q"][1:] - kwargs["cu_seq_lens_q"][:-1]).tolist()
                outputs = [torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=kwargs["scaling"]).transpose(1, 2)
                           for q, k, v in zip(*(torch.split(tensor, lengths, dim=2) for tensor in (query, key, value)))]
                return torch.cat(outputs, dim=1), None
            with mock.patch.object(module, "ALL_ATTENTION_FUNCTIONS", {"flash_attention_2": fake_flash}):
                original = _vision(module)
                original.config._attn_implementation = "flash_attention_2"
                changed = copy.deepcopy(original)
                adapter.install_qwen_vision_lengths(changed)
                pixels, grid = _inputs([[1, 4, 4]] * 3)
                with torch.no_grad():
                    a = original(pixels, grid)
                    expected = calls[:]
                    calls.clear()
                    with mock.patch.object(adapter, "_lengths_cpu", side_effect=AssertionError("Unexpected FA2 transfer")):
                        # Reinstall captures the spy helper into the compiled namespace.
                        adapter.uninstall_qwen_vision_lengths(changed)
                        adapter.install_qwen_vision_lengths(changed)
                        b = changed(pixels, grid)
                self.assert_nested_equal(a, b)
                self.assert_nested_equal(expected, calls)

    def test_standalone_attention_fallback_and_eager_backend(self):
        for backend in ("sdpa", "eager"):
            original = _vision(repository, backend)
            changed = copy.deepcopy(original)
            adapter.install_qwen_vision_lengths(changed)
            hidden = torch.randn(20, 16)
            positions = (torch.ones(20, 8), torch.zeros(20, 8))
            for lengths in ([4, 16], [8, 12]):
                cu = torch.tensor([0, lengths[0], 20], dtype=torch.int32)
                with torch.no_grad():
                    self.assert_nested_equal(
                        original.blocks[0].attn(hidden, cu, position_embeddings=positions),
                        changed.blocks[0].attn(hidden, cu, position_embeddings=positions),
                    )

    def test_idempotence_uninstall_and_other_instances_untouched(self):
        model = _vision(repository)
        other = _vision(repository)
        original_class_forward = type(model).forward
        inventory = _inventory(model)
        first = adapter.install_qwen_vision_lengths(model)
        installed = model.forward.__func__
        self.assertEqual(first, adapter.install_qwen_vision_lengths(model))
        self.assertIs(installed, model.forward.__func__)
        self.assertIs(type(model).forward, original_class_forward)
        self.assertIs(other.forward.__func__, original_class_forward)
        self.assert_nested_equal(inventory, _inventory(model))
        self.assertTrue(adapter.uninstall_qwen_vision_lengths(model))
        self.assertFalse(adapter.uninstall_qwen_vision_lengths(model))
        self.assertNotIn("forward", model.__dict__)
        self.assertTrue(all("forward" not in block.attn.__dict__ for block in model.blocks))
        self.assert_nested_equal(inventory, _inventory(model))
        self.assertEqual(first, adapter.install_qwen_vision_lengths(model))

    def test_unknown_version_source_or_existing_adapter_refused_before_mutation(self):
        for fault in ("version", "hash", "instance_forward", "class"):
            with self.subTest(fault=fault):
                model = _vision(repository)
                if fault == "instance_forward":
                    model.blocks[-1].attn.forward = MethodType(lambda self, *a, **k: None, model.blocks[-1].attn)
                if fault == "class":
                    class UnknownVision(type(model)): pass
                    model.__class__ = UnknownVision
                context = mock.patch.object(adapter.transformers, "__version__", "4.58.0") if fault == "version" else \
                    mock.patch.dict(adapter._MODULE_HASHES, {repository.__name__: "0" * 64}) if fault == "hash" else nullcontext()
                with context, self.assertRaises(ValueError): adapter.install_qwen_vision_lengths(model)
                self.assertNotIn(adapter._STATE, model.__dict__)
                self.assertNotIn("forward", model.__dict__)
                self.assertNotIn("forward", model.blocks[0].attn.__dict__)

    def test_external_adapter_replacement_is_not_overwritten(self):
        model = _vision(repository)
        adapter.install_qwen_vision_lengths(model)
        replacement = MethodType(lambda self, *a, **k: None, model)
        model.forward = replacement
        with self.assertRaises(ValueError): adapter.install_qwen_vision_lengths(model)
        with self.assertRaises(ValueError): adapter.uninstall_qwen_vision_lengths(model)
        self.assertIs(model.forward, replacement)

    def test_metadata_cannot_be_injected_or_reused_from_previous_forward(self):
        model = _vision(repository)
        adapter.install_qwen_vision_lengths(model)
        reference = _vision(repository)
        reference.load_state_dict(model.state_dict())
        for rows in ([[1, 4, 4]] * 3, [[1, 4, 6]] * 6, [[2, 6, 4], [1, 4, 4], [1, 6, 6]]):
            pixels, grid = _inputs(rows)
            with torch.no_grad(): self.assert_nested_equal(reference(pixels, grid), model(pixels, grid))
        with self.assertRaisesRegex(ValueError, "adapter-owned"):
            model(pixels, grid, _structured_world_lengths_cpu=(123,))

    def test_concurrent_different_grids_have_no_shared_metadata(self):
        reference = _vision(repository)
        model = copy.deepcopy(reference)
        adapter.install_qwen_vision_lengths(model)
        batches = [_inputs([[1, 4, 4]] * 3), _inputs([[1, 6, 4]] * 6)]
        with torch.no_grad(): expected = [reference(*args) for args in batches]
        def run(args):
            with torch.no_grad(): return model(*args)
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(run, batches))
        self.assert_nested_equal(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
