"""CPU-only tests of native GR00T ten-step K/V Linear reuse.

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B this_file.py
No model weights/network/GPU. Optional --output writes a NEW evidence receipt.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import copy
from contextlib import ExitStack, nullcontext, redirect_stdout
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
from types import MethodType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from omegaconf import OmegaConf
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.structured_world import proposal_execution as adapter

torch.set_num_threads(1)
EVIDENCE = {}


def source_digests():
    paths = {Path(inspect.getsourcefile(method)).resolve()
             for _, method, _, _ in adapter._SOURCES}
    paths.update((Path(adapter.__file__).resolve(), Path(__file__).resolve()))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def make_head(depth=4):
    config = OmegaConf.create({"framework": {"qwenvl": {"vl_hidden_dim": 32}, "action_model": {
        "hidden_size": 32, "DiTConfig": {"input_embedding_dim": 32, "attention_head_dim": 8,
            "num_attention_heads": 4, "num_layers": depth},
        "diffusion_model_cfg": {"cross_attention_dim": 32, "dropout": .2, "final_dropout": True,
            "interleave_self_attention": True, "norm_type": "ada_norm", "output_dim": 32,
            "positional_embeddings": None},
        "action_dim": 4, "action_horizon": 8, "num_inference_timesteps": 10,
        "add_pos_embed": True, "max_seq_len": 32, "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1., "noise_s": .999, "num_timestep_buckets": 1000,
    }}})
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        torch.manual_seed(819)
        return FlowmatchingActionHead(config).eval()


def inputs(batch=2, condition=8):
    generator = torch.Generator().manual_seed(818)
    return torch.randn(batch, condition, 32, generator=generator), torch.randn(batch, 8, 4, generator=generator)


def inventory(model):
    return {
        "parameters": [(name, id(value), tuple(value.shape), value.dtype, value.requires_grad) for name, value in model.named_parameters()],
        "buffers": [(name, id(value), tuple(value.shape), value.dtype) for name, value in model.named_buffers()],
        "values": {name: value.clone() for name, value in model.state_dict().items()},
        "modes": [(name, module.training) for name, module in model.named_modules()],
    }


class ProposalTests(unittest.TestCase):
    def assert_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for name in left: self.assert_equal(left[name], right[name])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_equal(a, b)
        else: self.assertEqual(left, right)

    def assert_no_projection_adapters(self, head):
        for block in head.model.transformer_blocks:
            for name in ("to_k", "to_v"):
                self.assertNotIn("forward", getattr(block.attn1, name).__dict__)
        self.assertFalse(adapter.proposal_kv_cache_status(head).get("active", False))

    def test_ten_step_fp32_bf16_outputs_rng_and_projection_count(self):
        for depth, condition in ((4, 8), (24, 8), (4, 152)):
            for bf16 in (False, True):
                with self.subTest(depth=depth, condition=condition, bf16=bf16):
                    original = make_head(depth)
                    changed = copy.deepcopy(original)
                    installed = adapter.install_proposal_kv_cache(changed, enabled=True)
                    context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
                    memory, noise = inputs(condition=condition)
                    weights = {id(getattr(block.attn1, name).weight) for block in original.model.transformer_blocks
                               if block.cross_attention_dim is not None for name in ("to_k", "to_v")}
                    changed_weights = {id(getattr(block.attn1, name).weight) for block in changed.model.transformer_blocks
                                       if block.cross_attention_dim is not None for name in ("to_k", "to_v")}
                    linear = torch.nn.functional.linear
                    counts = [0, 0]
                    def observe(value, weight, bias=None):
                        if id(weight) in weights: counts[0] += 1
                        if id(weight) in changed_weights: counts[1] += 1
                        return linear(value, weight, bias)
                    before = torch.get_rng_state().clone()
                    trajectories, action_inputs = [[], []], [[], []]
                    with ExitStack() as hooks, torch.no_grad(), context, \
                            mock.patch.object(torch.nn.functional, "linear", observe):
                        for index, model in enumerate((original, changed)):
                            def capture_dit(owner, args, kwargs, output, index=index):
                                trajectories[index].append({
                                    "timestep": kwargs["timestep"].clone(),
                                    "input": kwargs["hidden_states"].clone(),
                                    "output": output.clone(),
                                    "RNG": torch.get_rng_state().clone(),
                                })
                            def capture_actions(owner, args, index=index):
                                action_inputs[index].append(args[0].clone())
                            hooks.callback(model.model.register_forward_hook(
                                capture_dit, with_kwargs=True).remove)
                            hooks.callback(model.action_encoder.register_forward_pre_hook(
                                capture_actions).remove)
                        reference = original.predict_action(memory, initial_noise=noise)
                        actual = changed.predict_action(memory, initial_noise=noise)
                    self.assert_equal(reference, actual)
                    self.assert_equal(before, torch.get_rng_state())
                    self.assertEqual([len(trace) for trace in trajectories], [10, 10])
                    self.assertEqual([len(trace) for trace in action_inputs], [10, 10])
                    self.assert_equal(trajectories[0], trajectories[1])
                    self.assert_equal(action_inputs[0], action_inputs[1])
                    for step, sample in enumerate(trajectories[0]):
                        self.assert_equal(sample["timestep"], torch.full((2,), step*100))
                        self.assert_equal(sample["RNG"], before)
                    self.assertEqual(counts, [depth*10, depth])
                    status = adapter.proposal_kv_cache_status(changed)
                    self.assertEqual(status["last_scope"]["projection_calls_saved"], depth*9)
                    self.assert_no_projection_adapters(changed)
                    EVIDENCE.setdefault("numerical_cases", []).append({"depth": depth, "condition_tokens": condition,
                        "hidden_width": 32, "actual_DiT_forward_steps": 10,
                        "CPU_BF16_autocast": bf16, "native_KV_calls": counts[0], "cached_KV_calls": counts[1],
                        "projection_calls_saved": depth*9, "bitwise_output_and_RNG": True,
                        "bitwise_all_step_actions_DiT_inputs_outputs_and_RNG": True,
                        "native_timesteps": list(range(0, 1000, 100))})
                    EVIDENCE["provenance"] = installed["provenance"]

    def test_default_disabled_grad_train_and_inference_bypass(self):
        model = make_head()
        memory, noise = inputs()
        record = adapter.install_proposal_kv_cache(model)
        self.assertFalse(record["enabled"])
        with torch.no_grad(): model.predict_action(memory, initial_noise=noise)
        self.assertEqual(adapter.proposal_kv_cache_status(model)["last_scope"]["reason"], "disabled")
        adapter.install_proposal_kv_cache(model, enabled=True)
        model.predict_action(memory, initial_noise=noise)
        self.assertEqual(adapter.proposal_kv_cache_status(model)["last_scope"]["reason"], "grad_enabled")
        model.model.transformer_blocks[0].train()
        with torch.no_grad(): model.predict_action(memory, initial_noise=noise)
        self.assertEqual(adapter.proposal_kv_cache_status(model)["last_scope"]["reason"], "training_submodule")
        model.eval()
        with torch.inference_mode(): model.predict_action(memory, initial_noise=noise)
        self.assertEqual(adapter.proposal_kv_cache_status(model)["last_scope"]["reason"], "inference_mode")
        self.assert_no_projection_adapters(model)

    def test_single_call_lifetime_changed_memory_and_weights(self):
        model = make_head()
        adapter.install_proposal_kv_cache(model, enabled=True)
        memory, noise = inputs()
        with torch.no_grad():
            first = model.predict_action(memory, initial_noise=noise)
            second = model.predict_action(memory, initial_noise=noise)
            self.assert_equal(first, second)
            self.assertEqual(adapter.proposal_kv_cache_status(model)["last_scope"]["projection_calls_executed"], 4)
            memory.add_(.3)
            model.model.transformer_blocks[0].attn1.to_k.weight.add_(.1)
            adapter.install_proposal_kv_cache(model, enabled=False)
            expected = model.predict_action(memory, initial_noise=noise)
            adapter.install_proposal_kv_cache(model, enabled=True)
            actual = model.predict_action(memory, initial_noise=noise)
            self.assert_equal(expected, actual)
        self.assert_no_projection_adapters(model)

    def test_projection_in_place_input_weight_output_and_autocast_invalidation(self):
        layer = nn.Linear(32, 32).eval()
        import threading
        cache = adapter._Projection(layer, "owned", threading.get_ident())
        value = inputs()[0]
        with torch.no_grad():
            out = cache.forward(layer, value)
            self.assertIs(cache.forward(layer, value), out)
            out.add_(100)
            self.assert_equal(cache.forward(layer, value), layer(value))
            value.add_(.2)
            self.assert_equal(cache.forward(layer, value), layer(value))
            layer.weight.add_(.3)
            self.assert_equal(cache.forward(layer, value), layer(value))
            layer.weight = nn.Parameter(layer.weight.clone()+.1)
            self.assert_equal(cache.forward(layer, value), layer(value))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assert_equal(cache.forward(layer, value), layer(value))
        with torch.enable_grad():
            self.assertTrue(cache.forward(layer, value).requires_grad)
        with torch.inference_mode():
            self.assert_equal(cache.forward(layer, value), layer(value))
        cache.clear()
        self.assertIsNone(cache.input)
        self.assertIsNone(cache.output)

    def test_processor_mask_cross_norm_and_head_norm_stay_native(self):
        # Invoke the real processor under an owned projection scope. A cross
        # norm creates fresh input tensors, therefore safely misses the cache.
        for cross_norm in (None, "layer_norm"):
            original = Attention(32, cross_attention_dim=32, heads=4, dim_head=8,
                                 cross_attention_norm=cross_norm, qk_norm="layer_norm").eval()
            changed = copy.deepcopy(original)
            query, _ = inputs(condition=8)
            memory = query.clone()+.1
            mask = torch.zeros(2, 1, 8); mask[:, :, -2:] = -torch.inf
            import threading
            projections = [adapter._Projection(getattr(changed, name), name, threading.get_ident()) for name in ("to_k", "to_v")]
            for projection in projections: projection.layer.forward = MethodType(projection.forward, projection.layer)
            with torch.no_grad():
                for _ in range(3): self.assert_equal(original(query, memory, mask), changed(query, memory, mask))
            self.assertEqual(sum(p.hits for p in projections), 4 if cross_norm is None else 0)
            self.assertIs(type(changed.processor), AttnProcessor2_0)

    def test_native_fm_training_gradients_adam_and_rng_unchanged(self):
        original = make_head().train()
        changed = copy.deepcopy(original)
        adapter.install_proposal_kv_cache(changed, enabled=True)
        memory, actions = inputs()
        noise = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(811))
        time = torch.tensor([.1, .2])
        runs = []
        for model in (original, changed):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            torch.manual_seed(810)
            output = model(memory, actions, noise=noise, times=time)
            output.backward()
            gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
            optimizer.step()
            runs.append((output, gradients, model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
        self.assert_equal(runs[0], runs[1])
        self.assert_no_projection_adapters(changed)

    def test_native_initial_noise_rng_unchanged_and_model_state_portable(self):
        original = make_head()
        changed = make_head()
        changed.load_state_dict(original.state_dict(), strict=True)
        adapter.install_proposal_kv_cache(changed, enabled=True)
        memory, _ = inputs()
        runs = []
        with torch.no_grad():
            for model in (original, changed):
                torch.manual_seed(813)
                output = model.predict_action(memory)
                runs.append((output, torch.get_rng_state().clone()))
        self.assert_equal(runs[0], runs[1])
        # Plain state_dict stays independent of Python adapter state.
        loaded = make_head()
        loaded.load_state_dict(changed.state_dict(), strict=True)
        self.assertNotIn(adapter._STATE, loaded.__dict__)
        self.assert_equal(original.state_dict(), loaded.state_dict())

    def test_failure_cleanup_inventory_idempotence_and_isolation(self):
        model = make_head()
        other = make_head()
        before = inventory(model)
        rng = torch.get_rng_state().clone()
        first = adapter.install_proposal_kv_cache(model, enabled=True)
        self.assert_equal(before, inventory(model))
        self.assert_equal(rng, torch.get_rng_state())
        self.assert_equal(first, adapter.install_proposal_kv_cache(model, enabled=True))
        self.assertNotIn("predict_action", other.__dict__)
        memory, noise = inputs()
        with mock.patch.object(model.action_decoder, "forward", side_effect=RuntimeError("owned failure")), \
                torch.no_grad(), self.assertRaisesRegex(RuntimeError, "owned failure"):
            model.predict_action(memory, initial_noise=noise)
        self.assert_no_projection_adapters(model)
        self.assertFalse(adapter.proposal_kv_cache_status(model)["last_scope"]["succeeded"])
        with torch.no_grad(): model.predict_action(memory, initial_noise=noise)
        self.assertTrue(adapter.uninstall_proposal_kv_cache(model))
        self.assertFalse(adapter.uninstall_proposal_kv_cache(model))
        self.assertNotIn("predict_action", model.__dict__)
        self.assert_equal(before, inventory(model))

    def test_unknown_sources_processors_and_existing_adapters_refused(self):
        for fault in ("source", "processor", "linear", "steps"):
            with self.subTest(fault=fault):
                model = make_head()
                if fault == "processor": model.model.transformer_blocks[0].attn1.set_processor(object())
                if fault == "linear": model.model.transformer_blocks[0].attn1.to_k.forward = lambda x: x
                if fault == "steps": model.num_inference_timesteps = 8
                context = mock.patch.object(adapter, "_SOURCES", [(nn.Linear, nn.Linear.forward, "0"*64, "0"*64)]) if fault == "source" else nullcontext()
                with context, self.assertRaises(ValueError): adapter.install_proposal_kv_cache(model, enabled=True)
                self.assertNotIn("predict_action", model.__dict__)
                self.assertNotIn(adapter._STATE, model.__dict__)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists(): raise FileExistsError(args.output)
    sources_before = source_digests()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProposalTests))
    sources_after = source_digests()
    passed = result.wasSuccessful() and sources_before == sources_after
    record = {"schema": "proposal_KV_native_CPU_candidate_tests_v1", "passed": passed,
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "source_sha256_at_start": sources_before, "source_sha256_at_finish": sources_after,
        "sources_unchanged_during_tests": sources_before == sources_after,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "GPU_initialized": torch.cuda.is_initialized(),
        "evidence": EVIDENCE, "limits": ["tiny CPU models; no trained CUDA/ZeRO acceptance",
            "no GPU performance evidence; require >=20ms proposal improvement before rollout",
            "default disabled; no DiT/GR00T/trainer/integration source edits",
            "native strict numerical failures retained; CPU pass does not supersede them"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream: json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"passed": passed, "tests_run": result.testsRun, "receipt": str(args.output) if args.output else None}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
