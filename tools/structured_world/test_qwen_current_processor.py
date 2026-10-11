"""CPU parity checks for the real current processor and PreparedCurrent.

Uses local processor assets, the original wrapper, actual Qwen mRoPE and tiny
actual Qwen vision/language classes. Only accelerator pinning is mocked. No
pretrained model weights, network access, CUDA initialization or GPU kernels.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
import transformers
from transformers import AutoProcessor
from starVLA.cache.navsim_feature_cache import ROBOT_HISTORY_TOKEN, action_query_tokens
from starVLA.dataloader.structured_world import prepared_current
from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.structured_world import qwen_execution
from starVLA.model.modules.structured_world import queries
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

PROCESSOR_ROOT = Path("/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct")
EVIDENCE = {"processor_microbatches": [], "vision_language_comparisons": []}
torch.set_num_threads(1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def collate(rows):
    occupancy = torch.arange(len(rows)*9*6*8).reshape(len(rows), 9, 6, 8).remainder(3).to(torch.int8)
    valid = torch.ones_like(occupancy, dtype=torch.bool)
    road = torch.linspace(-2, 2, 48).reshape(6, 8).repeat(len(rows), 1, 1)
    # These SHA identities cover only the owned synthetic fixture bytes. They
    # do not stand in for the production Dataset's verified label-file hashes.
    hashes = [hashlib.sha256(occupancy[row].numpy().tobytes() + valid[row].numpy().tobytes() +
                          road[row].numpy().tobytes()).hexdigest() for row in range(len(rows))]
    return [dict(row) for row in rows], {
        "ego": torch.zeros(len(rows), 8, 4),
        "future_valid": torch.ones(len(rows), 8, dtype=torch.bool),
        "occupancy": occupancy, "occupancy_valid": valid, "road_distance": road,
        "label_sha256": hashes, "label_schema": ["owned_CPU_test_fixture_v1"]*len(rows),
    }


def still_image_rope_oracle(ids, grid, mask, config):
    """Independent Python-integer coordinates, including left-pad delta.

    Does not call get_rope_index or reuse its tensor expressions. Images arrive
    in the processor's sample/camera order; this suite has no video inputs.
    """
    positions = torch.ones((3, *ids.shape), dtype=ids.dtype)
    deltas, image_at = [], 0
    for row in range(ids.shape[0]):
        tokens = ids[row, mask[row].bool()].tolist()
        coordinates, cursor = [[], [], []], 0
        starts = [at for at, token in enumerate(tokens[:-1])
                  if token == config.vision_start_token_id and tokens[at+1] == config.image_token_id]
        for start in starts:
            first = start + 1
            base = max(max(axis) for axis in coordinates) + 1 if coordinates[0] else 0
            text_length = first - cursor
            for axis in coordinates:
                axis.extend(range(base, base + text_length))
            temporal, height, width = grid[image_at].tolist()
            image_at += 1
            height //= config.vision_config.spatial_merge_size
            width //= config.vision_config.spatial_merge_size
            for t in range(temporal):
                for h in range(height):
                    for w in range(width):
                        coordinates[0].append(base + text_length + t)
                        coordinates[1].append(base + text_length + h)
                        coordinates[2].append(base + text_length + w)
            cursor = first + temporal * height * width
        base = max(max(axis) for axis in coordinates) + 1 if coordinates[0] else 0
        for axis in coordinates:
            axis.extend(range(base, base + len(tokens) - cursor))
        positions[:, row, mask[row].bool()] = torch.tensor(coordinates, dtype=ids.dtype)
        deltas.append(max(max(axis) for axis in coordinates) + 1 - ids.shape[1])
    assert image_at == len(grid)
    return positions, torch.tensor(deltas, dtype=ids.dtype)[:, None]


class CurrentProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if transformers.__version__ != "4.57.0":
            raise ValueError("This CPU receipt requires the actual audited Transformers 4.57.0")
        cls.processor = AutoProcessor.from_pretrained(str(PROCESSOR_ROOT), local_files_only=True)
        cls.processor.tokenizer.padding_side = "left"
        cls.world = [f"<foresight_{i}>" for i in range(144)]
        cls.actions = list(action_query_tokens(8))
        cls.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": [ROBOT_HISTORY_TOKEN, *cls.actions, *cls.world],
        }, replace_additional_special_tokens=False)
        cls.config = Qwen3VLConfig.from_pretrained(str(PROCESSOR_ROOT), local_files_only=True)
        cls.source_paths = {
            "prepared_current": prepared_current.__file__, "adapter": qwen_execution.__file__,
            "queries": queries.__file__, "qwen": inspect.getsourcefile(Qwen3VLModel),
            "current_wrapper": inspect.getsourcefile(Qwenvl_OFT), "test": __file__,
        }
        EVIDENCE["source_sha256_at_start"] = {key: digest(path) for key, path in cls.source_paths.items()}
        EVIDENCE["sources"] = {
            "processor_root": str(PROCESSOR_ROOT.resolve()),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "processor_class": type(cls.processor).__module__ + "." + type(cls.processor).__name__,
            "image_processor_class": type(cls.processor.image_processor).__module__ + "." + type(cls.processor.image_processor).__name__,
            "processor_source_sha256": digest(inspect.getsourcefile(type(cls.processor))),
            "image_processor_source_sha256": digest(inspect.getsourcefile(type(cls.processor.image_processor))),
            "get_rope_index_source_sha256": hashlib.sha256(inspect.getsource(Qwen3VLModel.get_rope_index).encode()).hexdigest(),
            "qwen_source_sha256": digest(inspect.getsourcefile(Qwen3VLModel)),
            "prepared_current_sha256": digest(prepared_current.__file__),
            "queries_sha256": digest(queries.__file__),
            "adapter_sha256": digest(qwen_execution.__file__),
            "test_sha256": digest(__file__),
            "processor_asset_sha256": {name: digest(PROCESSOR_ROOT / name) for name in (
                "config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json")},
        }

    @classmethod
    def tearDownClass(cls):
        EVIDENCE["source_sha256_at_finish"] = {key: digest(path) for key, path in cls.source_paths.items()}
        if EVIDENCE["source_sha256_at_start"] != EVIDENCE["source_sha256_at_finish"]:
            raise AssertionError("Source changed during CPU validation; this receipt cannot accept that version")

    def setUp(self):
        # get_rope_index reads config only; no large model or model weights.
        self.core = SimpleNamespace(config=self.config, rope_deltas=None)
        self.core.get_rope_index = MethodType(Qwen3VLModel.get_rope_index, self.core)
        self.interface = SimpleNamespace(processor=self.processor,
            model=SimpleNamespace(device=torch.device("cpu"), model=self.core))
        self.interface.build_qwenvl_inputs = MethodType(_QWen3_VL_Interface.build_qwenvl_inputs, self.interface)
        self.controller = SimpleNamespace(qwen_vl_interface=self.interface, foresight_tokens=self.world, cameras=3,
            grid=GridSpec(xmin=-2, xmax=2, ymin=-1.5, ymax=1.5, dx=.5, dy=.5),
            act_query_tokens=self.actions, _find_token_positions=Qwenvl_OFT._find_token_positions,
            _special_token_ids={
                "history": (self.processor.tokenizer.convert_tokens_to_ids(ROBOT_HISTORY_TOKEN),),
                "action": tuple(self.processor.tokenizer.convert_tokens_to_ids(self.actions)),
                "foresight": tuple(self.processor.tokenizer.convert_tokens_to_ids(self.world)),
            })
        self.set_template(None)

    def set_template(self, template):
        options = {} if template is None else {"CoT_prompt": template}
        self.controller.config = OmegaConf.create({"datasets": {"vla_data": options}})
        self.interface.config = self.controller.config

    def rows(self, cameras, count=5):
        generator = np.random.default_rng(710 + cameras)
        result = []
        for at in range(count):
            images = []
            for camera in range(cameras):
                # One native image above min-pixels and two resized orientations
                # ensure different merged token counts as well as different h/w.
                h, w = [(288, 512), (128, 192), (192, 128)][(at+camera) % 3]
                images.append(Image.fromarray(generator.integers(0, 256, (h, w, 3), dtype=np.uint8)))
            result.append({"image": images,
                "lang": ["Turn left.", "Proceed cautiously at the junction. "*7,
                         "保持车道，注意前方车辆。", "Stop.", "Yield and proceed safely. "*3][at],
                "state": np.zeros((1, 4), dtype=np.float32),
                "geometry_images": torch.zeros(cameras, 3, 8, 8),
                "geometry_pixel_valid": torch.ones(cameras, 8, 8, dtype=torch.bool),
                "calibration": {key: torch.eye(4).repeat(cameras, 1, 1) for key in
                    ("sensor2ego", "intrinsics", "post_rots", "post_trans")}})
        return result

    def instructions(self, rows):
        return [row["lang"] + " " + ROBOT_HISTORY_TOKEN + "".join(self.world) + "".join(self.actions) for row in rows]

    def assert_same_rng(self, left, right):
        self.assertEqual(left[0], right[0])
        self.assertTrue(np.array_equal(left[1][1], right[1][1]))
        self.assertEqual(left[1][2:], right[1][2:])
        self.assertTrue(torch.equal(left[2], right[2]))

    def assert_tensor_equal(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(torch.equal(left, right))

    def prepare(self, rows, microbatch):
        self.controller.cameras = len(rows[0]["image"])
        collator = prepared_current.CurrentBatchCollator(self.controller, microbatch, collate)
        # pin_memory cannot be validated in a CPU-only process. All packing,
        # processor, geometry stacking, mRoPE and token-slot code remains real.
        with mock.patch.object(prepared_current, "pin", lambda value: value), \
                mock.patch.object(queries.PreparedQueryCandidates, "pin_memory", lambda item: item):
            observations, targets = collator(rows)
        return collator, observations, targets

    def test_identity_covers_resize_tokenizer_initialization_and_camera_count(self):
        before = rng_state()
        collator, observations, _ = self.prepare(self.rows(3, 2), 2)
        micro = observations[:2]
        payload = micro[0]["_prepared_current"]
        instructions = self.instructions(micro)
        baseline = prepared_current.processor_identity(self.controller)
        self.assertEqual(baseline, collator.identity)
        mutations = [
            ("image_resize", mock.patch.object(self.processor.image_processor, "size",
                dict(self.processor.image_processor.size, shortest_edge=262144))),
            ("tokenizer_init_kwargs", mock.patch.object(self.processor.tokenizer, "init_kwargs",
                dict(self.processor.tokenizer.init_kwargs, clean_up_tokenization_spaces=
                    not self.processor.tokenizer.init_kwargs.get("clean_up_tokenization_spaces", False)))),
            ("camera_count", mock.patch.object(self.controller, "cameras", 6)),
        ]
        for name, context in mutations:
            with self.subTest(identity_field=name), context:
                changed = prepared_current.processor_identity(self.controller)
                self.assertNotEqual(baseline, changed)
                with self.assertRaises(ValueError):
                    payload.validate(micro, instructions, changed)
            self.assertEqual(prepared_current.processor_identity(self.controller), baseline)
        self.assert_same_rng(before, rng_state())
        EVIDENCE["identity_field_regressions"] = {
            "image_resize": "changed digest and rejected stale payload",
            "tokenizer_init_kwargs": "changed digest and rejected stale payload",
            "camera_count": "changed digest and rejected stale payload",
            "restored_configuration_restores_digest": True,
        }

    def test_full_processor_actual_microbatch_integer_parity(self):
        for cameras in (3, 6):
            rows = self.rows(cameras)
            for template in (None, "Driving instruction: {instruction}\nUse only current cameras."):
                self.set_template(template)
                for microbatch in (2, 4):
                    before = rng_state()
                    collator, observations, _ = self.prepare(rows, microbatch)
                    for at in range(0, len(rows), microbatch):
                        micro = observations[at:at+microbatch]
                        payload = micro[0]["_prepared_current"]
                        self.assertTrue(all(row["_prepared_current"] is payload for row in micro))
                        instructions = self.instructions(micro)
                        native = self.interface.build_qwenvl_inputs([row["image"] for row in micro], instructions)
                        payload.validate(micro, instructions, collator.identity)
                        self.assertEqual(set(native), set(payload.inputs))
                        for key in native: self.assert_tensor_equal(native[key], payload.inputs[key])
                        positions, deltas = self.core.get_rope_index(native["input_ids"], native["image_grid_thw"], attention_mask=native["attention_mask"])
                        oracle_positions, oracle_deltas = still_image_rope_oracle(native["input_ids"], native["image_grid_thw"], native["attention_mask"], self.config)
                        self.assert_tensor_equal(payload.position_ids, positions)
                        self.assert_tensor_equal(positions, oracle_positions)
                        self.assert_tensor_equal(deltas, oracle_deltas)
                        self.assertEqual(positions.dtype, torch.int64)
                        self.assertFalse(positions.requires_grad)
                        padding = ~native["attention_mask"].bool()[None].expand_as(positions)
                        self.assertTrue(positions.masked_select(padding).eq(1).all())
                        for key, ids in self.controller._special_token_ids.items():
                            self.assert_tensor_equal(payload.slots[key], self.controller._find_token_positions(native["input_ids"], ids))
                        self.assertGreater(len(set((native["image_grid_thw"].prod(-1)//4).tolist())), 1)
                        EVIDENCE["processor_microbatches"].append({
                            "cameras": cameras, "CoT_template": template is not None, "microbatch": microbatch,
                            "actual_batch": len(micro), "sequence_shape": list(native["input_ids"].shape),
                            "active_lengths": native["attention_mask"].sum(-1).tolist(),
                            "grids": sorted({tuple(row) for row in native["image_grid_thw"].tolist()}),
                            "rope_deltas": deltas.flatten().tolist(), "pixel_dtype": str(native["pixel_values"].dtype),
                            "all_values_bitwise": True, "independent_integer_oracle": True,
                        })
                    self.assert_same_rng(before, rng_state())
        self.assertIsNone(self.core.rope_deltas)

    def tiny_model(self):
        config = Qwen3VLConfig(
            image_token_id=self.config.image_token_id, video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id, vision_end_token_id=self.config.vision_end_token_id,
            text_config={"vocab_size": len(self.processor.tokenizer), "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                "out_hidden_size": 16, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 2]})
        config._attn_implementation = "sdpa"
        return Qwen3VLModel(config).eval()

    def test_numeric_explicit_cpu_rope_vision_language_matches_native(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(808)
            original = self.tiny_model()
            changed = copy.deepcopy(original)
            EVIDENCE["adapter_receipt"] = qwen_execution.install_qwen_vision_lengths(changed.visual)
            fast_controller = SimpleNamespace(qwen_vl_interface=SimpleNamespace(
                model=SimpleNamespace(config=changed.config, model=changed)))
            for cameras in (3, 6):
                _, observations, _ = self.prepare(self.rows(cameras, 2), 2)
                payload = observations[0]["_prepared_current"]
                native = self.interface.build_qwenvl_inputs([row["image"] for row in observations], self.instructions(observations))
                for bf16 in (False, True):
                    context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
                    with torch.no_grad(), context:
                        before = rng_state()
                        reference = original(**native, use_cache=False).last_hidden_state
                        parts, deepstack = changed.get_image_features(payload.inputs["pixel_values"], payload.inputs["image_grid_thw"])
                        embeddings = changed.get_input_embeddings()(payload.inputs["input_ids"])
                        actual = Qwenvl_OFT._qwen_language_forward(fast_controller, payload.inputs["input_ids"], embeddings,
                            payload.inputs["attention_mask"], payload.position_ids, torch.cat(parts, 0), deepstack)
                        self.assert_tensor_equal(reference, actual)
                        self.assert_same_rng(before, rng_state())
                    EVIDENCE["vision_language_comparisons"].append({
                        "cameras": cameras, "CPU_BF16_autocast": bf16,
                        "merged_image_lengths": sorted(set((native["image_grid_thw"].prod(-1)//4).tolist())),
                        "output_shape": list(actual.shape), "dtype": str(actual.dtype), "max_absolute": 0, "bitwise": True,
                    })

    def test_ordered_cpu_worker_identity_and_transfer_aliases(self):
        rows = self.rows(3)
        before = rng_state()
        with ThreadPoolExecutor(max_workers=1) as pool:
            collator, observations, targets = pool.submit(self.prepare, rows, 2).result()
        self.assert_same_rng(before, rng_state())
        for at in (0, 2, 4):
            micro = observations[at:at+2]
            payload = micro[0]["_prepared_current"]
            native = self.interface.build_qwenvl_inputs([row["image"] for row in micro], self.instructions(micro))
            for key in native: self.assert_tensor_equal(payload.inputs[key], native[key])
            instructions = self.instructions(micro)
            with self.assertRaises(ValueError): payload.validate(micro, ["foreign"]*len(micro), collator.identity)
            with self.assertRaises(ValueError): payload.validate(micro, instructions, "foreign")
            changed = [dict(row) for row in micro]
            changed[0]["image"] = list(reversed(changed[0]["image"]))
            with self.assertRaises(ValueError): payload.validate(changed, instructions, collator.identity)
        transfer = prepared_current.PreparedCurrent.to
        transferred = []
        def observe(payload, device):
            transferred.append(payload)
            return transfer(payload, device)
        cpu_payloads = [row["_prepared_current"] for row in observations]
        with mock.patch.object(prepared_current.PreparedCurrent, "to", observe):
            moved, moved_targets = prepared_current.transfer_optimizer_batch(observations, targets, torch.device("cpu"))
        self.assertEqual(len(transferred), 3)
        for at, row in enumerate(moved):
            self.assertIs(row["_prepared_current_cpu"], cpu_payloads[at])
            self.assert_tensor_equal(row["_prepared_current"].position_ids, cpu_payloads[at].position_ids)
            for key in cpu_payloads[at].inputs:
                self.assert_tensor_equal(row["_prepared_current"].inputs[key], cpu_payloads[at].inputs[key])
            if at % 2: self.assertIs(row["_prepared_current"], moved[at-1]["_prepared_current"])
        for key in targets:
            if torch.is_tensor(targets[key]):
                self.assert_tensor_equal(targets[key], moved_targets[key])
            elif key == "query_candidates":
                for row, (left, right) in enumerate(zip(targets[key], moved_targets[key])):
                    self.assert_tensor_equal(left.global_indices, right.global_indices)
                    self.assert_tensor_equal(left.boundary_indices, right.boundary_indices)
                    right.validate(moved_targets, row, self.controller.grid)
                    self.assert_tensor_equal(right.global_indices, torch.nonzero(targets["occupancy_valid"][row, 1:], as_tuple=False))
                    boundary = queries.boundary_candidates(targets["road_distance"][row], targets["occupancy"][row], targets["occupancy_valid"][row])
                    self.assert_tensor_equal(right.boundary_indices, torch.nonzero(boundary, as_tuple=False))
            else:
                self.assertEqual(targets[key], moved_targets[key])
        self.assert_same_rng(before, rng_state())
        self.assertIsNone(self.core.rope_deltas)
        EVIDENCE["ordered_CPU_worker_identity_and_transfer_aliases"] = "passed; three transfers for five rows at microbatch two"

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


def main():
    global PROCESSOR_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-root", type=Path, default=PROCESSOR_ROOT)
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt; existing files are never overwritten")
    args = parser.parse_args()
    PROCESSOR_ROOT = args.processor_root
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CurrentProcessorTests))
    receipt = {"schema": "prepared_current_actual_processor_CPU_parity_v1",
        "created_UTC": datetime.now(timezone.utc).isoformat(), "passed": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "fixtures": {"image_rng": "numpy PCG64 seed 710+cameras", "tiny_model_torch_seed": 808,
            "camera_counts": [3, 6], "microbatches": [2, 4], "foresight_tokens": 144, "action_tokens": 8,
            "images": "owned synthetic PIL RGB; native above-min and two resized orientations",
            "tokenizer_padding_side": "left", "CoT_templates": "absent or explicit string"},
        "GPU_initialized": torch.cuda.is_initialized(), "evidence": EVIDENCE,
        "limitations": ["pin_memory mocked; CUDA transfers not exercised", "small random Qwen, no pretrained weights",
            "candidate label identities cover owned fixture bytes; production cache hash binding is not exercised",
            "CPU numerical evidence only; no trained CUDA, native Flash or ZeRO recovery acceptance",
            "no end-to-end speed measurement; prior strict numerical failures remain valid",
            "reference CoT_prompt:null is unsupported; this does not validate a widened prompt contract"]}
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream: stream.write(text)
    print(json.dumps({"passed": receipt["passed"], "tests_run": result.testsRun,
        "processor_microbatches": len(EVIDENCE["processor_microbatches"]),
        "vision_language_comparisons": len(EVIDENCE["vision_language_comparisons"]),
        "GPU_initialized": receipt["GPU_initialized"], "receipt": str(args.output) if args.output else None}))
    if not args.output: print(text, end="")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
