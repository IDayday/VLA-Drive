"""Bounded real-image frozen Qwen vision cache feasibility, not train acceptance.

One explicitly selected idle GPU, eight NAVSIM training scenes, all DeepStack
outputs at their original dtype. No training, no permanent cache, no labels.
The temporary bank is deleted on exit. Timing excludes language/FM/BEV/backward.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import benchmark_engineering_execution as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--current-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    args = parser.parse_args()
    bench.immutable_source(ROOT, args.source_sha)
    host = socket.gethostname().removesuffix('-worker-0')
    policy = bench.authorize(host)
    bench.require(os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid, 'Exactly one explicit GPU UUID required')
    def ready():
        raw = bench.checked(['nvidia-smi', '-i', args.gpu_uuid,
            '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
        uuid, used, free, util = [v.strip() for v in raw.split(',')]
        bench.require(uuid == args.gpu_uuid and int(used) < 64 and int(free) > 78000 and int(util) == 0,
                      'Selected GPU is not idle; no task will be stopped')
        apps = bench.checked(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'])
        bench.require(not any(line.startswith(args.gpu_uuid) for line in apps.splitlines()), 'Selected GPU has a compute owner')
        return {'gpu': raw, 'compute_apps': apps}
    with ExitStack() as stack:
        # Same physical-pool lock family as the full benchmark, plus this GPU.
        pool = bench.checked(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
        locks = Path('/mnt/project/structured-world-fgtr-execution-artifacts/20261011/physical_gpu_locks')
        for key in (hashlib.sha256(json.dumps(sorted(pool)).encode()).hexdigest(), args.gpu_uuid):
            bench.acquire(stack, locks / (key + '.lock'))
        admission = ready()
        args.output.mkdir(parents=True, exist_ok=False)
        import torch
        from safetensors import safe_open
        from transformers import AutoProcessor
        from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
        from starVLA.model.modules.vehicle_joint.initialization import verify_generic_source
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        model_root = Path('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
        manifest = Path('/mnt/project/ddp-foresight-artifacts/20260928/generic/sources.json')
        start = time.monotonic()
        source_identity = verify_generic_source(str(model_root), json.loads(manifest.read_text())['qwen'])
        config = Qwen3VLConfig.from_pretrained(str(model_root), local_files_only=True).vision_config
        config._attn_implementation = 'sdpa'
        vision = Qwen3VLVisionModel(config).float().eval().requires_grad_(False)
        prefix = 'model.visual.'
        with safe_open(model_root/'model.safetensors', framework='pt', device='cpu') as handle:
            weights = {k[len(prefix):]: handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix)}
        vision.load_state_dict(weights, strict=True)
        del weights
        model_load_seconds = time.monotonic() - start
        data = ForesightCurrentDataset(args.current_root)
        data.local_image_root = str(args.image_root)
        observations = [data[i] for i in range(8)]
        processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        def preprocess(rows):
            messages = [[{'role': 'user', 'content':
                [{'type': 'image', 'image': image} for image in row['image']] +
                [{'type': 'text', 'text': row['lang']}]}] for row in rows]
            # No language output is cached. Vision depends only on these images/grid.
            packed = processor.apply_chat_template(messages, tokenize=True, padding=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt')
            return {k: packed[k] for k in ('pixel_values', 'image_grid_thw')}
        start = time.monotonic()
        batches = [preprocess(observations[i:i+4]) for i in (0, 4)]
        scenes = [preprocess([row]) for row in observations]
        preprocessing_seconds = time.monotonic() - start
        for number, batch in enumerate(batches):
            singles = scenes[number*4:(number+1)*4]
            for key in batch:
                bench.require(torch.equal(batch[key], torch.cat([s[key] for s in singles])),
                              'Image preprocessing changes under scene batching: ' + key)
        ready()  # Admission again immediately before allocating CUDA.
        torch.cuda.set_device(0)
        vision.cuda()
        def forward(batch):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                output, deep = vision(batch['pixel_values'].to('cuda').type(vision.dtype),
                                      grid_thw=batch['image_grid_thw'].to('cuda'))
            return {'visual': output, **{f'deepstack_{i}': tensor for i, tensor in enumerate(deep)}}
        def cpu(values): return {k: v.cpu() for k, v in values.items()}
        def timed(fn, count=10):
            values = []
            for _ in range(count):
                torch.cuda.synchronize(); begin = time.monotonic()
                fn()
                torch.cuda.synchronize(); values.append(time.monotonic()-begin)
            return {'count': count, 'median_seconds': statistics.median(values), 'samples_seconds': values}
        for _ in range(3): forward(batches[0])
        torch.cuda.synchronize()
        online = timed(lambda: forward(batches[0]))
        torch.cuda.synchronize(); start = time.monotonic()
        multi = [cpu(forward(batch)) for batch in batches]
        torch.cuda.synchronize(); batched_encoding_seconds = time.monotonic() - start
        lengths = [int(scene['image_grid_thw'].prod(-1).sum()) // vision.spatial_merge_size**2
                   for scene in scenes]
        batch_encoded_scenes = []
        for number, values in enumerate(multi):
            parts = {key: tensor.split(lengths[number*4:(number+1)*4]) for key, tensor in values.items()}
            batch_encoded_scenes.extend({key: tensors[i].clone() for key, tensors in parts.items()} for i in range(4))
        # Same microbatch/grid shape, different scene companions/order. A cache
        # cannot assume this is equivalent just because the visual is frozen.
        regrouped_comparisons = []
        for indices in ((0, 4, 2, 6), (1, 5, 3, 7)):
            values = cpu(forward(preprocess([observations[i] for i in indices])))
            for key, tensor in values.items():
                for index, right in zip(indices, tensor.split([lengths[i] for i in indices])):
                    left = batch_encoded_scenes[index][key]
                    delta = (right.float()-left.float()).double()
                    regrouped_comparisons.append({'scene_index': index, 'output': key,
                        'bitwise': torch.equal(left, right), 'max_abs': delta.abs().max().item(),
                        'relative_L2': (delta.norm()/left.double().norm().clamp_min(1e-30)).item()})
        torch.cuda.synchronize(); start = time.monotonic()
        single = [cpu(forward(scene)) for scene in scenes]
        torch.cuda.synchronize(); encoding_seconds = time.monotonic() - start
        comparisons = []
        for batch_number, expected in enumerate(multi):
            for key, left in expected.items():
                right = torch.cat([row[key] for row in single[batch_number*4:(batch_number+1)*4]])
                delta = (right.float()-left.float()).double()
                comparisons.append({'batch': batch_number, 'output': key, 'shape': list(left.shape),
                    'dtype': str(left.dtype), 'bitwise': torch.equal(left, right),
                    'max_abs': delta.abs().max().item(),
                    'relative_L2': (delta.norm()/left.double().norm().clamp_min(1e-30)).item()})
        payload_bytes = sum(t.numel()*t.element_size() for row in single for t in row.values())
        bench.require(payload_bytes <= 512 * 1024**2, 'Probe must never exceed the half-GiB temporary bank')
        with tempfile.TemporaryDirectory(prefix='visual-cache-probe-', dir=args.output) as temp:
            bank = Path(temp)/'eight_current_scenes.pt'
            start = time.monotonic()
            with bank.open('wb') as stream:
                torch.save(batch_encoded_scenes, stream); stream.flush(); os.fsync(stream.fileno())
            write_seconds = time.monotonic() - start
            stored_bytes = bank.stat().st_size
            # First read after writing is already cache-warm. Do not claim cold/NFS throughput.
            loaded = torch.load(bank, map_location='cpu', weights_only=True)
            bench.require(all(torch.equal(a[k], b[k]) for a, b in zip(batch_encoded_scenes, loaded) for k in a),
                          'Lossless storage roundtrip failed')
            def read_transfer():
                loaded = torch.load(bank, map_location='cpu', weights_only=True)
                return [{k: v.pin_memory().to('cuda', non_blocking=True) for k, v in row.items()}
                        for row in loaded]
            read = timed(read_transfer, count=6)
        record = {'schema': 'frozen_visual_cache_feasibility_v1', 'source_sha': args.source_sha,
            'host': host, 'gpu_uuid': args.gpu_uuid, 'policy': policy, 'admission': admission,
            'Qwen_source_identity': source_identity, 'current_identity': data.identity,
            'samples': 8, 'cameras_per_sample': 3, 'population': len(data),
            'sample_tokens': [row['token'] for row in observations],
            'parameters': 'FP32 frozen/eval visual only', 'compute': 'BF16 autocast; original SDPA; TF32 off',
            'DeepStack_outputs_retained': len(single[0])-1, 'output_dtypes': {k: str(v.dtype) for k, v in single[0].items()},
            'model_verification_load_seconds': model_load_seconds,
            'input_preprocessing_seconds_eight_scenes_both_batchings': preprocessing_seconds,
            'online_four_scene_forward_and_H2D': online,
            'single_scene_encoding_D2H_seconds_eight_scenes': encoding_seconds,
            'four_scene_encoding_D2H_seconds_eight_scenes': batched_encoding_seconds,
            'single_vs_four_scene_comparisons': comparisons,
            'same_shape_rebatched_comparisons': regrouped_comparisons,
            'stored_payload_from_original_microbatch': 4,
            'cache_bank_bytes_eight_scenes': stored_bytes, 'tensor_payload_bytes': payload_bytes,
            'estimated_full_train_payload_bytes': payload_bytes*len(data)/8,
            'write_fsync_seconds': write_seconds, 'warm_read_pin_H2D_eight_scenes': read,
            'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_GPU_reserved_bytes': torch.cuda.max_memory_reserved(),
            'temporary_cache_removed': not bank.exists(), 'full_cache_built': False,
            'training_interface_changed': False, 'formal_authorized': False,
            'limits': ['Eight real training scenes only; not end-to-end training performance.',
                       'Page-cache-warm shared storage; cold and concurrent IO not measured.',
                       'Per-scene versus original microbatch output differences prohibit a claim of bitwise equivalence.',
                       'Same-shape rebatched checks cover only eight scenes; other grids, tail microbatch3 and world16 microbatch2 are unqualified.',
                       'No H_A/language/FM/BEV/backward, optimizer or planning acceptance.']}
        bench.write(args.output/'RESULT.json', record)
        print(json.dumps({'status': 'MEASURED_COMPONENT_ONLY', 'result': str(args.output/'RESULT.json'),
            'batching_bitwise': all(row['bitwise'] for row in comparisons),
            'same_shape_rebatching_bitwise': all(row['bitwise'] for row in regrouped_comparisons),
            'cache_bytes_per_scene': payload_bytes/8, 'formal_authorized': False}), flush=True)


if __name__ == '__main__':
    main()
