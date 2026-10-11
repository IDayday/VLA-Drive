"""One admitted GPU, real NAV G1 checkpoint: native vs single-proposal KV.

Run ONLY after the parent operator admits an actually idle GPU. This tool does
not dispatch, reserve devices, stop processes, or change training source.
CUDA_VISIBLE_DEVICES should select the admitted device; launch one process.
--output is a NEW JSON file. Default: 20 warmups and 50 timed calls PER MODE
in each of three alternating windows. Current Qwen is encoded exactly once.
--self-test checks scheduling/summary helpers on CPU without model imports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def tensor_digest(values):
    """Hash names, layout metadata and exact bytes; outside all timing."""
    import torch
    result = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        require(torch.is_tensor(tensor), 'State contains a non-tensor: '+name)
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        result.update(len(metadata).to_bytes(8, 'little'))
        result.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        result.update(memoryview(raw).cast('B'))
    return result.hexdigest()


def exact(left, right):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def inventory(head):
    return {'parameters': [(name, id(p), p._version, str(p.dtype), str(p.device), list(p.shape), p.requires_grad)
                           for name, p in head.named_parameters()],
            'buffers': [(name, id(p), p._version, str(p.dtype), str(p.device), list(p.shape))
                        for name, p in head.named_buffers()],
            'modes': [(name, module.training) for name, module in head.named_modules()]}


def windows(configure, invoke, measure, *, warmup=20, timed=50, count=3):
    """Alternating paired calls cancel mode order; observers run after timing."""
    require(warmup >= 0 and timed > 0 and count > 0, 'Invalid finite timing bounds')
    rows = []
    for window in range(count):
        order = ('IO', 'KV') if window % 2 == 0 else ('KV', 'IO')
        # Warm each mode equally before this window's alternating measurements.
        for mode in order:
            configure(mode)
            for _ in range(warmup):
                invoke(mode)
        row = {'window': window+1, 'first_mode': order[0], 'IO': [], 'KV': []}
        for _ in range(timed):
            for mode in order:
                configure(mode)  # installation/uninstallation is outside timing
                row[mode].append(measure(mode))
        rows.append(row)
    return rows


def summary(rows):
    result = {}
    for mode in ('IO', 'KV'):
        samples = [sample for row in rows for sample in row[mode]]
        require(samples, 'Missing timing samples')
        result[mode] = {key: statistics.median(sample[key] for sample in samples)
                        for key in ('wall_ms', 'CUDA_ms')}
    delta = [statistics.median(v['wall_ms'] for v in row['IO'])-
             statistics.median(v['wall_ms'] for v in row['KV']) for row in rows]
    return {'IO_wall_median_ms': result['IO']['wall_ms'], 'KV_wall_median_ms': result['KV']['wall_ms'],
            'IO_CUDA_median_ms': result['IO']['CUDA_ms'], 'KV_CUDA_median_ms': result['KV']['CUDA_ms'],
            'per_window_wall_saving_ms': delta, 'every_window_saves_at_least_20ms': all(v >= 20. for v in delta)}


def self_test():
    # No CUDA/model import. Verify balanced bounds, alternating order and exact
    # timing summary instead of pretending synthetic timing is GPU evidence.
    configured, calls = [], []
    def configure(mode): configured.append(mode)
    def invoke(mode): calls.append(('warm', mode))
    def measure(mode):
        calls.append(('timed', mode))
        return {'wall_ms': 100. if mode == 'IO' else 75., 'CUDA_ms': 90. if mode == 'IO' else 65.}
    rows = windows(configure, invoke, measure, warmup=2, timed=3, count=3)
    assert [r['first_mode'] for r in rows] == ['IO', 'KV', 'IO']
    assert all(len(row[mode]) == 3 for row in rows for mode in ('IO', 'KV'))
    assert all(calls.count((kind, mode)) == expected for kind, expected in [('warm', 6), ('timed', 9)] for mode in ('IO', 'KV'))
    assert summary(rows)['per_window_wall_saving_ms'] == [25., 25., 25.]
    assert summary(rows)['every_window_saves_at_least_20ms']
    print(json.dumps({'CPU_self_test': 'passed', 'GPU_launched': False}))


def run(args, report):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from starVLA.model.modules.structured_world.proposal_execution import (
        _attest_sources, install_proposal_kv_cache, uninstall_proposal_kv_cache, proposal_kv_cache_status)
    # Hard-pinned source proof BEFORE constructing weights or allocating CUDA.
    report['audited_sources'] = _attest_sources()
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Standalone single process required, not torchrun full8')
    require(Path(args.tag).name == args.tag and args.tag not in ('latest', '.', '..')
            and not args.tag.startswith('writing_'), 'Explicit immutable checkpoint tag required')
    identity = json.loads((args.run/'identity.json').read_text())
    folder = args.run/'checkpoints'/args.tag
    complete = json.loads((folder/'COMPLETE.json').read_text())
    require(identity['identity'] == complete['identity'] and complete['tag'] == args.tag, 'Foreign/incomplete checkpoint')
    require(identity['dataset'] == 'navsim' and identity['config']['structured_world']['group'] == 'G1_FULL_UNIFORM',
            'This probe is fixed to actual NAV G1')
    model_file = args.model_state_file or folder/'mp_rank_00_model_states.pt'
    require(file_digest(model_file) == complete['file_sha256']['mp_rank_00_model_states.pt'], 'Fixed checkpoint model hash changed')
    report['checkpoint'] = {'run': str(args.run.resolve()), 'tag': args.tag, 'completed': complete['completed'],
                            'identity': identity['identity'], 'model_file': str(model_file.resolve()),
                            'model_sha256': complete['file_sha256']['mp_rank_00_model_states.pt'],
                            'COMPLETE_sha256': file_digest(folder/'COMPLETE.json')}
    require(torch.cuda.is_available(), 'CUDA is required; CPU self-test uses --self-test')
    require(torch.cuda.device_count() == 1, 'Mask exactly one admitted GPU with CUDA_VISIBLE_DEVICES')
    device = torch.device(args.device)
    require(device.type == 'cuda', 'GPU timing requires an admitted CUDA device')
    torch.cuda.set_device(device)
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
    from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset
    from starVLA.model.modules.structured_world.execution import configure_execution
    from starVLA.model.modules.structured_world.fgtr import sample_proposal
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    saved = torch.load(model_file, map_location='cpu', mmap=True, weights_only=False)
    head_state = {name.removeprefix('action_model.'): tensor for name, tensor in saved['module'].items()
                  if name.startswith('action_model.')}
    require(head_state and all(not p.is_floating_point() or p.dtype == torch.float32 for p in head_state.values()),
            'Checkpoint must contain the native FP32 action master tensors')
    fixed_head_hash = tensor_digest(head_state)
    model = VLAStructuredFGTR(OmegaConf.create(identity['config'])).float()
    model.load_state_dict(saved['module'], strict=True)
    epoch, offset = saved['epoch'], saved['offset']
    del saved, head_state
    model.to(device).train(); model.inference_fp32 = False
    configure_execution(model, 'io_preserving_v1')
    head = model.action_model
    require(model.cameras == 3 and model.steps == 8 and head.num_inference_timesteps == 10,
            'Original NAV cameras/horizon/ten-step contract required')
    require(len(head.model.transformer_blocks) == 24 and head.model.inner_dim == 1536,
            'Actual 24-layer width-1536 head required')
    require(all(p.dtype == torch.float32 for p in head.parameters()), 'FP32 head masters required')
    require(tensor_digest(head.state_dict()) == fixed_head_hash, 'Restored action head differs from fixed checkpoint hash')
    data = StructuredNAVSIMDataset(args.cache, args.current_root, dino_root=args.dino_root, dino_index='',
                                  expected_dino=identity['DINO_identity'], image_root=args.image_root)
    require(data.identity['identity'] == identity['cache_identity'], 'Foreign NAV current/label population')
    batches = epoch_batches(len(data), 32, 42, epoch)
    if offset == len(batches):
        epoch += 1; offset = 0; batches = epoch_batches(len(data), 32, 42, epoch)
    require(0 <= offset < len(batches), 'Invalid native current batch offset')
    indices = batches[offset][::8][:4]  # Original next global batch, rank0 four scenes.
    require(len(indices) == 4, 'Checkpoint current batch must have four rank0 scenes')
    observations = [data[index][0] for index in indices]
    tokens = [data.index[index]['token'] for index in indices]
    with torch.no_grad():
        encoded = model.encode_current(observations)  # exactly ONE real current encode
        H_A = model.build_planner_condition(encoded).detach()
    require(H_A.shape[:2] == (4, 8), 'Current action-only eight-token planner memory required')
    del encoded, observations, data
    report['current'] = {'epoch': epoch, 'offset': offset, 'indices': indices,
                         'tokens': tokens,
                         'encode_current_calls': 1, 'H_A_shape': list(H_A.shape), 'H_A_dtype': str(H_A.dtype),
                         'H_A_sha256': tensor_digest({'H_A': H_A})}
    before = inventory(head)
    generator = torch.Generator(device=device).manual_seed(420300)
    noise = torch.randn((4, 8, 4), device=device, dtype=torch.float32, generator=generator)
    def rng():
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'CPU': torch.get_rng_state(),
                'CUDA_selected': torch.cuda.get_rng_state(device), 'proposal': generator.get_state()}
    baseline_rng = rng()
    reference = None
    counts = {'IO': 0, 'KV': 0}
    kv_stats = None
    def configure(mode):
        if mode == 'IO': uninstall_proposal_kv_cache(head)
        else: install_proposal_kv_cache(head, enabled=True)
    def invoke(mode):
        # Match the training proposal: native eval scope, outer no_grad, native
        # attention, FP32 masters and BF16 sampler. Keep autocast weight cache
        # lifetime identical in both arms (one context per proposal).
        with torch.no_grad(), model.amp():
            return sample_proposal(head, H_A, initial_noise=noise)
    def verify(mode, output):
        nonlocal reference, kv_stats
        require(torch.isfinite(output).all().item(), 'Nonfinite q0')
        if reference is None: reference = output.detach().clone()
        require(exact(reference, output), 'q0 bitwise mismatch: '+mode)
        require(exact(baseline_rng, rng()), 'Fixed-noise global/proposal RNG changed: '+mode)
        require(inventory(head) == before, 'Parameter/buffer/mode inventory changed: '+mode)
        counts[mode] += 1
        if mode == 'KV':
            status = proposal_kv_cache_status(head)
            require(not status['active'] and status['last_scope']['succeeded'], 'KV scope did not clear')
            kv_stats = status['last_scope']
    # A/A replay and A/B q0 proof before any timings.
    for mode in ('IO', 'IO', 'KV'):
        configure(mode); verify(mode, invoke(mode))
    report['adapter_provenance'] = proposal_kv_cache_status(head)['provenance']
    # Also exercise the REAL sampler's independent generator draw, not just
    # the fixed-noise timing path. Replay the generator start for both arms.
    generated = []
    for mode in ('IO', 'KV'):
        configure(mode)
        generator.set_state(baseline_rng['proposal'])
        with torch.no_grad(), model.amp():
            output = sample_proposal(head, H_A, generator=generator)
        generated.append((output.detach().clone(), rng()))
    require(exact(generated[0], generated[1]), 'Sampler generated-noise q0/RNG differs')
    require(not exact(generated[0][1]['proposal'], baseline_rng['proposal']), 'Sampler did not consume its independent RNG')
    require(all(exact(generated[0][1][key], baseline_rng[key]) for key in baseline_rng if key != 'proposal'),
            'Sampler changed global RNG')
    generator.set_state(baseline_rng['proposal'])
    report['sampler_generated_noise_q0_and_RNG_bitwise'] = True
    del generated
    def measured(mode):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter(); start.record()
        output = invoke(mode)
        end.record(); torch.cuda.synchronize(device)
        row = {'wall_ms': (time.perf_counter()-started)*1000., 'CUDA_ms': start.elapsed_time(end)}
        verify(mode, output)  # all host copies/checks OUTSIDE measured interval
        return row
    try:
        rows = windows(configure, invoke, measured, warmup=args.warmup, timed=args.timed, count=args.windows)
    finally:
        uninstall_proposal_kv_cache(head)
    require(inventory(head) == before, 'Final parameter/buffer/mode inventory changed')
    require(tensor_digest(head.state_dict()) == fixed_head_hash, 'Action head bytes changed after timing')
    require(report['audited_sources'] == _attest_sources(), 'Audited proposal sources changed during probe')
    report.update({'passed': True, 'timing': summary(rows), 'windows': rows,
                   'head_sha256_before_and_after': fixed_head_hash,
                   'head_parameter_inventory': before, 'q0_bitwise_and_RNG': True,
                   'verified_calls': counts, 'KV_last_scope': kv_stats,
                   'q0': reference.cpu().tolist(), 'GPU': torch.cuda.get_device_name(device),
                   'precision': 'FP32 masters/BF16 per-proposal autocast; native attention; TF32 disabled',
                   'warmup_per_mode_per_window': args.warmup, 'timed_per_mode_per_window': args.timed,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                   'limits': ['Single fixed current batch, no whole-training throughput or recovery claim',
                              'CUDA events include current-stream launch gaps, not kernel-only attribution',
                              'GPU admission is external and parent-owned; this script does not dispatch',
                              'Existing strict numerical failures are not superseded']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'cache', 'dino-root', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--current-root', type=Path, default=Path('/var/tmp/ddp-full-foresight-20260929/student_train_v1'))
    parser.add_argument('--image-root', type=Path, default=Path('/var/tmp/ddp-full-foresight-20260929/images'))
    parser.add_argument('--model-state-file', type=Path, help='Optional local byte-identical native model state')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--timed', type=int, default=50)
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    for name in ('run', 'tag', 'cache', 'dino_root', 'output'):
        if getattr(args, name) is None: parser.error('--'+name.replace('_', '-')+' is required')
    require(args.warmup >= 0 and 0 < args.timed <= 100 and 0 < args.windows <= 3 and args.warmup <= 20,
            'Bounded probe: <=20 warmup, <=100 timed, <=3 windows')
    require(not args.output.exists(), 'Output receipt already exists; never overwrite')
    report = {'schema': 'real_NAV_G1_current_proposal_KV_profile_v1', 'passed': False,
              'created_UTC': datetime.now(timezone.utc).isoformat(), 'script_sha256': file_digest(__file__)}
    try:
        run(args, report)
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'passed': report['passed'], **report['timing'], 'output': str(args.output)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
