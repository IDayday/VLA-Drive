"""Read-only historical Qwen/DDP latency through its pinned native loader."""
import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import types
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    for name in ('legacy-source', 'run', 'inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--scenes', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=4)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized read-only diagnostic host')
    if args.scenes < 16: raise ValueError('At least sixteen real current scenes required')
    sys.path.insert(0, str(args.legacy_source))
    from tools.foresight.checkpoints import checkpoint_identity, load_student, scene_noise
    import starVLA
    if Path(starVLA.__file__).resolve().parent != (args.legacy_source/'starVLA').resolve():
        raise ValueError('Historical implementation was shadowed by another worktree')
    # Load the unchanged current-only input files under an isolated namespace;
    # the historical starVLA package remains exclusively from its pinned root.
    adapter = types.ModuleType('cost_current_inputs')
    adapter.__path__ = [str(ROOT/'starVLA/dataloader/structured_world')]
    sys.modules[adapter.__name__] = adapter
    CurrentInputs = importlib.import_module('cost_current_inputs.current_inputs').CurrentInputs
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    identity, checkpoint = checkpoint_identity(args.run, args.tag)
    if identity['source_sha'] != '1493deda247107efe076b9294863a6753391298a':
        raise ValueError('Only the verified historical S0 source is eligible for this probe')
    current = CurrentInputs(args.inputs)
    if current.identity['dataset'] != 'navsim': raise ValueError('NAVSIM current inputs required')
    started = time.perf_counter()
    model = load_student(args.run, args.tag, identity, device='cuda', precision='fp32', strip=True)
    torch.cuda.synchronize(); loading = time.perf_counter()-started
    precision = 'canonical_fp32'
    rows, pending = [], []
    def wrap(name, original):
        def measured(*values, **keywords):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); result = original(*values, **keywords); end.record()
            pending.append((name, begin, end)); return result
        return measured
    model.encode_current = wrap('Qwen_current_inclusive', model.encode_current)
    model.action_model.predict_action = wrap('proposal_10steps', model.action_model.predict_action)
    for index in range(args.warmup+args.scenes):
        begin = time.perf_counter(); observation = current[index]
        read_seconds = time.perf_counter()-begin
        noise = scene_noise(observation['token'], 42, 'cuda')
        torch.cuda.synchronize(); begin = time.perf_counter()
        result = model.predict_action([observation], initial_noise=noise)
        torch.cuda.synchronize(); seconds = time.perf_counter()-begin
        if not torch.isfinite(result).all(): raise AssertionError('Invalid historical deployed trajectory')
        stages = {name: start.elapsed_time(end)/1000 for name, start, end in pending}; pending.clear()
        if index >= args.warmup:
            rows.append({'token': observation['token'], 'model_wall_seconds': seconds,
                         'current_input_read_seconds': read_seconds, 'stages': stages})
        del observation, result
    def summarize(values):
        return {'count': len(values), 'mean_s': statistics.mean(values), 'median_s': statistics.median(values),
                'p95_s': float(np.percentile(values, 95)), 'min_s': min(values), 'max_s': max(values)}
    report = {'schema': 'read_only_historical_S0_DDP_deployment_cost_v1', 'checkpoint': checkpoint,
        'legacy_source_root': str(args.legacy_source), 'source_sha': identity['source_sha'],
        'input_identity': current.identity['identity'], 'batch_size': 1, 'GPUs': 1,
        'hardware': torch.cuda.get_device_name(), 'precision_policy': precision,
        'precision': 'native FP32 optimizer master reconstruction, strict=True; FP32 inference; TF32 off',
        'sampling': 'original ten steps, single candidate, scene-bound CPU FP32 initial noise seed42',
        'warmup': args.warmup, 'model_loading_seconds': loading,
        'model_wall': summarize([r['model_wall_seconds'] for r in rows]),
        'serial_current_read_plus_model': summarize([r['model_wall_seconds']+r['current_input_read_seconds'] for r in rows]),
        'stage_CUDA_current_stream_medians_s': {name: statistics.median(r['stages'][name] for r in rows)
                                              for name in ('Qwen_current_inclusive', 'proposal_10steps')},
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'legacy_assets_modified': False, 'scope': 'read-only latency; no training, official score or quality comparison',
        'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError('Refusing to overwrite historical cost evidence')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('model_wall','serial_current_read_plus_model','peak_allocated_bytes')}),flush=True)


if __name__ == '__main__': main()
