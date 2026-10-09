"""Measure the actual current-only single-scene canonical deployment path."""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.dataloader.structured_world.current_inputs import CurrentInputs
from tools.structured_world.training_assets import file_digest
from tools.structured_world.stage_timing import TrainingStageTiming


def summarize(values):
    return {'count': len(values), 'mean_s': statistics.mean(values), 'median_s': statistics.median(values),
            'p95_s': float(np.percentile(values, 95)), 'min_s': min(values), 'max_s': max(values)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'model-state-file', 'inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=4)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized deployment measurement host')
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    identity = json.loads((args.run/'identity.json').read_text())
    tag = (args.run/'checkpoints/latest').read_text().strip()
    if Path(tag).name != tag: raise ValueError('Unsafe checkpoint pointer')
    complete = json.loads((args.run/'checkpoints'/tag/'COMPLETE.json').read_text())
    if complete['identity'] != identity['identity']: raise ValueError('Foreign checkpoint')
    if file_digest(args.model_state_file) != complete['file_sha256']['mp_rank_00_model_states.pt']:
        raise ValueError('Local FP32 model is not byte-identical to the saved formal model')
    model = VLAStructuredFGTR(OmegaConf.create(identity['config'])).float()
    saved = torch.load(args.model_state_file, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['module'], strict=True); del saved
    model.shared_geometry_identity = identity['shared_geometry_identity']
    model.strip_auxiliary_heads().cuda().eval(); model.inference_fp32 = True
    if any(p.dtype != torch.float32 for p in model.parameters()): raise AssertionError('Non-FP32 deployment parameters')
    current = CurrentInputs(args.inputs)
    if current.identity['dataset'] != identity['dataset']: raise ValueError('Foreign deployment dataset')
    if args.scenes+args.warmup > len(current) or args.scenes < 16: raise ValueError('At least sixteen real current scenes required')
    timer = TrainingStageTiming(model)
    samples, rows = [], []
    for index in range(args.scenes+args.warmup):
        begin = time.perf_counter(); observation = current[index]
        load_seconds = time.perf_counter()-begin
        torch.cuda.synchronize(); begin = time.perf_counter()
        result = model.predict_action([observation], sampling_seed=42, return_diagnostics=True)
        torch.cuda.synchronize(); seconds = time.perf_counter()-begin
        if not torch.isfinite(result['q_final']).all(): raise AssertionError('Invalid deployed trajectory')
        stages = timer.consume_after_synchronize()
        if index >= args.warmup:
            rows.append({'token': observation['token'], 'model_wall_seconds': seconds,
                         'current_input_read_seconds': load_seconds, 'stages': stages})
            samples.append(seconds)
        del observation, result
    report = {'schema': 'real_current_only_single_scene_deployment_cost_v1',
        'training_identity': identity['identity'], 'training_source_sha': identity['training_source_sha'],
        'checkpoint': tag, 'trained_updates': complete['completed'], 'input_identity': current.identity['identity'],
        'scope': 'cost-only on real current development inputs; no planning score or endpoint claim',
        'hardware': torch.cuda.get_device_name(), 'GPUs': 1, 'batch_size': 1, 'warmup': args.warmup,
        'precision': 'FP32 parameters and execution; TF32 disabled; no autocast',
        'proposal': 'original ten steps, one candidate, one FGTR residual',
        'cameras': model.cameras, 'future_points': model.steps,
        'training_labels_read_during_predict': False, 'supervision_heads_removed': True,
        'executed_parameter_elements': sum(p.numel() for p in model.parameters()),
        'model_wall': summarize(samples),
        'current_input_first_read': summarize([r['current_input_read_seconds'] for r in rows]),
        'serial_current_read_plus_model': summarize([r['current_input_read_seconds']+r['model_wall_seconds'] for r in rows]),
        'stage_CUDA_current_stream_medians_s': {name: statistics.median(r['stages'][name]['CUDA_stream_ms']/1000 for r in rows)
            for name in ('Qwen_current_inclusive','Qwen_vision','proposal_10steps','current_geometry','future_space','FGTR')},
        'vision_precision_observations': [r['stages']['observed_Qwen_vision_precision'] for r in rows],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'weights_loaded_allocated_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
        'limitations': 'FP32 research evaluation implementation; not a real-time vehicle deployment benchmark. Stage intervals include host launch gaps and inclusive parents overlap.',
        'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('checkpoint','trained_updates','model_wall','serial_current_read_plus_model','peak_allocated_bytes')}),flush=True)


if __name__ == '__main__': main()
