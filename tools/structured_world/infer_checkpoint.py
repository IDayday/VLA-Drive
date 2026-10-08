"""FP32 master restoration and current-only single-proposal inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.dataloader.structured_world.current_inputs import CurrentInputs
from tools.structured_world.build_cache import digest, atomic_json


def scene_noise(token, seed, steps, device):
    # The same scene-bound CPU noise contract as the historical canonical
    # helper, parameterized only for nuScenes's six-point horizon.
    bound = int.from_bytes(hashlib.sha256(f'foresight-action-v1:{seed}:{token}'.encode()).digest()[:8], 'little')%(2**63-1)
    generator = torch.Generator(device='cpu').manual_seed(bound)
    return torch.randn((1, steps, 4), generator=generator, dtype=torch.float32).to(device)


def load_model(run, tag, device):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    identity = json.loads((run/'identity.json').read_text())
    complete = json.loads((run/'checkpoints'/tag/'COMPLETE.json').read_text())
    if complete['identity'] != identity['identity']: raise ValueError('Foreign/incomplete checkpoint')
    config = OmegaConf.create(identity['config'])
    model = VLAStructuredFGTR(config).float()
    state = get_fp32_state_dict_from_zero_checkpoint(str(run/'checkpoints'), tag=tag)
    model.load_state_dict(state, strict=True); del state
    model.to(device).eval(); model.inference_fp32 = True
    model.shared_geometry_identity = identity['shared_geometry_identity']
    return model, identity, complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--scene-fields', action='store_true')
    parser.add_argument('--strip-supervision-heads', action='store_true')
    args = parser.parse_args()
    if args.scene_fields and args.strip_supervision_heads: raise ValueError('Scene diagnostics require their label heads')
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized inference host')
    rank, world, local = [int(os.environ.get(name, default)) for name, default in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data = CurrentInputs(args.inputs)
    model, source, checkpoint = load_model(args.run, args.tag, 'cuda')
    if source['dataset'] != data.identity['dataset']: raise ValueError('Input dataset differs from checkpoint')
    if args.strip_supervision_heads: model.strip_auxiliary_heads()
    contract = {'schema': 'structured_world_FP32_predictions_v1', 'training_identity': source['identity'],
        'training_source_sha': source['training_source_sha'], 'checkpoint': args.tag, 'updates': checkpoint['completed'],
        'evaluation_source_files': {p: digest(ROOT/p) for p in
            ('tools/structured_world/infer_checkpoint.py', 'starVLA/dataloader/structured_world/current_inputs.py')},
        'input_identity': data.identity['identity'], 'sampling_seed': args.sampling_seed,
        'sampling': 'independent CPU scene-bound Gaussian, original 10-step FM, one residual and one final candidate',
        'precision': 'FP32 native ZeRO master restore, strict=True; FP32 inference; TF32 disabled',
        'supervision_heads_removed': args.strip_supervision_heads, 'scene_diagnostics': args.scene_fields}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Foreign prediction population')
    if rank == 0:
        atomic_json(args.output/'identity.json', {'identity': identity, **contract})
        for name in ('predictions', 'records'): (args.output/name).mkdir(exist_ok=True)
    if world > 1:
        from torch import distributed as dist
        if not dist.is_initialized(): dist.init_process_group('nccl')
        dist.barrier()
    start = time.monotonic(); done = 0
    for index in range(rank, len(data), world):
        token = data.index[index]['token']; record = args.output/'records'/(token+'.json'); output = args.output/'predictions'/(token+'.npz')
        if record.exists():
            saved = json.loads(record.read_text())
            if saved['identity'] != identity or digest(output) != saved['sha256']: raise ValueError('Prediction identity/hash changed')
            done += 1; continue
        observation = data[index]; noise = scene_noise(token, args.sampling_seed, model.steps, 'cuda')
        with torch.no_grad():
            planned = model.plan_from_current([observation], initial_noise=noise)
            values = {key: planned[key][0].cpu().numpy() for key in ('q0', 'q_final', 'out_of_range')}
            if args.scene_fields:
                fields = model.scene_semantics(planned['current']['B0'], planned['Bt'])
                values.update({key: fields[key][0].cpu().numpy() for key in
                              ('road_distance_m', 'p0', 'pt', 'p_enter', 'p_release')})
        if not np.isfinite(values['q_final']).all(): raise FloatingPointError('Invalid complete trajectory; scene is not removed from population')
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream, **values)
        temporary.replace(output)
        atomic_json(record, {'token': token, 'identity': identity, 'sha256': digest(output)})
        done += 1
    torch.cuda.synchronize()
    count = torch.tensor(done, device='cuda'); cost = torch.tensor(time.monotonic()-start, device='cuda')
    if world > 1:
        dist.all_reduce(count); dist.all_reduce(cost, op=dist.ReduceOp.MAX); dist.barrier()
    if int(count) != len(data): raise ValueError('Prediction population incomplete')
    if rank == 0:
        atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': int(count),
            'seconds': float(cost), 'GPU_hours': float(cost)*world/3600., 'full_population': True})
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
