"""Fixed historical C1 target recipe, six current nuScenes images only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
from torch import distributed as dist
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.foresight.dinov3_target_encoder import DINOv3TargetEncoder
from starVLA.model.modules.foresight.tradeoff import CANDIDATES, preprocess_current, pool_patches
from tools.structured_world.build_cache import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-images', type=int, default=16)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized host')
    rank, world, local = [int(os.environ.get(name, default)) for name, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    if world not in (1, 8): raise ValueError('One-card preparation or full authorized eight-card host required')
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cache = json.loads((args.cache/'identity.json').read_text())
    complete = json.loads((args.cache/'COMPLETE.json').read_text())
    if complete['identity'] != cache['identity'] or complete['scenes'] != complete['expected_scenes']:
        raise ValueError('Incomplete label population')
    rows = json.loads((args.cache/'index.json').read_text())
    model = DINOv3TargetEncoder(args.model_root, 'cuda')
    if model.recipe['timm'] != '1.0.20': raise ValueError('Use the locked historical C1 teacher environment')
    recipe = {k: v for k, v in model.recipe.items() if k not in
              ('source_rgb_range', 'short_side', 'padding', 'pooling', 'implementation_sha256', 'original_user_recipe')}
    recipe.update(source_rgb_range='full original16:9 image resized anisotropically to4:3; no crop',
                  padding='none', actual_tensor_chw=[3, 192, 256], pooling='2x2 independent per-view mean',
                  normalization='ImageNet RGB once; last post-norm patches; no extra L2',
                  teacher_input='current images only, no future targets generated')
    contract = {'schema': 'structured_world_current_C1_v1', 'dataset': cache['dataset'],
                'structured_cache_identity': cache['identity'], 'recipe': recipe, 'cameras': 6,
                'grid_hw': [6, 8], 'feature_dim': 1024, 'dtype': 'float16 storage',
                'writer_sha256': digest(Path(__file__)),
                'preprocess_sha256': digest(ROOT/'starVLA/model/modules/foresight/tradeoff.py')}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
            raise ValueError('Refusing to overwrite another teacher contract')
        atomic_json(args.output/'identity.json', {'identity': identity, **contract})
        for name in ('records', 'targets'): (args.output/name).mkdir(exist_ok=True)
    if world > 1: dist.barrier()
    start = time.monotonic(); written = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in rows[rank::world]:
            token = row['token']; metadata = args.output/'records'/(token+'.json'); target = args.output/'targets'/(token+'.npz')
            if metadata.exists():
                saved = json.loads(metadata.read_text())
                if saved['identity'] != identity or digest(target) != saved['sha256']: raise ValueError('Corrupt current teacher target')
                written += 1; continue
            record = json.loads((args.cache/'records'/(token+'.json')).read_text())['current_record']
            paths = record['image_paths']
            if len(paths) != 6: raise ValueError('Six current cameras required')
            pixels = list(pool.map(lambda path: preprocess_current(path, 256, 192)[0], paths))
            features = []
            for offset in range(0, len(pixels), args.batch_images):
                inputs = torch.stack(pixels[offset:offset+args.batch_images]).cuda()
                features.append(pool_patches(model(inputs), CANDIDATES['C1']).cpu().half().numpy())
            feature = np.concatenate(features)
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, current_dino=feature, valid=np.ones((6, 6, 8), bool))
            temporary.replace(target)
            atomic_json(metadata, {'token': token, 'identity': identity, 'sha256': digest(target),
                                   'source_image_sha256': [digest(Path(p)) for p in paths]})
            written += 1
            if written % 64 == 0: print(json.dumps({'rank': rank, 'scenes': written, 'seconds': time.monotonic()-start}), flush=True)
    cost = torch.tensor(time.monotonic()-start, device='cuda')
    count = torch.tensor(written, device='cuda')
    if world > 1:
        dist.all_reduce(cost, op=dist.ReduceOp.MAX); dist.all_reduce(count); dist.barrier()
    if int(count) != len(rows): raise ValueError('Teacher cache changed the population')
    if rank == 0:
        atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': int(count), 'seconds': float(cost),
                                                'GPU_hours': float(cost)*world/3600., 'current_only': True})
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
