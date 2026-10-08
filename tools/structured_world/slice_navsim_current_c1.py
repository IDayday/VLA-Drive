"""Lossless current-C1 extraction; new runs never require future DINO assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import socket
from pathlib import Path
import sys
import threading
import time
import numpy as np
from safetensors import safe_open
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.build_cache import atomic_json
from tools.structured_world.training_assets import file_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'source', 'dino-index', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized teacher preparation host')
    cache = json.loads((args.cache/'identity.json').read_text())
    rows = json.loads((args.cache/'index.json').read_text())
    legacy = json.loads((args.source/'identity.json').read_text())
    if legacy['identity'] != '7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb':
        raise ValueError('Only the locked historical current C1 targets are authorized')
    scene_index = args.dino_index/'train_scenes.json'
    index_identity = json.loads((args.dino_index/'identity.json').read_text())
    if legacy['index'] != index_identity['identity'] or file_digest(scene_index) != index_identity['files']['train']:
        raise ValueError('Wrong teacher index')
    lookup = {r['token']: r for r in json.loads(scene_index.read_text())}
    contract = {'schema': 'structured_world_current_C1_v1', 'dataset': 'navsim', 'cameras': 3,
        'structured_cache_identity': cache['identity'], 'grid_hw': [6, 8], 'feature_dim': 1024,
        'dtype': 'float16 lossless copy from legacy current targets', 'recipe': legacy['recipe'],
        'legacy_source_identity': legacy['identity'], 'writer_sha256': file_digest(Path(__file__)),
        'source_teacher_index_sha256': file_digest(scene_index), 'future_teacher_required_by_training': False}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Foreign current-only teacher cache')
    atomic_json(args.output/'identity.json', {'identity': identity, **contract})
    for name in ('records', 'targets'): (args.output/name).mkdir(exist_ok=True)
    verified, lock = {}, threading.Lock()
    def image(index):
        if index < 0: raise ValueError('Missing current teacher image')
        chunk, at = divmod(index, legacy['chunk_size'])
        file = args.source/f'chunk_{chunk:06d}.safetensors'
        with lock:
            if chunk not in verified:
                record = json.loads((file.with_suffix('.json')).read_text())
                actual = file_digest(file)
                if record['identity'] != legacy['identity'] or actual != record['sha256']:
                    raise ValueError('Legacy teacher chunk changed')
                verified[chunk] = actual
        with safe_open(str(file), framework='pt', device='cpu') as stream:
            if stream.metadata()['identity'] != legacy['identity']: raise ValueError('Foreign teacher tensors')
            return stream.get_slice('features')[at].numpy(), stream.get_slice('valid')[at].numpy()
    def write(row):
        token = row['token']; target = args.output/'targets'/(token+'.npz'); meta = args.output/'records'/(token+'.json')
        if meta.exists():
            record = json.loads(meta.read_text())
            if record['identity'] != identity or file_digest(target) != record['sha256']: raise ValueError('Changed current teacher target')
            return token
        pairs = [image(index) for index in lookup[token]['images'][0]]
        features, valid = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        if features.shape != (3, 1024, 6, 8) or valid.shape != (3, 6, 8): raise ValueError('Current teacher layout changed')
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream, current_dino=features, valid=valid)
        temporary.replace(target)
        with np.load(target, allow_pickle=False) as reread:
            if not np.array_equal(reread['current_dino'], features): raise ValueError('Teacher copy changed values')
        atomic_json(meta, {'token': token, 'identity': identity, 'sha256': file_digest(target),
            'legacy_current_image_indices': lookup[token]['images'][0], 'bitwise_source_values_retained': True})
        return token
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        count = sum(1 for _ in pool.map(write, rows))
    atomic_json(args.output/'SOURCE_CHUNK_HASHES.json', verified)
    atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': count, 'seconds': time.monotonic()-start,
        'current_only': True, 'lossless_legacy_values': True, 'GPU_hours': 0.})
    print(json.dumps({'identity': identity, 'scenes': count, 'seconds': time.monotonic()-start}), flush=True)


if __name__ == '__main__': main()
