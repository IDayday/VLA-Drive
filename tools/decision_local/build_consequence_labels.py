"""Compact labels for fixed complete queries; reuse original fields, not RGB/BEV."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, identity, read, source_sha
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig, trajectory_queries
from starVLA.model.modules.decision_local.consequence_geometry import read_consequences, current_cv_occupancy, carry_static_ground_support
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.dataloader.foresight_dataset import decode_ego


@lru_cache(maxsize=2)
def current_log(root, log):
    with (Path(root)/(log+'.pkl')).open('rb') as stream:
        frames = pickle.load(stream)
    return {f['token']: f for f in frames}


def scene(task):
    token, log, q0, state, anchor, split, query_config, clip, ground_policy = task
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    root = Path(anchor['data'][split]['label_root'])
    metadata = read(root/'records'/(token+'.json'))
    path = root/'labels'/(token+'.npz')
    if metadata['cache_identity'] != anchor['data'][split]['label_identity'] or digest(path) != metadata['label_sha256']:
        raise ValueError('Foreign or changed original geometry label')
    with np.load(path, allow_pickle=False) as f:
        labels = {k: f[k].copy() for k in ('road_distance', 'road_valid', 'occupancy', 'occupancy_valid', 'instances', 'future_valid')}
    support_receipt={'rule':'strict existing G label validity'}
    if ground_policy=='current_static_ground':
        labels['occupancy'],labels['occupancy_valid'],support_receipt=carry_static_ground_support(
            labels['occupancy'],labels['occupancy_valid'],labels['instances'],labels['future_valid'],metadata['height_quality'])
    T = q0.shape[1]
    if labels['occupancy'].shape[0] != T+1:
        raise ValueError('Auxiliary physical-time labels differ from query horizon')
    frame = current_log('/mnt/navsim/trainval_navsim_logs/trainval', log)[token]
    cv_occ, cv_valid, cv_receipt = current_cv_occupancy(frame, labels['occupancy'][0], labels['occupancy_valid'][0],
                                                       np.arange(1, T+1)*.5)
    current_occ = np.repeat(labels['occupancy'][:1], T, axis=0)
    current_valid = np.repeat(labels['occupancy_valid'][:1], T, axis=0)
    sources = [(current_occ, current_valid), (cv_occ, cv_valid), (labels['occupancy'][1:], labels['occupancy_valid'][1:])]
    velocity = decode_ego(torch.from_numpy(state)[None])[0, :2].numpy()/.5
    body = metadata['ego_body']
    rows = []
    for proposal in q0:
        query = trajectory_queries(proposal, velocity, QueryConfig(**query_config))
        values, valid, support = [], [], []
        for occ, mask in sources:
            v, m, s = read_consequences(labels['road_distance'], labels['road_valid'], occ, mask,
                                        query['poses'], body, clip_m=clip)
            values.append(v); valid.append(m); support.append(s)
        # Input support for every GT information model is the SAME intersection.
        common = np.logical_and.reduce(valid)
        rows.append({'poses': query['poses'], 'kinematic_valid': query['kinematic_valid'],
            'in_grid': support[0]['body_in_grid'], 'relations': np.stack(values), 'valid': np.stack(valid),
            'common_valid': common, 'kinematics': np.array([[k['speed_max_mps'], k['acceleration_max_mps2'],
                                                          k['yaw_rate_max_rps']] for k in query['kinematics']], dtype=np.float32)})
    ego_root = Path(anchor['data'][split]['current_root'])
    gt = torch.load(ego_root/'ego'/(token+'.pt'), map_location='cpu', weights_only=True)
    ego_identity = read(ego_root/'ego_identity.json')['identity']
    if gt['identity'] != ego_identity or gt['token'] != token:
        raise ValueError('Original unmodified planning target mismatch')
    result = {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}
    result.update(ego=gt['ego'].float(), future_valid=torch.ones(T, dtype=torch.bool),
                   body=torch.tensor([body[k] for k in ('length', 'width', 'rear_axle_to_center')]),
                   cv_receipt=cv_receipt, support_receipt=support_receipt, label_sha256=metadata['label_sha256'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor', type=Path, required=True); p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-seconds', type=float, required=True); p.add_argument('--query-config', type=Path)
    p.add_argument('--ground-policy',choices=('strict','current_static_ground'),default='strict')
    p.add_argument('--follow-cache',action='store_true',help='Consume newly committed anchor chunks within the same finite time budget')
    args = p.parse_args()
    resources = check_host(); anchor = read(args.anchor); cache = read(args.cache/'identity.json')
    if cache['anchor_identity'] != anchor['identity'] or cache['split'] not in ('train', 'dev'):
        raise ValueError('Registered train/dev anchor required')
    config = read(args.query_config) if args.query_config else QueryConfig().to_dict()
    contract = {'schema': 'decision_local_query_consequences_v1', 'anchor_identity': anchor['identity'],
        'anchor_cache_identity': cache['identity'], 'source_sha': source_sha(ROOT),
        'code_files': {name: digest(ROOT/name) for name in ('tools/decision_local/build_consequence_labels.py',
            'starVLA/model/modules/decision_local/consequence_geometry.py', 'starVLA/model/modules/decision_local/trajectory_queries.py')},
        'split': cache['split'], 'query_config': config, 'distance_clip_m': 10., 'body_spacing_m': .5,
        'ground_support_policy':args.ground_policy,
        'sources': ['CURRENT', 'CURRENT_CV', 'TRUE_FUTURE'], 'channels': ['road_margin_m', 'obstacle_clearance_m', 'geometric_overlap'],
        'validity': 'original unknown/height masks; distance cannot cross closer unknown; complete body; oracle uses source intersection',
        'future_times': 'true physical time; never warp environment', 'cost': 'CPU labels only; no feature/model evaluation'}
    ident = identity(contract); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')['identity'] != ident:
        raise ValueError('Refusing incompatible consequence labels')
    atomic_json(args.output/'identity.json', {'identity': ident, **contract})
    for name in ('chunks', 'records'): (args.output/name).mkdir(exist_ok=True)
    started = time.monotonic(); done = 0
    def available_receipts():
        seen=set()
        while time.monotonic()-started<args.max_seconds:
            for source in sorted((args.cache/'records').glob('*.json')):
                if source.name in seen:continue
                seen.add(source.name);yield source
            if not args.follow_cache:break
            if (args.cache/'COMPLETE.json').exists():
                # The last producer may commit between our scan and this check.
                # Rescan once against the committed complete manifest.
                for source in sorted((args.cache/'records').glob('*.json')):
                    if source.name not in seen:seen.add(source.name);yield source
                break
            time.sleep(min(10.,max(0.,args.max_seconds-(time.monotonic()-started))))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for source in available_receipts():
            chunk = source.stem; receipt = read(source)
            out = args.output/'chunks'/(chunk+'.safetensors'); meta = args.output/'records'/(chunk+'.json')
            if time.monotonic()-started >= args.max_seconds: break
            if meta.exists():
                saved = read(meta)
                if saved['identity'] != ident or digest(out) != saved['sha256']: raise ValueError('Changed consequence chunk')
                done += len(saved['tokens']); continue
            features_path = args.cache/'chunks'/(chunk+'.safetensors')
            if receipt['identity'] != cache['identity'] or digest(features_path) != receipt['sha256']:
                raise ValueError('Changed frozen features')
            tensors = load_file(str(features_path)); by_token = {r['token']: r for r in cache['index']}
            tasks = [(token, by_token[token]['log'], tensors['q0'][i].numpy(), tensors['state'][i].numpy(),
                      anchor, cache['split'], config, 10., args.ground_policy) for i, token in enumerate(receipt['tokens'])]
            results = list(pool.map(scene, tasks))
            numeric = {k: torch.stack([r[k] for r in results]).contiguous() for k in results[0] if torch.is_tensor(results[0][k])}
            temporary = out.with_name(out.name+f'.{os.getpid()}.tmp')
            save_file(numeric, str(temporary), metadata={'identity': ident}); temporary.replace(out)
            atomic_json(meta, {'identity': ident, 'tokens': receipt['tokens'], 'sha256': digest(out),
                'bytes': out.stat().st_size, 'label_sha256': [r['label_sha256'] for r in results],
                'current_cv': [r['cv_receipt'] for r in results],
                'support_extension': [r['support_receipt'] for r in results],
                'valid_fraction_sources_channels': numeric['valid'].float().mean((0, 1, 3, 4)).tolist(),
                'common_valid_fraction': float(numeric['common_valid'].float().mean())})
            done += len(results); print(json.dumps({'scenes': done, 'chunk': chunk, 'seconds': time.monotonic()-started}), flush=True)
    status={'identity': ident, 'scenes': done, 'seconds': time.monotonic()-started,
        'CPU_worker_wall_hours_including_wait': (time.monotonic()-started)*args.workers/3600., 'resource_check': resources}
    atomic_json(args.output/'STATUS.json',status)
    atomic_json(args.output/'attempts'/(str(time.time_ns())+'.json'),status)
    if done == len(cache['index']):
        atomic_json(args.output/'COMPLETE.json', {'identity': ident, 'scenes': done, 'full_cache_population': True})


if __name__ == '__main__': main()
