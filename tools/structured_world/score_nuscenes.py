"""Score both recorded plans with the unchanged UniAD v2.0 metric class."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import numpy as np
from nuscenes.nuscenes import NuScenes
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.structured_world.nuscenes_adapter import adapted_nuscenes_scene, planning_metadata
from starVLA.model.modules.structured_world.nuscenes_evaluation import native_planning_segmentation, scene_metrics, PROTOCOL
from tools.structured_world.build_cache import digest, atomic_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'population', 'predictions', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--debug-only', action='store_true')
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized evaluation host')
    import torch
    torch.set_num_threads(2)
    prediction_identity = json.loads((args.predictions/'identity.json').read_text())
    complete = json.loads((args.predictions/'COMPLETE.json').read_text())
    if complete['identity'] != prediction_identity['identity']: raise ValueError('Incomplete predictions')
    rows = json.loads(args.population.read_text())
    tokens = {p.stem for p in (args.predictions/'records').glob('*.json')}
    if len(tokens) != complete['scenes']: raise ValueError('Prediction population incomplete')
    if args.debug_only: rows = [r for r in rows if r['token'] in tokens]
    if {r['token'] for r in rows} != tokens or len(rows) != len(tokens):
        raise ValueError('Scoring population differs from predictions')
    if not args.debug_only and (args.root/'PROVISIONAL_DEBUG_ONLY.json').exists():
        raise ValueError('Provisional data cannot produce formal validation results')
    contract = {'predictions': prediction_identity, 'population_sha256': digest(args.population),
        'protocol': PROTOCOL, 'scope': 'engineering_debug_not_formal' if args.debug_only else 'formal_validation',
        'code_sha256': {p: digest(ROOT/p) for p in ('tools/structured_world/score_nuscenes.py',
            'starVLA/model/modules/structured_world/nuscenes_evaluation.py', 'third_party/uniad/ported/planning_metrics.py')},
        'failure_policy': 'abort on missing or invalid labels/trajectories; never remove those scenes from denominator'}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Foreign evaluation output')
    atomic_json(args.output/'identity.json', {'identity': identity, **contract})
    (args.output/'records').mkdir(exist_ok=True)
    nusc = NuScenes('v1.0-trainval', dataroot=str(args.root), verbose=False)
    start = time.monotonic(); scored = []
    for row in rows:
        token = row['token']; output = args.output/'records'/(token+'.json')
        record = json.loads((args.predictions/'records'/(token+'.json')).read_text())
        prediction = args.predictions/'predictions'/(token+'.npz')
        if record['identity'] != prediction_identity['identity'] or digest(prediction) != record['sha256']:
            raise ValueError('Prediction changed after inference')
        if output.exists():
            saved = json.loads(output.read_text())
            if saved['identity'] != identity or saved['prediction_sha256'] != record['sha256']: raise ValueError('Foreign score record')
            scored.append(saved); continue
        sample = nusc.get('sample', token); scene = adapted_nuscenes_scene(nusc, sample)
        gt = planning_metadata(nusc, sample)['original_VAD_lidar_xy']
        segmentation = native_planning_segmentation(scene)
        with np.load(prediction, allow_pickle=False) as plans:
            metrics = {key: scene_metrics(plans[key], gt, segmentation) for key in ('q0', 'q_final')}
        saved = {'identity': identity, 'token': token, 'scene': scene.log,
            'prediction_sha256': record['sha256'], 'metrics': metrics}
        atomic_json(output, saved); scored.append(saved)
    summary = {'identity': identity, 'scenes': len(scored), 'seconds': time.monotonic()-start, 'protocol': PROTOCOL,
        'scope': contract['scope'], 'main_result': 'q_final', 'sampling_seed_is_not_training_seed': True}
    for variant in ('q0', 'q_final'):
        summary[variant] = {name: float(np.mean([r['metrics'][variant][name] for r in scored]))
            for name in scored[0]['metrics'][variant] if name != 'per_timestep'}
    atomic_json(args.output/'COMPLETE.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
