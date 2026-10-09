"""Whole official training GT/body coverage, using mature VAD sensor poses.

Reads metadata only; no validation/test sampling or map-size search. Native
planning endpoints remain eligible regardless of auxiliary timestamp jitter.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nuscenes.nuscenes import NuScenes
from starVLA.dataloader.structured_world.nuscenes_adapter import planning_metadata
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points
from tools.structured_world.build_cache import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'population', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    population = json.loads((args.population/'identity.json').read_text())
    path = args.population/'train_scenes.json'
    if digest(path) != population['file_sha256']['train']: raise ValueError('Foreign population')
    rows = json.loads(path.read_text())
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.root), verbose=False)
    grid = GridSpec(); torch.set_num_threads(1)
    # Shared body reader calls this offset rear_axle_to_center; here its
    # reference is the explicitly disclosed current LiDAR origin.
    body = {'length': 4.084, 'width': 1.85, 'rear_axle_to_center': .5}
    trajectories, outside = [], []
    body_min, body_max = np.full(2, np.inf), np.full(2, -np.inf)
    start = time.monotonic()
    for index, row in enumerate(rows):
        metadata = planning_metadata(nusc, nusc.get('sample', row['token']))
        trajectory = torch.from_numpy(metadata['ego_physical'])
        points = body_points(trajectory, body)
        _, out = grid.normalized_reference(points)
        body_min = np.minimum(body_min, points.amin((0, 1)).numpy())
        body_max = np.maximum(body_max, points.amax((0, 1)).numpy())
        if out.any(): outside.append({'token': row['token'], 'steps': int(out.any(-1).sum())})
        trajectories.append(metadata['ego_physical'])
        if (index+1) % 2048 == 0: print(json.dumps({'scenes': index+1, 'seconds': time.monotonic()-start}), flush=True)
    xy = np.stack(trajectories)[..., :2]
    atomic_json(args.output, {'scope': 'whole official train metadata GT coverage, no planning result',
        'population_identity': population['identity'], 'scenes': len(rows), 'grid': asdict(grid),
        'body': body, 'origin': 'current LiDAR, internal x forward/y left',
        'trajectory_xy_min': xy.min((0, 1)).tolist(), 'trajectory_xy_max': xy.max((0, 1)).tolist(),
        'body_xy_min': body_min.tolist(), 'body_xy_max': body_max.tolist(),
        'scenes_with_body_out_of_range': len(outside), 'out_of_range_records': outside,
        'native_scenes_removed': 0, 'grid_changed': False, 'seconds': time.monotonic()-start,
        'code_sha256': digest(Path(__file__))})


if __name__ == '__main__':
    main()
