"""Measure fixed 0.5m EDT and full-body reads against Shapely geometry."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from shapely import affinity, contains_xy, distance, points
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.structured_world.labels import road_distance, RoadMap
from starVLA.dataloader.structured_world.adapters import navsim_scene
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points
from tools.structured_world.build_cache import atomic_json


def measure(polygon, grid, *, poses):
    centers = grid.centers().numpy()
    signed = distance(polygon.boundary, points(centers))
    signed = np.where(contains_xy(polygon, centers[..., 0], centers[..., 1]), signed, -signed)
    oracle = np.clip(signed, -10., 10.)
    field = road_distance(polygon, grid)
    error = np.abs(field-oracle)
    boundary = np.abs(signed) <= 1.
    body = dict(length=4.084, width=1.85, rear_axle_to_center=.5)
    sample = body_points(torch.from_numpy(poses).float(), body)
    read, valid = grid.sample(torch.from_numpy(field)[None, None], sample.reshape(1, -1, 2))
    margins = read.reshape(len(poses), -1).amin(1).numpy()
    supported = valid.reshape(len(poses), -1).all(1).numpy()
    exact_containment, inside_margins, inside_errors = [], [], []
    for pose, approximation, ok in zip(poses, margins, supported):
        vehicle = box(-body['length']/2+body['rear_axle_to_center'], -body['width']/2,
                       body['length']/2+body['rear_axle_to_center'], body['width']/2)
        vehicle = affinity.rotate(vehicle, pose[2], origin=(0., 0.), use_radians=True)
        vehicle = affinity.translate(vehicle, pose[0], pose[1])
        inside = polygon.covers(vehicle); exact_containment.append(inside)
        if inside and ok:
            margin = vehicle.distance(polygon.boundary)
            inside_margins.append(margin); inside_errors.append(abs(approximation-margin))
    exact_containment = np.asarray(exact_containment)
    proxy_inside = margins >= 0.
    return {'grid': dict(H=grid.height, W=grid.width, dx=grid.dx, dy=grid.dy),
        'point_SDF_MAE_m': float(error.mean()), 'point_SDF_max_error_m': float(error.max()),
        'boundary_1m_SDF_MAE_m': float(error[boundary].mean()) if boundary.any() else None,
        'body_poses': len(poses), 'body_in_grid': int(supported.sum()),
        'body_inside_SDF_margin_MAE_m': float(np.mean(inside_errors)) if inside_errors else None,
        'body_inside_SDF_margin_max_error_m': float(np.max(inside_errors)) if inside_errors else None,
        'false_nonnegative_raster_body_margin': int((supported & proxy_inside & ~exact_containment).sum()),
        'false_negative_raster_body_margin': int((supported & ~proxy_inside & exact_containment).sum()),
        'out_of_range_body_queries_retained': int((~supported).sum()),
        'scope': 'raster/body-point approximation; proxy is not official DAC or collision probability'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--current-root', type=Path); parser.add_argument('--real-scenes', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    grid = GridSpec(-10., 10., -10., 10., .5, .5)
    poses = np.column_stack((rng.uniform(-7., 7., 512), rng.uniform(-7., 7., 512), rng.uniform(-np.pi, np.pi, 512)))
    polygons = {'straight': box(-30., -3., 30., 3.),
        'corner': unary_union([box(-8., -3., 3., 3.), box(-3., -3., 3., 8.)]),
        'hole': Polygon(box(-9., -9., 9., 9.).exterior.coords, holes=[box(-1., -1., 1., 1.).exterior.coords]),
        'rotated': affinity.rotate(box(-30., -3., 30., 3.), 31., origin=(0., 0.))}
    report = {'kind': 'geometry_approximation_QA_no_planning_score_selection',
        'synthetic': {name: measure(polygon, grid, poses=poses) for name, polygon in polygons.items()}, 'real_train': []}
    if args.current_root:
        road = RoadMap('navsim', '/mnt/navsim/maps')
        rows = json.loads((args.current_root/'index.json').read_text())
        ids = sorted(rng.choice(len(rows), min(args.real_scenes, len(rows)), replace=False))
        for index in ids:
            record = json.loads((args.current_root/'current'/(rows[index]['token']+'.json')).read_text())
            scene = navsim_scene(record)
            anchors = scene.ego_physical[rng.integers(8, size=128)].copy()
            anchors[:, :2] += rng.uniform(-3., 3., (len(anchors), 2)); anchors[:, 2] += rng.uniform(-.3, .3, len(anchors))
            values = measure(road.polygon(scene, GridSpec()), GridSpec(), poses=anchors)
            values['token'] = scene.token; report['real_train'].append(values)
    atomic_json(args.output, report)
    print(json.dumps({'synthetic': report['synthetic'], 'real_train_scenes': len(report['real_train'])}), flush=True)


if __name__ == '__main__': main()
