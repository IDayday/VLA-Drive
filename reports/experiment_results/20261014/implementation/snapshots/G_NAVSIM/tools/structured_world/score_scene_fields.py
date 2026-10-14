"""Fixed-population scene diagnostics, separate from canonical planning scores.

Dense valid cells and the SAME logged-GT body neighborhood are evaluated for
every group. The latter is an offline metric reference, never a model input.
Field reads reuse the training grid/body operators. Missing/OOR support is
reported, never converted to free space. Endpoint events are not first-entry
events, and body max-occupancy is a proxy, not an official collision probability.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points, body_relations, read_time_field
from tools.structured_world.build_cache import atomic_json, digest


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def binary_counts(probability, truth, valid):
    predicted = probability >= .5
    truth, valid = truth.bool(), valid.bool()
    return {'TP': int((predicted & truth & valid).sum()),
            'FP': int((predicted & ~truth & valid).sum()),
            'FN': int((~predicted & truth & valid).sum()),
            'TN': int((~predicted & ~truth & valid).sum()),
            'valid': int(valid.sum()), 'positive': int((truth & valid).sum()),
            'Brier_sum': float(((probability-truth.float()).square())[valid].sum())}


def summarize_counts(counts):
    if 'TP' in counts:
        tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
        return {**counts, 'IoU': ratio(tp, tp+fp+fn), 'precision': ratio(tp, tp+fp),
                'recall': ratio(tp, tp+fn), 'Brier': ratio(counts['Brier_sum'], counts['valid'])}
    result = dict(counts)
    for prefix in ('road', 'road_boundary', 'body_road', 'body_occupancy', 'temporal_change', 'unchanged_temporal_change',
                   'GT_to_prediction_endpoint_time', 'prediction_to_GT_endpoint_time'):
        if prefix+'_error_sum' in counts:
            result[prefix+'_MAE'] = ratio(counts[prefix+'_error_sum'], counts[prefix+'_count'])
    return result


def add_counts(total, row):
    for name, value in row.items(): total[name] = total.get(name, 0)+value


def fixed_neighborhood(labels, body, grid):
    """Union of complete body and +/-2m lateral offsets at each valid GT time."""
    times = len(labels['ego_physical'])
    masks = torch.zeros(times, grid.height, grid.width, dtype=torch.bool)
    for t, pose in enumerate(labels['ego_physical']):
        if not labels['future_valid'][t]: continue
        for lateral in (-2., 0., 2.):
            points = body_points(pose[None], body, lateral_offset_m=lateral)[0]
            _, out = grid.normalized_reference(points)
            cell = ((points-points.new_tensor([grid.xmin, grid.ymin]))/points.new_tensor([grid.dx, grid.dy])).floor().long()
            cell = cell[~out]
            masks[t, cell[:, 1], cell[:, 0]] = True
    return masks


def current_camera_support(labels, height_quality, grid):
    """Calibration/ROI support proxy; NOT object visibility or occlusion GT."""
    ground = height_quality[0].get('ground_level') if height_quality else None
    if ground is None: return None
    centers = grid.centers().reshape(-1, 2)
    points = torch.cat((centers, torch.full((len(centers), 1), float(ground)), torch.ones(len(centers), 1)), 1)
    pixel_valid = labels['geometry_pixel_valid']
    support = torch.zeros(len(centers), dtype=torch.bool)
    height, width = pixel_valid.shape[-2:]
    for camera in range(len(pixel_valid)):
        xyz = (points @ torch.linalg.inv(labels['calibration_sensor2ego'][camera]).T)[:, :3]
        pixels = xyz @ labels['calibration_intrinsics'][camera].T
        pixels = pixels/pixels[:, 2:].clamp_min(1e-8)
        pixels = pixels @ labels['calibration_post_rots'][camera].T+labels['calibration_post_trans'][camera]
        x, y = pixels[:, 0].round().long(), pixels[:, 1].round().long()
        inside = (xyz[:, 2] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        support[inside] |= pixel_valid[camera, y[inside], x[inside]].bool()
    return support.reshape(grid.height, grid.width)


def endpoint_time_counts(probability, truth, valid, dt=.5):
    """Nearest positive endpoint time, in BOTH directions, with miss counts.

    Only cells valid at every time participate. No accumulated first-entry
    interpretation is made. A missing predicted event has no timing error but
    is retained in unmatched counts and binary recall/precision.
    """
    predicted, truth = probability >= .5, truth.bool()
    complete = valid.all(0)
    p, g = predicted[:, complete], truth[:, complete]
    times = torch.arange(len(p), dtype=torch.float32)*dt
    result = {'complete_time_cells': int(complete.sum())}
    for name, source, target in [('GT_to_prediction_endpoint_time', g, p), ('prediction_to_GT_endpoint_time', p, g)]:
        available = target.any(0)
        differences = (times[:, None]-times[None, :]).abs()
        distances = torch.where(target[None], differences[:, :, None], torch.inf).amin(1)
        matched = source & available[None]
        result[name+'_error_sum'] = float(distances[matched].sum())
        result[name+'_count'] = int(matched.sum())
        result[name+'_unmatched_positive_endpoints'] = int((source & ~available[None]).sum())
    return result


def stationary_track_mask(labels, *, threshold_m=.5, grid):
    """Conservative raster-centroid approximation, disclosed separately."""
    instances, occupancy, valid = labels['instances'], labels['occupancy'], labels['occupancy_valid']
    complete_times = labels['future_valid'].all()
    result = torch.zeros_like(occupancy[1:], dtype=torch.bool)
    if not complete_times: return result
    current_ids = torch.unique(instances[0][(occupancy[0] == 1) & valid[0]])
    centers = grid.centers()
    for track in current_ids:
        if track <= 0 or track == 255: continue
        positions = []
        for frame in instances:
            mask = frame == track
            if not mask.any(): break
            positions.append(centers[mask].mean(0))
        if len(positions) != len(instances): continue
        if torch.linalg.vector_norm(torch.stack(positions)-positions[0], dim=1).amax() <= threshold_m:
            result |= (instances[1:] == track) & (occupancy[1:] == 1) & valid[1:]
    return result


def body_counts(prediction, labels, body, grid):
    times = len(labels['ego_physical'])
    poses = labels['ego_physical'].repeat_interleave(3, 0).clone()
    lateral = poses.new_tensor([-2., 0., 2.]).repeat(times)
    poses[:, 0] -= poses[:, 2].sin()*lateral
    poses[:, 1] += poses[:, 2].cos()*lateral
    time_index = torch.arange(times).repeat_interleave(3)[None]
    predicted = body_relations(prediction['road_distance_m'][None], prediction['pt'][None], poses[None], time_index, [body], grid=grid)
    truth = body_relations(labels['road_distance'][None, None].float(), (labels['occupancy'][1:, None] == 1)[None].float(), poses[None], time_index, [body], grid=grid)
    points = body_points(poses, body)
    coverage = points.shape[1]
    xy = points.reshape(1, -1, 2)
    sampled_road, in_road = grid.sample(labels['road_valid'][None, None].float(), xy)
    t = time_index[:, :, None].expand(-1, -1, coverage).reshape(1, -1)
    sampled_occ, in_occ = read_time_field(labels['occupancy_valid'][None, 1:, None].float(), xy, t, grid)
    road_valid = ((sampled_road[..., 0] >= 1.-1e-5) & in_road).reshape(1, -1, coverage).all(-1)
    occ_valid = ((sampled_occ[..., 0] >= 1.-1e-5) & in_occ).reshape(1, -1, coverage).all(-1)
    eligible_time = labels['future_valid'].repeat_interleave(3)[None]
    result = {'registered_body_queries': len(poses), 'invalid_auxiliary_time_queries': int((~eligible_time).sum()),
        'out_of_range_body_queries': int((~predicted['in_range']).sum())}
    for name, key, mask in [('body_road', 'road_margin_m', road_valid), ('body_occupancy', 'occupancy_overlap_proxy', occ_valid)]:
        mask = mask & eligible_time & predicted['in_range']
        result[name+'_error_sum'] = float((predicted[key]-truth[key]).abs()[mask].sum())
        result[name+'_count'] = int(mask.sum())
        result[name+'_unknown_or_out_of_range_queries'] = int((eligible_time & ~mask).sum())
    return result


def score_scene(prediction, labels, metadata, grid):
    h, w = grid.height, grid.width
    steps = len(labels['ego_physical'])
    for name, shape in [('road_distance_m', (1, h, w)), ('p0', (1, h, w)),
                        ('pt', (steps, 1, h, w)), ('p_enter', (steps, 1, h, w)), ('p_release', (steps, 1, h, w))]:
        if prediction[name].shape != shape or not torch.isfinite(prediction[name]).all():
            raise ValueError('Invalid prediction field layout or values: '+name)
        if name != 'road_distance_m' and ((prediction[name] < 0) | (prediction[name] > 1)).any():
            raise ValueError('Uncalibrated probability range: '+name)
    O, valid = labels['occupancy'], labels['occupancy_valid'].bool() & (labels['occupancy'] != 255)
    p0, pt = prediction['p0'][0], prediction['pt'][:, 0]
    pair = valid[:1] & valid[1:]
    enter, release = (O[:1] == 0) & (O[1:] == 1), (O[:1] == 1) & (O[1:] == 0)
    road_error = (prediction['road_distance_m'][0]-labels['road_distance']).abs()
    neighbor = fixed_neighborhood(labels, metadata['ego_body'], grid)
    scopes = {'global_valid_cells': torch.ones_like(neighbor), 'fixed_logged_GT_body_neighborhood': neighbor}
    support = current_camera_support(labels, metadata.get('height_quality'), grid)
    if support is not None:
        scopes['current_geometric_camera_support_proxy'] = support[None].expand_as(neighbor)
        scopes['outside_current_geometric_camera_support_proxy'] = (~support)[None].expand_as(neighbor)
    stationary = stationary_track_mask(labels, grid=grid)
    groups = {}
    for scope, region in scopes.items():
        current_region = region.any(0)
        road_mask = labels['road_valid'].bool() & current_region
        boundary_mask = road_mask & (labels['road_distance'].abs() <= 1.)
        groups[scope+'/road'] = {'road_error_sum': float(road_error[road_mask].sum()), 'road_count': int(road_mask.sum()),
            'road_boundary_error_sum': float(road_error[boundary_mask].sum()), 'road_boundary_count': int(boundary_mask.sum())}
        groups[scope+'/current_occupancy'] = binary_counts(p0, O[0] == 1, valid[0] & current_region)
        groups[scope+'/future_occupancy'] = binary_counts(pt, O[1:] == 1, valid[1:] & region)
        groups[scope+'/paired_future_occupancy'] = binary_counts(pt, O[1:] == 1, pair & region)
        groups[scope+'/enter'] = binary_counts(prediction['p_enter'][:, 0], enter, pair & region)
        groups[scope+'/release'] = binary_counts(prediction['p_release'][:, 0], release, pair & region)
        groups[scope+'/raster_stationary_track_retention'] = binary_counts(pt, torch.ones_like(pt, dtype=torch.bool), stationary & region)
        for t in range(steps):
            groups[scope+f'/future_occupancy_{(t+1)*.5:g}s'] = binary_counts(pt[t], O[t+1] == 1, valid[t+1] & region[t])
        for name, truth in [('enter', enter), ('release', release)]:
            groups[scope+'/'+name+'_endpoint_timing'] = endpoint_time_counts(prediction['p_'+name][:, 0], truth, pair & region)
        transition_valid = valid[1:-1] & valid[2:] & region[:-1] & region[1:]
        delta = pt[1:]-pt[:-1]
        change_error = (delta-((O[2:] == 1).float()-(O[1:-1] == 1).float())).abs()
        unchanged = transition_valid & (O[2:] == O[1:-1])
        groups[scope+'/temporal_consistency'] = {'temporal_change_error_sum': float(change_error[transition_valid].sum()),
            'temporal_change_count': int(transition_valid.sum()), 'unchanged_temporal_change_error_sum': float(delta.abs()[unchanged].sum()),
            'unchanged_temporal_change_count': int(unchanged.sum())}
    groups['fixed_logged_GT_body_relations'] = body_counts(prediction, labels, metadata['ego_body'], grid)
    deployment = {}
    for name in ('q0', 'q_final'):
        trajectory = prediction[name]
        if trajectory.shape != (steps, 3) or not torch.isfinite(trajectory).all(): raise ValueError('Invalid full trajectory')
        _, out = grid.normalized_reference(trajectory[:, :2])
        points = body_points(trajectory, metadata['ego_body'])
        _, body_out = grid.normalized_reference(points)
        deployment[name+'_points'] = steps
        deployment[name+'_out_of_range_points'] = int(out.sum())
        deployment[name+'_body_out_of_range_steps'] = int(body_out.any(-1).sum())
    groups['complete_deployment_trajectory_range'] = deployment
    groups['label_coverage'] = {'current_cells': h*w, 'current_valid_cells': int(valid[0].sum()),
        'future_cells': steps*h*w, 'future_valid_cells': int(valid[1:].sum()), 'future_times': steps,
        'auxiliary_eligible_future_times': int(labels['future_valid'].sum()),
        'unknown_current_camera_support_scenes': int(support is None)}
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads((args.predictions/'identity.json').read_text())
    label_source = json.loads((args.labels/'identity.json').read_text())
    done = json.loads((args.predictions/'COMPLETE.json').read_text())
    label_done = json.loads((args.labels/'COMPLETE.json').read_text())
    rows = json.loads((args.labels/'index.json').read_text())
    if not source.get('scene_diagnostics') or done['identity'] != source['identity'] or done['scenes'] != len(rows):
        raise ValueError('Complete, identical-population scene-field predictions required')
    if label_done['identity'] != label_source['identity'] or label_done['scenes'] != len(rows) or label_done['expected_scenes'] != len(rows):
        raise ValueError('Incomplete scene labels')
    tokens = {row['token'] for row in rows}
    if {p.stem for p in (args.predictions/'records').glob('*.json')} != tokens:
        raise ValueError('Scene populations differ; no easy-subset evaluation allowed')
    grid = GridSpec(**label_source['grid'])
    contract = {'schema': 'structured_world_scene_metrics_v1', 'prediction_identity': source['identity'],
        'label_identity': label_source['identity'], 'evaluator_sha256': digest(Path(__file__)), 'grid': asdict(grid),
        'binary_threshold': .5, 'road_boundary_band_m': 1., 'fixed_body_lateral_offsets_m': [-2., 0., 2.],
        'body_spacing_m': .5, 'sampling': 'all valid cells; identical offline GT-body neighborhood for every group',
        'event_time': 'bidirectional nearest active endpoint time, complete-time cells, unmatched events retained',
        'visibility': 'current calibration/ROI ground-plane support proxy, not object visibility or occlusion annotation',
        'stationary': 'same raster track at every keyframe, centroid displacement <=0.5m; approximate not true static GT',
        'relations': 'bilinear same fields, complete body min road/max occupancy; not official DAC/collision probability',
        'canonical_planning_metrics_modified': False}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Foreign scene metric output')
    atomic_json(args.output/'identity.json', {'identity': identity, **contract})
    aggregate, scene_rows = {}, []
    start = time.monotonic()
    for row in rows:
        token = row['token']
        prediction_file = args.predictions/'predictions'/f'{token}.npz'
        label_file = args.labels/'labels'/f'{token}.npz'
        prediction_record = json.loads((args.predictions/'records'/f'{token}.json').read_text())
        meta = json.loads((args.labels/'records'/f'{token}.json').read_text())
        if prediction_record['identity'] != source['identity'] or digest(prediction_file) != prediction_record['sha256']:
            raise ValueError('Prediction identity/hash mismatch')
        if meta['cache_identity'] != label_source['identity'] or digest(label_file) != meta['label_sha256']:
            raise ValueError('Label identity/hash mismatch')
        with np.load(prediction_file, allow_pickle=False) as data: prediction = {k: torch.from_numpy(data[k].copy()) for k in data}
        keys = ['road_distance', 'road_valid', 'occupancy', 'occupancy_valid', 'instances', 'ego_physical', 'future_valid',
                'geometry_pixel_valid', 'calibration_sensor2ego', 'calibration_intrinsics', 'calibration_post_rots', 'calibration_post_trans']
        with np.load(label_file, allow_pickle=False) as data:
            labels = {k: torch.from_numpy(data[k].copy()) for k in keys}
            if 'future_label_time_valid' in data:
                labels['future_valid'] &= torch.from_numpy(data['future_label_time_valid'].copy())
        # nuScenes's native planning mask remains complete; its separate time
        # eligibility is already applied to occupancy_valid in the label cache.
        groups = score_scene(prediction, labels, meta, grid)
        for name, counts in groups.items(): add_counts(aggregate.setdefault(name, {}), counts)
        scene_rows.append({'token': token, 'log': meta['log'], 'counts': groups,
                           'metrics': {name: summarize_counts(counts) for name, counts in groups.items()}})
    summary = {'identity': identity, 'scenes': len(rows), 'seconds': time.monotonic()-start,
        'pooling': 'cell/query-weighted counts; per-scene counts and metrics retained for clustered scene-weighted comparisons',
        'metrics': {name: summarize_counts(counts) for name, counts in aggregate.items()}}
    atomic_json(args.output/'SCENE_METRICS.json', summary)
    atomic_json(args.output/'SCENE_ROWS.json', scene_rows)
    atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': len(rows)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
