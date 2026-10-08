"""Locked UniAD v2.0 open-loop metric, separate from derived auxiliary labels."""
import numpy as np
import torch
from third_party.uniad.ported.planning_metrics import PlanningMetric
from starVLA.dataloader.structured_world.nuscenes_adapter import to_lidar


PROTOCOL = {
    'name': 'UniAD-v2.0-style open-loop planning',
    'commit': '609ee083ea51c3521c323f1279dfc4cee0e60467',
    'time_aggregation': 'cumulative mean of first 2/4/6 positions at 1/2/3 seconds',
    'collision': 'native fixed axis-aligned 4.084 x 1.85 m raster rectangle with +0.5 m forward offset',
    'GT_self_collision': 'native per-timestep exclusion of predicted collision where GT collides; denominator remains all eligible scenes',
    'L2': 'native masked Euclidean XY distance, complete six-keyframe population',
    'grid': 'native 200 x 200, +/-50 m, 0.5 m; distinct from auxiliary grid',
    'out_of_range': 'native metric preserves its clipping behavior; all scenes also receive explicit out-of-range diagnostics',
    'ego_state': 'legal previous/current sensor poses provided identically',
    'obstacles': 'original UniAD only_vehicle=True: cars, buses, trucks, trailers, construction vehicles, bicycles, motorcycles; filter_invisible=False',
    'not_closed_loop': True,
}


def native_planning_segmentation(scene):
    """Original UniAD raster/box transforms on its native evaluation grid.

    Evaluation does not reuse auxiliary unknown masks, grid, or class mapping.
    This is the original vehicle collision protocol, not all environment
    obstacles and not a new official nuScenes planning challenge.
    """
    from third_party.uniad.ported.occflow_label import GenerateOccFlowLabels
    from starVLA.dataloader.structured_world.nuscenes_adapter import INTERNAL_TO_LIDAR
    generator = GenerateOccFlowLabels(dict(xbound=[-50., 50., .5], ybound=[-50., 50., .5],
        zbound=[-10., 10., 20.]), only_vehicle=True, filter_invisible=False, compute_flow=False)
    if len(scene.frames) != 7 or any(frame is None for frame in scene.frames):
        raise ValueError('Native evaluation requires the common full-future population')
    rotation = np.eye(4); rotation[:3, :3] = INTERNAL_TO_LIDAR.T
    reference = scene.inverse_transform@rotation
    identities = np.eye(3, dtype=np.float32)
    tracks = {name: index+1 for index, name in enumerate(sorted({t for f in scene.frames for t in f.track_tokens}))}
    results = {'future_gt_bboxes_3d': [], 'future_gt_labels_3d': [], 'future_gt_inds': [],
               'future_gt_vis_tokens': [], 'occ_l2e_r_mats': [], 'occ_l2e_t_vecs': [],
               'occ_e2g_r_mats': [], 'occ_e2g_t_vecs': [], 'occ_has_invalid_frame': False,
               'occ_img_is_valid': np.ones(7, bool)}
    # An explicit empty native reference timestep makes all later boxes reframe
    # into LiDAR(t0), rather than into the adapted current internal frame.
    from starVLA.dataloader.structured_world.adapters import box_container
    for index, frame in enumerate(scene.frames):
        relative = np.linalg.inv(reference)@frame.ego2global
        results['future_gt_bboxes_3d'].append(box_container([]) if index == 0 else frame.boxes)
        results['future_gt_labels_3d'].append(np.zeros(0, np.int64) if index == 0 else frame.labels)
        results['future_gt_inds'].append(np.zeros(0, np.int64) if index == 0 else np.asarray([tracks[t] for t in frame.track_tokens]))
        results['future_gt_vis_tokens'].append(np.zeros(0, np.int64) if index == 0 else frame.visibility)
        results['occ_l2e_r_mats'].append(identities); results['occ_l2e_t_vecs'].append(np.zeros(3, np.float32))
        results['occ_e2g_r_mats'].append(identities if index == 0 else relative[:3, :3].astype(np.float32))
        results['occ_e2g_t_vecs'].append(np.zeros(3, np.float32) if index == 0 else relative[:3, 3].astype(np.float32))
    return generator(results)['gt_segmentation'][1:].numpy()


def scene_metrics(internal_prediction, original_lidar_gt, segmentation):
    prediction = torch.as_tensor(to_lidar(internal_prediction[..., :2])).float()[None]
    gt = torch.as_tensor(np.asarray(original_lidar_gt)).float()[None]
    occupancy = torch.as_tensor(segmentation).long()
    if prediction.shape != (1, 6, 2) or gt.shape != (1, 6, 2) or occupancy.shape != (6, 200, 200):
        raise ValueError('Locked six-step UniAD metric shape mismatch')
    if not torch.isfinite(prediction).all() or not torch.isfinite(gt).all(): raise ValueError('Invalid trajectory')
    metric = PlanningMetric(n_future=6)
    # The upstream evaluator mutates trajectory inputs. Isolate those mutations
    # at this boundary; raw predictions/GT and auxiliary labels stay unchanged.
    metric.update(prediction.clone(), gt.clone(), torch.ones_like(gt), occupancy[None])
    values = {name: value.cpu().numpy() for name, value in metric.compute().items()}
    result = {'per_timestep': {name: value.tolist() for name, value in values.items()},
              'native_out_of_range_steps': int((prediction.abs() >= 50.).any(-1).sum())}
    for seconds, end in ((1, 2), (2, 4), (3, 6)):
        result[f'L2_{seconds}s'] = float(values['L2'][:end].mean())
        result[f'box_collision_{seconds}s'] = float(values['obj_box_col'][:end].mean())
        result[f'point_collision_{seconds}s'] = float(values['obj_col'][:end].mean())
    return result
