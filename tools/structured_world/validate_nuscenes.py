"""Real six-camera calibration, native VAD targets, boxes and UniAD metric QA."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.structured_world.adapters import CLASS_NAMES, pose_matrix
from starVLA.dataloader.structured_world.nuscenes_adapter import adapted_nuscenes_scene, planning_metadata, to_lidar
from starVLA.model.modules.structured_world.nuscenes_evaluation import native_planning_segmentation, scene_metrics, PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=8)
    args = parser.parse_args()
    nusc = NuScenes('v1.0-trainval', dataroot=str(args.root), verbose=False)
    rows = json.loads((args.cache/'index.json').read_text())[:args.scenes]
    records = []
    for row in rows:
        sample = nusc.get('sample', row['token']); scene = adapted_nuscenes_scene(nusc, sample)
        planning = planning_metadata(nusc, sample)
        if not np.allclose(to_lidar(scene.ego_physical[:, :2]), planning['original_VAD_lidar_xy'], atol=2e-5, rtol=1e-6):
            raise AssertionError('Original VAD target inverse changed')
        calibration_error = []
        for index, name in enumerate(scene.protocol['camera_order']):
            sd = nusc.get('sample_data', sample['data'][name]); ep = nusc.get('ego_pose', sd['ego_pose_token'])
            cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            expected = np.linalg.inv(scene.inverse_transform)@pose_matrix(ep['translation'], ep['rotation'])@pose_matrix(cs['translation'], cs['rotation'])
            actual = scene.observations['calibration']['sensor2ego'][index].numpy()
            calibration_error.append(float(np.max(np.abs(expected-actual))))
        if max(calibration_error) > 2e-5: raise AssertionError('Current camera timestamp/extrinsic mismatch')
        box_center_errors = []
        current = sample
        from nuscenes.eval.detection.utils import category_to_detection_name
        for frame in scene.frames:
            expected = []
            for token in current['anns']:
                annotation = nusc.get('sample_annotation', token)
                if category_to_detection_name(annotation['category_name']) in CLASS_NAMES:
                    center = np.linalg.inv(frame.ego2global)@np.array([*annotation['translation'], 1.])
                    expected.append(center[:3])
            if expected:
                actual = frame.boxes.gravity_center.numpy()
                box_center_errors.extend(np.linalg.norm(actual-np.asarray(expected), axis=-1).tolist())
            current = nusc.get('sample', current['next']) if current['next'] else None
        if max(box_center_errors, default=0.) > 2e-4:
            raise AssertionError('Mature box transform changed the source gravity center')
        segmentation = native_planning_segmentation(scene)
        gt_metric = scene_metrics(scene.ego_physical, planning['original_VAD_lidar_xy'], segmentation)
        if gt_metric['L2_3s'] != 0.: raise AssertionError('GT planning replay L2 changed')
        records.append({'token': scene.token, 'camera_calibration_max_absolute_error': max(calibration_error),
            'source_VAD_GT_max_absolute_xy_error': float(np.max(np.abs(to_lidar(scene.ego_physical[:, :2])-planning['original_VAD_lidar_xy']))),
            'mature_box_planar_center_error_m_max': max(box_center_errors, default=0.),
            'mature_box_planar_center_error_m_mean': float(np.mean(box_center_errors)) if box_center_errors else 0.,
            'GT_native_metric': gt_metric, 'protocol': scene.protocol})
    report = {'scope': 'real_data_coordinate_and_evaluation_validation', 'scenes': len(records),
              'root_is_provisional': (args.root/'PROVISIONAL_DEBUG_ONLY.json').exists(),
              'native_metric_protocol': PROTOCOL, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'scenes': len(records), 'maximum_camera_error': max(r['camera_calibration_max_absolute_error'] for r in records),
        'maximum_planar_box_center_error_m': max(r['mature_box_planar_center_error_m_max'] for r in records), 'output': str(args.output)}), flush=True)


if __name__ == '__main__': main()
