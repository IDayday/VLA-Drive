"""Official train700/val150, identical previous/full-future sample eligibility."""
import argparse
import hashlib
import json
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    nusc = NuScenes('v1.0-trainval', dataroot=str(args.root), verbose=True)
    official = create_splits_scenes()
    scene_names = {s['token']: s['name'] for s in nusc.scene}
    split_names = {name: set(official[name]) for name in ('train', 'val')}
    records, reasons = {'train': [], 'val': []}, {'no_legal_past_pose': 0, 'incomplete_six_keyframe_future': 0}
    timing_errors = []
    for sample in nusc.sample:
        scene_name = scene_names[sample['scene_token']]
        split = next((k for k, value in split_names.items() if scene_name in value), None)
        if split is None:
            continue
        if not sample['prev']:
            reasons['no_legal_past_pose'] += 1
            continue
        origin = nusc.get('sample_data', sample['data']['LIDAR_TOP'])['timestamp']
        cursor, eligible, times = sample, True, []
        for step in range(1, 7):
            if not cursor['next']:
                reasons['incomplete_six_keyframe_future'] += 1; eligible = False; break
            cursor = nusc.get('sample', cursor['next'])
            timestamp = nusc.get('sample_data', cursor['data']['LIDAR_TOP'])['timestamp']
            times.append((timestamp-origin)/1e6)
        if eligible:
            timing_errors.extend(np.asarray(times)-np.arange(1, 7)*.5)
            records[split].append({'token': sample['token'], 'scene': scene_name, 'timestamp_us': origin,
                                   'actual_future_times_s': times})
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, values in records.items():
        values.sort(key=lambda value: (value['scene'], value['timestamp_us']))
        path = args.output/(split+'_scenes.json')
        path.write_text(json.dumps(values, indent=2)+'\n')
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {'schema': 'structured_world_nuscenes_population_v2', 'official_scene_counts': {k: len(v) for k, v in split_names.items()},
                'sample_counts': {k: len(v) for k, v in records.items()}, 'file_sha256': hashes,
                'eligibility': 'previous sample LiDAR pose and six complete successor annotations; original VAD future-valid protocol', 'excluded': reasons,
                'timing_protocol': 'nominal 2 Hz annotations; real timestamp jitter retained, not exact resampled physical instants',
                'absolute_timestamp_error_s_quantiles': {str(p): float(np.quantile(np.abs(timing_errors), p)) for p in [.5, .9, .95, .99, 1.]},
                'navigation': 'VAD high-level intention derived from logged future, fixed text only',
                'inputs': 'six current cameras, local calibration, previous/current ego pose; maps/LiDAR GT are labels only',
                'training': 'all eligible scenes from official train700; no development/val images in geometry preparation'}
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (args.output/'identity.json').exists():
        previous = json.loads((args.output/'identity.json').read_text())
        if previous['identity'] != identity['identity']:
            raise ValueError('Use a new output for a changed population protocol')
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(identity, indent=2), flush=True)


if __name__ == '__main__':
    main()
