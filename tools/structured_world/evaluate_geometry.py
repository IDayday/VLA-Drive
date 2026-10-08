"""Dense fixed-training-set diagnostics before/after shared geometry learning.

These metrics are perception engineering checks, never formal planning results.
Predictions use only cached current RGB, camera support and local calibration.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.train_geometry import GeometryCache, GeometryPreparation


@torch.no_grad()
def evaluate(model, data, limit, *, visualization=None):
    totals = dict(road_abs=0., road_count=0, boundary_abs=0., boundary_count=0,
                  TP=0, FP=0, FN=0, depth_abs=0., depth_count=0)
    rows = []
    model.eval().cuda()
    for index in range(min(limit, len(data))):
        batch = {key: value[None].cuda() for key, value in data[index].items()}
        rgb = batch['geometry_rgb'].float()/255.
        images = (rgb-rgb.new_tensor([.485, .456, .406])[None, None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, None, :, None, None]
        current = model.geometry(images, {k: batch['calibration_'+k] for k in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}, batch['geometry_pixel_valid'])
        maps = model.current_head(current['B0'])
        road, probability = 10*maps[:, 0].tanh(), maps[:, 1].sigmoid()
        road_valid = batch['road_valid'].bool()
        error = (road-batch['road_distance']).abs()
        boundary = road_valid & (batch['road_distance'].abs() <= 1.)
        occ_valid = batch['occupancy_valid'][:, 0].bool()
        pred, truth = probability >= .5, batch['occupancy'][:, 0] == 1
        row = {'token': data.index[index]['token'], 'road_abs': float(error[road_valid].sum()),
            'road_count': int(road_valid.sum()), 'boundary_abs': float(error[boundary].sum()),
            'boundary_count': int(boundary.sum()), 'TP': int((pred & truth & occ_valid).sum()),
            'FP': int((pred & ~truth & occ_valid).sum()), 'FN': int((~pred & truth & occ_valid).sum())}
        labels = model.geometry.view.get_downsampled_gt_depth(batch['depth'], 16)
        probabilities = current['depth'][0].permute(0, 2, 3, 1).reshape_as(labels)
        valid = labels.sum(1) > 0
        depth_prediction = (probabilities*torch.arange(1, 100, device=probabilities.device)[None]).sum(1)
        depth_gt = labels.argmax(1)+1
        row.update(depth_abs=float((depth_prediction-depth_gt).abs()[valid].sum()), depth_count=int(valid.sum()))
        for key in totals: totals[key] += row[key]
        rows.append(row)
        if visualization and index < 8:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots(1, 4, figsize=(14, 4))
            maps_to_plot = [batch['road_distance'][0], road[0], truth[0].float(), probability[0]]
            masks = [road_valid[0], road_valid[0], occ_valid[0], occ_valid[0]]
            for axis, field, mask, title in zip(axes, maps_to_plot, masks, ['derived GT road SDF', 'predicted road SDF', 'derived GT obstacle', 'predicted obstacle']):
                values = field.cpu().numpy().copy(); values[~mask.cpu().numpy()] = np.nan
                axis.imshow(values, origin='lower', extent=(-20, 80, -40, 40),
                    vmin=-10 if 'SDF' in title else 0, vmax=10 if 'SDF' in title else 1)
                axis.set_title(title); axis.set_xlabel('x forward (m)'); axis.set_ylabel('y left (m)')
            figure.tight_layout()
            figure.savefig(visualization/f'train_{index:02d}_derived_fields.png', dpi=130)
            plt.close(figure)
    def ratio(a, b): return a/b if b else None
    return {'scenes': len(rows), 'road_MAE_m': ratio(totals['road_abs'], totals['road_count']),
        'road_boundary_MAE_m': ratio(totals['boundary_abs'], totals['boundary_count']),
        'current_obstacle_IoU': ratio(totals['TP'], totals['TP']+totals['FP']+totals['FN']),
        'current_obstacle_precision': ratio(totals['TP'], totals['TP']+totals['FP']),
        'current_obstacle_recall': ratio(totals['TP'], totals['TP']+totals['FN']),
        'projected_depth_bin_expectation_MAE_m': ratio(totals['depth_abs'], totals['depth_count']),
        'totals': totals, 'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--imagenet', type=Path, required=True)
    parser.add_argument('--shared-geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=64)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data = GeometryCache(args.cache, debug=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model = GeometryPreparation(args.imagenet).float()
    report = {'kind': 'fixed_training_subset_perception_learning_not_formal_planning',
        'label_identity': data.identity['identity'], 'before': evaluate(model, data, args.scenes)}
    state = torch.load(args.shared_geometry, map_location='cpu', weights_only=False)
    if state['label_cache_identity'] != data.identity['identity']:
        raise ValueError('Geometry checkpoint and metric labels have different identities')
    model.geometry.load_state_dict(state['geometry'], strict=True)
    model.current_head.load_state_dict(state['current_head'], strict=True)
    report['shared_geometry_identity'] = state['identity']
    visual = args.output/'controlled_derived_field_visualizations'; visual.mkdir(exist_ok=True)
    report['after'] = evaluate(model, data, args.scenes, visualization=visual)
    (args.output/'GEOMETRY_LEARNING.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v if not isinstance(v, dict) else {n: x for n, x in v.items() if n != 'rows'} for k, v in report.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
