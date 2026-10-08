"""Training-only physical coverage audit before freezing the single BEV grid."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    index = json.loads((args.current_root/'index.json').read_text())
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    vehicle = get_pacifica_parameters()
    body = dict(length=vehicle.length, width=vehicle.width, rear_axle_to_center=vehicle.rear_axle_to_center)
    def read(row):
        label = torch.load(args.current_root/'ego'/(row['token']+'.pt'), map_location='cpu', weights_only=True)
        return decode_ego(label['ego']).numpy()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=12) as pool:
        trajectories = np.stack(list(pool.map(read, index)))
    positions = body_points(torch.from_numpy(trajectories), body).numpy()
    grid = GridSpec()
    outside = ((positions[..., 0] < grid.xmin) | (positions[..., 0] >= grid.xmax) |
               (positions[..., 1] < grid.ymin) | (positions[..., 1] >= grid.ymax))
    report = {'scope': 'original NAVSIM training GT only; no dev/test map search', 'scenes': len(index),
        'grid': grid.__dict__, 'GT_body_points_out_fraction': float(outside.mean()),
        'GT_scenes_with_any_body_point_out_fraction': float(outside.reshape(len(index), -1).any(1).mean()),
        'body_xy_min': positions.min((0, 1, 2)).tolist(), 'body_xy_max': positions.max((0, 1, 2)).tolist(),
        'GT_last_x_quantiles': dict(zip(['50', '90', '95', '99', '99.9'], np.quantile(trajectories[:, -1, 0], [.5, .9, .95, .99, .999]).tolist())),
        'seconds': time.monotonic()-start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
