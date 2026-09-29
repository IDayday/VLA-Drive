"""Register the existing 30-epoch MAE recipe against locally rebuilt GT data."""
import argparse
import datetime
import json
import math
from pathlib import Path
import subprocess

from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.foresight.teacher_runtime import TeacherDataset


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-hours-cap', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.gpu_hours_cap <= 0 or not math.isfinite(args.gpu_hours_cap):
        raise ValueError('Explicit finite teacher budget required')
    if subprocess.check_output(['git', 'status', '--porcelain']).strip():
        raise ValueError('Lock clean teacher source before registration')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Teacher registration is immutable')
    train, dev = TeacherDataset(args.data, 'train'), TeacherDataset(args.data, 'dev')
    complete = json.loads((Path(args.data) / 'COMPLETE.json').read_text())
    if (complete['identity'] != train.identity['identity'] or complete['failed'] or
            complete['scenes'] != len(train) + len(dev)):
        raise ValueError('A complete locally prepared GT population is required')
    if any(not (Path(args.data) / 'records' / (row['token'] + '.pt')).is_file()
           for dataset in (train, dev) for row in dataset.index):
        raise ValueError('Missing GT trajectory record')
    if {r['log'] for r in train.index} & {r['log'] for r in dev.index}:
        raise ValueError('Teacher train/dev log leakage')
    record = {'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'data_identity': train.identity['identity'], 'train_scenes': len(train), 'dev_scenes': len(dev),
              'epochs': 30, 'batch': 256, 'seed': args.seed, 'lr': .0003, 'xy_scale': 20,
              'extra_companion_mask_probability': 0, 'smallfit_weights_reused': False,
              'maximum_optimizer_updates': math.ceil(len(train) / 256) * 30,
              'phase_gpu_hour_cap': args.gpu_hours_cap, 'finite_reruns_max': 2,
              'freeze_selection': 'After30epochs, fixed1/2/4/8/16/24/30 milestones; minimum0.5*(ego_with_peer ADE+vehicle_with_peer ADE), tie later; no Navtest',
              'teacher_loss': 'valid xy SmoothL1; balanced nominal ego/neighbor target, ego-only fallback retained',
              'early_stop': 'correctness/user stop/allocation budget only',
              'teacher_initialization': 'random; no old teacher or JointSceneFlow checkpoint'}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, record)
    print(json.dumps({'registration': str(output), 'updates': record['maximum_optimizer_updates']}))


if __name__ == '__main__':
    main()
