"""Read real A samples through original versus staged current-image paths.

All original complete target-cache checks remain enabled. This does not load
a VLA, fit scales, or change teacher/target files.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, read, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('command-record', 'population-contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier verification')
    command = read(args.command_record)['command']
    def option(name):
        return command[command.index('--' + name) + 1]
    spec = importlib.util.spec_from_file_location('_stage_runtime', Path(__file__).with_name('train_shared_a_base.py'))
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    _, contract = runtime.prepare_runtime(Path(option('legacy-source')), args.population_contract)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
    cfg = OmegaConf.load(option('config'))
    common = dict(candidate=cfg.foresight.candidate, current=True, future=True,
        future_type=cfg.foresight.future_target_type, expected_teacher=cfg.foresight.video_teacher_identity,
        dino_index=option('dino-index'), allow_partial=False)
    for name in ('dino', 'clip', 'interaction'):
        common[name + '_root'] = option(name + '-root')
        common['expected_' + name] = option(name + '-identity')
    original = ActionVideoForesightDataset(option('data'), **common)
    staged = ActionVideoForesightDataset(option('data'), image_root=option('local-image-root'), **common)
    if len(original) != len(staged) or original.index != staged.index or original.identity != staged.identity:
        raise ValueError('Image staging changed the A training population')
    checked = []
    for i in (0, 1, 2, 3, len(original) - 1):
        left, left_targets = original[i]
        right, right_targets = staged[i]
        if left['token'] != right['token'] or left['lang'] != right['lang']:
            raise ValueError('Current scene or prompt changed')
        np.testing.assert_array_equal(left['state'], right['state'])
        if len(left['image']) != 3 or len(right['image']) != 3:
            raise ValueError('Camera inventory changed')
        for a, b in zip(left['image'], right['image']):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        if left_targets.keys() != right_targets.keys():
            raise ValueError('Target inventory changed')
        for key in left_targets:
            torch.testing.assert_close(left_targets[key], right_targets[key], rtol=0, atol=0, equal_nan=True)
        checked.append(dict(index=i, token=left['token'], target_fields=sorted(left_targets),
                            image_shapes=[list(np.asarray(image).shape) for image in left['image']]))
    try:
        ActionVideoForesightDataset(option('data'), image_root=str(Path(option('local-image-root')) / 'foreign'), **common)
    except ValueError as error:
        if 'explicit A-only image stage' not in str(error):
            raise
    else:
        raise AssertionError('Foreign staged root was accepted')
    atomic_json(args.output, dict(passed=True, source_sha=source_sha(ROOT),
        dataset_adapter_sha256=digest(ROOT / 'starVLA/dataloader/a_only_base_view.py'),
        population_contract_sha256=digest(args.population_contract),
        population_scenes=len(original), checked=checked, current_pixels_and_targets_bitwise_equal=True,
        foreign_stage_rejected=True, full_target_cache_checks_enabled=True,
        model_calls=0, optimizer_updates=0, resources=check_host()))
    print({'passed': True, 'population_scenes': len(original), 'real_checked_scenes': len(checked)})


if __name__ == '__main__':
    main()
