"""Prepare/check two fixed configs with the existing S-P2 runtime; never launch."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNTIME = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
SOURCE = 'fdf25f99b702754787904215bf1c2a2b7515ea75'
BASIS = '268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46'
OUTPUT_ROOT = '/mnt/project/st-world-policy-20261016/formal/teacher_channel_control_v1/runs'
CONFIGS = REPO / 'configs/st_world_policy/teacher_channel_control_v1'


def small_json(path, expected=None):
    """Read only bounded regular control files, including through an atomic rename."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
            raise ValueError('Only control JSON <=1MiB is permitted')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            content = stream.read(1024 * 1024 + 1)
        after = os.fstat(fd)
        if len(content) != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Control changed during read')
    finally:
        os.close(fd)
    if expected and hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Control checksum mismatch: ' + str(path))
    return json.loads(content)


def runtime_api():
    # Use the exact runtime of the completed384 control, not a modified live tree.
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip() != SOURCE:
        raise ValueError('Reference runtime source changed')
    if subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=no'], cwd=RUNTIME, text=True):
        raise ValueError('Reference runtime has tracked modifications')
    sys.path.insert(0, str(RUNTIME)) if str(RUNTIME) not in sys.path else None
    names = ['starVLA.dataloader.st_world_schema', 'starVLA.dataloader.st_world_dataset',
             'tools.st_world_policy.training_config']
    modules = [importlib.import_module(name) for name in names]
    if any(not Path(m.__file__).resolve().is_relative_to(RUNTIME) for m in modules):
        raise ValueError('A different runtime was imported before this fixed study')
    return modules[0].TeacherFeatureContract, modules[1].projected_teacher_contract, modules[2].validate_config


def load_metadata():
    pins = small_json(HERE / 'INPUT_PINS.json')
    return {name: small_json(row['path'], row['sha256']) for name, row in pins.items()}


def expected_pair(metadata):
    Contract, project, validate = runtime_api()
    original = metadata['reference_config']
    receipt = metadata['pca_receipt']
    if (receipt['identity'] != BASIS or receipt['fit_role'] != 'train'
            or receipt['dimensions'] != [256, 384] or receipt['whiten'] is not False
            or receipt['standardize_channels'] is not True):
        raise ValueError('Only the original train-only nonwhitened basis is permitted')
    scope = metadata['pca_channel_scope']
    matches = [row for row in scope['allowed_caches'] if row['cache_root'] == original['data']['teacher_root']]
    if len(matches) != 1 or scope['pca_identity'] != BASIS or scope['statistics_refitted'] is not False:
        raise ValueError('Original target population is not bound to this fixed basis')
    raw = Contract.from_dict(matches[0]['contract'])
    if raw.identity != matches[0]['teacher_contract_identity'] or raw.channels != 1024:
        raise ValueError('Raw teacher contract changed')
    if original['space_time']['teacher'] != project(raw, pca_identity=BASIS, dimension=384).to_dict():
        raise ValueError('Completed384 control does not use this exact transform')
    if (metadata['reference_registration']['source_sha'] != SOURCE
            or metadata['reference_registration']['execution'] != original['execution']):
        raise ValueError('Completed control source or execution changed')
    if (original['space_time']['cell'] != 'S-P2' or raw.image_hw != (288, 384)
            or raw.output_grid_hw != (18, 24) or original['space_time']['world_queries_per_view'] != 48
            or original['space_time']['sampled_positions'] != 96):
        raise ValueError('This pair is only native432/W48/K96')
    configs = {}
    for width in (256, 384):
        value = deepcopy(original)
        value['space_time']['teacher'] = project(raw, pca_identity=BASIS, dimension=width).to_dict()
        if width == 256:
            value['runtime']['output_root'] = OUTPUT_ROOT
        validate(value)
        configs[width] = value
    return configs


def verify_pair(configs, metadata):
    expected = expected_pair(metadata)
    if set(configs) != {256, 384} or any(configs[k] != expected[k] for k in expected):
        raise ValueError('Pair changed a fixed input, basis, budget, output identity, or derived contract')
    return {'status': 'CONFIGURATION_PASS', 'formal_launch_allowed': False,
            'source_sha': SOURCE, 'pca_identity': BASIS,
            'reference384': 'REUSE_EXISTING_SP2_5000_NOT_NEW_TRAINING',
            'new256': 'NOT_RUN_REQUIRES_NEW_REGISTRATION_AND_REVIEWS',
            'model_checkpoint_tensors_read': False, 'teacher_cache_tensors_read': False,
            'shared_modules_runtime_changed': False}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_once(path, value):
    content = encoded(value)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Refusing to replace an existing different study artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-configs', action='store_true', help='Write only this repository\'s four small JSON files, idempotently')
    args = parser.parse_args()
    metadata = load_metadata()
    if args.write_configs:
        for width, cfg in expected_pair(metadata).items():
            write_once(CONFIGS / f'PCA{width}.json', cfg)
            write_once(CONFIGS / f'TARGET_CONTRACT_{width}.json', cfg['space_time']['teacher'])
    configs = {width: small_json(CONFIGS / f'PCA{width}.json') for width in (256, 384)}
    result = verify_pair(configs, metadata)
    for width, cfg in configs.items():
        if small_json(CONFIGS / f'TARGET_CONTRACT_{width}.json') != cfg['space_time']['teacher']:
            raise ValueError('Registered target file differs from the actual training config')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
