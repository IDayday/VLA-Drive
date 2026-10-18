#!/usr/bin/env python3
"""CPU-only immutable T0 timing metadata from the completed capacity binding.

No model/cache tensors, index traversal, GPU query, trainer fork or registration.
The native runtime still validates the new recipe/provider before training.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'prepared_inputs'
ROOT = Path('/mnt/project/st-world-policy-20261016')
PARENT = ROOT / 'profile/temporal_global32_qualification_v1/T0_capacity4/INPUTS.json'
PARENT_SHA = '940f30904a84277dff8e27b4813100758bd114f27600676ce4cc3b1d2ef79d51'
RUNTIME = Path('/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018')
SOURCE = '38f86a15a92171e583bb5019cce89a63fb7a1d43'
OUT = ROOT / 'profile/temporal_global32_qualification_v1/T0_timing240/training_control'
RUN_ID = 'T0_fulltrain_seed42_profile240_world4_micro1_global32_v1'


def digest(path):
    p = Path(path)
    if not p.is_file() or p.stat().st_size > 32 << 20:
        raise ValueError('Small existing metadata only')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def read(pin_value):
    if set(pin_value) != {'path', 'sha256'} or digest(pin_value['path']) != pin_value['sha256']:
        raise ValueError('Parent metadata identity mismatch')
    return json.loads(Path(pin_value['path']).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Immutable preparation conflict: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(raw)


def prepare():
    parent = read({'path': str(PARENT), 'sha256': PARENT_SHA})
    if parent['source_sha'] != SOURCE or parent['key'] != 'T0_capacity4' or parent['training_started']:
        raise ValueError('Wrong immutable parent binding')
    cfg = read(parent['pins']['config'])
    descriptor = read(parent['pins']['teacher_manifest'])
    scope = read(parent['pins']['provider_pca_scope'])
    counts = deepcopy(parent['consumption'])
    counts.update(cumulative_optimizer_updates=240, cumulative_scene_exposures=7680,
                  per_rank_scene_exposures=[1920] * 4, resume_budget='fresh_complete_windows_no_resume')
    cfg['execution']['stop_after'] = 240
    cfg['runtime'].update(output_root=str(OUT / 'runs'), profile_mode='timing', save_every=240)
    # Preserve the validated 3600s native cap and the original20k/warmup1000 clock.
    recipe = deepcopy(cfg)
    recipe['data'].pop('teacher_provider')
    DEST.mkdir(parents=True, exist_ok=True)
    write(DEST / 'RECIPE.json', recipe)
    admission = descriptor['temporal_profile_admission']
    admission.update(recipe=pin(DEST / 'RECIPE.json'), consumption=counts)
    descriptor['identity'] = identity({k: v for k, v in descriptor.items() if k != 'identity'})
    write(DEST / 'PROVIDER.json', descriptor)
    scope['provider_descriptor'] = pin(DEST / 'PROVIDER.json')
    scope['identity'] = identity({k: v for k, v in scope.items() if k != 'identity'})
    write(DEST / 'PROVIDER_CHANNEL_SCOPE.json', scope)
    cfg['data']['teacher_provider'] = {'descriptor': pin(DEST / 'PROVIDER.json'),
                                     'pca_channel_scope': pin(DEST / 'PROVIDER_CHANNEL_SCOPE.json')}
    write(DEST / 'training_config.json', cfg)
    inputs = deepcopy(parent)
    inputs.update(key='T0_timing240', run_id=RUN_ID, output_root=str(OUT), execution=cfg['execution'],
                  consumption=counts, status='TIMING_METADATA_BOUND_NOT_ADMITTED',
                  parent_capacity_inputs={'path': str(PARENT), 'sha256': PARENT_SHA},
                  timing_plan={'warmup': 20, 'steady': 100, 'windows': 2, 'updates': 240},
                  next_required=['independent correctness review of exact pins',
                                 'fresh independent resource admission including live concurrent storage holds',
                                 'Lead-only prepare and single run; no resume or retry'])
    inputs['native_argv'][-1] = str(OUT / 'REGISTRATION.json')
    for key, name in [('config', 'training_config.json'), ('teacher_manifest', 'PROVIDER.json'),
                      ('provider_pca_scope', 'PROVIDER_CHANNEL_SCOPE.json')]:
        inputs['pins'][key] = pin(DEST / name)
    write(DEST / 'INPUTS.json', inputs)
    write(DEST / 'METADATA_LINEAGE.json', {
        'schema': 'st_T0_timing240_metadata_lineage_v1', 'status': 'PREPARED_NOT_ADMITTED',
        'parent': {'path': str(PARENT), 'sha256': PARENT_SHA},
        'child': pin(DEST / 'INPUTS.json'), 'runtime': str(RUNTIME), 'source_sha': SOURCE,
        'changed': ['execution.stop_after:4->240', 'runtime.profile_mode:none->timing',
                    'runtime.save_every:4->240', 'run/output namespace',
                    'matching provider admission consumption/recipe pins and descriptor/PCA-scope hashes'],
        'unchanged': ['original Base binding', 'optimizer and original20k learning-rate schedule',
                      'global32/world4/micro1/accumulation8', 'sample order/seed',
                      'full101592 TRAIN manifest', 'teacher/PCA numerical function',
                      'precision and all scientific losses'],
        'tensor_bytes_read': 0, 'gpu_calls': 0, 'training_started': False,
        'weights_usable_for_formal_init': False,
    })
    return pin(DEST / 'INPUTS.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write only small immutable metadata beside this script')
    args = parser.parse_args()
    if not args.write:
        parser.error('Use --write explicitly; this command never starts a GPU job')
    print(json.dumps({'status': 'METADATA_PREPARED_NOT_ADMITTED', 'inputs': prepare()}))
