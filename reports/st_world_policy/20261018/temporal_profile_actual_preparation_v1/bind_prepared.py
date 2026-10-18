#!/usr/bin/env python3
"""Read completed native temporal metadata; emit inputs, never launch or admit work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'PLAN.json'


def document(path, expected=None):
    p = Path(path)
    if not p.is_absolute() or p.resolve() != p:
        raise ValueError('Canonical absolute metadata file required')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        first = os.fstat(stream.fileno())
        if not stat.S_ISREG(first.st_mode) or first.st_size > 8 << 20:
            raise ValueError('Only small regular metadata is allowed')
        raw = stream.read((8 << 20) + 1)
        last = os.fstat(stream.fileno())
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if version(first) != version(last) or len(raw) != first.st_size:
        raise ValueError('Metadata changed while reading')
    pin = dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest())
    if expected is not None and pin != expected:
        raise ValueError('Metadata checksum/path changed')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Metadata object required')
    return value, pin


def artifact(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('Exact file pin required')
    return document(pin['path'], pin)[0]


def bind(plan, key):
    case = plan['cases'][key]
    runtime = Path(plan['runtime']['path'])
    source = plan['runtime']['source_sha']
    if subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=runtime,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise ValueError('Prepared runtime is not the fixed integrated source')
    if subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=runtime):
        raise ValueError('Frozen runtime must be clean')
    for relative, pin in plan['execution_files'].items():
        p = runtime / relative
        if hashlib.sha256(p.read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('Execution source changed: ' + relative)
    # Native prepare publishes one local symlink to an immutable staging version.
    public = Path(case['preparation_output'])
    target = public.resolve(strict=True)
    if (not public.is_symlink() or target.parent != public.parent
            or not target.name.startswith('.' + public.name + '.staging-') or target.is_symlink()):
        raise ValueError('Require the native atomic preparation publication')
    done, done_pin = document(target / 'PREPARATION_COMPLETE.json')
    if done.get('schema') != 'st_temporal_formal5k_preparation_v1':
        raise ValueError('Unknown native preparation receipt')
    inputs, result = done['inputs'], done['result']
    for name, expected in case['expected_inputs'].items():
        if inputs.get(name) != expected:
            raise ValueError('Prepared input differs: ' + name)
    if (result.get('scope') != 'PROFILE' or result.get('updates') != case['updates']
            or result.get('training_started') is not False or result.get('formal_approved') is not False
            or result.get('new_teacher_tensor_bytes') != 0 or result.get('diagnostic_weights_allowed') is not False):
        raise ValueError('Preparation is not this finite diagnostic')
    artifacts = done.get('artifacts', [])
    expected_names = {'RECIPE.json','PROVIDER.json','PROVIDER_CHANNEL_SCOPE.json',
                      'TARGET_CONTRACT.json','training_config.json'}
    if len(artifacts) != len(expected_names) or {Path(p['path']).name for p in artifacts} != expected_names:
        raise ValueError('Incomplete native preparation artifacts')
    docs = {}
    pins = {}
    for pin in artifacts:
        p = Path(pin['path'])
        if p.parent != target:
            raise ValueError('Mixed preparation staging versions')
        docs[p.name] = artifact(pin)
        pins[p.name] = pin
    cfg = docs['training_config.json']
    expected_execution = dict(scope='PROFILE', family='temporal', world_size=4, micro_batch=1,
        global_batch=32, max_updates=20000, stop_after=case['updates'], sample_limit=101592,
        seed=42, precision_policy='S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off')
    if cfg['execution'] != expected_execution or cfg['optimizer']['warmup'] != 1000:
        raise ValueError('Global32/original20k recipe changed')
    rt = cfg['runtime']
    if (rt['output_root'] != case['training_output_root'] or rt['profile_mode'] != case['profile_mode']
            or rt['max_seconds'] != 3600 or rt['save_every'] != case['updates']
            or rt['loader_workers'] != 0 or rt['gradient_checkpointing'] is not False
            or rt['deterministic'] is not True):
        raise ValueError('Runtime limits/backend changed')
    if cfg['space_time']['cell'] != case['cell']:
        raise ValueError('Wrong cell')
    provider = docs['PROVIDER.json']
    admission = provider.get('temporal_profile_admission', {})
    if (admission.get('scope') != 'PROFILE' or admission.get('cell') != case['cell']
            or admission.get('max_scenes') != 101592 or admission.get('max_updates') != 256
            or admission.get('formal_approved') is not False
            or admission.get('data_manifest') != plan['data_manifest']
            or admission.get('consumption') != result.get('consumption')):
        raise ValueError('Full-population admission differs')
    consumption = result['consumption']
    if any(consumption.get(k) != v for k,v in dict(cumulative_optimizer_updates=case['updates'],
            cumulative_scene_exposures=case['updates']*32, world_size=4, micro_batch=1,
            global_batch=32, gradient_accumulation_steps=8).items()):
        raise ValueError('Consumption budget changed')
    if case['cell'] == 'T4':
        history = admission.get('history')
        if not isinstance(history, dict):
            raise ValueError('History is unresolved')
        h = artifact(history['history_complete'])
        if (h.get('status') != 'COMPLETE' or h.get('scenes') != 101592
                or h.get('history_identity') != history.get('history_identity')
                or h.get('history_manifest_sha256') != history.get('history_manifest_sha256')):
            raise ValueError('Actual complete TRAIN history required')
    elif admission.get('history') is not None:
        raise ValueError('T0 cannot introduce history')
    if result['config'] != pins['training_config.json'] or result['teacher_manifest'] != pins['PROVIDER.json']:
        raise ValueError('Receipt/artifact mismatch')
    if (cfg['data']['teacher_provider']['descriptor'] != pins['PROVIDER.json']
            or cfg['data']['teacher_provider']['pca_channel_scope'] != pins['PROVIDER_CHANNEL_SCOPE.json']):
        raise ValueError('Provider/config mismatch')
    artifact(plan['base_binding'])
    return dict(schema='st_temporal_global32_profile_prepared_binding_v1',
        status='METADATA_BOUND_NOT_ADMITTED', source_sha=source, runtime=str(runtime), key=key,
        native_preparation=done_pin, pins=dict(config=pins['training_config.json'],
            teacher_manifest=pins['PROVIDER.json'], provider_pca_scope=pins['PROVIDER_CHANNEL_SCOPE.json'],
            target_contract=pins['TARGET_CONTRACT.json'], data_manifest=plan['data_manifest'],
            base_binding=plan['base_binding']), execution=expected_execution, consumption=consumption,
        run_id=case['run_id'], output_root=case['control_output'],
        native_argv=case['native_training_argv'], native_preparation_is_not_resource_admission=True,
        weights_usable_for_formal_init=False, training_started=False, formal_launch_allowed=False,
        next_required=['bounded metadata supervisor clean completion', 'fresh independent resource admission',
                       'reviewed thin binding of the existing profile controller',
                       'Lead-only actual-host registration and one-shot dispatch'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('T0_capacity4','T4_capacity4','T0_timing240','T4_timing240'), required=True)
    p.add_argument('--output', type=Path, help='Optional new INPUTS.json; no overwrite')
    args = p.parse_args()
    plan, _ = document(PLAN)
    value = bind(plan, args.case)
    if args.output:
        if not args.output.is_absolute() or args.output.name != 'INPUTS.json':
            p.error('An absolute, new INPUTS.json output is required')
        with args.output.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
