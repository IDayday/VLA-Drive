"""Read-only metadata audit and command handoff; never prepares a trainer lease.

Run with PYTHONPATH pointing to the immutable qualified runtime. This script
does not execute the generated provider-preparation or training commands.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    def rchar():
        return int(dict(x.split(':', 1) for x in Path('/proc/self/io').read_text().splitlines())['rchar'])
    initial_read = rchar()
    def guard():
        if time.monotonic() - started > 180:
            raise TimeoutError('Metadata preparation exceeds 180 seconds')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2 << 30:
            raise MemoryError('Metadata preparation exceeds 2 GiB RSS')
        if rchar() - initial_read > 2 << 30:
            raise RuntimeError('Metadata preparation exceeds 2 GiB logical reads')
    forbidden = ('.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.jpg', '.jpeg', '.png', '.pkl', '.pickle', '.npy', '.npz')
    def deny_payload(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            if os.fsdecode(values[0]).lower().endswith(forbidden):
                raise RuntimeError('This preparation cannot read model/image/tensor payloads: ' + os.fsdecode(values[0]))
    sys.addaudithook(deny_payload)

    import torch
    torch.set_num_threads(2)
    from tools.st_world_policy import teacher_provider as tp
    from tools.st_world_policy.qualify_indexed_provider import validate_receipt
    from tools.st_world_policy.training_config import validate_config
    from tools.st_world_policy.teacher_pca import teacher_channel_family
    from starVLA.dataloader.st_world_dataset import projected_teacher_contract
    from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
    from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader, IndexedBoundedTeacherProvider
    from starVLA.dataloader.st_world_targets import atomic_json

    root = Path('/mnt/project/st-world-policy-20261016')
    runtime = Path(tp.__file__).resolve().parents[2]
    qualification = root / 'audits/indexed_provider_composition_run_v1/RESULT.json'
    selected = root / 'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json'
    old = root / 'smoke/world_fixed_provider_native2_v1/continuous'
    namespace = root / 'smoke/world_indexed_provider_native2_v1'
    output = Path(args.output).resolve()
    loaded = {}
    def load(path):
        path = Path(path).resolve()
        if path not in loaded:
            body = path.read_bytes()
            if len(body) > 32 << 20:
                raise ValueError('Preparation expects small pinned JSON')
            loaded[path] = (json.loads(body), {'path': str(path), 'sha256': hashlib.sha256(body).hexdigest()})
        guard()
        return loaded[path]
    result, result_pin = load(qualification)
    if result_pin['sha256'] != 'a4f4c86f0b5c09d18a58d02e8919dd73e659b48a53ed0d2a25c7d68e497b7a2f':
        raise ValueError('Different actual composition qualification')
    plan, plan_pin = load(qualification.parent / 'RESOLVED_PLAN.json')
    selection, selected_pin = load(selected)
    if selected_pin['sha256'] != '7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8':
        raise ValueError('Different exact pinned dataset')
    future_plan, future_pin = load(selection['provenance']['plan_path'])
    if (future_pin['sha256'] != selection['provenance']['plan_sha256'] or
            selection['scenes'] != future_plan['selected_rows'] or len(selection['scenes']) != 32 or
            identity_hash(selection['scenes']) != future_plan['selection_identity']):
        raise ValueError('Pinned dataset is not the unchanged future32 selection')
    cfg, cfg_pin = load(old / 'CONFIG.json')
    validate_config(cfg)
    raw = TeacherFeatureContract.from_dict(plan['contract'])
    descriptor = {'schema': tp.INDEXED_SCHEMA, 'requests': plan['requests'],
        'request_index': plan['request_index'],
        'sources': [{'index': plan['source_index'], 'cache_identity': plan['current_identity']}],
        'teacher_config': plan['teacher_config'], 'qualification': result_pin,
        'qualification_plan': plan_pin, 'execution': plan['execution'],
        'limits': plan['limits'], 'provider': plan['provider']}
    descriptor['identity'] = identity_hash(descriptor)
    validate_receipt(plan, result, descriptor)
    pca, pca_pin = load(Path(cfg['data']['pca_path']) / 'PCA.json')
    fit_root = root / 'caches/original_pca_fit64_metadata_index_v1'
    fit_receipt, fit_pin = load(fit_root / 'COMPLETE.json')
    fit_binding = {'complete': fit_pin, 'request_identity': fit_receipt['request_identity'],
        'index_identity': fit_receipt['identity'], 'max_metadata_cache_bytes': 16 << 20}
    indices = []
    try:
        for binding in (plan['request_index'], plan['source_index'], fit_binding):
            indices.append(tp.open_metadata_index(binding)); guard()
        request, current, fit = indices
        teacher = IndexedTeacherCacheReader(current, expected_identity=plan['current_identity'], expected_contract=raw)
        placeholder = SimpleNamespace(contract_identity=raw.identity, execution_identity=plan['execution_identity'])
        provider = IndexedBoundedTeacherProvider(request, raw, [teacher], executor=placeholder,
            execution_identity=plan['execution_identity'], **plan['limits'])
        tp._validate_indexed_descriptor(descriptor, existing=provider)
        header = request.receipt['manifest_header']
        source, source_pin = load(Path(cfg['data']['root']) / 'identity.json')
        if (selection['schema'] != 'st_fixed_scene_list_v1' or selection['split'] != 'train' or
                source['identity'] != identity_hash({k: v for k, v in source.items() if k != 'identity'}) or
                header['source_student_identity'] != source['identity'] or
                header['source_index_sha256'] != source['index_sha256'] or
                selection['source_index_hash'] != source['index_sha256'] or
                header['partition_sha256'] != source['partition_sha256']):
            raise ValueError('Current student / request / selected role identity mismatch')
        if [row['token'] for row in selection['scenes']] != plan['scene_ids']:
            raise ValueError('Prepared sample order differs from the exact qualified order')
        for row in selection['scenes']:
            if request.scenes[row['token']]['log'] != row['log']:
                raise ValueError('Scene log join changed')
            for name in ('current_record_sha256', 'ego_record_sha256'):
                tp.require_hash(row[name], name)
        if (fit.receipt['cache_identity'] != pca['fit_cache_identity'] or
                TeacherFeatureContract.from_dict(fit.receipt['contract']).identity != pca['teacher_contract_identity'] or
                teacher_channel_family(fit.receipt['cache_header']) !=
                teacher_channel_family({'contract': raw.to_dict(), 'execution': plan['execution']})):
            raise ValueError('PCA fitted source/channel family differs')
        projected = projected_teacher_contract(raw, pca_identity=pca['identity'], dimension=384)
        if projected.to_dict() != cfg['space_time']['teacher']:
            raise ValueError('Existing target is not the same frozen PCA384 basis')
        guard()
    finally:
        for index in indices:
            index.close()
    new = deepcopy(cfg)
    new['execution']['sample_limit'] = 32
    new['runtime']['output_root'] = str(namespace / 'configured/runs')
    validate_config(new)
    expected = dict(cfg['execution'], sample_limit=32)
    if (new['execution'] != expected or new['execution']['stop_after'] != 2 or
            new['execution']['max_updates'] != 20000 or new['optimizer']['warmup'] != 1000 or
            [new['execution'][k] for k in ('world_size','micro_batch','global_batch')] != [1,1,2]):
        raise ValueError('Diagnostic recipe changed')
    atomic_json(output / 'INPUT_CONFIG.json', new, immutable=True)
    command = ['/root/miniconda3/envs/ddp/bin/python', '-m', 'tools.st_world_policy.teacher_provider',
        '--request-index', str(Path(plan['request_index']['complete']['path']).parent),
        '--source-index', str(Path(plan['source_index']['complete']['path']).parent),
        '--teacher-config', plan['teacher_config']['path'], '--qualification-result', str(qualification),
        '--output', str(namespace / 'PROVIDER.json'), '--max-cached-bytes', str(plan['limits']['max_cached_bytes']),
        '--max-scene-mib', '64', '--max-batch-images', '8', '--max-metadata-cache-mib', '16',
        '--training-config', str(output / 'INPUT_CONFIG.json'),
        '--pca-fit-cache-root', fit_receipt['cache_root'], '--pca-fit-cache-index', str(fit_root),
        '--configured-output', str(namespace / 'configured')]
    report = {'schema': 'st_indexed_smoke_handoff_v1', 'status': 'METADATA_CHECKED_NOT_ADMITTED',
        'runtime': str(runtime), 'source_config': cfg_pin, 'data_manifest': selected_pin,
        'qualification': result_pin, 'plan': plan_pin, 'source_student': source_pin,
        'pca_receipt': pca_pin, 'fit_index': fit_binding, 'execution': new['execution'],
        'target_contract': projected.to_dict(), 'provider_descriptor_expected_identity': descriptor['identity'],
        'namespace': str(namespace), 'prepare_command': command,
        'training_command': ['/root/miniconda3/envs/ddp/bin/torchrun', '--standalone', '--nproc_per_node=1',
            '-m', 'tools.st_world_policy.train_world', '--registration', str(namespace / 'REGISTRATION.json')],
        'new_inputs': {'selected_scenes': 32, 'unique_images': plan['unique_images'],
            'current_hits': len(plan['hit_keys']), 'on_demand_misses': len(plan['miss_keys'])},
        'unchanged': ['optimizer', 'space_time', 'max_updates_20000', 'warmup_1000', 'precision_policy',
            'micro1_global2_world1', 'PCA_basis', 'original_current_ego_content_pins'],
        'pending_runtime_checks': ['PCA tensor SHA/load', 'current/ego actual bytes', 'images and teacher weights',
            'current runtime environment and GPU/resource lease', 'native trainer two-update execution'],
        'full_current_index_opens': 1, 'pca_refitted': False, 'tensor_payload_reads': 0,
        'gpu_execution': False, 'resource_registered': False, 'training_started': False,
        'formal_approved': False, 'weights_usable_for_formal_init': False,
        'prepare_command_executed': False, 'configured_output_exists': (namespace / 'configured/training_config.json').exists(),
        'measurements': {'elapsed_seconds': time.monotonic() - started,
            'logical_read_rchar_bytes': rchar() - initial_read,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    guard()
    atomic_json(output / 'HANDOFF.json', report, immutable=True)
    print(json.dumps({'status': report['status'], **report['measurements'], 'prepare': shlex.join(command)}))


if __name__ == '__main__':
    main()
