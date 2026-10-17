"""Metadata/stat-only Lead handoff. Does not admit resources or launch training."""
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    start = time.monotonic()
    output = Path(__file__).resolve().parent
    root = Path('/mnt/project/st-world-policy-20261016')
    runtime = Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017')
    source = '31c3dca2fbc697b419bf343ce05d10be0a4fe13f'
    old_source = '8c1e3f86268ef5ed79f46d69abf33d5c9ca74bdf'
    new_root = root / 'smoke/world_indexed_provider_training2_v1'
    factory = root / 'smoke/world_indexed_provider_native2_v1'
    old = root / 'smoke/world_fixed_provider_native2_v1/continuous'
    requested_bytes = 0
    loaded = {}
    def canonical(d):
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    def check():
        if time.monotonic()-start > 180 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 2 << 30:
            raise RuntimeError('Metadata CPU/RAM/time allowance exceeded')
        if requested_bytes > 32 << 20:
            raise RuntimeError('Small metadata read allowance exceeded')
    def read(path):
        nonlocal requested_bytes
        path = Path(path).resolve()
        if path not in loaded:
            if path.suffix not in ('.json', '.jsonl') or path.stat().st_size > 32 << 20:
                raise ValueError('Only bounded JSON evidence is read')
            body = path.read_bytes(); requested_bytes += len(body); check()
            loaded[path] = (json.loads(body), {'path': str(path), 'sha256': hashlib.sha256(body).hexdigest()})
        return loaded[path]
    cfg, cfg_pin = read(factory / 'configured/training_config.json')
    desc, desc_pin = read(factory / 'PROVIDER.json')
    scope, scope_pin = read(factory / 'configured/PROVIDER_CHANNEL_SCOPE.json')
    lineage, lineage_pin = read(factory / 'configured/PROVIDER_CONFIG_LINEAGE.json')
    prepared, prepared_pin = read(factory / 'preparation_control_v1/RESULT.json')
    for pin, expected in [(cfg_pin, 'f6b8bd3ad59a763355130ef8bb0ef9228473f424516c62e630def80e87db948d'),
        (desc_pin, 'bb53252f1aea91ed54fa7b3917314d2d7f6c1c251d1f89f6a079f0f582f3874d'),
        (scope_pin, 'dc9c7c5f6be9213a9d755f17d9c12a34a46c4ddf2f10b3c80c952a03bc6ae41e'),
        (lineage_pin, 'e5203bdfcb7eef1b9fb7ffd59b42d524c037e23488daf60830a75509e3d429d1'),
        (prepared_pin, 'afb0c77ccea1b6e368995ad68efa7140e30e1f3a8fa1ab97e716a04f27c704d4')]:
        if pin['sha256'] != expected:
            raise ValueError('Actual completed factory output differs: '+pin['path'])
    if prepared['status'] != 'COMPLETE' or prepared['training_started'] is not False:
        raise ValueError('Unexpected factory state')
    if cfg['data']['teacher_provider'] != {'descriptor':desc_pin, 'pca_channel_scope':scope_pin}:
        raise ValueError('Config provider/scope pins differ')
    if lineage['new_config'] != cfg_pin or lineage['pca_refitted'] is not False:
        raise ValueError('Factory lineage differs')
    selected, selected_pin = read(root/'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json')
    base, base_pin = read(root/'bindings/BASE_BINDING.json')
    if selected_pin['sha256'] != '7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8' or len(selected['scenes']) != 32:
        raise ValueError('Different exact32 manifest')
    if base_pin['sha256'] != '84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38':
        raise ValueError('Base metadata differs')
    original = json.loads(json.dumps(cfg))
    cfg['runtime']['output_root'] = str(new_root/'runs')
    if (cfg['execution'] != {'family':'temporal','global_batch':2,'max_updates':20000,'micro_batch':1,
        'precision_policy':'S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off','sample_limit':32,
        'scope':'SMOKE','seed':42,'stop_after':2,'world_size':1} or cfg['optimizer']['warmup'] != 1000 or
        cfg['runtime']['loader_workers'] != 0 or cfg['runtime']['save_every'] != 1 or cfg['runtime']['max_seconds'] != 900):
        raise ValueError('Unexpected execution budget')
    old_cfg, _ = read(old/'CONFIG.json')
    if any(cfg[k] != old_cfg[k] for k in ('optimizer','space_time')):
        raise ValueError('Scientific target or optimizer recipe changed')
    records = []
    checkpoint_root = old/'runs/fixed_provider_continuous_updates2_v1/checkpoints'
    for path in sorted(checkpoint_root.glob('*/COMPLETE.json')):
        receipt, pin = read(path)
        if canonical({k:v for k,v in receipt.items() if k!='receipt_identity'}) != receipt['receipt_identity']:
            raise ValueError('Old native receipt identity changed')
        if receipt['world_size'] != 1 or receipt['state']['completed'] not in (1,2):
            raise ValueError('Unexpected reference native checkpoint')
        for name, ref in receipt['files'].items():
            if Path(name).name != name or (path.parent/name).stat().st_size != ref['bytes']:
                raise ValueError('Reference native inventory stat mismatch')
        all_files = {str(p.relative_to(path.parent)):p.stat().st_size for p in path.parent.rglob('*') if p.is_file()}
        records.append({'receipt':pin,'step':receipt['state']['completed'],'files':receipt['files'],
            'native_tensor_bytes':sum(v['bytes'] for v in receipt['files'].values()),
            'all_directory_bytes':sum(all_files.values()), 'all_file_sizes':all_files,
            'tensor_contents_rehashed':False})
    if [r['step'] for r in records] != [1,2]:
        raise ValueError('Need both real reference checkpoints')
    per_checkpoint = max(r['all_directory_bytes'] for r in records)
    paths = ['tools/st_world_policy/train_world.py','tools/st_world_policy/training_state.py',
        'tools/st_world_policy/fp32_master_init.py','tools/st_world_policy/official_runtime.py',
        'tools/st_world_policy/teacher_pca.py','starVLA/dataloader/st_world_dataset.py']
    reuse = {}
    for path in paths:
        blobs = [subprocess.check_output(['git','show',sha+':'+path],cwd=runtime) for sha in (old_source, source)]
        requested_bytes += sum(map(len,blobs)); check()
        reuse[path] = {'unchanged':blobs[0] == blobs[1], 'sha256':hashlib.sha256(blobs[1]).hexdigest()}
    if not all(v['unchanged'] for v in reuse.values()):
        raise ValueError('A supposedly inherited training component changed')
    sys.path.insert(0,str(runtime))
    from tools.st_world_policy.check_formal_approval import source_fingerprint
    fingerprint = source_fingerprint(runtime,source)
    def write(name, value):
        path = output/name
        if path.exists():
            raise FileExistsError('Preserve existing handoff evidence')
        body = (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        path.write_bytes(body)
        return {'path':str(path),'sha256':hashlib.sha256(body).hexdigest()}
    config_pin = write('TRAINING_CONFIG.json',cfg)
    target_pin = write('TARGET_CONTRACT.json',cfg['space_time']['teacher'])
    inputs = {'schema':'st_single_indexed_smoke_registration_inputs_v1','status':'LEAD_PREPARATION_ONLY',
        'runtime':str(runtime),'source_sha':source,'source_fingerprint':fingerprint,
        'run_id':'indexed_provider_single_updates2_v1','registration_output':str(new_root/'REGISTRATION.json'),
        'execution':cfg['execution'],'weights_usable_for_formal_init':False,
        'known_artifacts':{'config':config_pin,'target_contract':target_pin,'base_binding':base_pin,
            'data_manifest':selected_pin,'teacher_manifest':desc_pin},
        'missing_artifacts':['fresh actual environment','fresh actual resource_lease'],
        'execution_config_lineage':{'parent':cfg_pin,'new':config_pin,'changed_field':'runtime.output_root',
            'before':original['runtime']['output_root'],'after':cfg['runtime']['output_root']},
        'factory_qualification':prepared_pin,'provider_scope':scope_pin,'factory_lineage':lineage_pin,
        'proposed_topology':{'physical_gpu':'0','world_size':1,'micro_batch':1,'global_batch':2,'accumulation':2},
        'live_registration_created':False,'gpu_inspected_or_acquired':False,'training_launched':False,
        'proposed_checkpoint_budget':{'reservation_bytes':120*(1<<30),'separate_free_floor_bytes':1<<40,
            'normal_native_copies':2,'conservative_slots_for_step0_step1_step2':3,
            'observed_per_complete_directory_bytes':per_checkpoint,
            'three_slot_bytes':3*per_checkpoint,'remaining_for_partial_metadata_variation_bytes':120*(1<<30)-3*per_checkpoint,
            'quota_or_real_admission':False,'no_deletion':True,'new_teacher_tensor_bytes':0,
            'factory_root_is_separate_32MiB_metadata_namespace':True},
        'old_native_receipts':records,'inherited_unchanged_source':reuse,
        'measurement':{'seconds':time.monotonic()-start,'explicit_small_metadata_read_bytes':requested_bytes,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
        'pending_test':'One two-update student integration on the new exact32 indexed source; no repeated resume pair'}
    check(); write('REGISTRATION_INPUTS.json',inputs)
    print(json.dumps({'status':inputs['status'],'native_bytes':records[0]['native_tensor_bytes'],
        'three_slot_bytes':3*per_checkpoint,'config':config_pin, **inputs['measurement']}))


if __name__ == '__main__':
    main()
