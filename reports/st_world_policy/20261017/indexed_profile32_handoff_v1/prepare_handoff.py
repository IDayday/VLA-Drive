"""Prepare local proposal JSON only; no index/tensor/model access or live admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016')
RUNTIME = Path('/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017')
SOURCE = '2d5e9483a8c8bc80b71348211756f3cab6aa21b1'
FACTORY = ROOT/'smoke/world_indexed_provider_native2_v1'
NEW = FACTORY/'profile32_v1'
TRAIN = ROOT/'profile/world_indexed_provider_profile32_updates64_v1'


def read(path, sha=None):
    path = Path(path)
    if path.stat().st_size > 4 << 20:
        raise ValueError('Small metadata only')
    body = path.read_bytes()
    pin = {'path': str(path.resolve()), 'sha256': hashlib.sha256(body).hexdigest()}
    if sha is not None and pin['sha256'] != sha:
        raise ValueError('Changed actual input: '+str(path))
    return json.loads(body), pin


def write(name, value):
    path = OUT/name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return read(path)[1]


def main():
    start = time.monotonic()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip()!=SOURCE:
        raise ValueError('Runtime source changed')
    if subprocess.check_output(['git','status','--porcelain=v1'],cwd=RUNTIME,text=True).strip():
        raise ValueError('Runtime dirty')
    original, source_pin = read(FACTORY/'configured/training_config.json',
        'f6b8bd3ad59a763355130ef8bb0ef9228473f424516c62e630def80e87db948d')
    descriptor, desc_pin = read(FACTORY/'PROVIDER.json',
        'bb53252f1aea91ed54fa7b3917314d2d7f6c1c251d1f89f6a079f0f582f3874d')
    manifest, manifest_pin = read(ROOT/'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json',
        '7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8')
    base, base_pin = read(ROOT/'bindings/BASE_BINDING.json',
        '84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38')
    admitted, admitted_pin = read(FACTORY/'preparation_control_v1/ADMITTED.json')
    assert len(manifest['scenes']) == 32
    cfg = deepcopy(original)
    cfg['execution'].update(scope='PROFILE',stop_after=64)
    cfg['runtime'].update(output_root=str(TRAIN/'runs'),save_every=64,max_seconds=1500,profile_mode='none')
    cfg['runtime']['official_execution']['adaptation_reason'] = (
        'One bounded 64-update exact32 population supply/memory stability diagnostic. '
        'World1 micro1 global2; preserve the actual 20000-update/warmup1000 schedule, '
        'original objectives, BF16/FA2/TorchAdamW/ZeRO2 FP32 master backend, '
        'FP32 fixed8 teacher/PCA384, deterministic=true and checkpointing=false. '
        'Repeated32 is not full-population throughput or a formal recipe qualification.')
    # A PROFILE config must not present the old SMOKE descriptor as its active
    # provider. The factory later installs the NEW reviewed descriptor/scope.
    cfg['data'].pop('teacher_provider')
    cfg['data']['teacher_root'] = str(ROOT/'audits/history_data/native384_train64_cost_v1')
    assert cfg['optimizer']==original['optimizer'] and cfg['space_time']==original['space_time']
    cfg_pin=write('INPUT_PROFILE_CONFIG.json',cfg)
    target_pin=write('TARGET_CONTRACT.json',cfg['space_time']['teacher'])
    prior,_=read(OUT.parent/'indexed_training2_handoff_v1/REGISTRATION_INPUTS.json')
    size=prior['proposed_checkpoint_budget']['observed_per_complete_directory_bytes']
    handoff={'schema':'st_indexed_profile32_handoff_v1','status':'PROPOSED_NOT_APPROVED',
        'runtime':str(RUNTIME),'source_sha':SOURCE,'run_id':'indexed_provider_profile32_updates64_v1',
        'metadata_output_root':str(NEW),'checkpoint_output_root':str(TRAIN),
        'input_config':cfg_pin,'reference_config':source_pin,'parent_descriptor':desc_pin,
        'data_manifest':manifest_pin,'base_binding':base_pin,'target_contract':target_pin,
        'execution':cfg['execution'],'max_seconds':1500,'profile_mode':'none',
        'preparation':{'cpu':2,'rss_bytes':2<<30,'wall_seconds':180,'logical_read_budget_bytes':4<<30,
            'output_bytes':32<<20,'gpu':False,'known_original_pca_payload_read_bytes':1590864,
            'new_teacher_tensor_bytes':0,'full_index_read_required':True},
        'metadata_permission':{'prior_one_shot_admission':admitted_pin,
            'prior_status':admitted['status'],'extension_approved':False,
            'required':'New explicit one-shot script/subpath admission; existing headroom alone is not authorization',
            'preferred':'Account profile32_v1 and existing files together within original 32MiB ceiling',
            'fallback_minimum_new_reservation_bytes':32<<20},
        'native_storage':{'reservation_bytes':80<<30,'observed_native_directory_bytes':size,
            'normal_complete_copies':1,'conservative_slots_including_partial':2,
            'two_slot_bytes':2*size,'remaining_bytes':(80<<30)-2*size,
            'actual_reservation':False,'existing_files_deleted':False,'teacher_tensor_disk_bytes':0},
        'pending':['independent new-source correctness closure','explicit metadata one-shot resource admission',
            'actual factory descriptor/config pins','fresh training registration/reviews/environment/GPU lease',
            'separate <=80GiB training storage admission'],
        'training_launched':False,'gpu_queried_or_acquired':False,'live_ledger_changed':False,
        'formal_approved':False,'new_population_qualified':False,'data_reused_exact32':True,
        'local_metadata_preparation_wall_seconds':time.monotonic()-start}
    write('HANDOFF.json',handoff)
    print(json.dumps({'status':handoff['status'],'config':cfg_pin,'checkpoint_two_slot_bytes':2*size}))


if __name__=='__main__':
    main()
