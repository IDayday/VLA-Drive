#!/usr/bin/env python3
"""Pin actual T3 CPU outputs for the existing PROFILE registrar. Never dispatch."""
import argparse, hashlib, importlib.util, json, math, os, stat, subprocess, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/mnt/project/st-world-policy-20261016')
PREP=BASE/'audits/temporal_T3_indexed32_preparation_v2'
RUNTIME=Path('/mnt/project/VLA-Drive-st-temporal-packed-migration-fix-20261018')
SOURCE='cf516259c8efc24503727f14b9e330ed32231293'
OUT=BASE/'profile/temporal_T3_indexed32_updates4_v2'
RUN_ID='temporal_T3_indexed32_updates4_v2'
HOST='training-vlawm-tsinghua-zt2-worker-0'
GPU_UUID='GPU-cdd3d7ed-6094-0ef4-282c-2de8dbf86e00'
EXECUTION=dict(scope='PROFILE',family='temporal',world_size=1,micro_batch=1,global_batch=2,
    max_updates=20000,stop_after=4,sample_limit=32,seed=42,
    precision_policy='S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off')
BUDGET=dict(gpu_devices=1,gpu_capacity_bytes=80<<30,rss_bytes=64<<30,
    checkpoint_output_bytes=80<<30,logical_read_proposed_bytes=256<<30,
    native_seconds=1800,outer_seconds=1920,cleanup_seconds=180,
    cpu_threads=2,loader_workers=0,new_teacher_tensor_disk_bytes=0,
    cumulative_updates=4,cumulative_scene_exposures=8)
SELECTED={'path':str(BASE/'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json'),
    'sha256':'7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8'}
BASE_PIN={'path':str(BASE/'bindings/BASE_BINDING.json'),
    'sha256':'84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38'}

def document(path,expected=None):
    p=Path(path)
    if not p.is_absolute() or p.resolve()!=p:raise ValueError('Canonical absolute metadata path required')
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>4<<20:raise ValueError('Small regular metadata required')
        raw=stream.read((4<<20)+1);after=os.fstat(stream.fileno())
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if fields(before)!=fields(after) or len(raw)!=before.st_size:raise ValueError('Metadata changed during read')
    pin={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
    if expected is not None and pin!=expected:raise ValueError('Metadata pin mismatch')
    return json.loads(raw),pin

def validate_factory(result):
    res=result.get('resources',{});cli=result.get('cli_result',{})
    if (result.get('status')!='PREPARATION_COMPLETE' or res.get('returncode')!=0
        or res.get('failure') is not None or res.get('cleanup',{}).get('live_owned_remaining')!=0
        or cli.get('status')!='PROVIDER_PREPARATION_COMPLETE' or cli.get('new_teacher_tensor_bytes')!=0
        or result.get('training_started') is not False or result.get('ledger_written') is not False):
        raise ValueError('Actual COMPLETE factory with successful cleanup required')
    expected={'SOURCE_MIGRATION.json','configured/PREPARATION_COMPLETE.json','configured/PROVIDER.json',
        'configured/PROVIDER_CHANNEL_SCOPE.json','configured/T2_REFERENCE.json','configured/T3_RECIPE.json','configured/training_config.json'}
    if set(cli.get('outputs',{}))!=expected:raise ValueError('Exact seven original factory artifacts required')
    return cli['outputs']

def validate_recipe(cfg):
    rt=cfg['runtime'];st=cfg['space_time']
    if (cfg['execution']!=EXECUTION or rt['max_seconds']!=1800 or rt['save_every']!=4
        or rt['loader_workers']!=0 or rt['output_root']!=str(OUT/'runs')
        or rt['deterministic'] is not True or rt['gradient_checkpointing'] is not False
        or rt.get('profile_mode')!='none' or cfg['optimizer']['warmup']!=1000
        or st['cell']!='T3' or st['history'] is not True or st['future'] is not True
        or st['delta'] is not False or st['delta_weight']!=0):
        raise ValueError('Exact T3 four-update/original schedule recipe required')

def actual_inputs():
    # Fail before importing model code or creating an output if preparation is absent/failed.
    result,result_pin=document(PREP/'controls/RESULT.json')
    outputs=validate_factory(result)
    docs={}
    stages=set()
    for name,pin in outputs.items():
        path=Path(pin['path'])
        if name=='SOURCE_MIGRATION.json':
            if path!=PREP/name:raise ValueError('Foreign source migration')
        else:
            if path.parent.parent!=PREP or not path.parent.name.startswith('.configured.staging-') or path.name!=Path(name).name:
                raise ValueError('Foreign immutable composition stage')
            stages.add(str(path.parent))
        docs[name],_=document(path,pin)
    if len(stages)!=1:raise ValueError('Mixed composition stages')
    cfg=docs['configured/training_config.json'];validate_recipe(cfg)
    done=docs['configured/PREPARATION_COMPLETE.json']
    if (done.get('schema')!='st_temporal_composition_preparation_v1'
        or done.get('result',{}).get('config')!=outputs['configured/training_config.json']
        or done['result'].get('training_started') is not False or done['result'].get('formal_approved') is not False
        or done['result'].get('updates')!=4 or done['inputs'].get('cell')!='T3'
        or done['inputs'].get('data_manifest')!=SELECTED
        or done['inputs'].get('source_migration')!=outputs['SOURCE_MIGRATION.json']):
        raise ValueError('Actual composition completion/inputs mismatch')
    selected,_=document(SELECTED['path'],SELECTED);document(BASE_PIN['path'],BASE_PIN)
    if len(selected['scenes'])!=32:raise ValueError('Exact32 required')
    if cfg['data']['teacher_provider']['descriptor']!=outputs['configured/PROVIDER.json']:raise ValueError('Provider mismatch')
    if docs['configured/PROVIDER.json']['profile_admission']['data_manifest']!=SELECTED:raise ValueError('Provider population mismatch')
    if subprocess.check_output(['git','-C',str(RUNTIME),'rev-parse','HEAD'],text=True).strip()!=SOURCE or subprocess.check_output(['git','-C',str(RUNTIME),'status','--porcelain=v1']):
        raise ValueError('Exact clean frozen runtime required')
    return dict(schema='st_T3_profile4_binding_v1',status='ACTUAL_FACTORY_PINNED_NOT_DISPATCHED',
        source_sha=SOURCE,runtime=str(RUNTIME),factory_result=result_pin,outputs=outputs,
        pins=dict(config=outputs['configured/training_config.json'],teacher_manifest=outputs['configured/PROVIDER.json'],
            data_manifest=SELECTED,base_binding=BASE_PIN),execution=EXECUTION,
        host=HOST,gpu='0',gpu_uuid=GPU_UUID,run_id=RUN_ID,output_root=str(OUT),
        proposed_budget=BUDGET,weights_usable_for_formal_init=False,training_started=False,
        registration_created=False,resource_admission_created=False)

def registrar():
    p=RUNTIME/'tools/st_world_policy/check_formal_approval.py'
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='e46632f71b2385a431d4dc58003177e682620f29dfc239dd9088e135647e903e':
        raise ValueError('Existing registration validator changed')
    spec=importlib.util.spec_from_file_location('original_st_registrar',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def registration_value(inputs,environment_pin,resource_lease_pin,target_contract_pin,now):
    # The original prepare helper writes these actual files on the target host.
    # This function only constructs its unchanged schema and never signs an admission.
    if inputs.get('source_sha')!=SOURCE or inputs.get('execution')!=EXECUTION or inputs.get('run_id')!=RUN_ID:
        raise ValueError('Exact original T3 binding required')
    cfg,_=document(inputs['pins']['config']['path'],inputs['pins']['config']);validate_recipe(cfg)
    env,_=document(environment_pin['path'],environment_pin)
    lease,_=document(resource_lease_pin['path'],resource_lease_pin)
    target,_=document(target_contract_pin['path'],target_contract_pin)
    if target!=cfg['space_time']['teacher'] or env.get('schema')!='st_training_environment_v1':raise ValueError('Actual target/environment required')
    if any(type(lease.get(k)) not in (int,float) or not math.isfinite(lease[k]) for k in ('valid_from_unix','valid_until_unix')):
        raise ValueError('Finite actual resource lease times required')
    if (lease.get('schema')!='st_training_resource_lease_v1' or lease.get('hostnames')!=[HOST]
        or lease.get('world_size')!=1 or lease.get('cuda_visible_devices_by_host')!={HOST:'0'}
        or lease.get('gpu_uuids_by_host')!={HOST:[GPU_UUID]} or lease.get('max_invocation_seconds')!=1800
        or lease.get('cpu_threads')!=2 or lease.get('tensor_cache_write_bytes')!=0
        or lease.get('checkpoint_reservation_bytes')!=80<<30
        or not lease.get('valid_from_unix',float('inf'))<=now
        or not lease.get('valid_until_unix',0)>=now+2100):raise ValueError('Fresh exact target-host resource lease required')
    m=registrar()
    artifacts={**inputs['pins'],'environment':environment_pin,'resource_lease':resource_lease_pin,'target_contract':target_contract_pin}
    m.validate_execution(inputs['execution'])
    return dict(schema='st_world_run_v1',run_id=RUN_ID,source_sha=SOURCE,
        source_fingerprint=m.source_fingerprint(RUNTIME),artifacts=artifacts,execution=EXECUTION,
        implementer_agents=['/root','/root/st_indexprep_boundary_fix'],weights_usable_for_formal_init=False)

def native_argv():
    return ['/root/miniconda3/envs/ddp/bin/python','-m','torch.distributed.run','--standalone',
        '--nnodes=1','--nproc_per_node=1','-m','tools.st_world_policy.train_world','--registration',str(OUT/'REGISTRATION.json')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='Optional new small binding JSON; never registration or launch')
    a=p.parse_args();value=actual_inputs();value['native_argv_for_existing_supervisor']=native_argv()
    if a.output:
        # Evidence is immutable and cannot be confused with the actual training registration.
        if a.output.name!='INPUTS.json' or a.output.resolve().parent!=HERE:raise ValueError('Output is only this owned INPUTS.json')
        with a.output.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'status':value['status'],'output':str(a.output) if a.output else None,'dispatch':False}))
if __name__=='__main__':main()
