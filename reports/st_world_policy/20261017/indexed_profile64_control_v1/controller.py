"""Lead-only one-shot PROFILE control. Default is read-only preflight; no trainer fork."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path('/mnt/project/st-world-policy-20261016')
REPO=Path('/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017')
SOURCE='2d5e9483a8c8bc80b71348211756f3cab6aa21b1'
OUT=ROOT/'profile/world_indexed_provider_profile32_updates64_v1'
RUN_ID='indexed_provider_profile32_updates64_v1'
HOST='training-vla-zt-worker-0'
GPU='0'
GPU_LOCK=Path('/tmp/foresight_gpu_0.lock')
PYTHON='/root/miniconda3/envs/ddp/bin/python'
OLD=ROOT/'resources/qualify_fixed_provider_training_v1.py'
OLD_SHA='8d7049bd2e7557121c78d02674590004ab923a8129dd9c30ffd3cbc7e8ef8925'
POLICY=Path('/mnt/project/server_dispatch_policy.json')
AUTH=ROOT/'resources/AUTHORIZATION_20261016T203442.json'
INPUT=Path(__file__).resolve().with_name('INPUTS.json')
RESERVE=80*(1<<30)
OTHER_COMMITMENTS=2103*(1<<30)
FREE_FLOOR=1<<40
OUTER_SECONDS=1620
CLEANUP_SECONDS=180


def read(path):
    p=Path(path)
    if p.stat().st_size>32<<20: raise ValueError('Bounded metadata only')
    return json.loads(p.read_text())


def digest(path):
    p=Path(path)
    if p.stat().st_size>32<<20: raise ValueError('No tensor hashing from controller')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(value,message):
    if not value: raise RuntimeError(message)


def legacy():
    require(digest(OLD)==OLD_SHA,'Inherited helper source changed')
    spec=importlib.util.spec_from_file_location('fixed_provider_control_helpers',OLD)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.REPO=REPO
    return module


def validate_recipe(cfg):
    require(cfg['execution']=={
        'scope':'PROFILE','family':'temporal','world_size':1,'micro_batch':1,'global_batch':2,
        'max_updates':20000,'stop_after':64,'sample_limit':32,'seed':42,
        'precision_policy':'S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off'},'Wrong finite PROFILE identity')
    rt=cfg['runtime']
    require(rt['max_seconds']==1500 and rt['save_every']==64 and rt['loader_workers']==0
        and rt['deterministic'] is True and rt['gradient_checkpointing'] is False
        and rt.get('profile_mode','none')=='none' and rt['output_root']==str(OUT/'runs')
        and cfg['optimizer']['warmup']==1000,'Runtime/schedule changed')


def command():
    return ['timeout','--signal=TERM','--kill-after=180s','1620s',PYTHON,'-m','torch.distributed.run',
        '--standalone','--nnodes=1','--nproc_per_node=1','-m','tools.st_world_policy.train_world',
        '--registration',str(OUT/'REGISTRATION.json')]


def storage_requirement(used):
    require(type(used) is int and 0<=used<=RESERVE,'Checkpoint reservation exceeded')
    return FREE_FLOOR+OTHER_COMMITMENTS+RESERVE-used


def storage():
    used=0
    if OUT.exists():
        for path in OUT.rglob('*'):
            require(not path.is_symlink(),'Unexpected symlink in owned output')
            if path.is_file(): used+=max(path.stat().st_size,path.stat().st_blocks*512)
    free=shutil.disk_usage(ROOT).free
    required=storage_requirement(used)
    require(free>=required,'Insufficient space after 2103GiB existing commitments + 1TiB floor + own remaining80GiB')
    return {'schema':'st_indexed_profile64_storage_check_v1','checked_at_unix':time.time(),
        'actual_free_bytes':free,'minimum_free_bytes':required,'reserved_bytes':RESERVE,
        'existing_task_bytes':used,'other_registered_commitments_bytes':OTHER_COMMITMENTS,
        'additional_free_floor_bytes':FREE_FLOOR,'remaining_reserved_bytes':RESERVE-used,
        'ledger_mutated':False,'reservation_kind':'externally_admitted_advisory_not_quota',
        'quota_verified':False,'automatic_cleanup':False,'new_teacher_tensor_bytes':0}


def gpu_record(idle):
    rows=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    require(len(rows)==1,'Exactly physical GPU0 required')
    uuid,memory,util=[x.strip() for x in rows[0].split(',')]
    if idle:require(int(memory)<64 and int(util)<=5,'GPU0 busy; never stop another job')
    return {'uuid':uuid,'memory_mib':int(memory),'utilization_percent':int(util)}


def common():
    require(socket.gethostname()==HOST,'Wrong actual dispatch host')
    policy=read(POLICY)
    require(HOST in policy['allowed_hosts']['training-vla-zt']['aliases'] and HOST in policy['allowed_aliases'],
        'Host no longer authorized')
    for key,item in policy.get('global_dispatch_prohibitions',{}).items():
        if key in (HOST,'training-vla-zt') or HOST in item.get('aliases',[]):
            require(item.get('new_task_dispatch_allowed') is True,'Host dispatch explicitly prohibited')
    require('training-vla-zt' in read(AUTH)['allowed_hosts'],'Campaign authorization missing')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SOURCE,'Wrong runtime')
    require(not subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO).strip(),'Dirty runtime')
    inputs=read(INPUT)
    for name,pin in inputs['pins'].items():
        require(digest(pin['path'])==pin['sha256'],'Changed actual factory/dependency: '+name)
    require(read(inputs['pins']['factory_result']['path'])['status']=='COMPLETE','Factory not complete')
    cfg=read(inputs['pins']['config']['path']);validate_recipe(cfg)
    require(cfg['data']['teacher_provider']['descriptor']==inputs['pins']['teacher_manifest'],
        'Config and provider pins disagree')
    descriptor=read(inputs['pins']['teacher_manifest']['path'])
    require(descriptor['profile_admission']['data_manifest']==inputs['pins']['data_manifest'],
        'Registered population differs from actual PROFILE descriptor')
    require(len(read(inputs['pins']['data_manifest']['path'])['scenes'])==32,'Changed exact32 population')
    return inputs,cfg


def admission(path,sha):
    require(path and sha and digest(path)==sha,'Explicit externally admitted resource receipt is required')
    a=read(path)
    expected={'schema':'st_indexed_profile64_resource_admission_v1','status':'ADMITTED','host':HOST,'gpu':GPU,
        'source_sha':SOURCE,'run_id':RUN_ID,'output_root':str(OUT),'controller_sha256':digest(__file__),
        'inputs_sha256':digest(INPUT),'checkpoint_reservation_bytes':RESERVE,
        'other_commitments_bytes':OTHER_COMMITMENTS,'free_floor_bytes':FREE_FLOOR,
        'max_invocation_seconds':1500,'outer_timeout_seconds':OUTER_SECONDS,'cleanup_seconds':CLEANUP_SECONDS}
    require(all(a.get(k)==v and type(a.get(k)) is type(v) for k,v in expected.items()),'Resource admission does not bind this exact run')
    now=time.time()
    require(type(a.get('valid_from_unix')) in (int,float) and type(a.get('valid_until_unix')) in (int,float)
        and math.isfinite(a['valid_from_unix']) and math.isfinite(a['valid_until_unix'])
        and a['valid_from_unix']<=now and now+OUTER_SECONDS+CLEANUP_SECONDS<=a['valid_until_unix'],
        'Resource admission expired or too short including final cleanup')
    return a,{'path':str(Path(path).resolve()),'sha256':sha}


def prepare(inputs,cfg,approval,approval_pin):
    m=legacy(); envrec,validate,official,fingerprint,verify,write,_=m.modules()
    with m.lock(ROOT/'resources/indexed_profile64_preparation.lock'),m.lock(GPU_LOCK):
        require(not OUT.exists(),'Existing output preserved; no overwrite or automatic retry')
        gpu=gpu_record(True);disk=storage();validate(cfg)
        require(official(cfg) is not None,'Official execution policy missing')
        now=time.time()
        lease={'schema':'st_training_resource_lease_v1','hostnames':[HOST],'world_size':1,
            'cuda_visible_devices_by_host':{HOST:GPU},'gpu_uuids_by_host':{HOST:[gpu['uuid']]},
            'valid_from_unix':now-1,'valid_until_unix':approval['valid_until_unix'],'max_invocation_seconds':1500,
            'purpose':'One64-update exact32 supply/memory diagnostic; no fullpopulation/timing qualification',
            'authorization':str(AUTH),'resource_admission':approval_pin,'checkpoint_reservation_bytes':RESERVE,
            'other_commitments_bytes':OTHER_COMMITMENTS,'separate_free_floor_bytes':FREE_FLOOR,
            'cpu_threads':2,'tensor_cache_write_bytes':0,'pressure_workers_killed':False,'real_jobs_stopped':False}
        OUT.mkdir(parents=True)
        for name,value in [('CONFIG.json',cfg),('TARGET_CONTRACT.json',cfg['space_time']['teacher']),
            ('ENVIRONMENT.json',envrec()),('RESOURCE_LEASE.json',lease),('STORAGE_CHECK.json',disk)]:
            write(OUT/name,value,immutable=True)
        paths={k:Path(inputs['pins'][k]['path']) for k in ('base_binding','data_manifest','teacher_manifest')}
        paths.update(config=OUT/'CONFIG.json',target_contract=OUT/'TARGET_CONTRACT.json',
            environment=OUT/'ENVIRONMENT.json',resource_lease=OUT/'RESOURCE_LEASE.json')
        reg={'schema':'st_world_run_v1','run_id':RUN_ID,'source_sha':SOURCE,'source_fingerprint':fingerprint(REPO),
            'artifacts':{k:{'path':str(v),'sha256':digest(v)} for k,v in paths.items()},'execution':cfg['execution'],
            'implementer_agents':['/root','/root/st_indexed_composition','/root/st_bounded_future_provider'],
            'weights_usable_for_formal_init':False}
        write(OUT/'REGISTRATION.json',reg,immutable=True);verify(OUT/'REGISTRATION.json',repo=REPO)
        write(OUT/'PREPARED.json',{'status':'PREPARED_NOT_LAUNCHED','source_sha':SOURCE,'host':HOST,'gpu':gpu,
            'script_sha256':digest(__file__),'inherited_helpers_sha256':digest(OLD),'inputs_sha256':digest(INPUT),
            'registration_sha256':digest(OUT/'REGISTRATION.json'),'resource_admission':approval_pin,
            'dispatch_policy_sha256':digest(POLICY),'authorization_sha256':digest(AUTH),'formal_approved':False},immutable=True)
    print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','registration':str(OUT/'REGISTRATION.json')}),flush=True)


def execute(approval_pin):
    m=legacy();envrec,_,_,_,verify,write,_=m.modules()
    verify(OUT/'REGISTRATION.json',repo=REPO)
    prepared=read(OUT/'PREPARED.json');lease=read(OUT/'RESOURCE_LEASE.json')
    require(prepared['script_sha256']==digest(__file__) and prepared['inputs_sha256']==digest(INPUT)
        and prepared['inherited_helpers_sha256']==digest(OLD),'Controller inputs changed')
    require(prepared['registration_sha256']==digest(OUT/'REGISTRATION.json') and prepared['host']==HOST,
        'Registered identity changed')
    require(prepared['resource_admission']==approval_pin and lease['resource_admission']==approval_pin,
        'Different resource admission; no implicit extension')
    require(read(OUT/'ENVIRONMENT.json')==envrec(),'Actual environment changed')
    require(lease['valid_from_unix']<=time.time() and time.time()+1800<=lease['valid_until_unix'],'Lease too short')
    run=OUT/'runs'/RUN_ID
    with m.lock(OUT/'DISPATCH.lock'),m.lock(GPU_LOCK) as gpu_lock:
        gpu=gpu_record(True);disk=storage()
        require(gpu['uuid']==lease['gpu_uuids_by_host'][HOST][0],'Physical GPU0 UUID changed')
        require(not (OUT/'LAUNCH_initial.json').exists() and not run.exists(),'Duplicate/failed run preserved; no retry')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'OPENBLAS_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','TOKENIZERS_PARALLELISM':'false',
            'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
        with (OUT/'TRAIN_initial.log').open('xb') as log:
            child=subprocess.Popen(command(),cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(gpu_lock.fileno(),))
        write(OUT/'LAUNCH_initial.json',{'host':HOST,'pid':child.pid,'process_starttime':m.proc_start(child.pid),
            'launcher_pid':os.getpid(),'launcher_starttime':m.proc_start(os.getpid()),'argv':command(),
            'source_sha':SOURCE,'script_sha256':digest(__file__),'started_unix':time.time(),'scope':'PROFILE',
            'cuda_visible_devices':GPU,'gpu_lock_path':str(GPU_LOCK),'storage':disk,'formal':False,'resume':False,
            'max_completed_updates':64,'sample_limit':32,'resource_admission':approval_pin},immutable=True)
        print(json.dumps({'status':'LAUNCHED','pid':child.pid,'run_id':RUN_ID}),flush=True)
        code=child.wait()
        m.archive_run_metadata(OUT,run,'initial',write)
        result=read(run/'RESULT.json') if (run/'RESULT.json').exists() else None
        write(OUT/'EXIT_initial.json',{'returncode':code,'ended_unix':time.time(),'result':result},immutable=True)
        require(code==0,'Invocation failed; preserved evidence, no automatic retry')
        require(result is not None and result.get('completed')==64 and result.get('status')=='PAUSED',
            'Earlier boundary or incorrect terminal result; do not claim64 completed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('preflight','prepare','run'),default='preflight')
    p.add_argument('--resource-admission');p.add_argument('--resource-admission-sha256')
    args=p.parse_args();inputs,cfg=common()
    if args.action=='preflight':
        print(json.dumps({'status':'METADATA_PREFLIGHT_ONLY','source':SOURCE,'scope':'PROFILE',
            'updates':64,'gpu_queried':False,'output_created':False}));return
    approved,approval_pin=admission(args.resource_admission,args.resource_admission_sha256)
    if args.action=='prepare':prepare(inputs,cfg,approved,approval_pin)
    else:execute(approval_pin)


if __name__=='__main__':
    main()
