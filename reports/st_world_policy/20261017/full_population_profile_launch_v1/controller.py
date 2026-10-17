"""Lead-only one-shot PROFILE control. Default is read-only preflight; no trainer fork."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
from types import ModuleType
import socket
import subprocess
import sys
import time

ROOT=Path('/mnt/project/st-world-policy-20261016')
REPO=Path('/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017')
SOURCE='a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a'
OUT=ROOT/'profile/world_indexed_provider_fulltrain_updates64_world2_v1'
FACTORY=ROOT/'audits/full_population_provider_preparation_v1'
RUN_ID='indexed_provider_fulltrain_updates64_world2_v1'
HOST='training-vla-zt-worker-0'
GPU='0,1'
GPU_LOCKS=(Path('/tmp/foresight_gpu_0.lock'),Path('/tmp/foresight_gpu_1.lock'))
PYTHON='/root/miniconda3/envs/ddp/bin/python'
OLD=ROOT/'resources/qualify_fixed_provider_training_v1.py'
OLD_SHA='8d7049bd2e7557121c78d02674590004ab923a8129dd9c30ffd3cbc7e8ef8925'
POLICY=Path('/mnt/project/server_dispatch_policy.json')
AUTH=ROOT/'resources/AUTHORIZATION_20261016T203442.json'
INPUT=Path(__file__).resolve().with_name('INPUTS.json')
RESERVE=80*(1<<30)
OTHER_COMMITMENTS=2303*(1<<30)
FREE_FLOOR=1<<40
OUTER_SECONDS=1620
CLEANUP_SECONDS=180
RSS_LIMIT=64*(1<<30)
RSS_INTERVAL=1.
STORAGE_INTERVAL=10.
MONITOR_SOURCES={
 'prepare_bindings':('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/prepare_bindings.py','31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84'),
 'full_profile_legacy_supervisor':('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py','d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902'),
 'full_profile_ancestry':('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/torchrun_descendant_scope_v1/owned_descendants.py','b8d38e1e66036fc09886368f4a35bb21b5ea76306291a84588a082069ee26c85')}
METADATA_SHA='46233cf232a61e73e7e24f967b1cf316298473b017f49f4e96432d7e1b1a3591'


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
        'scope':'PROFILE','family':'temporal','world_size':2,'micro_batch':1,'global_batch':2,
        'max_updates':20000,'stop_after':64,'sample_limit':101592,'seed':42,
        'precision_policy':'S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off'},'Wrong finite PROFILE identity')
    rt=cfg['runtime']
    require(rt['max_seconds']==1500 and rt['save_every']==64 and rt['loader_workers']==0
        and rt['deterministic'] is True and rt['gradient_checkpointing'] is False
        and rt.get('profile_mode','none')=='none' and rt['output_root']==str(FACTORY/'configured/runs')
        and cfg['optimizer']['warmup']==1000,'Runtime/schedule changed')


def command(resume=False):
    argv = ['timeout','--signal=TERM','--kill-after=180s','1620s',PYTHON,'-m','torch.distributed.run',
        '--standalone','--nnodes=1','--nproc_per_node=2','-m','tools.st_world_policy.train_world',
        '--registration',str(OUT/'REGISTRATION.json')]
    return argv + (['--resume','--acknowledge-stop'] if resume else [])


def storage_requirement(used):
    require(type(used) is int and 0<=used<=RESERVE,'Checkpoint reservation exceeded')
    return FREE_FLOOR+OTHER_COMMITMENTS+RESERVE-used


def storage():
    used=0
    if OUT.exists():
        for count,path in enumerate(OUT.rglob('*'),1):
            require(count<=2048,'Owned output entry bound exceeded')
            require(not path.is_symlink(),'Unexpected symlink in owned output')
            if path.is_file(): used+=max(path.stat().st_size,path.stat().st_blocks*512)
    free=shutil.disk_usage(ROOT).free
    required=storage_requirement(used)
    require(free>=required,'Insufficient space after 2303GiB existing commitments + 1TiB floor + own remaining80GiB')
    return {'schema':'st_indexed_profile64_storage_check_v1','checked_at_unix':time.time(),
        'actual_free_bytes':free,'minimum_free_bytes':required,'reserved_bytes':RESERVE,
        'existing_task_bytes':used,'other_registered_commitments_bytes':OTHER_COMMITMENTS,
        'additional_free_floor_bytes':FREE_FLOOR,'remaining_reserved_bytes':RESERVE-used,
        'ledger_mutated':False,'reservation_kind':'externally_admitted_advisory_not_quota',
        'quota_verified':False,'automatic_cleanup':False,'new_teacher_tensor_bytes':0}


def gpu_record(idle):
    rows=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    require(len(rows)==2,'Exactly physical GPU0/1 required')
    result=[]
    for gpu,row in zip(GPU.split(','),rows):
        uuid,memory,util=[x.strip() for x in row.split(',')]
        if idle:require(int(memory)<64 and int(util)<=5,'GPU'+gpu+' busy; never stop another job')
        result.append({'id':gpu,'uuid':uuid,'memory_mib':int(memory),'utilization_percent':int(util)})
    require(len({x['uuid'] for x in result})==2,'Duplicate physical GPU UUID')
    return result


def validate_population(inputs, cfg):
    require(inputs.get('schema')=='st_full_index_profile64_controller_inputs_v1'
        and inputs.get('source_sha')==SOURCE and inputs.get('status')=='ACTUAL_FACTORY_PINNED_NOT_DISPATCHED',
        'Actual completed factory and resource input pins required; templates cannot prepare')
    pins=inputs['pins']; factory=read(pins['factory_result']['path'])
    require(factory.get('status')=='PREPARATION_COMPLETE'
        and factory.get('training_started') is False
        and factory.get('cli_result',{}).get('status')=='PROVIDER_PREPARATION_COMPLETE',
        'Full-population provider factory not complete')
    outputs=factory['cli_result']['outputs']
    for key,name in [('config','configured/training_config.json'),('teacher_manifest','PROVIDER.json'),
        ('provider_pca_scope','configured/PROVIDER_CHANNEL_SCOPE.json'),
        ('config_lineage','configured/PROVIDER_CONFIG_LINEAGE.json'),('source_migration','SOURCE_MIGRATION.json')]:
        require(outputs.get(name)==pins[key],'Factory result output pin differs: '+key)
    require(cfg['data']['teacher_provider']['descriptor']==pins['teacher_manifest']
        and cfg['data']['teacher_provider']['pca_channel_scope']==pins['provider_pca_scope'],
        'Config and actual provider/PCA pins disagree')
    descriptor=read(pins['teacher_manifest']['path']); population=read(pins['request_index_complete']['path'])
    admission=descriptor.get('profile_admission',{})
    require(admission.get('schema')=='st_full_index_provider_profile_admission_v1'
        and admission.get('data_manifest')==pins['data_manifest'] and admission.get('max_scenes')==101592
        and descriptor.get('request_index',{}).get('complete')==pins['request_index_complete']
        and population.get('status')=='COMPLETE' and population.get('scene_count')==101592
        and population.get('manifest_header',{}).get('split')=='train',
        'Actual complete full original TRAIN population must match the provider')
    require(admission.get('consumption')=={
        'schema':'st_full_index_profile_consumption_v1','initial_completed_updates':0,'initial_epoch':0,'initial_offset':0,
        'cumulative_optimizer_updates':64,'cumulative_scene_exposures':128,'world_size':2,'micro_batch':1,'global_batch':2,
        'gradient_accumulation_steps':1,'per_rank_scene_exposures':[64,64],
        'resume_budget':'same_run_cumulative_not_per_invocation','formal_global32_throughput_qualified':False},
        'Whole-run64/128 budget changed')
    # The factory and native provider validate every row/order/hash. This control
    # uses its pinned COMPLETE count and never parses the full scene list again.


def registered_config(cfg):
    # Same output-only relocation used by the prior relocate_config.py. The
    # provider recipe identity deliberately excludes only this output namespace.
    result=deepcopy(cfg);result['runtime']['output_root']=str(OUT/'runs')
    return result


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
    require(inputs.get('status')=='ACTUAL_FACTORY_PINNED_NOT_DISPATCHED','Required actual input pins are not bound')
    for name,pin in inputs['pins'].items():
        require(isinstance(pin,dict) and set(pin)=={'path','sha256'}
            and isinstance(pin['sha256'],str) and len(pin['sha256'])==64,
            'Missing exact input SHA: '+name)
        require(digest(pin['path'])==pin['sha256'],'Changed actual factory/dependency: '+name)
    cfg=read(inputs['pins']['config']['path']);validate_recipe(cfg);validate_population(inputs,cfg)

    return inputs,cfg


def admission(path,sha,action):
    require(path and sha and digest(path)==sha,'Explicit externally admitted resource receipt is required')
    a=read(path)
    expected={'schema':'st_full_index_profile64_resource_admission_v1','status':'ADMITTED','host':HOST,'gpu':GPU,
        'source_sha':SOURCE,'run_id':RUN_ID,'output_root':str(OUT),'controller_sha256':digest(__file__),
        'inputs_sha256':digest(INPUT),'checkpoint_reservation_bytes':RESERVE,
        'other_commitments_bytes':OTHER_COMMITMENTS,'free_floor_bytes':FREE_FLOOR,
        'max_invocation_seconds':1500,'outer_timeout_seconds':OUTER_SECONDS,'cleanup_seconds':CLEANUP_SECONDS,
        'rss_limit_bytes':RSS_LIMIT,'rss_interval_seconds':RSS_INTERVAL,'storage_interval_seconds':STORAGE_INTERVAL}
    require(all(a.get(k)==v and type(a.get(k)) is type(v) for k,v in expected.items()),'Resource admission does not bind this exact run')
    require(action in a.get('allowed_actions',[]),'This action is outside its resource admission')
    cpus=a.get('cpu_affinity')
    require(isinstance(cpus,list) and len(cpus) in (2,4) and len(set(cpus))==len(cpus)
        and all(type(x) is int and x>=0 for x in cpus),'Explicit bounded CPU affinity required')
    cp_pin=a.get('checkpoint_commitments',{})
    require(isinstance(cp_pin,dict) and set(cp_pin)=={'path','sha256'}
        and digest(cp_pin['path'])==cp_pin['sha256'],'Fresh exact other-checkpoint commitments required')
    cp=read(cp_pin['path']);now=time.time()
    require(cp.get('remaining_bytes')==OTHER_COMMITMENTS and cp.get('free_floor_bytes')==FREE_FLOOR
        and type(cp.get('valid_until_unix')) in (int,float) and math.isfinite(cp['valid_until_unix'])
        and now+OUTER_SECONDS+CLEANUP_SECONDS<=cp['valid_until_unix'],
        'Checkpoint obligations changed or expire during this invocation')
    require(type(a.get('valid_from_unix')) in (int,float) and type(a.get('valid_until_unix')) in (int,float)
        and math.isfinite(a['valid_from_unix']) and math.isfinite(a['valid_until_unix'])
        and a['valid_from_unix']<=now and now+OUTER_SECONDS+CLEANUP_SECONDS<=a['valid_until_unix'],
        'Resource admission expired or too short including final cleanup')
    return a,{'path':str(Path(path).resolve()),'sha256':sha}


def prepare(inputs,cfg,approval,approval_pin):
    m=legacy(); envrec,validate,official,fingerprint,verify,write,_=m.modules()
    with ExitStack() as stack:
        stack.enter_context(m.lock(ROOT/'resources/indexed_profile64_preparation.lock'))
        for path in GPU_LOCKS: stack.enter_context(m.lock(path))
        require(not OUT.exists(),'Existing output preserved; no overwrite or automatic retry')
        gpu=gpu_record(True);disk=storage();cfg=registered_config(cfg);validate(cfg)
        require(official(cfg) is not None,'Official execution policy missing')
        now=time.time()
        lease={'schema':'st_training_resource_lease_v1','hostnames':[HOST],'world_size':2,
            'cuda_visible_devices_by_host':{HOST:GPU},'gpu_uuids_by_host':{HOST:[row['uuid'] for row in gpu]},
            'valid_from_unix':now-1,'valid_until_unix':approval['valid_until_unix'],'max_invocation_seconds':1500,
            'purpose':'Full original population, cumulative64 updates/128 exposures, world2/global2 diagnostic only; not formal global32 qualification',
            'authorization':str(AUTH),'resource_admission':approval_pin,'checkpoint_reservation_bytes':RESERVE,
            'other_commitments_bytes':OTHER_COMMITMENTS,'separate_free_floor_bytes':FREE_FLOOR,
            'cpu_threads':2,'cpu_affinity':approval['cpu_affinity'],'rss_limit_bytes':RSS_LIMIT,'tensor_cache_write_bytes':0,'pressure_workers_killed':False,'real_jobs_stopped':False}
        OUT.mkdir(parents=True)
        for name,value in [('CONFIG.json',cfg),('TARGET_CONTRACT.json',cfg['space_time']['teacher']),
            ('ENVIRONMENT.json',envrec()),('RESOURCE_LEASE.json',lease),('STORAGE_CHECK.json',disk)]:
            write(OUT/name,value,immutable=True)
        write(OUT/'EXECUTION_CONFIG_LINEAGE.json',{'parent':inputs['pins']['config'],
            'child':{'path':str(OUT/'CONFIG.json'),'sha256':digest(OUT/'CONFIG.json')},
            'changed_field':'runtime.output_root','before':str(FACTORY/'configured/runs'),'after':str(OUT/'runs'),
            'teacher_and_mathematics_changed':False},immutable=True)
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


def monitor_helpers():
    # Execute only the exact reviewed bytes, without importing donor trainers or
    # changing an already loaded supervisor. This is the existing PPID/pidfd
    # ownership primitive, with an explicit 180s deadline supplied below.
    modules={}
    for name,(path,sha) in MONITOR_SOURCES.items():
        p=Path(path);require(p.stat().st_size<=128<<10,'Oversized monitor source')
        payload=p.read_bytes();require(hashlib.sha256(payload).hexdigest()==sha,'Monitor source changed')
        if name in sys.modules:
            existing=sys.modules[name]
            require(getattr(existing,'__file__',None)==str(p) and getattr(existing,'_source_sha',None)==sha,
                'Mixed pre-imported monitor dependency')
            module=existing
        else:
            module=ModuleType(name);module.__file__=str(p);module._source_sha=sha
            sys.modules[name]=module
            try:exec(compile(payload,str(p),'exec'),module.__dict__)
            except BaseException:del sys.modules[name];raise
        modules[name]=module
    require(digest(REPO/'tools/st_world_policy/metadata_storage.py')==METADATA_SHA,'RSS source changed')
    from tools.st_world_policy import metadata_storage
    require(Path(metadata_storage.__file__).resolve()==(REPO/'tools/st_world_policy/metadata_storage.py').resolve(),
        'Mixed pre-imported RSS dependency')
    process_tree_rss=metadata_storage.process_tree_rss
    facade=modules['full_profile_ancestry'].ancestry_supervisor(modules['full_profile_legacy_supervisor'])
    return facade,process_tree_rss


def proc_io():
    return {k:int(v) for k,v in (line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())}


def rank_telemetry(run):
    values={}
    for rank in range(2):
        p=run/('teacher_provider_rank'+str(rank)+'.json')
        if p.exists():
            require(p.stat().st_size<=64<<10,'Oversized original per-rank telemetry')
            values[str(rank)]=read(p)
    return values


def monitor_training(child, owner, stopped, rss_reader, run, phase, *, started, absolute_end):
    next_rss=next_storage=started; peak=0; reason=None; last_storage=None
    initial_io=proc_io(); monitor_error=None; cleanup_error=None; code=None; closed=False
    samples=OUT/('MONITOR_'+phase+'.jsonl')
    log=None
    def sample(active):
        # Also used inside the existing helper's cleanup discovery loop. This
        # observes only; it never changes membership, signals or cleanup budget.
        nonlocal next_rss,next_storage,peak,reason,last_storage,monitor_error
        now=time.monotonic()
        if stopped['signal'] is not None:reason=reason or 'SUPERVISOR_SIGNAL_'+str(stopped['signal'])
        if now<next_rss:return
        next_rss=now+RSS_INTERVAL
        try:
            peak=max(peak,rss_reader(os.getpid()))
            if peak>RSS_LIMIT:reason=reason or 'RSS_LIMIT'
            if now>=next_storage:
                next_storage=now+STORAGE_INTERVAL;last_storage=storage()
            row={'elapsed_seconds':now-started,'rss_peak_bytes':peak,'active_owned':len(active),
                'ranks':rank_telemetry(run),'storage':last_storage,'reason':reason}
            if log is not None and log.tell()<=16<<20:
                log.write(json.dumps(row,sort_keys=True)+'\n');log.flush()
            else:reason=reason or 'MONITOR_LOG_LIMIT'
        except Exception as exc:
            reason=reason or 'MONITOR_ERROR'
            monitor_error=monitor_error or {'class':type(exc).__name__,'message':str(exc)[:1024]}
    try:
        log=samples.open('x')
        while True:
            active=owner.discover();done=owner.leader_done()
            if done and active:active=owner.discover()  # close normal exit between the two snapshots
            now=time.monotonic();sample(active)
            if done and active:reason=reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
            if not active:break
            if now-started>=OUTER_SECONDS:reason=reason or 'WALL_LIMIT'
            if reason:break
            time.sleep(.2)
    except BaseException as exc:
        reason=reason or 'MONITOR_ERROR'
        monitor_error=monitor_error or {'class':type(exc).__name__,'message':str(exc)[:1024]}
    # Explicitly retain the original 180s cleanup, not the donor's default40s.
    # Wrap this one owner's read-only discovery return only while finishing.
    # First wait/reap remains in the reviewed ownership helper.
    discover=owner.discover
    def sampled_discover():
        active=discover();sample(active);return active
    owner.discover=sampled_discover
    try:
        code=owner.finish(terminate=reason is not None,deadline=min(absolute_end,time.monotonic()+CLEANUP_SECONDS))
        closed=True
    except BaseException as exc:
        reason=reason or 'OWNERSHIP_CLEANUP_FAILURE'
        cleanup_error={'class':type(exc).__name__,'message':str(exc)[:1024]}
    finally:
        owner.discover=discover;owner.close()
        if log is not None:log.close()
    try:
        final_storage=storage();telemetry=rank_telemetry(run)
    except Exception as exc:
        reason=reason or 'FINAL_MONITOR_ERROR';final_storage=last_storage;telemetry={}
        monitor_error=monitor_error or {'class':type(exc).__name__,'message':str(exc)[:1024]}
    final_io=proc_io()
    return code,{'reason':reason,'wall_seconds':time.monotonic()-started,'rss_peak_bytes':peak,
        'rss_limit_bytes':RSS_LIMIT,'rss_interval_seconds':RSS_INTERVAL,'storage_interval_seconds':STORAGE_INTERVAL,
        'storage':final_storage,'rank_telemetry':telemetry,'verified_owned_processes':owner.verified,
        'live_owned_remaining':0 if closed else None,'monitor_error':monitor_error,'cleanup_error':cleanup_error,
        'proc_supervisor_after_wait_delta':{k:final_io[k]-initial_io[k] for k in final_io},
        'io_scope':'supervisor self plus waited descendants after reap; rank telemetry is not added again; incomplete if cleanup fails',
        'sampling_boundary':'1s RSS including cleanup,10s output/free-space; scheduler/filesystem delay means trigger limits, not hard OS quotas'}


def validate_resume_state(previous):
    require(previous.get('status')=='PAUSED' and previous.get('source_sha')==SOURCE
        and type(previous.get('completed')) is int and 0<previous['completed']<64
        and previous.get('exposure')==2*previous['completed'],
        'Resume requires a native interior consumed boundary; completed64 cannot be extended')


def archive_rank1(run,phase):
    # The inherited archive already preserves rank0 and native RESULT/status.
    # Keep rank1's exact bytes as well before any native resume can replace it.
    source=run/'teacher_provider_rank1.json'
    if source.exists():
        require(source.stat().st_size<=64<<10,'Oversized rank1 telemetry')
        payload=source.read_bytes();target=OUT/'invocations'/phase/source.name
        if target.exists():require(target.read_bytes()==payload,'Existing rank1 evidence differs')
        else:
            with target.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())


def require_resume_slot(run):
    # Count partial slots too; COMPLETE is not a disk-allocation boundary.
    folder=run/'checkpoints';slots=[]
    require(folder.is_dir() and not folder.is_symlink(),'Native checkpoint directory missing/aliased')
    for count,p in enumerate(folder.iterdir(),1):
        require(count<=16 and not p.is_symlink(),'Unexpected native checkpoint entries')
        if p.is_dir():slots.append(p.name)
    require(len(slots)==1,'80GiB allows one existing native slot plus one final slot; no third slot')
    return slots


def execute(approval,approval_pin,resume=False):
    m=legacy();envrec,_,_,_,verify,write,_=m.modules()
    verify(OUT/'REGISTRATION.json',repo=REPO)
    prepared=read(OUT/'PREPARED.json');lease=read(OUT/'RESOURCE_LEASE.json')
    require(prepared['script_sha256']==digest(__file__) and prepared['inputs_sha256']==digest(INPUT)
        and prepared['inherited_helpers_sha256']==digest(OLD),'Controller inputs changed')
    require(prepared['registration_sha256']==digest(OUT/'REGISTRATION.json') and prepared['host']==HOST,
        'Registered identity changed')
    require(lease['resource_admission']==prepared['resource_admission'],'Registered initial admission changed')
    if resume:
        require(approval.get('initial_resource_admission')==prepared['resource_admission']
            and approval.get('registration_sha256')==prepared['registration_sha256']
            and approval.get('allowed_actions')==['resume'],'Resume needs a new explicit single-invocation admission')
    else:
        require(prepared['resource_admission']==approval_pin,'Different initial resource admission')
    require(read(OUT/'ENVIRONMENT.json')==envrec(),'Actual environment changed')
    require(lease['valid_from_unix']<=time.time() and time.time()+1800<=lease['valid_until_unix'],'Lease too short')
    require(set(lease['cpu_affinity']).issubset(os.sched_getaffinity(0)),'Admitted CPUs not available')
    os.sched_setaffinity(0,set(lease['cpu_affinity']))
    run=OUT/'runs'/RUN_ID
    phase='resume' if resume else 'initial'
    with ExitStack() as stack:
        stack.enter_context(m.lock(OUT/'DISPATCH.lock'))
        gpu_locks=[stack.enter_context(m.lock(path)) for path in GPU_LOCKS]
        gpu=gpu_record(True);disk=storage()
        require([row['uuid'] for row in gpu]==lease['gpu_uuids_by_host'][HOST],'Physical GPU0/1 UUID changed')
        require(not (OUT/('LAUNCH_'+phase+'.json')).exists(),'This explicit invocation already exists; no automatic retry')
        if resume:
            previous=read(run/'RESULT.json')
            require((OUT/'LAUNCH_initial.json').exists(),'Original launch evidence missing')
            validate_resume_state(previous);require_resume_slot(run)
            m.archive_run_metadata(OUT,run,'before_resume',write);archive_rank1(run,'before_resume')
        else:
            require(not run.exists(),'Duplicate/failed run preserved; no retry')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'OPENBLAS_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','TOKENIZERS_PARALLELISM':'false',
            'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
        mon,rss_reader=monitor_helpers()
        with mon.lifetime_scope() as stopped:
            invoked=time.monotonic()
            absolute_end=min(invoked+OUTER_SECONDS+CLEANUP_SECONDS,invoked+lease['valid_until_unix']-time.time())
            child=None;owner=None;leader_fd=None;entered_monitor=False
            try:
                with (OUT/('TRAIN_'+phase+'.log')).open('xb') as log:
                    child=subprocess.Popen(command(resume),cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(lock.fileno() for lock in gpu_locks))
                leader_fd=mon.pidfd_open(child.pid)
                owner=mon.OwnedSession(child)
                write(OUT/('LAUNCH_'+phase+'.json'),{'host':HOST,'pid':child.pid,'process_starttime':m.proc_start(child.pid),
                    'launcher_pid':os.getpid(),'launcher_starttime':m.proc_start(os.getpid()),'argv':command(resume),
                    'source_sha':SOURCE,'script_sha256':digest(__file__),'started_unix':time.time(),'scope':'PROFILE',
                    'cuda_visible_devices':GPU,'gpu_lock_paths':[str(p) for p in GPU_LOCKS],'storage':disk,'formal':False,'resume':resume,
                    'max_completed_updates':64,'max_total_scene_exposures':128,'sample_limit':101592,'resource_admission':approval_pin},immutable=True)
                print(json.dumps({'status':'LAUNCHED','pid':child.pid,'run_id':RUN_ID}),flush=True)
                entered_monitor=True
                code,monitor=monitor_training(child,owner,stopped,rss_reader,run,phase,started=invoked,absolute_end=absolute_end)
            except BaseException:
                if child is not None and not entered_monitor:
                    if owner is not None:
                        try:owner.finish(terminate=True,deadline=min(absolute_end,time.monotonic()+CLEANUP_SECONDS))
                        finally:owner.close()
                    else:mon.abort_spawn_setup(child,leader_fd,deadline=min(absolute_end,time.monotonic()+CLEANUP_SECONDS))
                raise
            finally:
                if leader_fd is not None:os.close(leader_fd)
            write(OUT/('MONITOR_RESULT_'+phase+'.json'),monitor,immutable=True)
        m.archive_run_metadata(OUT,run,phase,write);archive_rank1(run,phase)
        result=read(run/'RESULT.json') if (run/'RESULT.json').exists() else None
        write(OUT/('EXIT_'+phase+'.json'),{'returncode':code,'ended_unix':time.time(),'result':result,
            'status':'FAIL' if monitor['reason'] is not None or code!=0 else 'NATIVE_EXIT',
            'monitor':monitor},immutable=True)
        require(monitor['reason'] is None,'Monitored failure preserved even if native save/returncode succeeded')
        require(code==0,'Invocation failed; preserved evidence, no automatic retry')
        require(result is not None and result.get('completed')==64 and result.get('exposure')==128 and result.get('status')=='PAUSED',
            'Earlier boundary or incorrect terminal result; do not claim64 completed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('preflight','prepare','run','resume'),default='preflight')
    p.add_argument('--resource-admission');p.add_argument('--resource-admission-sha256')
    args=p.parse_args();inputs,cfg=common()
    if args.action=='preflight':
        print(json.dumps({'status':'METADATA_PREFLIGHT_ONLY','source':SOURCE,'scope':'PROFILE',
            'updates':64,'total_exposures':128,'world_size':2,'sample_limit':101592,'gpu_queried':False,'output_created':False}));return
    approved,approval_pin=admission(args.resource_admission,args.resource_admission_sha256,args.action)
    if args.action=='prepare':prepare(inputs,cfg,approved,approval_pin)
    else:execute(approved,approval_pin,resume=args.action=='resume')


if __name__=='__main__':
    main()
