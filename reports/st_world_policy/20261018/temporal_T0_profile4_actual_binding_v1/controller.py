#!/usr/bin/env python3
"""Lead-only fixed T0 world4 PROFILE binding of the existing one-shot supervisor."""
import time
START=time.monotonic()
with open('/proc/self/io') as f:
    BOOT_IO={k:int(v) for k,v in (line.split(':') for line in f)}
import ast
import hashlib
import math
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

HERE=Path(__file__).resolve().parent
RUNTIME=Path('/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018')
SOURCE='38f86a15a92171e583bb5019cce89a63fb7a1d43'
ROOT=Path('/mnt/project/st-world-policy-20261016')
BASE=ROOT/'profile/temporal_global32_qualification_v1/T0_capacity4'
INPUT=BASE/'INPUTS.json'
INPUT_SHA='940f30904a84277dff8e27b4813100758bd114f27600676ce4cc3b1d2ef79d51'
FACTORY=BASE/'metadata/controls/RESULT.json'
FACTORY_SHA='ac9c921c1608064ead2507290fa75d89d685fba445ec13252994773b233fa430'
OUT=BASE/'training_control'
RUN_ID='T0_fulltrain_seed42_profile4_world4_micro1_global32_v1'
HOST='training-vlawm-tsinghua-zt4-worker-0'
HOST_KEY='training-vlawm-tsinghua-zt4'
GPU='0,1,2,3'
DONOR=RUNTIME/'reports/st_world_policy/20261017/full_population_profile_launch_v1/controller.py'
DONOR_SHA='bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1'
GUARD_SOURCE=RUNTIME/'reports/st_world_policy/20261018/temporal_T3_profile4_binding_v1/controller.py'
GUARD_SHA='1993f4b7afb601e375efe701f9bf25d1a209b4943bbfea8911b7f97abfa1443d'
IO_SOURCE=DONOR.parent.parent/'full_future_metadata_preparation_v1/run_build_once.py'
IO_SHA='4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2'
READ_CAP=1<<40
CACHE_RESERVE=768<<30
METADATA_RESERVE=10<<30
EXECUTION=dict(scope='PROFILE',family='temporal',world_size=4,micro_batch=1,global_batch=32,
    max_updates=20000,stop_after=4,sample_limit=101592,seed=42,
    precision_policy='S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off')


def source(path,sha):
    if path.stat().st_size>64<<10:raise ValueError('Only bounded pinned control source')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Inherited control source changed')
    return raw.decode()


def inherited_guard():
    ns={'Path':Path,'SimpleNamespace':SimpleNamespace,'os':os}
    for path,sha,name in [(GUARD_SOURCE,GUARD_SHA,'OwnedReadGuard'),(IO_SOURCE,IO_SHA,'owned_proc_io')]:
        tree=ast.parse(source(path,sha))
        node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['OwnedReadGuard'],ns['owned_proc_io']


def adapted_source(raw):
    changes=[
      ("'world_size':2,\n            'cuda_visible_devices_by_host'","'world_size':4,\n            'cuda_visible_devices_by_host'",1),
      ("'max_invocation_seconds':1500","'max_invocation_seconds':3600",2),
      ("'Full original population, cumulative64 updates/128 exposures, world2/global2 diagnostic only; not formal global32 qualification'","'T0 actual full TRAIN, cumulative4 updates/128 exposures, world4/micro1/global32 capacity only; no formal permission'",1),
      ("'before':str(FACTORY/'configured/runs')","'before':str(OUT/'runs')",1),
      ("time.time()+1800<=lease['valid_until_unix']","time.time()+OUTER_SECONDS+CLEANUP_SECONDS<=lease['valid_until_unix']",1),
      ("'max_completed_updates':64,'max_total_scene_exposures':128,'sample_limit':101592","'max_completed_updates':4,'max_total_scene_exposures':128,'sample_limit':101592",1),
      ("result.get('completed')==64 and result.get('exposure')==128","result.get('completed')==4 and result.get('exposure')==128",1),
      ("choices=('preflight','prepare','run','resume')","choices=('preflight','prepare','run')",1),
      ("'updates':64,'total_exposures':128,'world_size':2,'sample_limit':101592","'updates':4,'total_exposures':128,'world_size':4,'sample_limit':101592",1),
      ("for rank in range(2):","for rank in range(4):",1),
      ("Insufficient space after 2303GiB existing commitments + 1TiB floor + own remaining80GiB","Insufficient space after actual checkpoint/cache/metadata obligations + 1TiB floor + own remaining80GiB",1),
      ("Physical GPU0/1 UUID changed","Four admitted physical GPU UUIDs changed",1),
      ("Earlier boundary or incorrect terminal result; do not claim64 completed","Earlier boundary or incorrect terminal result; do not claim4 completed",1),
      ("'implementer_agents':['/root','/root/st_indexed_composition','/root/st_bounded_future_provider']","'implementer_agents':['/root','/root/st_indexprep_boundary_fix','/root/temporal_profile_prepare']",1),
      ("'other_commitments_bytes':OTHER_COMMITMENTS,'separate_free_floor_bytes':FREE_FLOOR,","'other_commitments_bytes':OTHER_COMMITMENTS,'separate_free_floor_bytes':FREE_FLOOR,\n            'retained_cache_reservation_bytes':CACHE_RESERVE,'separate_metadata_buffer_bytes':METADATA_RESERVE,'logical_read_limit_bytes':READ_CAP,",1),
      ("initial_io=proc_io(); monitor_error=None","initial_io=BOOT_IO; read_guard=OwnedReadGuard(owner,IO_LEGACY,READ_OWNED,BOOT_IO); monitor_error=None",1),
      ("            peak=max(peak,rss_reader(os.getpid()))","            read_high=read_guard.sample()\n            if read_high>READ_CAP:reason=reason or 'LOGICAL_READ_LIMIT'\n            peak=max(peak,rss_reader(os.getpid()))",1),
      ("    final_io=proc_io()\n    return code,","    final_io=proc_io()\n    if final_io['rchar']-initial_io['rchar']>READ_CAP:reason=reason or 'LOGICAL_READ_LIMIT'\n    return code,",1),
      ("'rss_limit_bytes':RSS_LIMIT,'rss_interval_seconds':RSS_INTERVAL,'storage_interval_seconds':STORAGE_INTERVAL,\n        'storage':final_storage","'rss_limit_bytes':RSS_LIMIT,'rss_interval_seconds':RSS_INTERVAL,'storage_interval_seconds':STORAGE_INTERVAL,\n        'sampled_conservative_read_stop_bytes':READ_CAP,'read_conservative_high_water_bytes':read_guard.high,\n        'storage':final_storage",1),
    ]
    for old,new,count in changes:
        if raw.count(old)!=count:raise ValueError('Mature supervisor changed: '+old)
        raw=raw.replace(old,new)
    return raw


def controller():
    c=ModuleType('original_PROFILE_T0');c.__file__=str(HERE/'controller.py')
    exec(compile(adapted_source(source(DONOR,DONOR_SHA)),str(DONOR),'exec'),c.__dict__)
    guard,read_owned=inherited_guard()
    c.__dict__.update(REPO=RUNTIME,SOURCE=SOURCE,OUT=OUT,RUN_ID=RUN_ID,HOST=HOST,GPU=GPU,
        GPU_LOCKS=tuple(Path('/tmp/foresight_gpu_'+i+'.lock') for i in GPU.split(',')),INPUT=INPUT,
        RESERVE=80<<30,OTHER_COMMITMENTS=None,FREE_FLOOR=1<<40,OUTER_SECONDS=3720,CLEANUP_SECONDS=180,
        RSS_LIMIT=192<<30,READ_CAP=READ_CAP,CACHE_RESERVE=CACHE_RESERVE,METADATA_RESERVE=METADATA_RESERVE,
        BOOT_IO=BOOT_IO,OwnedReadGuard=guard,READ_OWNED=read_owned,
        METADATA_SHA='65644ac00b6a395bf2e1eb498e9f221a3cdebee63a027857513b1f50118151a6')
    c.registered_config=lambda cfg:c.deepcopy(cfg)
    c.command=lambda resume=False: (['timeout','--signal=TERM','--kill-after=180s','3720s',c.PYTHON,
        '-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4','-m',
        'tools.st_world_policy.train_world','--registration',str(OUT/'REGISTRATION.json')]
        if not resume else (_ for _ in ()).throw(ValueError('Fresh T0 capacity only; no resume admitted')))
    def validate_recipe(cfg):
        rt=cfg['runtime'];st=cfg['space_time']
        c.require(cfg['execution']==EXECUTION and cfg['optimizer']['warmup']==1000,'Wrong T0/global32/20k recipe')
        c.require(rt['max_seconds']==3600 and rt['save_every']==4 and rt['loader_workers']==0
            and rt['deterministic'] is True and rt['gradient_checkpointing'] is False
            and rt['profile_mode']=='none' and rt['output_root']==str(OUT/'runs'),'Changed capacity runtime')
        c.require(st['cell']=='T0' and not st['history'] and not st['future'] and not st['delta'],'Changed scientific cell')
    c.validate_recipe=validate_recipe
    def common():
        c.require(c.socket.gethostname()==HOST,'Wrong target host')
        policy=c.read(c.POLICY)
        c.require(HOST in policy['allowed_hosts'][HOST_KEY]['aliases'] and HOST in policy['allowed_aliases'],'Host not allowed')
        for key,value in policy.get('global_dispatch_prohibitions',{}).items():
            if key in (HOST,HOST_KEY) or HOST in value.get('aliases',[]):c.require(value.get('new_task_dispatch_allowed') is True,'Host withdrawn')
        c.require(HOST_KEY in c.read(c.AUTH)['allowed_hosts'],'Campaign authorization missing')
        c.require(c.subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip()==SOURCE
            and not c.subprocess.check_output(['git','status','--porcelain=v1'],cwd=RUNTIME),'Frozen runtime changed')
        c.require(c.digest(INPUT)==INPUT_SHA and c.digest(FACTORY)==FACTORY_SHA,'Actual CPU preparation changed')
        factory=c.read(FACTORY);r=factory['resources']
        c.require(factory['status']=='PREPARATION_COMPLETE' and r['returncode']==0 and r['failure'] is None
            and r['cleanup']['live_owned_remaining']==0 and factory['training_started'] is False,'CPU preparation incomplete')
        inputs=c.read(INPUT)
        c.require(inputs['execution']==EXECUTION and inputs['source_sha']==SOURCE and inputs['run_id']==RUN_ID
            and inputs['output_root']==str(OUT) and inputs['weights_usable_for_formal_init'] is False,'Input identity changed')
        for pin in inputs['pins'].values():c.require(c.digest(pin['path'])==pin['sha256'],'Prepared artifact changed')
        cfg=c.read(inputs['pins']['config']['path']);validate_recipe(cfg)
        return inputs,cfg
    c.common=common
    def admission(path,sha,action):
        c.require(path and sha and c.digest(path)==sha,'Actual independent resource admission required')
        a=c.read(path)
        expected=dict(schema='st_T0_global32_profile4_resource_admission_v1',status='ADMITTED',host=HOST,gpu=GPU,
            source_sha=SOURCE,run_id=RUN_ID,output_root=str(OUT),controller_sha256=c.digest(HERE/'controller.py'),
            inputs_sha256=INPUT_SHA,checkpoint_reservation_bytes=80<<30,free_floor_bytes=1<<40,
            retained_cache_reservation_bytes=CACHE_RESERVE,separate_metadata_buffer_bytes=METADATA_RESERVE,
            max_invocation_seconds=3600,outer_timeout_seconds=3720,cleanup_seconds=180,
            rss_limit_bytes=192<<30,rss_interval_seconds=1.,storage_interval_seconds=10.,
            logical_read_limit_bytes=READ_CAP,cpu_threads_per_rank=2,exclusive_devices=True,exclusive_performance_claim=False,
            automatic_retry=False,automatic_resume=False,wrapper_outer_seconds=3900,wrapper_kill_grace_seconds=180,
            launcher_sha256=c.digest(HERE/'launch_once.sh'))
        c.require(all(a.get(k)==v and type(a.get(k)) is type(v) for k,v in expected.items()),'Admission differs from exact four-step capacity')
        c.require(isinstance(a.get('reviewer'),str) and a.get('reviewer') not in (None,'','/root','/root/temporal_profile_prepare'), 'Independent reviewer required')
        c.require(action in a.get('allowed_actions',[]) and set(a['allowed_actions'])<= {'prepare','run'},'Action not admitted')
        cpus=a.get('cpu_affinity');uuids=a.get('gpu_uuids')
        c.require(isinstance(cpus,list) and len(cpus)==8 and len(set(cpus))==8
            and all(type(x) is int and x>=0 for x in cpus) and set(cpus)<=os.sched_getaffinity(0),'Eight actual CPUs required')
        c.require(isinstance(uuids,list) and len(uuids)==4 and len(set(uuids))==4
            and all(isinstance(x,str) and x.startswith('GPU-') for x in uuids),'Four actual GPU UUIDs required')
        cp=a.get('checkpoint_commitments',{})
        c.require(set(cp)=={'path','sha256'} and c.digest(cp['path'])==cp['sha256'],'Actual remaining commitments required')
        commitments=c.read(cp['path']);remaining=commitments.get('remaining_bytes')
        c.require(type(remaining) is int and remaining>=0 and a.get('other_commitments_bytes')==remaining
            and commitments.get('free_floor_bytes')==1<<40,'Checkpoint remaining/floor mismatch')
        now=time.time()
        for value in (a.get('valid_from_unix'),a.get('valid_until_unix'),commitments.get('valid_until_unix')):
            c.require(type(value) in (int,float) and math.isfinite(value),'Finite resource validity required')
        c.require(a['valid_from_unix']<=now and now+4080<=min(a['valid_until_unix'],commitments['valid_until_unix']),
            'Lease too short for complete invocation and cleanup')
        c.OTHER_COMMITMENTS=remaining;c.EXPECTED_UUIDS=uuids
        return a,dict(path=str(Path(path).resolve()),sha256=sha)
    c.admission=admission
    def storage_requirement(used):
        c.require(type(used) is int and 0<=used<=c.RESERVE,'Own inclusive80GiB output bound exceeded')
        c.require(type(c.OTHER_COMMITMENTS) is int,'Storage requires actual independent admission first')
        return c.FREE_FLOOR+c.OTHER_COMMITMENTS+CACHE_RESERVE+METADATA_RESERVE+c.RESERVE-used
    c.storage_requirement=storage_requirement
    original_storage=c.storage
    def storage():
        value=original_storage()
        value.update(retained_cache_reservation_bytes=CACHE_RESERVE,separate_metadata_buffer_bytes=METADATA_RESERVE)
        return value
    c.storage=storage
    def gpu_record(idle):
        rows=c.subprocess.check_output(['nvidia-smi','--id='+GPU,
            '--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).strip().splitlines()
        c.require(len(rows)==4,'Exactly four physical GPUs required');records=[]
        for index,row in zip(GPU.split(','),rows):
            uuid,total,used,free,util=[x.strip() for x in row.split(',')]
            if idle:c.require(int(used)<64 and int(free)>=76*1024 and int(util)<=5,'Device busy; never stop other work')
            records.append(dict(id=index,uuid=uuid,total_mib=int(total),memory_mib=int(used),free_mib=int(free),utilization_percent=int(util)))
        c.require([r['uuid'] for r in records]==getattr(c,'EXPECTED_UUIDS',None),'Admitted physical GPUs changed')
        return records
    c.gpu_record=gpu_record
    original_helpers=c.monitor_helpers
    def monitor_helpers():
        c.sys.path.insert(0,str(RUNTIME));module,rss=original_helpers();c.IO_LEGACY=module;return module,rss
    c.monitor_helpers=monitor_helpers
    def archive_ranks(run,phase):
        for rank in range(4):
            source_path=run/('teacher_provider_rank'+str(rank)+'.json')
            if source_path.exists():
                c.require(source_path.stat().st_size<=64<<10,'Oversized rank telemetry')
                raw=source_path.read_bytes();target=OUT/'invocations'/phase/source_path.name
                if target.exists():c.require(target.read_bytes()==raw,'Rank evidence changed')
                else:
                    with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    c.archive_rank1=archive_ranks
    return c


if __name__=='__main__':controller().main()
