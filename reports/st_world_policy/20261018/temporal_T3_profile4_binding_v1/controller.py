#!/usr/bin/env python3
"""Fixed T3 adapter of the existing PROFILE registrar/owned supervisor. No retries."""
import time
START=time.monotonic()
with open('/proc/self/io') as f:BOOT_IO={k:int(v) for k,v in (line.split(':') for line in f)}
import ast, hashlib, importlib.util, json, os
from pathlib import Path
from types import ModuleType, SimpleNamespace
HERE=Path(__file__).resolve().parent
BINDING_SHA='78495993e59efa4edf9395d6f13743414c227948b5e47398be040d2ad87753af'
if hashlib.sha256((HERE/'binding_factory.py').read_bytes()).hexdigest()!=BINDING_SHA:raise ValueError('Pinned binding factory changed')
spec=importlib.util.spec_from_file_location('T3_binding',HERE/'binding_factory.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
DONOR=Path('/mnt/project/VLA-Drive-st-temporal-combination-20261017/reports/st_world_policy/20261017/full_population_profile_launch_v1/controller.py')
DONOR_SHA='bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1'
IO_SOURCE=DONOR.parent.parent/'full_future_metadata_preparation_v1/run_build_once.py'
IO_SHA='4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2'
READ_CAP=256<<30
ASSIGNMENT=Path('/mnt/project/st-world-policy-20261016/resources/history32_actual_20261017T2345/CHECKPOINT_HOLD80_ASSIGNMENT.json')

def source(path,sha):
    if path.stat().st_size>64<<10:raise ValueError('Small pinned control source required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Pinned mature controller changed')
    return raw.decode()

def io_function():
    parsed=ast.parse(source(IO_SOURCE,IO_SHA))
    node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='owned_proc_io')
    ns={'os':os};exec(compile(ast.Module(body=[node],type_ignores=[]),str(IO_SOURCE),'exec'),ns)
    return ns['owned_proc_io']

class OwnedReadGuard:
    """A conservative leaf-to-root cut over witnessed identities, never arbitrary PIDs.

    Reads consumed/reaped descendants are inherited by a later ancestor read.
    Reaping during the cut can overcount (early stop), not grant extra budget.
    Final exact charged bytes still come from the original post-wait parent IO.
    """
    def __init__(self,owner,legacy,read_owned,baseline):
        self.owner=owner;self.legacy=legacy;self.read_owned=read_owned;self.baseline=baseline;self.high=0
    def sample(self):
        owner=self.owner;records={};parents={}
        key=lambda r:(r['pid'],r['start_ticks'])
        for edge in owner.evidence:
            records[key(edge['child'])]=edge['child'];parents[key(edge['child'])]=key(edge['parent'])
        records[key(owner.anchor)]=owner.anchor
        if len(records)>4096:raise ValueError('Original owned history bound exceeded')
        def depth(k):
            seen=set()
            while k in parents:
                if k in seen:raise ValueError('Invalid owned ancestry cycle')
                seen.add(k);k=parents[k]
            return len(seen)
        total=0
        for _,known in sorted(records.items(),key=lambda item:depth(item[0]),reverse=True):
            try:
                actual=self.legacy.proc_identity(known['pid'])
                if not owner.same(known,actual):continue  # Old identity reaped; never read recycled PID.
                proxy=SimpleNamespace(anchor=known,same=owner.same)
                total+=self.read_owned(proxy,self.legacy)['rchar']
            except (FileNotFoundError,ProcessLookupError):
                continue  # A later ancestor read inherits its waited IO.
        own={k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
        total+=own['rchar']-self.baseline['rchar']
        if total<0:raise ValueError('Nonmonotonic process IO')
        self.high=max(self.high,total);return self.high

def adapted_source(raw):
    changes=[
      ("'st_full_index_profile64_resource_admission_v1'","'st_T3_profile4_resource_admission_v1'",1),
      ("'max_invocation_seconds':1500","'max_invocation_seconds':1800",2),
      ("'world_size':2,\n            'cuda_visible_devices_by_host'","'world_size':1,\n            'cuda_visible_devices_by_host'",1),
      ("len(rows)==2","len(rows)==1",1),
      ("len({x['uuid'] for x in result})==2","len({x['uuid'] for x in result})==1",1),
      ("len(cpus) in (2,4)","len(cpus)==2",1),
      ("cp.get('remaining_bytes')==OTHER_COMMITMENTS","cp.get('remaining_bytes')==OTHER_COMMITMENTS+RESERVE",1),
      ("'before':str(FACTORY/'configured/runs')","'before':str(OUT/'runs')",1),
      ("'Full original population, cumulative64 updates/128 exposures, world2/global2 diagnostic only; not formal global32 qualification'","'T3 exact32, cumulative4 updates/8 exposures, world1/global2 diagnostic only'",1),
      ("time.time()+1800<=lease['valid_until_unix']","time.time()+OUTER_SECONDS+CLEANUP_SECONDS<=lease['valid_until_unix']",1),
      ("'max_completed_updates':64,'max_total_scene_exposures':128,'sample_limit':101592","'max_completed_updates':4,'max_total_scene_exposures':8,'sample_limit':32",1),
      ("result.get('completed')==64 and result.get('exposure')==128","result.get('completed')==4 and result.get('exposure')==8",1),
      ("choices=('preflight','prepare','run','resume')","choices=('preflight','prepare','run')",1),
      ("'updates':64,'total_exposures':128,'world_size':2,'sample_limit':101592","'updates':4,'total_exposures':8,'world_size':1,'sample_limit':32",1),
      ("initial_io=proc_io(); monitor_error=None", "initial_io=BOOT_IO; read_guard=OwnedReadGuard(owner,IO_LEGACY,READ_OWNED,BOOT_IO); monitor_error=None",1),
      ("            peak=max(peak,rss_reader(os.getpid()))","            read_high=read_guard.sample()\n            if read_high>READ_CAP:reason=reason or 'LOGICAL_READ_LIMIT'\n            peak=max(peak,rss_reader(os.getpid()))",1),
      ("    final_io=proc_io()\n    return code,", "    final_io=proc_io()\n    if final_io['rchar']-initial_io['rchar']>READ_CAP:reason=reason or 'LOGICAL_READ_LIMIT'\n    return code,",1),
      ("'rss_limit_bytes':RSS_LIMIT,'rss_interval_seconds':RSS_INTERVAL,'storage_interval_seconds':STORAGE_INTERVAL,\n        'storage':final_storage", "'rss_limit_bytes':RSS_LIMIT,'rss_interval_seconds':RSS_INTERVAL,'storage_interval_seconds':STORAGE_INTERVAL,\n        'sampled_conservative_read_stop_bytes':READ_CAP,'read_conservative_high_water_bytes':read_guard.high,\n        'storage':final_storage",1),
      ("for rank in range(2):","for rank in range(1):",1),
      ("'cpu_threads':2,'cpu_affinity':approval['cpu_affinity'],'rss_limit_bytes':RSS_LIMIT",
       "'shared_gpu':True,'protected_external_processes':approval['protected_external_processes'],'exclusive_performance_claim':False,\n            'cpu_threads':2,'cpu_affinity':approval['cpu_affinity'],'rss_limit_bytes':RSS_LIMIT",1),
    ]
    for old,new,n in changes:
        if raw.count(old)!=n:raise ValueError('Original PROFILE template changed: '+old)
        raw=raw.replace(old,new)
    return raw

def controller():
    c=ModuleType('original_PROFILE_T3');c.__file__=str(HERE/'controller.py')
    exec(compile(adapted_source(source(DONOR,DONOR_SHA)),str(DONOR),'exec'),c.__dict__)
    c.__dict__.update(REPO=b.RUNTIME,SOURCE=b.SOURCE,OUT=b.OUT,FACTORY=b.PREP,RUN_ID=b.RUN_ID,
        HOST=b.HOST,GPU='0',GPU_LOCKS=(Path('/tmp/foresight_gpu_0.lock'),),INPUT=HERE/'INPUTS.json',
        OTHER_COMMITMENTS=2623<<30,OUTER_SECONDS=1920,CLEANUP_SECONDS=180,
        METADATA_SHA='aaa36f5383451eadcd92c160c6f3ffb8704110ab44eb12ee433b3207995e71ba',
        BOOT_IO=BOOT_IO,READ_CAP=READ_CAP,OwnedReadGuard=OwnedReadGuard,READ_OWNED=io_function())
    c.validate_recipe=b.validate_recipe
    c.registered_config=lambda cfg:c.deepcopy(cfg)
    c.command=lambda resume=False:(['timeout','--signal=TERM','--kill-after=180s','1920s',*b.native_argv()]
        if not resume else (_ for _ in ()).throw(ValueError('No resume authorized for this fresh four-step profile')))
    def gpu_record(require_capacity):
        rows=c.subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        c.require(len(rows)==1,'One actual shared GPU0 required')
        uuid,total,used,free,util=[x.strip() for x in rows[0].split(',')]
        c.require(uuid==b.GPU_UUID,'Exact admitted physical GPU identity required')
        if require_capacity:c.require(int(free)>=68*1024,'Shared GPU0 needs at least68GiB actual free memory')
        processes=protected_gpu_processes(c)
        c.require(processes==getattr(c,'EXPECTED_PROTECTED_PROCESSES',None),'Fresh shared GPU processes differ from actual admission')
        return [dict(uuid=uuid,memory_mib=int(used),free_mib=int(free),total_mib=int(total),utilization_percent=int(util),
            shared=True,exclusive_performance_claim=False,protected_external_processes=processes)]
    c.gpu_record=gpu_record
    original_helpers=c.monitor_helpers
    def monitor_helpers():
        c.sys.path.insert(0,str(b.RUNTIME))
        m,rss=original_helpers();c.IO_LEGACY=m;return m,rss
    c.monitor_helpers=monitor_helpers
    def common():
        c.require(c.socket.gethostname()==b.HOST,'Actual tsinghua-zt2 host required')
        policy=c.read(c.POLICY);key='training-vlawm-tsinghua-zt2'
        c.require(b.HOST in policy['allowed_hosts'][key]['aliases'] and b.HOST in policy['allowed_aliases'],'Host no longer authorized')
        for k,v in policy.get('global_dispatch_prohibitions',{}).items():
            if k in (b.HOST,key) or b.HOST in v.get('aliases',[]):c.require(v.get('new_task_dispatch_allowed') is True,'Host withdrawn')
        c.require(key in c.read(c.AUTH)['allowed_hosts'],'Campaign host authorization missing')
        inputs=c.read(c.INPUT);actual=b.actual_inputs()
        recorded=dict(inputs);recorded.pop('native_argv_for_existing_supervisor',None)
        c.require(recorded==actual,'Exact actual factory binding changed')
        cfg=c.read(inputs['pins']['config']['path']);b.validate_recipe(cfg)
        return inputs,cfg
    c.common=common
    original_admission=c.admission
    def admission(path,sha,action):
        a,pin=original_admission(path,sha,action)
        c.require(a.get('logical_read_limit_bytes')==READ_CAP and a.get('gpu_uuid')==b.GPU_UUID,'Read/physical GPU bounds missing')
        c.require(a.get('shared_gpu') is True and a.get('minimum_free_gpu_bytes')==68<<30
            and a.get('exclusive_performance_claim') is False and isinstance(a.get('protected_external_processes'),list),
            'Explicit shared placement and protected external identities required')
        for r in a['protected_external_processes']:
            c.require(set(r)=={'pid','start_ticks','cmdline_sha256'} and type(r['pid']) is int and r['pid']>0
                and isinstance(r['start_ticks'],str) and r['start_ticks'].isdigit()
                and isinstance(r['cmdline_sha256'],str) and len(r['cmdline_sha256'])==64,'Real external process identity required')
        c.EXPECTED_PROTECTED_PROCESSES=a['protected_external_processes']
        ap=a.get('checkpoint_hold_assignment',{})
        c.require(ap.get('path')==str(ASSIGNMENT) and c.digest(ap['path'])==ap.get('sha256'),'Unique existing80GiB hold assignment required')
        v=c.read(ap['path']);cp=c.read(a['checkpoint_commitments']['path'])
        required=dict(parent_checkpoint_commitments=a['checkpoint_commitments'],source_slot='additional_conservative_unissued_GiB',assigned_run_id=b.RUN_ID,
            output_root=str(b.OUT),inputs_sha256=c.digest(c.INPUT),config=inputs_config_pin(c),runtime={'path':str(b.RUNTIME),'source_sha':b.SOURCE},
            assigned_bytes=c.RESERVE,aggregate_bytes=c.OTHER_COMMITMENTS+c.RESERVE,other_bytes=c.OTHER_COMMITMENTS,
            additional_bytes=0,no_reassignment=True,no_parallel_reuse=True)
        c.require(all(v.get(k)==val for k,val in required.items()),'80GiB slot assignment differs')
        c.require(v.get('valid_until_unix',0)>=a['valid_until_unix'] and v['aggregate_bytes']==cp['remaining_bytes'],'Assignment expired or aggregate changed')
        return a,pin
    c.admission=admission
    return c

def protected_gpu_processes(c):
    rows=c.subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(rows)>128:raise ValueError('Bounded shared GPU process census exceeded')
    result=[]
    for row in rows:
        uuid,pid=[x.strip() for x in row.split(',')]
        if uuid!=b.GPU_UUID:continue
        pid=int(pid);p=Path('/proc')/str(pid)
        before=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
        raw=(p/'cmdline').read_bytes()
        if len(raw)>64<<10:raise ValueError('Bounded external command identity exceeded')
        after=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
        if before!=after:raise ValueError('External PID changed during placement snapshot')
        result.append(dict(pid=pid,start_ticks=before,cmdline_sha256=hashlib.sha256(raw).hexdigest()))
    return sorted(result,key=lambda r:r['pid'])

def inputs_config_pin(c):return c.read(c.INPUT)['pins']['config']
if __name__=='__main__':controller().main()
