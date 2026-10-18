#!/usr/bin/env python3
"""Fixed diagnostic71011454 under the unchanged4b2f one-shot monitor; Lead only."""
import time
START=time.monotonic()
with open('/proc/self/io') as _io:BOOT_IO={k:int(v) for k,v in (line.split(':') for line in _io)}
import argparse
from contextlib import ExitStack,redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import socket
import stat
import subprocess
import sys
from types import ModuleType

HERE=Path(__file__).absolute().parent
ROOT=Path('/mnt/project/st-world-policy-20261016/audits/teacher_channel_prediction_diagnostic_v1')
CONTROL=ROOT.with_name(ROOT.name+'_controls')
RUNTIME=Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
PARENT_RUNTIME=Path('/mnt/project/VLA-Drive-st-temporal-packed-migration-fix-20261018')
HELPER_ROOT=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017/reports/st_world_policy/20261017')
BOOTSTRAP=Path('/mnt/project/VLA-Drive-st-full-population-provider-preparation-20261017/reports/st_world_policy/20261017/future_index_sequential_control_v1/prepare_once.py')
BOOTSTRAP_SHA='49b21405d1dacee17405598c1a7165727db9855af4de66c2feff70ab30b93ce5'
DIAGNOSTIC_SHA='3a7d5a45cbe5cddf64880f5e230228a6d6e6c3e23b5a51363698440ad77fb918'
BINDING_SHA='5dbc27344a531ea3a8c4eeb25bb17e4c349e209d29db856e2804bd95e1f70710'
LIMITS=dict(wall_seconds=1800,cleanup_seconds=40,read_bytes=320<<30,rss_bytes=96<<30)
OUTPUT_CAP=15<<20
CONTROL_CAP=1<<20
GPU_FREE=68<<30
PYTHON='/root/miniconda3/envs/ddp/bin/python'


def proc_io():
    with open('/proc/self/io') as f:return {k:int(v) for k,v in (line.split(':') for line in f)}


def bootstrap(deadline):
    # GNU timeout bounds Python bootstrap; this alarm covers the first source read.
    if signal.getitimer(signal.ITIMER_REAL)!=(0.,0.):raise ValueError('Existing alarm must not be replaced')
    previous=signal.getsignal(signal.SIGALRM)
    def expire(*_):raise TimeoutError('Whole diagnostic wall deadline')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        fd=os.open(BOOTSTRAP,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as f:
            info=os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>64<<10:raise ValueError('Small regular bootstrap required')
            raw=f.read((64<<10)+1)
        if len(raw)>64<<10 or hashlib.sha256(raw).hexdigest()!=BOOTSTRAP_SHA:raise ValueError('Bootstrap pin differs')
        b=ModuleType('diagnostic_bootstrap');b.__file__=str(BOOTSTRAP)
        exec(compile(raw,str(BOOTSTRAP),'exec'),b.__dict__)
        finite=b.load(b.ADAPTER,b.ADAPTER_SHA,'diagnostic_finite_controls')
        return b,finite
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def validate(a,now,affinity,hostname,child=False):
    cpus=a.get('cpu_affinity')
    if (a.get('schema')!='st_teacher_channel_diagnostic_admission_v1' or a.get('status')!='APPROVED_DIAGNOSTIC_ONLY'
        or a.get('reviewer') in (None,'','/root/st_indexprep_boundary_fix') or a.get('hostname')!=hostname
        or a.get('limits')!=LIMITS or a.get('output_root')!=str(ROOT) or a.get('control_root')!=str(CONTROL)
        or a.get('output_bytes')!=OUTPUT_CAP or a.get('control_bytes')!=CONTROL_CAP
        or a.get('diagnostic')!={'path':str(HERE/'diagnose.py'),'sha256':DIAGNOSTIC_SHA}
        or a.get('binding')!={'path':str(HERE/'BINDING.json'),'sha256':BINDING_SHA}
        or a.get('automatic_retry') is not False or a.get('training') is not False
        or a.get('new_checkpoint_reservation') is not False or a.get('ledger_writes') is not False
        or a.get('minimum_gpu_free_bytes')!=GPU_FREE):raise ValueError('Fixed independently reviewed diagnostic admission required')
    if (not isinstance(cpus,list) or len(cpus)!=2 or len(set(cpus))!=2
        or any(type(x) is not int or x<0 for x in cpus) or not set(cpus)<=set(affinity)):
        raise ValueError('Exactly two available CPUs required')
    if type(a.get('gpu_index')) is not int or a['gpu_index']<0 or not str(a.get('gpu_uuid','')).startswith('GPU-'):
        raise ValueError('Caller must bind one actual GPU UUID/index')
    for key in ('issued_at_unix','latest_start_unix','valid_until_unix'):
        if type(a.get(key)) not in (int,float) or not math.isfinite(a[key]):raise ValueError('Finite fresh lease required')
    if not a['issued_at_unix']<=now<=a['latest_start_unix'] or a['valid_until_unix']<now+(0 if child else 1840):
        raise ValueError('Expired or insufficient lease')
    expected={str(HERE/'control.py'),str(HERE/'launch_once.sh')}
    pins=a.get('controller_sources',[])
    if len(pins)!=2 or {p.get('path') for p in pins}!=expected:raise ValueError('Exact controller/bootstrap pins required')
    for p in pins:
        if set(p)!={'path','sha256'} or len(str(p['sha256']))!=64:raise ValueError('Non-null source pins required')


def fresh_capacity(a):
    fields=subprocess.check_output(['nvidia-smi','--id='+str(a['gpu_index']),
        '--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    if len(fields)!=4:raise ValueError('Expected exactly one physical GPU')
    uuid,name,total,free=(s.strip() for s in fields)
    if uuid!=a['gpu_uuid'] or 'A800' not in name or int(total)*(1<<20)<80<<30 or int(free)*(1<<20)<GPU_FREE:
        raise ValueError('Fresh admitted A80080GiB UUID/free capacity unavailable')
    with open('/proc/meminfo') as f:memory={s.split(':')[0]:int(s.split()[1])*1024 for s in f if ':' in s}
    if memory['MemAvailable']<LIMITS['rss_bytes']:raise ValueError('96GiB currently available host RAM required')
    return dict(uuid=uuid,name=name,total_bytes=int(total)<<20,free_bytes=int(free)<<20,
        host_mem_available_bytes=memory['MemAvailable'],observed_at_unix=time.time(),exclusive_performance_claim=False)


def original_cli(path):
    prior=sys.argv
    try:
        sys.argv=[str(path),'--execute']
        try:
            with redirect_stdout(sys.stderr):runpy.run_path(str(path),run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None,0):raise RuntimeError('Original diagnostic failed: '+str(exc.code)) from exc
    finally:sys.argv=prior


def child_envelope(action,baseline):
    result=None;error=None
    try:result=action()
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)[:2048]}
    final=proc_io()
    return dict(schema='st_full_future_child_result_v2',cli_result=result,error=error,
        counters=dict(proc_self_io_baseline=baseline,proc_self_io_final=final,
            proc_self_io_delta={k:final[k]-baseline[k] for k in final}))


def census(metadata,reserve=0):
    # Same-mount checks also cover the roots themselves, not just descendants.
    for root,cap,extra in ((ROOT,OUTPUT_CAP,0),(CONTROL,CONTROL_CAP,reserve)):
        if root.exists() and root.lstat().st_dev!=root.parent.stat().st_dev:raise ValueError('Root mount escape')
        if metadata.tree_inventory(root,max_depth=0,max_entries=32)['charged_bytes']+extra>cap:
            raise ValueError('Diagnostic/control output allocation exceeded')


def compact_resources(observed):
    result={k:v for k,v in observed.items() if k not in ('stdout','stderr')}
    for key in ('stdout','stderr'):
        raw=observed.get(key,'').encode()
        result[key]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'tail':raw[-4096:].decode('utf8','replace')}
    return result


def parent_helpers(b):
    helper=HELPER_ROOT/'spatial_probe_actual_preparation_v1'
    for name,sha in {'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
        'supervise_once.py':'d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
        'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'}.items():b.read_source(helper/name,sha)
    for name,sha in {'metadata_storage.py':'aaa36f5383451eadcd92c160c6f3ffb8704110ab44eb12ee433b3207995e71ba',
        'cache_storage_budget.py':'2103a082271706513127404b35fb1ff9154309b5e5fbe56306b8e831a509b1d5'}.items():
        b.read_source(PARENT_RUNTIME/'tools/st_world_policy'/name,sha)
    sys.path.insert(0,str(helper));sys.path.insert(0,str(PARENT_RUNTIME))
    import supervise_once as s
    import observed_cli as obs
    from tools.st_world_policy import metadata_storage as metadata
    if Path(metadata.__file__).resolve()!=PARENT_RUNTIME/'tools/st_world_policy/metadata_storage.py':raise ValueError('Mixed parent control runtime')
    return s,obs,metadata


def run(args,b,finite,deadline):
    a=json.loads(b.read_source(Path(args.admission),args.admission_sha256))
    validate(a,time.time(),os.sched_getaffinity(0),socket.gethostname(),args.child)
    for pin in [a['diagnostic'],a['binding'],*a['controller_sources']]:b.read_source(Path(pin['path']),pin['sha256'])
    if args.child:
        os.fstat(int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']))
        if os.getpid()!=os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a['gpu_index']):raise ValueError('Owned single-GPU child required')
        if ROOT.exists():raise ValueError('Child output must be claimed by original diagnostic')
        def action():
            original_cli(HERE/'diagnose.py')
            path=ROOT/'RESULT.json'
            with path.open('rb') as f:raw=f.read(OUTPUT_CAP+1)
            if len(raw)>OUTPUT_CAP or json.loads(raw).get('status')!='COMPLETE_DIAGNOSTIC_ONLY':raise ValueError('Native diagnostic did not complete')
            return {'status':'COMPLETE_DIAGNOSTIC_ONLY','result':{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}}
        envelope=child_envelope(action,BOOT_IO);print(json.dumps(envelope,sort_keys=True));return 2 if envelope['error'] else 0
    if ROOT.exists() or CONTROL.exists():raise ValueError('One shot only; retain previous outputs/failure evidence')
    core=b.load(HELPER_ROOT/'full_future_metadata_preparation_v1/run_build_once.py',b.CORE_SHA,'diagnostic_original_monitor')
    core.BOOT_IO=BOOT_IO;core.START=START
    s,obs,metadata=parent_helpers(b);s.shared_mount();os.sched_setaffinity(0,set(a['cpu_affinity']))
    finite.CONTROL_CAP=CONTROL_CAP;observed=None
    with ExitStack() as stack:
        lock=s.acquire(s.LOCK,stack)
        # Reuse the existing nonblocking flock implementation; external jobs are never owned or signalled.
        s.acquire('/tmp/foresight_gpu_'+str(a['gpu_index'])+'.lock',stack)
        capacity=fresh_capacity(a);validate(a,time.time(),os.sched_getaffinity(0),socket.gethostname())
        CONTROL.mkdir(parents=True,exist_ok=False);stack.enter_context(finite.control_lock(CONTROL))
        def publish(name,value):
            census(metadata,len(json.dumps(value,sort_keys=True,allow_nan=False).encode())+8192)
            finite.control_once(CONTROL,name,value);census(metadata)
        publish('INVOCATION.json',{'schema':'st_teacher_channel_diagnostic_invocation_v1','admission':{'path':args.admission,'sha256':args.admission_sha256},
            'limits':LIMITS,'capacity':capacity,'cpu_affinity':a['cpu_affinity'],'automatic_retry':False,
            'read_semantics':'sampled cumulative rchar stop threshold; mmap page reads not included;320GiB also conservative inventory allowance'})
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a['gpu_index']),PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(RUNTIME),
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
                ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()),ST_CHANNEL_DIAGNOSTIC_DEADLINE=str(deadline))
            observed=core.monitor([PYTHON,str(Path(__file__).absolute()),*sys.argv[1:],'--child'],runtime=str(RUNTIME),
                env=env,fd=lock.fileno(),root=ROOT,s=s,m=metadata,obs=obs,budget=LIMITS,
                cpu_affinity=tuple(a['cpu_affinity']),storage_guard=lambda:census(metadata,256<<10),lease_valid_until=a['valid_until_unix'])
            charged=core.charge_after_reap(observed['read_accounting'],obs.proc_io())
            result=(observed.get('child_accounting') or {}).get('cli_result') or {}
            ok=(observed['returncode']==0 and observed['failure'] is None and observed['cleanup']['live_owned_remaining']==0
                and charged<=LIMITS['read_bytes'] and result.get('status')=='COMPLETE_DIAGNOSTIC_ONLY')
            publish('RESULT.json',{'status':'COMPLETE_DIAGNOSTIC_ONLY' if ok else 'FAILED_NO_AUTO_RETRY','automatic_retry':False,
                'resources':compact_resources(observed),'charged_parent_child_rchar_bytes':charged,'cli_result':result,
                'training_loss_claimed':False,'mmap_physical_IO_measured':False,'new_teacher_cache_bytes':0})
            return 0 if ok else 2
        except BaseException as exc:
            with finite.whole_invocation_alarm(min(deadline+40,time.monotonic()+40),terminal=True):
                # A child-output overrun must not prevent its failure receipt in the separately reserved control slice.
                finite.control_once(CONTROL,'RESULT.json',{'status':'FAILED_NO_AUTO_RETRY','automatic_retry':False,'error_type':type(exc).__name__,
                    'error':str(exc)[:2048],'charged_parent_child_rchar_bytes':LIMITS['read_bytes'],
                    'cleanup':observed.get('cleanup') if observed else 'MONITOR_RAISED_AFTER_OWNED_CLEANUP'})
            return 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS);args=p.parse_args();sys.dont_write_bytecode=True
    deadline=START+LIMITS['wall_seconds']
    if args.child:deadline=min(deadline,float(os.environ['ST_CHANNEL_DIAGNOSTIC_DEADLINE']))
    b,finite=bootstrap(deadline)
    with finite.whole_invocation_alarm(deadline):return run(args,b,finite,deadline)


if __name__=='__main__':raise SystemExit(main())
