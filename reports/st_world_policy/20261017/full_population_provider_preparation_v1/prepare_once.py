#!/usr/bin/env python3
"""One bounded original full-population provider preparation; never trains or retries."""
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
import socket
import sys
from types import ModuleType

HERE=Path(__file__).absolute().parent
BASE=Path('/mnt/project/st-world-policy-20261016')
ROOT=BASE/'audits/full_population_provider_preparation_v1'
CONTROL=ROOT/'controls'
RUNTIME=Path('/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017')
COMMIT='a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a'
CONTROL_RUNTIME=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
LIB=CONTROL_RUNTIME/'reports/st_world_policy/20261017'
BOOTSTRAP=HERE.parent/'future_index_sequential_control_v1/prepare_once.py'
BOOTSTRAP_SHA='49b21405d1dacee17405598c1a7165727db9855af4de66c2feff70ab30b93ce5'
PLAN_SHA='d8c76c0445975362969efed9e506ae84956a29e8851a8eac90ffb9fd5ed004f4'
LIMITS=dict(wall_seconds=1500,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30)
OUTPUT_LIMIT=32<<20
PYTHON='/root/miniconda3/envs/ddp/bin/python'


def bootstrap():
    # GNU timeout in launch_once.sh covers these earliest imports/source reads.
    fd=os.open(BOOTSTRAP,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        import stat
        before=os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>64<<10:raise ValueError('Small regular pinned bootstrap required')
        raw=f.read((64<<10)+1)
    if len(raw)>64<<10 or hashlib.sha256(raw).hexdigest()!=BOOTSTRAP_SHA:raise ValueError('Bootstrap changed')
    module=ModuleType('reviewed_index_bootstrap');module.__file__=str(BOOTSTRAP)
    exec(compile(raw,str(BOOTSTRAP),'exec'),module.__dict__)
    return module


def commands(plan):
    p=plan['inputs'];path=lambda key:p[key]['path']
    return [
      ('tools.st_world_policy.qualify_indexed_provider',[
        'prepare-profile-migration','--parent-descriptor',path('parent_descriptor'),
        '--parent-runtime',plan['parent_runtime'],'--output',str(ROOT/'SOURCE_MIGRATION.json')]),
      ('tools.st_world_policy.teacher_provider',[
        '--request-index',str(Path(path('train_index_complete')).parent),
        '--source-index',str(Path(path('current_index_complete')).parent),
        '--teacher-config',path('teacher_config'),'--qualification-result',path('qualification'),
        '--max-cached-bytes','14159232','--max-scene-mib','64','--max-batch-images','8','--max-metadata-cache-mib','16',
        '--profile-parent-descriptor',path('parent_descriptor'),
        '--profile-source-migration',str(ROOT/'SOURCE_MIGRATION.json'),
        '--profile-data-manifest',plan['data_manifest']['path'],'--profile-reference-config',path('reference_config'),
        '--profile-full-population','--profile-current-registration',path('current_registration'),
        '--profile-current-complete',path('current_complete'),'--profile-future-complete',path('future_complete'),
        '--profile-consumption-budget',str(ROOT/'CONSUMPTION.json'),
        '--training-config',str(ROOT/'PROFILE_RECIPE.json'),'--output',str(ROOT/'PROVIDER.json'),
        '--pca-fit-cache-root',plan['pca_fit_cache_root'],
        '--pca-fit-cache-index',str(Path(path('pca_fit_index_complete')).parent),
        '--configured-output',str(ROOT/'configured')])]


def validate(admission,plan,args,reads,now,child=False):
    if (admission.get('schema')!='st_full_population_provider_preparation_admission_v1'
            or admission.get('status')!='APPROVED_PROVIDER_METADATA_PREPARATION_ONLY'
            or admission.get('reviewer') in (None,'','/root/st_indexprep_boundary_fix')
            or admission.get('hostname')!=socket.gethostname()
            or admission.get('limits')!=LIMITS or admission.get('cpu_affinity')!=[0,1]
            or admission.get('output_root')!=str(ROOT) or admission.get('max_output_bytes')!=OUTPUT_LIMIT
            or admission.get('control_reservation')!=plan.get('control_reservation')
            or admission.get('gpu') is not False or admission.get('automatic_retry') is not False
            or admission.get('ledger_written') is not False or admission.get('checkpoint_reservation_created') is not False
            or admission.get('plan')!={'path':str(HERE/'PLAN.json'),'sha256':PLAN_SHA}):
        raise ValueError('Exact independent metadata-only resource admission required')
    for key in ('issued_at_unix','latest_start_unix','valid_until_unix'):
        if type(admission.get(key)) not in (int,float) or not math.isfinite(admission[key]):raise ValueError('Finite lease required')
    if (not admission['issued_at_unix']<=now<=admission['latest_start_unix']
            or admission['valid_until_unix']<now+(0 if child else 1540)):
        raise ValueError('Expired or insufficient full preparation lease')
    if (plan.get('schema')!='st_full_population_provider_preparation_plan_v1'
            or plan.get('runtime')!={'path':str(RUNTIME),'source_sha':COMMIT}
            or plan.get('output_root')!=str(ROOT) or plan.get('limits')!=LIMITS
            or plan.get('max_output_bytes')!=OUTPUT_LIMIT or plan.get('cpu_affinity')!=[0,1]):
        raise ValueError('Fixed plan/runtime/output/resource identity changed')
    expected={str(HERE/'prepare_once.py'),str(HERE/'launch_once.sh')}
    sources=admission.get('controller_sources',[])
    if len(sources)!=2 or {p.get('path') for p in sources}!=expected:raise ValueError('Exact wrapper and GNU bootstrap pins required')
    for pin in sources:
        value=pin.get('sha256')
        if (set(pin)!={'path','sha256'} or not isinstance(value,str) or len(value)!=64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Exact nonnull controller SHA256 required')
        reads.raw(pin['path'],value)
    for pin in plan['source_files'].values():reads.raw(pin['path'],pin['sha256'])
    for pin in plan['inputs'].values():reads.document(pin)
    fit=reads.document(plan['inputs']['fit_result'])
    if fit.get('status')!='COMPLETE' or fit.get('stage')!='fit':raise ValueError('Original fit300 has not completed')
    if not {0,1}.issubset(os.sched_getaffinity(0)):raise ValueError('Leased CPU0/1 unavailable')


def original_cli(module,argv):
    before=sys.argv
    try:
        sys.argv=[module,*argv]
        # Original CLI JSON/logs go to the existing bounded stderr stream.
        # stdout is reserved for the sole monitor accounting envelope.
        try:
            with redirect_stdout(sys.stderr):runpy.run_module(module,run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None,0):raise RuntimeError('Original CLI failed: '+module+' '+str(exc.code)) from exc
    finally:sys.argv=before


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();sys.dont_write_bytecode=True
    b=bootstrap();deadline=START+LIMITS['wall_seconds']
    if args.child:deadline=min(deadline,float(os.environ['ST_PROVIDER_PREPARATION_DEADLINE']))
    with b.first_source_alarm(deadline):
        finite=b.load(b.ADAPTER,b.ADAPTER_SHA,'reviewed_finite_provider_controls')
        common=b.load(HERE.parent/'w_storage_bridge_v1/storage_bridge.py',
            'a6aa3a2ba5430ef588cd9811a4da373d64d05c17f042cade40324c810909089f','reviewed_bounded_metadata_reader')
    with finite.whole_invocation_alarm(deadline):return run(args,b,finite,common,deadline)


def run(args,b,finite,common,deadline):
    reads=common.Reads()
    plan=reads.document({'path':str(HERE/'PLAN.json'),'sha256':PLAN_SHA})
    admission=reads.document({'path':args.admission,'sha256':args.admission_sha256})
    validate(admission,plan,args,reads,time.time(),args.child)
    core=b.load(LIB/'full_future_metadata_preparation_v1/run_build_once.py',b.CORE_SHA,'reviewed_provider_monitor')
    core.BOOT_IO=BOOT_IO;core.START=START
    helper=LIB/'spatial_probe_actual_preparation_v1'
    for name,sha in {'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
        'supervise_once.py':'d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
        'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'}.items():b.read_source(helper/name,sha)
    sys.path.insert(0,str(helper));sys.path.insert(0,str(RUNTIME))
    import supervise_once as s
    import observed_cli as obs
    from tools.st_world_policy import metadata_storage as metadata
    def census(reserve=0):
        count=metadata.tree_inventory(ROOT,max_depth=3,max_entries=64)['charged_bytes']+reserve
        if count>OUTPUT_LIMIT:raise ValueError('32 MiB administrative output slice exceeded')
        return count
    if args.child:
        os.fstat(int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']))
        if os.getpid()!=os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Owned CPU-only child required')
        result=None;error=None
        try:
            from starVLA.dataloader.st_world_targets import atomic_json
            atomic_json(ROOT/'PROFILE_RECIPE.json',plan['recipe'],immutable=True)
            atomic_json(ROOT/'CONSUMPTION.json',plan['consumption'],immutable=True)
            for module,argv in commands(plan):
                if time.monotonic()>=deadline:raise TimeoutError('Shared preparation deadline exhausted')
                original_cli(module,argv);census(1<<20)
            # Its helper window is small metadata only; aggregate factory reads
            # remain charged once by the outer 16 GiB monitor across both CLIs.
            outputs={};output_reads=common.Reads()
            for name in ('SOURCE_MIGRATION.json','PROVIDER.json','configured/training_config.json',
                         'configured/PROVIDER_CHANNEL_SCOPE.json','configured/PROVIDER_CONFIG_LINEAGE.json'):
                raw=output_reads.raw(ROOT/name);outputs[name]={'path':str(ROOT/name),'sha256':hashlib.sha256(raw).hexdigest()}
            result={'status':'PROVIDER_PREPARATION_COMPLETE','outputs':outputs,'training_started':False,
                'new_teacher_tensor_bytes':0,'ledger_written':False,'checkpoint_reservation_created':False}
        except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)[:2048]}
        final=obs.proc_io()
        print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':result,'error':error,
            'counters':{'proc_self_io_baseline':BOOT_IO,'proc_self_io_final':final,
                'proc_self_io_delta':{k:final[k]-BOOT_IO[k] for k in final}}},sort_keys=True))
        return 2 if error else 0
    if ROOT.exists():raise ValueError('One shot only; preserve all prior output and failed evidence')
    if reads.git(RUNTIME,'rev-parse','HEAD').decode().strip()!=COMMIT or reads.git(RUNTIME,'status','--porcelain'):
        raise ValueError('Original frozen provider runtime changed')
    s.shared_mount();os.sched_setaffinity(0,{0,1})
    finite.CONTROL_CAP=1<<20;observed=None
    with ExitStack() as stack:
        lock=s.acquire(s.LOCK,stack)
        validate(admission,plan,args,reads,time.time())
        ROOT.mkdir(exist_ok=False);CONTROL.mkdir()
        stack.enter_context(finite.control_lock(CONTROL))
        def publish(name,value):
            census(len(json.dumps(value,sort_keys=True,allow_nan=False).encode())+8192)
            finite.control_once(CONTROL,name,value);census()
        publish('INVOCATION.json',{'schema':'st_full_population_provider_preparation_invocation_v1',
            'admission':{'path':args.admission,'sha256':args.admission_sha256},'plan':admission['plan'],
            'argv':sys.argv,'limits':LIMITS,'cpu_affinity':[0,1],'max_output_bytes':OUTPUT_LIMIT,'automatic_retry':False})
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=str(RUNTIME),
                ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()),ST_PROVIDER_PREPARATION_DEADLINE=str(deadline))
            observed=core.monitor([PYTHON,str(Path(__file__).absolute()),*sys.argv[1:],'--child'],
                runtime=str(RUNTIME),env=env,fd=lock.fileno(),root=ROOT,s=s,m=metadata,obs=obs,
                budget=LIMITS,cpu_affinity=(0,1),storage_guard=lambda:census(1<<20),
                lease_valid_until=admission['valid_until_unix'])
            result=(observed.get('child_accounting') or {}).get('cli_result') or {}
            charged=core.charge_after_reap(observed['read_accounting'],obs.proc_io())
            ok=(observed['returncode']==0 and observed['failure'] is None
                and observed['cleanup']['live_owned_remaining']==0 and charged<=LIMITS['read_bytes']
                and result.get('status')=='PROVIDER_PREPARATION_COMPLETE')
            publish('RESULT.json',{'schema':'st_full_population_provider_preparation_invocation_v1',
                'status':'PREPARATION_COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY','resources':observed,
                'cli_result':result,'charged_rchar_bytes':charged,'automatic_retry':False,
                'training_started':False,'ledger_written':False,'checkpoint_reservation_created':False})
            return 0 if ok else 2
        except BaseException as exc:
            with finite.whole_invocation_alarm(min(deadline+40,time.monotonic()+40),terminal=True):
                publish('RESULT.json',{'status':'FAILED_NO_AUTO_RETRY','automatic_retry':False,
                    'error_type':type(exc).__name__,'error':str(exc)[:2048],
                    'cleanup':observed.get('cleanup') if observed else 'UNKNOWN_OR_NOT_STARTED',
                    'charged_rchar_bytes':LIMITS['read_bytes'],'training_started':False,'ledger_written':False})
            return 2


if __name__=='__main__':raise SystemExit(main())
