#!/usr/bin/env python3
"""Lead-only finite dev/train future metadata build; same ledger, no automatic retry."""
with open('/proc/self/io') as _io:
    BOOT_IO={k:int(v) for k,v in (line.split(':') for line in _io)}
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

START=time.monotonic()
BASE=Path('/mnt/project/st-world-policy-20261016')
PREP=BASE/'audits/full_future_metadata_preparation_v1/prep_phase_v1/PREPARATION.json'
PREP_SHA='dce5b37f391f37c3076a232510e554f382c3a61ad8d983ec381d63c59970855c'
PYTHON='/root/miniconda3/envs/ddp/bin/python'
MiB=1<<20
GiB=1<<30
RUNTIMES={'dev':('/mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017','1265b2376864ee929d95513f4f91e58d0910d147'),
    'train':('/mnt/project/VLA-Drive-st-metadata-build-train-v1-20261017','838dd21f')}
MONITORS={'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
    'supervise_once.py':'a29527dfbe23a4563a66266532f44644f5f184f4a8f59842883524632a63f2d6',
    'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'}


def limits(role):
    return {'wall_seconds':1800 if role=='dev' else 14400,'cleanup_seconds':40,
        'read_bytes':(16 if role=='dev' else 512)*GiB,'rss_bytes':4*GiB,'cpu_affinity':[0,1],
        'stdout_bytes':MiB,'stderr_bytes':MiB,'attempts':1,'automatic_retry':False,'gpu':False}


def pinned(pin,*,decode=True):
    path=Path(pin['path'])
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Absolute non-symlink controls required')
    before=path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>4*MiB:raise ValueError('Small pinned controls required')
    raw=path.read_bytes();after=path.stat()
    ver=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if ver(before)!=ver(after) or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
        raise ValueError('Control/source changed')
    return json.loads(raw) if decode else raw


def validate(a,role,now,*,child=False):
    if role not in RUNTIMES:raise ValueError('Only dev/train metadata builds')
    if (a.get('schema')!='st_full_future_build_resource_admission_v1'
            or a.get('status')!='PASS_BOUNDED_'+role.upper()+'_BUILD_ONLY'
            or a.get('role')!=role or json.dumps(a.get('limits'),sort_keys=True,allow_nan=False)!=json.dumps(limits(role),sort_keys=True)
            or a.get('hostname')!=socket.gethostname()
            or a.get('reviewer') in (None,'','/root/st_indexed_composition')
            or a.get('preparation')!={'path':str(PREP),'sha256':PREP_SHA}):
        raise ValueError('Exact independent role/resource admission required')
    for name in ('issued_at_unix','latest_start_unix','valid_until_unix'):
        if type(a.get(name)) not in (int,float) or not math.isfinite(a[name]):raise ValueError('Finite lease required')
    if not a['issued_at_unix']<=now<=a['latest_start_unix'] or a['valid_until_unix']<now+(0 if child else limits(role)['wall_seconds']+40):
        raise ValueError('Expired/insufficient resource window')
    if not {0,1}.issubset(os.sched_getaffinity(0)):raise ValueError('Admitted CPUs unavailable')


def controls(args,*,child=False):
    a=pinned({'path':args.admission,'sha256':args.admission_sha256});role=a['role'];validate(a,role,time.time(),child=child)
    if a['controller']['path']!=str(Path(__file__).absolute()):raise ValueError('Wrong controller path')
    pinned(a['controller'],decode=False)
    p=pinned(a['preparation']);regpin=p['registrations'][role];reg=pinned(regpin)
    runtime,commit=RUNTIMES[role]
    source=pinned(p['source_inputs']['roles'][role]['source_pins'])
    if reg['cwd']!=runtime or source['runtime']!=runtime or not source['source_commit'].startswith(commit):
        raise ValueError('Fixed role build source changed')
    for pin in source['files'].values():pinned(pin,decode=False)
    if a['policy']!=p['proposed_policy'] or a['registration']!=regpin:raise ValueError('Actual preparation bindings changed')
    pinned(a['policy']);migration=pinned(a['migration'])
    if (migration.get('schema')!='st_metadata_storage_migration_v1' or migration.get('status')!='MIGRATED'
            or migration.get('new_policy_sha256')!=a['policy']['sha256']):
        raise ValueError('Actual completed same-ledger migration required')
    cp=pinned(a['checkpoint_receipt'])
    if (cp.get('schema')!='st_checkpoint_storage_commitments_v1' or cp.get('status')!='CURRENT'
            or cp.get('remaining_bytes')!=2183*GiB or cp.get('free_floor_bytes')!=1<<40
            or type(cp.get('valid_until_unix')) not in (int,float) or not math.isfinite(cp['valid_until_unix'])
            or cp['valid_until_unix']<a['valid_until_unix']):
        raise ValueError('Complete fresh checkpoint/floor reservation must cover lease')
    if set(a['monitor_helpers'])!=set(MONITORS):raise ValueError('Original monitor helpers required')
    parent=None
    for name,sha in MONITORS.items():
        pin=a['monitor_helpers'][name]
        if pin['sha256']!=sha:raise ValueError('Original helper changed')
        pinned(pin,decode=False)
        if parent is None:parent=Path(pin['path']).parent
        if Path(pin['path']).parent!=parent or Path(pin['path']).name!=name:raise ValueError('Mixed monitor helpers')
    sys.path.insert(0,str(parent));import supervise_once as s;import observed_cli as obs
    sys.path.insert(0,runtime)
    from tools.st_world_policy import metadata_storage as m,cache_storage_budget as b
    guard={'policy_path':a['policy']['path'],'policy_sha256':a['policy']['sha256'],
        'registration_path':regpin['path'],'registration_sha256':regpin['sha256'],
        'checkpoint_receipt':a['checkpoint_receipt']['path'],'checkpoint_sha256':a['checkpoint_receipt']['sha256']}
    root=Path(reg['output_root'])
    if root!=BASE/'caches'/('full_future_metadata_'+role+'_v1'):raise ValueError('Fixed role output root required')
    return a,p,reg,root,guard,s,obs,m,b


@contextmanager
def counter_lock(path,*,create=False):
    if path.is_symlink():raise ValueError('Counter-lock alias forbidden')
    with path.open('a+b' if create else 'rb') as f:
        deadline=time.monotonic()+2
        while True:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('Counter snapshot writer-lock deadline')
                time.sleep(.01)
        yield


def read_counter(root):
    path=root/'CONTROL_BUILD_COUNTERS.json'
    if not path.exists():return None
    with counter_lock(root/'CONTROL_BUILD_COUNTER_LOCK.json'):
        with path.open('rb') as stream:value=stream.read(65537)
        if len(value)>65536:raise ValueError('Counter size')
        return json.loads(value)


def append_pipe(buffers,name,data):
    buf=buffers[name];remaining=MiB-len(buf);buf.extend(data[:remaining])
    return len(data)<=remaining


def publish_guarded(m,b,guard,path,value):
    """Original single-ledger marker; source/output checks occur before replace."""
    with m.metadata_writing(**guard) as ctx:
        if path.exists():raise ValueError('One-attempt marker already exists; no retry')
        observed=ctx.check(before_commit=True,path=path)
        if observed['charged_disk_bytes']+b.json_write_bound(value)>ctx.limits['max_disk_bytes']:
            raise ValueError('Retained metadata lacks control/staging headroom')
        b._atomic(path,value);ctx.check(before_commit=True)


def child_run(args):
    a,p,reg,root,guard,s,obs,m,b=controls(args,child=True)
    os.fstat(int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']))
    if os.getpid()!=os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Owned CPU child required')
    # Accounting baseline preceded all imports; no torch/model is required here.
    class NoCUDA:
        @staticmethod
        def is_initialized():return False
    class CPUOnly:cuda=NoCUDA
    cap=a['limits']['read_bytes']-128*MiB
    counter=obs.Counters(root/'CONTROL_BUILD_COUNTERS.json',CPUOnly,cap,BOOT_IO)
    original_context=m.metadata_writing
    @contextmanager
    def observed_context(**kw):
        with original_context(**kw) as ctx:
            lock=root/'CONTROL_BUILD_COUNTER_LOCK.json';ctx.check(before_commit=True,path=lock)
            with counter_lock(lock,create=True):pass
            def store(path,value,*,replace=False):
                if Path(path)!=counter.target:raise ValueError('Only this bounded counter may replace')
                raw=json.dumps(value,allow_nan=False)
                if len(raw)>65536:raise ValueError('Counter size limit')
                with counter_lock(lock):
                    observed=ctx.check(before_commit=True,path=counter.target)
                    if observed['charged_disk_bytes']+b.json_write_bound(value)>ctx.limits['max_disk_bytes']:
                        raise ValueError('Counter/staging headroom exhausted')
                    b._atomic(counter.target,value);ctx.check(before_commit=True)
            obs.write_control=store
            counter.data['phase']='REGISTERED_FUTURE_BUILD';counter.snapshot()
            thread=threading.Thread(target=counter.monitor,daemon=True);thread.start()
            try:yield ctx
            finally:
                counter.stop.set();thread.join(2)
                if counter.snapshot()['limit_exceeded']:raise RuntimeError('Inclusive child read budget exceeded')
    m.metadata_writing=observed_context
    result=m.run_registered_metadata(**guard)
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0


def monitor(command,*,runtime,env,fd,root,s,m,obs,budget):
    """Original bounded pipe loop with existing unreaped-session/pidfd ownership."""
    buffers={name:bytearray() for name in ('stdout','stderr')};peak=0;failure=None;counter=None;child=None;owner=None;leader=None
    start=time.monotonic();selector=selectors.DefaultSelector()
    with s.lifetime_scope() as stopped:
        try:
            child=subprocess.Popen(['taskset','-c','0,1',*command],cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,pass_fds=(fd,))
            leader=s.pidfd_open(child.pid);owner=s.OwnedSession(child)
            for name,stream in (('stdout',child.stdout),('stderr',child.stderr)):
                os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
            next_sample=0.;term_at=None
            while True:
                active=owner.discover();done=owner.leader_done()
                if stopped['signal'] is not None:failure=failure or 'SUPERVISOR_SIGNAL_'+str(stopped['signal'])
                if done and active:failure=failure or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
                if not active and not selector.get_map():break
                for key,_ in selector.select(.25):
                    data=os.read(key.fd,65536)
                    if not data:selector.unregister(key.fileobj);continue
                    if not append_pipe(buffers,key.data,data):failure=failure or key.data.upper()+'_LIMIT'
                now=time.monotonic()
                if now>=next_sample:
                    next_sample=now+1
                    try:
                        peak=max(peak,m.process_tree_rss(os.getpid()))
                        path=root/'CONTROL_BUILD_COUNTERS.json'
                        if path.exists():
                            counter=read_counter(root)
                            if counter['limit_exceeded']:failure=failure or counter['limit_exceeded']
                            if not done and time.time()-path.stat().st_mtime>15:failure=failure or 'STALE_COUNTER'
                            total=counter['proc_self_io_delta']['rchar']+obs.proc_io()['rchar']-BOOT_IO['rchar']
                            if total>budget['read_bytes']:failure=failure or 'PARENT_CHILD_READ_LIMIT'
                    except Exception:failure=failure or 'MONITOR_UNREADABLE'
                    if peak>budget['rss_bytes']:failure=failure or 'RSS_LIMIT'
                    if now-START>=budget['wall_seconds']:failure=failure or 'WALL_LIMIT'
                if failure and term_at is None:owner.send(signal.SIGTERM);term_at=now
                if term_at is not None and now-term_at>=30:owner.send(signal.SIGKILL)
                if term_at is not None and now-term_at>=40:raise RuntimeError('Owned cleanup exceeded grace')
            code=owner.finish(terminate=failure is not None,deadline=term_at+40 if term_at else None)
            counter=read_counter(root)  # final snapshot, not a potentially older polling sample
            return {'returncode':code,'failure':failure,'peak_aggregate_rss_bytes':peak,
                'invocation_seconds':time.monotonic()-START,'supervised_child_seconds':time.monotonic()-start,
                'counters':counter,'stdout':bytes(buffers['stdout']).decode('utf8','replace'),
                'stderr':bytes(buffers['stderr']).decode('utf8','replace'),
                'cleanup':{'verified_processes':owner.verified,'live_owned_remaining':0}}
        except BaseException:
            if owner is not None:owner.finish(terminate=True)
            elif child is not None:s.abort_spawn_setup(child,leader)
            raise
        finally:
            selector.close()
            if owner is not None:owner.close()
            if leader is not None:os.close(leader)
            if child is not None:child.stdout.close();child.stderr.close()


def run(args):
    a,p,reg,root,guard,s,obs,m,b=controls(args)
    runtime=reg['cwd'];budget=a['limits']
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()!=pinned(p['source_inputs']['roles'][a['role']]['source_pins'])['source_commit']:
        raise ValueError('Immutable runtime changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=runtime).strip():raise ValueError('Immutable runtime dirty')
    dispatch=pinned(a['dispatch_policy'])
    if not any(socket.gethostname()==h or socket.gethostname().startswith(h+'-worker-') for h in dispatch['allowed_hosts']):
        raise ValueError('Current host not allowed by current pinned dispatch policy')
    s.shared_mount();os.sched_setaffinity(0,[0,1])
    started=root/'CONTROL_BUILD_STARTED.json';result_path=root/'CONTROL_BUILD_RESULT.json'
    def publish(path,value):return publish_guarded(m,b,guard,path,value)
    with ExitStack() as stack:
        lock=s.acquire(s.LOCK,stack)
        validate(a,a['role'],time.time())
        start={'schema':'st_full_future_build_invocation_v1','role':a['role'],
            'admission':{'path':args.admission,'sha256':args.admission_sha256},'controller':a['controller'],
            'preparation':a['preparation'],'registered_unix':time.time(),'limits':budget,'automatic_retry':False}
        publish(started,start)  # existing metadata_writing owns same global lock through atomic marker
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=runtime,
                ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()))
            command=[PYTHON,str(Path(__file__).absolute()),'--child','--admission',args.admission,
                '--admission-sha256',args.admission_sha256]
            observed=monitor(command,runtime=runtime,env=env,fd=lock.fileno(),root=root,s=s,m=m,obs=obs,budget=budget)
            try:output=json.loads(observed['stdout'])
            except Exception:output=None
            c=observed['counters'];total=budget['read_bytes'] if c is None else c['proc_self_io_delta']['rchar']+obs.proc_io()['rchar']-BOOT_IO['rchar']
            ok=(observed['returncode']==0 and observed['failure'] is None and c is not None
                and isinstance(output,dict) and output.get('status')=='GUARDED_METADATA_RETURN'
                and output.get('artifact',{}).get('status')=='COMPLETE' and total<=budget['read_bytes']
                and observed['invocation_seconds']<=budget['wall_seconds'])
            result={**start,'status':'BUILD_COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY',
                'resources':observed,'charged_parent_child_rchar_bytes':total,'cli_result':output,
                'scientific_result':False,'training_authorized':False,'new_teacher_tensor_bytes':0}
            publish(result_path,result);print(json.dumps({'status':result['status'],'result':str(result_path)}))
            return 0 if ok else 2
        except BaseException as exc:
            if not result_path.exists():publish(result_path,{**start,'status':'FAILED_NO_AUTO_RETRY',
                'error_type':type(exc).__name__,'error':str(exc)[:2048],'charged_source_bytes':budget['read_bytes']})
            return 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    return child_run(a) if a.child else run(a)


if __name__=='__main__':raise SystemExit(main())
