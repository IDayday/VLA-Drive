#!/usr/bin/env python3
"""Lead-invoked index controls only; reuse the finite monitor, never build or migrate."""
import time
START=time.monotonic()
with open('/proc/self/io') as _io:BOOT_IO={k:int(v) for k,v in (line.split(':') for line in _io)}
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
from types import ModuleType,SimpleNamespace

RUNTIME=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
LIB=RUNTIME/'reports/st_world_policy/20261017'
ADAPTER=Path('/mnt/project/VLA-Drive-st-future-finalization-fix-20261017/reports/st_world_policy/20261017/future_finalization_actual_proposal_v1/run_finalize_once.py')
ADAPTER_SHA='82477551e0e335bc25c20556387d0b8d98512e1728481ed1e00e8218976c1bb2'
CORE_SHA='4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2'
ALLOWED=Path('/mnt/project/st-world-policy-20261016/audits/full_future_index_preparation_v1')
LIMITS={'wall_seconds':120,'cleanup_seconds':40,'read_bytes':384<<20,'rss_bytes':1<<30}
PYTHON='/root/miniconda3/envs/ddp/bin/python'


def read_source(path,sha):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):
        raise ValueError('Small canonical source required')
    def identity(s):
        return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before=p.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>1<<20:
        raise ValueError('Small regular source required')
    # Nonblocking/no-follow also closes the lstat-to-open FIFO/symlink race.
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        opened=os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or identity(before)!=identity(opened):
            raise ValueError('Pinned source changed before open')
        chunks=[];size=0
        while size<=1<<20:
            chunk=os.read(fd,min(64<<10,(1<<20)+1-size))
            if not chunk:break
            chunks.append(chunk);size+=len(chunk)
        raw=b''.join(chunks)
        after=os.fstat(fd);named=p.lstat()
    finally:os.close(fd)
    if (size>1<<20 or len(raw)!=before.st_size or identity(before)!=identity(after)
            or identity(after)!=identity(named) or hashlib.sha256(raw).hexdigest()!=sha):
        raise ValueError('Pinned source changed')
    return raw


@contextmanager
def first_source_alarm(deadline):
    """Bound bootstrap reads before the pinned finite adapter can be loaded."""
    previous=signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)!=(0.,0.):
        raise ValueError('Existing caller alarm must not be replaced')
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Whole wrapper wall deadline')
    def expire(*_):raise TimeoutError('Whole wrapper wall deadline')
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,remaining)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def load(path,sha,name):
    module=ModuleType(name);module.__file__=str(path)
    exec(compile(read_source(path,sha),str(path),'exec'),module.__dict__)
    return module


def options():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('wrapper-sha256','preparer','preparer-sha256','policy','policy-sha256','ledger-sha256',
                 'output','train-index-source-pins','train-index-source-pins-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    deadline=START+LIMITS['wall_seconds']
    with first_source_alarm(deadline):
        args=options();os.sched_setaffinity(0,{0,1});sys.dont_write_bytecode=True
        finite=load(ADAPTER,ADAPTER_SHA,'reviewed_finite_controls')
    with finite.whole_invocation_alarm(deadline):
        return run(args,finite)


def run(args,finite):
    read_source(Path(__file__).absolute(),args.wrapper_sha256)
    read_source(args.preparer,args.preparer_sha256)
    output=Path(args.output)
    if not output.is_absolute() or '..' in output.parts or not output.is_relative_to(ALLOWED) or output==ALLOWED:
        raise ValueError('New isolated index preparation directory required')
    control=output.with_name(output.name+'_controls')
    if any(p.is_symlink() for p in (output,control,*output.parents)):raise ValueError('Plain control/output paths required')
    core=load(LIB/'full_future_metadata_preparation_v1/run_build_once.py',CORE_SHA,'reviewed_index_prepare_monitor')
    core.BOOT_IO=BOOT_IO;core.START=START
    helper=LIB/'spatial_probe_actual_preparation_v1'
    pins={'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
        'supervise_once.py':'d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
        'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'}
    for name,sha in pins.items():read_source(helper/name,sha)
    sys.path.insert(0,str(helper));sys.path.insert(0,str(RUNTIME))
    import supervise_once as s
    import observed_cli as obs
    from tools.st_world_policy import metadata_storage as metadata
    def census(reserve=0):
        amount=reserve
        for path in (output,control):
            amount+=metadata.tree_inventory(path,max_depth=0,max_entries=16)['charged_bytes']
        if amount>1<<20:raise ValueError('Combined original administrative one-MiB allowance exceeded')
        return amount
    if args.child:
        os.fstat(int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']))
        if os.getpid()!=os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Owned CPU preparation child required')
        result=None;error=None
        try:
            preparer=load(args.preparer,args.preparer_sha256,'reviewed_index_preparer')
            result=preparer.prepare(SimpleNamespace(**{**vars(args),'check_only':False}))
            census()
        except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)[:2048]}
        final=obs.proc_io()
        print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':result,'error':error,
            'counters':{'proc_self_io_baseline':BOOT_IO,'proc_self_io_final':final,
                'proc_self_io_delta':{k:final[k]-BOOT_IO[k] for k in final}}},sort_keys=True))
        return 2 if error else 0
    if output.exists() or control.exists():raise ValueError('One invocation only; retain all prior evidence')
    control.mkdir(parents=True,exist_ok=False);finite.CONTROL_CAP=128<<10
    def publish(name,value):
        census(len(json.dumps(value,sort_keys=True,allow_nan=False).encode())+8192)
        finite.control_once(control,name,value);census()
    observed=None;deadline=START+LIMITS['wall_seconds']
    with finite.control_lock(control) as lock:
        publish('INVOCATION.json',{'schema':'st_future_index_preparation_invocation_v1','argv':sys.argv,
            'limits':LIMITS,'cpu_affinity':[0,1],'max_write_bytes':1<<20,'automatic_retry':False})
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                PYTHONPATH=str(RUNTIME),ST_PROBE_SUPERVISOR_LOCK_FD=str(lock))
            observed=core.monitor([PYTHON,str(Path(__file__).absolute()),*sys.argv[1:],'--child'],
                runtime=str(RUNTIME),env=env,fd=lock,root=output,s=s,m=metadata,obs=obs,
                budget=LIMITS,cpu_affinity=(0,1),storage_guard=census)
            result=(observed.get('child_accounting') or {}).get('cli_result') or {}
            charged=core.charge_after_reap(observed['read_accounting'],obs.proc_io())
            ok=(observed['returncode']==0 and observed['failure'] is None and
                observed['cleanup']['live_owned_remaining']==0 and charged<=LIMITS['read_bytes'] and
                result.get('status')=='PREPARED_NOT_MIGRATED_NOT_AUTHORIZED')
            publish('RESULT.json',{'schema':'st_future_index_preparation_invocation_v1',
                'status':'PREPARATION_COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY','resources':observed,
                'cli_result':result,'charged_rchar_bytes':charged,'automatic_retry':False,
                'ledger_written':False,'index_built':False})
            return 0 if ok else 2
        except BaseException as exc:
            with finite.whole_invocation_alarm(min(deadline+40,time.monotonic()+40),terminal=True):
                publish('RESULT.json',{'status':'FAILED_NO_AUTO_RETRY','automatic_retry':False,
                    'error_type':type(exc).__name__,'error':str(exc)[:2048],
                    'cleanup':observed.get('cleanup') if observed else 'UNKNOWN_OR_NOT_STARTED',
                    'charged_rchar_bytes':LIMITS['read_bytes'],'ledger_written':False,'index_built':False})
            return 2


if __name__=='__main__':raise SystemExit(main())
