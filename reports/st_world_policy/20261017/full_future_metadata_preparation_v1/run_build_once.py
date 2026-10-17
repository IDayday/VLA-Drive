#!/usr/bin/env python3
"""Lead-only finite dev/train future metadata build; same ledger, no automatic retry."""
with open('/proc/self/io') as _io:
    BOOT_IO={k:int(v) for k,v in (line.split(':') for line in _io)}
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import traceback
import socket
import stat
import subprocess
import sys
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
PRIOR_DEV_RESULT={'path':str(BASE/'caches/full_future_metadata_dev_v1/CONTROL_BUILD_RESULT.json'),
    'sha256':'60b7ba6177b51081fdb323b36a0821f3848e6ddf2c2bb625b97e9a8c850aefda'}
PRIOR_DEV_CHARGE=2202810181
MONITORS={'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
    'supervise_once.py':'a29527dfbe23a4563a66266532f44644f5f184f4a8f59842883524632a63f2d6',
    'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'}


def limits(role,attempt=1):
    return {'wall_seconds':1800 if role=='dev' else 14400,'cleanup_seconds':40,
        'read_bytes':(16 if role=='dev' else 512)*GiB-(PRIOR_DEV_CHARGE if attempt==2 else 0),'rss_bytes':4*GiB,'cpu_affinity':[0,1],
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
    attempt=a.get('attempt',1)
    if type(attempt) is not int or attempt not in (1,2) or (attempt==2 and role!='dev'):
        raise ValueError('Only original attempt or explicit dev attempt2')
    if attempt==2 and a.get('prior_attempt')!=PRIOR_DEV_RESULT:
        raise ValueError('Exact retained failed dev attempt required')
    if (a.get('schema')!='st_full_future_build_resource_admission_v1'
            or a.get('status')!='PASS_BOUNDED_'+role.upper()+('_BUILD_ONLY' if attempt==1 else '_BUILD_ATTEMPT2_ONLY')
            or a.get('role')!=role or json.dumps(a.get('limits'),sort_keys=True,allow_nan=False)!=json.dumps(limits(role,attempt),sort_keys=True)
            or a.get('hostname')!=socket.gethostname()
            or a.get('reviewer') in (None,'','/root/st_indexed_composition')
            or a.get('preparation')!={'path':str(PREP),'sha256':PREP_SHA}):
        raise ValueError('Exact independent role/resource admission required')
    for name in ('issued_at_unix','latest_start_unix','valid_until_unix'):
        if type(a.get(name)) not in (int,float) or not math.isfinite(a[name]):raise ValueError('Finite lease required')
    if not a['issued_at_unix']<=now<=a['latest_start_unix'] or a['valid_until_unix']<now+(0 if child else limits(role,attempt)['wall_seconds']+40):
        raise ValueError('Expired/insufficient resource window')
    if not {0,1}.issubset(os.sched_getaffinity(0)):raise ValueError('Admitted CPUs unavailable')


def controls(args,*,child=False):
    a=pinned({'path':args.admission,'sha256':args.admission_sha256});role=a['role'];validate(a,role,time.time(),child=child)
    if a['controller']['path']!=str(Path(__file__).absolute()):raise ValueError('Wrong controller path')
    pinned(a['controller'],decode=False)
    if a.get('attempt',1)==2:
        prior=pinned(a['prior_attempt'])
        if (prior.get('status')!='FAILED_NO_AUTO_RETRY' or prior.get('charged_parent_child_rchar_bytes')!=PRIOR_DEV_CHARGE
                or prior.get('resources',{}).get('cleanup',{}).get('live_owned_remaining')!=0):
            raise ValueError('Prior failed attempt/accounting/cleanup changed')
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
    # Existing Counters are sampled once into RAM; no observer writes into the builder tree.
    class NoCUDA:
        @staticmethod
        def is_initialized():return False
    class CPUOnly:cuda=NoCUDA
    counter=obs.Counters(root/'UNUSED_RAM_ONLY_COUNTER',CPUOnly,a['limits']['read_bytes'],BOOT_IO)
    original_write=obs.write_control
    obs.write_control=lambda *args,**kwargs:None
    result=None;error=None;code=0
    try:
        result=m.run_registered_metadata(**guard)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)[:2048]};code=1
        traceback.print_exc()  # retain actual failure before the final accounting snapshot
    finally:
        try:
            counters=counter.snapshot()
            current=obs.proc_io()
            counters['proc_self_io_delta']={k:current[k]-BOOT_IO.get(k,0) for k in current}
            counters['proc_self_io_final']=current
            counters['proc_self_io_baseline']=BOOT_IO
            counters['read_accounting_scope']='child pre-import baseline delta; parent separately charges whole owned-process rchar including startup'
        finally:obs.write_control=original_write
    print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':result,
        'error':error,'counters':counters},sort_keys=True,allow_nan=False))
    return code


def owned_proc_io(owner,s):
    """Read only the current unreaped spawn anchor, with both-sided identity checks."""
    anchor=owner.anchor;pid=anchor['pid']
    if not owner.same(anchor,s.proc_identity(pid)):raise RuntimeError('Owned IO identity changed')
    with open('/proc/'+str(pid)+'/io') as stream:raw=stream.read(4097)
    if len(raw)>4096:raise ValueError('Owned IO record exceeds bound')
    value={k:int(v) for k,v in (line.split(':') for line in raw.splitlines())}
    if 'rchar' not in value or any(type(v) is not int or v<0 for v in value.values()):raise ValueError('Invalid owned IO counters')
    if not owner.same(anchor,s.proc_identity(pid)):raise RuntimeError('Owned IO identity changed during read')
    return value


def decode_final(stdout):
    value=json.loads(stdout)
    if not isinstance(value,dict) or value.get('schema')!='st_full_future_child_result_v2':raise ValueError('Final child accounting schema')
    c=value.get('counters')
    if not isinstance(c,dict):raise ValueError('Missing final child counters')
    for key in ('proc_self_io_baseline','proc_self_io_final','proc_self_io_delta'):
        row=c.get(key)
        if not isinstance(row,dict) or 'rchar' not in row or any(type(v) is not int or v<0 for v in row.values()):
            raise ValueError('Final child counters must be nonnegative exact integers')
    before,after,delta=(c[k] for k in ('proc_self_io_baseline','proc_self_io_final','proc_self_io_delta'))
    if set(after)!=set(delta) or any(after[k]-before.get(k,0)!=delta[k] for k in after):
        raise ValueError('Final child IO delta mismatch')
    return value



def terminal_read_accounting(baseline,pre_reap,post_reap,child_rchar,read_cap,*,final_child_available):
    """Linux parent /proc/io after wait already contains this reaped child once."""
    value={'schema':'st_wait_boundary_read_accounting_v1','parent_baseline':baseline,
        'parent_pre_reap':pre_reap,'parent_post_reap':post_reap,'child_terminal_rchar_bytes':child_rchar}
    try:
        counts=[]
        for row in (baseline,pre_reap,post_reap):
            if not isinstance(row,dict) or type(row.get('rchar')) is not int or row['rchar']<0:
                raise ValueError('Missing/invalid parent wait-boundary IO')
            counts.append(row['rchar'])
        if (not final_child_available or type(child_rchar) is not int or child_rchar<0):
            raise ValueError('Missing/invalid final child IO')
        base,before,after=counts
        if before<base or after<before+child_rchar:
            raise ValueError('Parent post-wait IO did not consistently absorb child')
        value.update(status='COMPLETE',parent_pre_reap_delta_rchar_bytes=before-base,
            parent_post_reap_delta_rchar_bytes=after-base,
            parent_activity_during_wait_rchar_bytes=after-before-child_rchar,
            charged_parent_child_rchar_bytes=after-base)
    except ValueError as exc:
        value.update(status='INCOMPLETE_CONSERVATIVE',error=str(exc),charged_parent_child_rchar_bytes=read_cap)
    return value


def charge_after_reap(accounting,current_parent):
    """Add only later parent activity; never re-add the already absorbed child."""
    previous=accounting.get('parent_post_reap')
    if (not isinstance(current_parent,dict) or type(current_parent.get('rchar')) is not int
            or not isinstance(previous,dict) or type(previous.get('rchar')) is not int
            or current_parent['rchar']<previous['rchar']):
        raise ValueError('Missing/nonmonotonic post-reap parent IO')
    return accounting['charged_parent_child_rchar_bytes']+current_parent['rchar']-previous['rchar']

def monitor(command,*,runtime,env,fd,root,s,m,obs,budget, cpu_affinity=(0,1),
            storage_guard=None, lease_valid_until=None):
    """Original bounded pipe loop with existing unreaped-session/pidfd ownership."""
    buffers={name:bytearray() for name in ('stdout','stderr')};peak=0;failure=None;counter=None;child=None;owner=None;leader=None
    finished=False;monitor_error=None;last_io=None;terminal_io=None;accounting=None
    start=time.monotonic();selector=selectors.DefaultSelector()
    wall_started=START if lease_valid_until is None else start
    absolute_end=wall_started+budget["wall_seconds"]+40
    if lease_valid_until is not None:
        if lease_valid_until-time.time()<budget["wall_seconds"]+40: raise ValueError("Insufficient actual-start lease")
        absolute_end=min(absolute_end,start+lease_valid_until-time.time())
    cleanup_until=None
    with s.lifetime_scope() as stopped:
        try:
            if storage_guard: storage_guard()
            child=subprocess.Popen(['taskset','-c',','.join(map(str,cpu_affinity)),*command],cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,pass_fds=(fd,))
            leader=s.pidfd_open(child.pid);owner=s.OwnedSession(child)
            for name,stream in (('stdout',child.stdout),('stderr',child.stderr)):
                os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
            next_sample=0.;term_at=None
            while True:
                active=owner.discover();done=owner.leader_done()
                if stopped['signal'] is not None:failure=failure or 'SUPERVISOR_SIGNAL_'+str(stopped['signal'])
                if done and active:
                    # discover() may have seen the leader just before it exited.
                    # Revalidate all pidfd-owned identities after WNOWAIT confirms exit.
                    active=owner.discover()
                    if active:failure=failure or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
                if not active and not selector.get_map():
                    try:terminal_io=owned_proc_io(owner,s)
                    except Exception as exc:
                        monitor_error=monitor_error or {'type':type(exc).__name__,'message':str(exc)[:2048]}
                    break
                for key,_ in selector.select(.25):
                    data=os.read(key.fd,65536)
                    if not data:selector.unregister(key.fileobj);continue
                    if not append_pipe(buffers,key.data,data):failure=failure or key.data.upper()+'_LIMIT'
                now=time.monotonic()
                if now>=next_sample:
                    next_sample=now+1
                    try:
                        peak=max(peak,m.process_tree_rss(os.getpid()))
                        last_io=owned_proc_io(owner,s)
                        total=last_io['rchar']+obs.proc_io()['rchar']-BOOT_IO['rchar']
                        if total>budget['read_bytes']:failure=failure or 'PARENT_CHILD_READ_LIMIT'
                    except Exception as exc:
                        failure=failure or 'MONITOR_UNREADABLE'
                        monitor_error=monitor_error or {'type':type(exc).__name__,'message':str(exc)[:2048]}
                    if peak>budget['rss_bytes']:failure=failure or 'RSS_LIMIT'
                    if storage_guard: storage_guard()
                    if now-wall_started>=budget['wall_seconds']:failure=failure or 'WALL_LIMIT'
                if failure and term_at is None:
                    owner.send(signal.SIGTERM);term_at=now;cleanup_until=min(absolute_end,now+40)
                if cleanup_until is not None and now>=cleanup_until-10:owner.send(signal.SIGKILL)
                if cleanup_until is not None and now>=cleanup_until:raise RuntimeError('Owned cleanup exceeded grace')
            parent_pre_reap=obs.proc_io()
            code=owner.finish(terminate=failure is not None,deadline=cleanup_until or absolute_end)
            finished=True  # leader is reaped; later receipt errors must not clean it twice
            parent_post_reap=obs.proc_io()
            try:accounting=decode_final(bytes(buffers['stdout']).decode('utf8','replace'))
            except Exception as exc:
                failure=failure or 'CHILD_ACCOUNTING_INVALID'
                monitor_error=monitor_error or {'type':type(exc).__name__,'message':str(exc)[:2048]}
            final_io=accounting['counters']['proc_self_io_final'] if accounting else None
            if terminal_io is None and final_io is None:failure=failure or 'FINAL_IO_MISSING'
            # Whole /proc lifetime absolute counts cover taskset/python startup too.
            # Missing both final sources is charged at the entire remaining allowance.
            charged=max([x['rchar'] for x in (terminal_io,final_io,last_io) if x is not None],default=0)
            if terminal_io is None and final_io is None:charged=budget['read_bytes']
            counter=accounting['counters'] if accounting else None
            read_accounting=terminal_read_accounting(BOOT_IO,parent_pre_reap,parent_post_reap,charged,budget['read_bytes'],
                final_child_available=terminal_io is not None or final_io is not None)
            if read_accounting['status']!='COMPLETE':failure=failure or 'TERMINAL_IO_INCONSISTENT'
            return {'returncode':code,'failure':failure,'monitor_error':monitor_error,'peak_aggregate_rss_bytes':peak,
                'invocation_seconds':time.monotonic()-START,'supervised_child_seconds':time.monotonic()-start,
                'counters':counter,'child_accounting':accounting,'owned_proc_io_baseline':0,
                'owned_proc_io_last_active':last_io,'owned_proc_io_terminal':terminal_io,
                'charged_owned_child_rchar_bytes':charged,'read_accounting':read_accounting,'stdout':bytes(buffers['stdout']).decode('utf8','replace'),
                'stderr':bytes(buffers['stderr']).decode('utf8','replace'),
                'cleanup':{'verified_processes':owner.verified,'live_owned_remaining':0}}
        except BaseException:
            if owner is not None and not finished:owner.finish(terminate=True,deadline=cleanup_until or min(absolute_end,time.monotonic()+40))
            elif owner is None and child is not None:s.abort_spawn_setup(child,leader,deadline=cleanup_until or absolute_end)
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
    prefix='CONTROL_BUILD_' if a.get('attempt',1)==1 else 'CONTROL_BUILD_ATTEMPT2_'
    started=root/(prefix+'STARTED.json');result_path=root/(prefix+'RESULT.json')
    def publish(path,value):return publish_guarded(m,b,guard,path,value)
    with ExitStack() as stack:
        lock=s.acquire(s.LOCK,stack)
        validate(a,a['role'],time.time())
        start={'schema':'st_full_future_build_invocation_v1','role':a['role'],'attempt':a.get('attempt',1),
            'prior_attempt':a.get('prior_attempt'),
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
            envelope=observed['child_accounting'];output=envelope.get('cli_result') if envelope else None
            c=observed['counters'];total=charge_after_reap(observed['read_accounting'],obs.proc_io())
            ok=(observed['returncode']==0 and observed['failure'] is None and c is not None
                and isinstance(output,dict) and output.get('status')=='GUARDED_METADATA_RETURN'
                and output.get('artifact',{}).get('status')=='COMPLETE' and total<=budget['read_bytes']
                and observed['invocation_seconds']<=budget['wall_seconds'])
            result={**start,'status':'BUILD_COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY',
                'resources':observed,'charged_parent_child_rchar_bytes':total,'cli_result':output,
                'cumulative_parent_child_rchar_bytes':total+(PRIOR_DEV_CHARGE if a.get('attempt',1)==2 else 0),
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
