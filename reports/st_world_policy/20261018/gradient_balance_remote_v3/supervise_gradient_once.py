#!/usr/bin/env python3
"""One fixed C1 diagnostic, external resource observation only.

Authored by the systems reviewer; requires independent reviews before Lead
dispatch. No resource release, retry, optimizer, score call or feature workflow.
Reuses the existing reviewed owned-process helper, never its SP0 main function.
RSS/output are observed limits with a 0.2-second reaction interval, not cgroups.
"""
import argparse
from contextlib import ExitStack
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time

RUNTIME=Path('/mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018')
SOURCE='83679937fe9702be39f0184154fdf9ebdc4eb98a'
REGISTRATION=Path('/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v3_controls/REGISTRATION.json')
REG_SHA='6a48f942e8b57f568d74be0bd92cab389daa6140d0a0307896acd40da3358770'
RUNTIME_ENVIRONMENT={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
POLICY=Path('/mnt/project/server_dispatch_policy.json')
GPU_LOCK=Path('/mnt/project/st-world-policy-20261016/resources/locks/GPU-c0a3d651-8691-b507-5df6-6c7f0cc6c897.lock')
ATTEMPT=REGISTRATION.parent/'RUNTIME_ATTEMPT'
HELPERS=RUNTIME/'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1'
PYTHON='/root/miniconda3/envs/ddp/bin/python'
READ_LIMIT=192<<30
RSS_LIMIT=64<<30
OUTPUT_LIMIT=8<<20
WALL=1800
RESERVE=256<<10


def small_json(path, expected=None):
    path=Path(path)
    with path.open('rb') as f: data=f.read((1<<20)+1)
    if len(data)>1<<20: raise ValueError('Oversize control')
    sha=hashlib.sha256(data).hexdigest()
    if expected is not None and sha!=expected: raise ValueError('Changed control '+str(path))
    return json.loads(data)


def write_json(path,value,immutable=False):
    path=Path(path);data=json.dumps(value,sort_keys=True,allow_nan=False,indent=2).encode()
    if len(data)>RESERVE: raise ValueError('Oversize supervisor receipt')
    if immutable:
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)


def ownership_helpers():
    sys.path.insert(0,str(HELPERS))
    spec=importlib.util.spec_from_file_location('_gradient_owned_session',HELPERS/'supervise_once.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ReadAdmission:
    """Precharge every full hash/load, including possible mmap pages.

    Counts repeated operations conservatively. This is a logical admission,
    not physical bandwidth; rchar/read_bytes remain separately observable.
    """
    def __init__(self,path,limit=READ_LIMIT):
        self.path=Path(path);self.limit=limit
        self.values={'hash_nominal_bytes':0,'torch_load_nominal_bytes':0,
            'safetensors_nominal_bytes':0,'calls':0,'rejected_admission':None}
        self.save()
    def save(self):write_json(self.path,self.values)
    def total(self):return sum(self.values[k] for k in ('hash_nominal_bytes','torch_load_nominal_bytes','safetensors_nominal_bytes'))
    def wrap(self,original,kind):
        @functools.wraps(original)
        def wrapped(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):size=os.stat(path).st_size
            elif hasattr(path,'fileno'):size=os.fstat(path.fileno()).st_size
            else:raise ValueError('Unaccountable model/hash input refused')
            if size<0 or self.total()+size>self.limit:
                self.values['rejected_admission']={'kind':kind,'bytes':size,'admitted_before':self.total()};self.save()
                raise RuntimeError('Read admission exhausted before '+kind)
            self.values[kind]+=size;self.values['calls']+=1;self.save()
            return original(path,*args,**kwargs)
        return wrapped


def replace_aliases(original,wrapped):
    # Loaded from-import aliases must also be measured. New imports inherit the
    # patched defining module. No mathematical or loader behavior is replaced.
    for module in tuple(sys.modules.values()):
        if module is None:continue
        for key,value in tuple(vars(module).items()):
            if value is original:setattr(module,key,wrapped)


def validate_runtime_environment(environ=None,*,before_torch=True):
    env=os.environ if environ is None else environ
    if any(env.get(k)!=v for k,v in RUNTIME_ENVIRONMENT.items()):
        raise ValueError('C1 deterministic runtime requires explicit CUBLAS_WORKSPACE_CONFIG=:4096:8')
    if before_torch and 'torch' in sys.modules:
        raise ValueError('Runtime environment must be checked before importing torch')
    return dict(RUNTIME_ENVIRONMENT)


def child_environment(parent,admission,lock_fd):
    validate_runtime_environment(parent)
    return dict(parent,CUDA_VISIBLE_DEVICES=admission['gpu_uuid'],OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',ST_GRADIENT_LOCK_FD=str(lock_fd),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',**RUNTIME_ENVIRONMENT)


def child_main():
    runtime_environment=validate_runtime_environment()
    # Internal fixed child entry. The parent's inherited lock FD proves this is
    # its single active invocation; merely setting an environment variable fails.
    fd=int(os.environ['ST_GRADIENT_LOCK_FD']);os.fstat(fd)
    lock_stat=os.stat(REGISTRATION.parent/'RUN.lock')
    if (os.fstat(fd).st_dev,os.fstat(fd).st_ino)!=(lock_stat.st_dev,lock_stat.st_ino):
        raise RuntimeError('Foreign lock descriptor')
    validate_admission(ATTEMPT/'ADMISSION.json')
    write_json(ATTEMPT/'RUNTIME_ENVIRONMENT.json',{
        'runtime_environment':runtime_environment,'torch_imported':False,
        'checked_before_import':True,'original_C1_determinism_preserved':True},True)
    os.chdir(RUNTIME);sys.path.insert(0,str(RUNTIME))
    counters=ReadAdmission(ATTEMPT/'COUNTERS.json')
    import torch
    import safetensors
    import safetensors.torch
    from starVLA.model.modules.vehicle_joint import initialization
    replace_aliases(torch.load,counters.wrap(torch.load,'torch_load_nominal_bytes'))
    replace_aliases(safetensors.safe_open,counters.wrap(safetensors.safe_open,'safetensors_nominal_bytes'))
    replace_aliases(initialization.file_sha256,counters.wrap(initialization.file_sha256,'hash_nominal_bytes'))
    from tools.st_world_policy.audit_flow_gradient_balance import main
    try:return main(['--registration',str(REGISTRATION),'--execute'])
    finally:counters.save()


def directory_bytes(paths):
    total=0
    for root in paths:
        if not Path(root).exists():continue
        for path in Path(root).rglob('*'):
            try:st=path.lstat()
            except FileNotFoundError:continue
            if path.is_symlink():raise RuntimeError('Symlink inside owned outputs')
            if __import__('stat').S_ISREG(st.st_mode):total+=st.st_size
    return total


def process_sample(owner,observed):
    rss=0
    owner.discover()
    # Keep the unreaped anchor's final /proc/io even after it is a zombie.
    # Fast native exit must not erase its reads between monitor samples.
    for record,_ in tuple(owner.members.values()):
        pid=record['pid']
        try:
            status=Path(f'/proc/{pid}/status').read_text()
            values=[line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')]
            if not values:
                if ownership_helpers_cached.proc_identity(pid)['state'] not in ('Z','X'):
                    raise RuntimeError('Missing live RSS counter')
                current=0
            else:current=int(values[0])*1024
            io={k:int(v) for k,v in (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
            actual=ownership_helpers_cached.proc_identity(pid)
            if not owner.same(record,actual):raise RuntimeError('Monitor identity changed')
        except (FileNotFoundError,ProcessLookupError):continue
        rss+=current
        key=(pid,record['start_ticks'])
        previous=observed.setdefault(key,{'rchar':0,'read_bytes':0})
        for field in previous:previous[field]=max(previous[field],io[field])
    return rss,sum(v['rchar'] for v in observed.values()),sum(v['read_bytes'] for v in observed.values())


ownership_helpers_cached=None


def supervise(argv,*,attempt,output,env,pass_fds=(),wall=WALL,rss_limit=RSS_LIMIT,
              output_limit=OUTPUT_LIMIT,read_limit=READ_LIMIT,sample=None):
    """Fixtures may supply fake children/limits; production argv is fixed below."""
    global ownership_helpers_cached
    helper=ownership_helpers_cached=ownership_helpers()
    started=time.monotonic();owner=None;child=None;leader_fd=None;selector=None
    observed={};peak=0;reason=None;exit_code=None;emitted=0;logical=physical=nominal=0
    sample=sample or process_sample
    receipt={'status':'STARTING','real_scoring_calls':0,'optimizer_steps':0}
    with helper.lifetime_scope() as stopped:
        try:
            child=subprocess.Popen(argv,cwd=RUNTIME,env=env,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(pass_fds))
            leader_fd=helper.pidfd_open(child.pid);owner=helper.OwnedSession(child)
            write_json(Path(attempt)/'CHILD.json',{'pid':child.pid,'identity':owner.anchor,'argv':argv},True)
            selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
            with (Path(attempt)/'CHILD.log').open('xb',buffering=0) as log:
                while True:
                    active=owner.discover();done=owner.leader_done()
                    if done and active:reason='OWNED_DESCENDANTS_AFTER_LEADER_EXIT';break
                    if not active and not selector.get_map():break
                    for key,_ in selector.select(.2):
                        block=os.read(key.fileobj.fileno(),65536)
                        if not block:selector.unregister(key.fileobj);key.fileobj.close();continue
                        if emitted+len(block)>2<<20:reason='STDOUT_LIMIT';break
                        log.write(block);emitted+=len(block)
                    if reason:break
                    current,logical,physical=sample(owner,observed)
                    supervisor_rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
                    peak=max(peak,current+supervisor_rss)
                    counter=Path(attempt)/'COUNTERS.json'
                    if counter.exists():
                        values=small_json(counter)
                        nominal=sum(values[k] for k in ('hash_nominal_bytes','torch_load_nominal_bytes','safetensors_nominal_bytes'))
                    elapsed=time.monotonic()-started
                    if stopped['signal']:reason='SUPERVISOR_SIGNAL_'+str(stopped['signal'])
                    elif elapsed>=max(0.,wall-40):reason='WALL_LIMIT'
                    elif peak>rss_limit:reason='RSS_LIMIT'
                    elif max(logical,nominal)>read_limit:reason='READ_LIMIT'
                    elif directory_bytes((attempt,output))>output_limit-RESERVE:reason='OUTPUT_LIMIT'
                    if reason:break
        except BaseException as exc:
            reason='MONITOR_OR_CHILD_SETUP_FAILURE'
            receipt['error']={'type':type(exc).__name__,'message':str(exc)[:2048]}
        finally:
            try:
                if owner is not None:
                    try:
                        current,logical,physical=process_sample(owner,observed)
                        peak=max(peak,current)
                    except BaseException as exc:
                        reason=reason or 'FINAL_MONITOR_FAILURE'
                        receipt['final_monitor_error']={'type':type(exc).__name__,'message':str(exc)[:2048]}
                    exit_code=owner.finish(terminate=bool(reason),deadline=min(started+wall,time.monotonic()+40))
                    receipt['live_owned_remaining']=0
                elif child is not None:
                    helper.abort_spawn_setup(child,leader_fd,deadline=min(started+wall,time.monotonic()+10))
                    exit_code=child.returncode;receipt['live_owned_remaining']=0
            except BaseException as exc:
                reason='CLEANUP_FAILED';receipt['cleanup_error']={'type':type(exc).__name__,'message':str(exc)[:2048]}
            finally:
                if selector is not None:selector.close()
                if owner is not None:owner.close()
                if leader_fd is not None:os.close(leader_fd)
                if child is not None and child.stdout is not None:child.stdout.close()
    # A fast child can finish between monitoring samples. Do not accept its
    # final admitted read/output totals merely because exit status was zero.
    try:
        counter=Path(attempt)/'COUNTERS.json'
        if counter.exists():
            values=small_json(counter)
            nominal=sum(values[k] for k in ('hash_nominal_bytes','torch_load_nominal_bytes','safetensors_nominal_bytes'))
        output_bytes=directory_bytes((attempt,output))
        if max(logical,nominal)>read_limit:reason=reason or 'READ_LIMIT'
        if output_bytes>output_limit-RESERVE:reason=reason or 'OUTPUT_LIMIT'
        if time.monotonic()-started>wall:reason=reason or 'WALL_LIMIT'
    except BaseException as exc:
        reason=reason or 'FINAL_ACCOUNTING_FAILED';output_bytes=None
        receipt['final_accounting_error']={'type':type(exc).__name__,'message':str(exc)[:2048]}
    receipt.update(status='EXITED' if reason is None and exit_code==0 else 'STOPPED_NO_AUTO_RETRY',
        reason=reason,exit_code=exit_code,wall_seconds=time.monotonic()-started,
        peak_observed_RSS_bytes=peak,observed_rchar_bytes=logical,observed_physical_read_bytes=physical,
        known_nominal_read_bytes=nominal,charged_read_bytes=max(logical,nominal),
        accounting='max(full-file hash + torch.load + safetensors admissions, child-tree rchar); mmap charged at load admission',
        no_RLIMIT_AS=True,output_bytes_before_receipt=output_bytes)
    write_json(Path(attempt)/'SUPERVISOR_RESULT.json',receipt,True)
    return receipt


def validate_admission(path):
    validate_runtime_environment()
    r=small_json(REGISTRATION,REG_SHA);a=small_json(path)
    policy=small_json(POLICY,a.get('dispatch_policy_sha256'))
    if not a.get('dispatch_policy_sha256') or socket.gethostname() not in policy['allowed_aliases']:
        raise ValueError('Fresh pinned allowed-host dispatch policy required')
    self_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_hashes={n:hashlib.sha256((HELPERS/n).read_bytes()).hexdigest() for n in ('supervise_once.py','prepare_bindings.py')}
    if (a.get('status')!='APPROVED' or a.get('registration_sha256')!=REG_SHA
        or a.get('source_sha')!=SOURCE or a.get('wrapper_sha256')!=self_hash
        or a.get('helper_sha256')!=helper_hashes or a.get('runtime_environment')!=RUNTIME_ENVIRONMENT
        or a.get('host')!=socket.gethostname()
        or a.get('host')!=r['resource']['host'] or a.get('gpu_uuid')!=r['resource']['gpu_uuid']
        or a.get('valid_until_unix',0)<time.time()+WALL):raise ValueError('Exact unexpired Lead admission required')
    for kind in ('correctness','resources'):
        pin=a['reviews'][kind];review=small_json(pin['path'],pin['sha256'])
        if (review.get('status')!='APPROVE_GRADIENT_BALANCE_EXECUTION' or review.get('review_kind')!=kind
            or review.get('reviewer') in (None,'','/root/road_proxy_systems_review')
            or review.get('source_sha')!=SOURCE or review.get('registration_sha256')!=REG_SHA
            or review.get('wrapper_sha256')!=self_hash or review.get('runtime_environment')!=RUNTIME_ENVIRONMENT):
            raise ValueError('Independent exact '+kind+' review required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True,timeout=10).strip()!=SOURCE:
        raise ValueError('Frozen runtime changed')
    if subprocess.run(['git','diff','--quiet','HEAD'],cwd=RUNTIME,timeout=10).returncode:raise ValueError('Dirty frozen runtime')
    cpus=a.get('cpu_affinity')
    if not isinstance(cpus,list) or len(cpus)!=2 or len(set(cpus))!=2 or not set(cpus)<=os.sched_getaffinity(0):
        raise ValueError('Two available CPU IDs required')
    # Lead, not this wrapper, frees positively identified pressure workers.
    apps=subprocess.check_output(['nvidia-smi','-i',a['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
    memory=int(subprocess.check_output(['nvidia-smi','-i',a['gpu_uuid'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    if apps or memory>128:raise ValueError('Registered GPU is not empty; never preempt real/unknown tasks')
    return r,a


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission');p.add_argument('--execute',action='store_true')
    p.add_argument('--fixed-child',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if args.fixed_child:return child_main()
    if not args.execute:
        print(json.dumps({'status':'NOT_RUN','registration':str(REGISTRATION),'registration_sha256':REG_SHA,
            'independent_reviews_required':True,'runtime_environment':RUNTIME_ENVIRONMENT,
            'wrapper_author':'/root/road_proxy_systems_review'}));return 0
    if not args.admission:raise ValueError('Explicit independently reviewed Lead admission required')
    # All preflight waits count toward the same 1800-second allocation. Native
    # git/nvidia helpers have separate 10-second subprocess timeouts.
    old_alarm=signal.getsignal(signal.SIGALRM)
    def preflight_timeout(*_):raise TimeoutError('Fixed launch preflight exceeded 60 seconds')
    signal.signal(signal.SIGALRM,preflight_timeout);signal.setitimer(signal.ITIMER_REAL,60)
    try:r,a=validate_admission(args.admission)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
    with ExitStack() as stack:
        GPU_LOCK.parent.mkdir(parents=True,exist_ok=True)
        gpu_lock=stack.enter_context(GPU_LOCK.open('a+b'))
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=stack.enter_context((REGISTRATION.parent/'RUN.lock').open('a+b'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if Path(r['output']).exists():raise ValueError('Diagnostic output already exists; no retry')
        ATTEMPT.mkdir(exist_ok=False)
        write_json(ATTEMPT/'ADMISSION.json',a,True)
        env=child_environment(os.environ,a,lock.fileno())
        argv=['taskset','-c',','.join(map(str,a['cpu_affinity'])),PYTHON,str(Path(__file__).resolve()),'--fixed-child']
        remaining=WALL-(time.monotonic()-started)
        if remaining<40:raise TimeoutError('No diagnostic/cleanup budget remains')
        receipt=supervise(argv,attempt=ATTEMPT,output=r['output'],env=env,pass_fds=(lock.fileno(),),wall=remaining)
        print(json.dumps(receipt));return 0 if receipt['status']=='EXITED' else 1


if __name__=='__main__':raise SystemExit(main())
