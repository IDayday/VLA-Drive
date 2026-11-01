"""One finite administrative lease guard; no model, scorer or training loop."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import hashlib
import signal

POLICY=Path('/mnt/project/server_dispatch_policy.json')
REMOTE=r'''
import hashlib,json,os,signal,time
from pathlib import Path
request=__REQUEST__
roots=tuple(request['owned_roots']);modules=set(request['owned_modules']);rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  stat=(p/'stat').read_text().split()
  if stat[2]in('Z','X'):continue
  raw=(p/'cmdline').read_bytes();args=[x for x in raw.decode().split(chr(0))if x]
  entries=[args[i+1]for i,x in enumerate(args[:-1])if x=='-m']
  if not any(x in modules for x in entries):continue
  env=dict(x.split('=',1)for x in(p/'environ').read_bytes().decode().split(chr(0))if '='in x)
  value=env.get('WORLD_POSTCOT_STOP_FILE');stop=Path(value)if value else None
  if stop is None or stop.name!='STOP_REQUESTED'or not any(str(stop).startswith(root+'/')for root in roots):continue
  row=dict(pid=int(p.name),start_ticks=int(stat[21]),cmd_sha256=hashlib.sha256(raw).hexdigest(),entry=entries,
    stop_file=str(stop),output=args[args.index('--output')+1]if '--output'in args else None,action='OBSERVED')
  if request['mode']=='stop':stop.touch(exist_ok=True);row['action']='COOPERATIVE_STOP_REQUESTED'
  elif request['mode']=='kill':
   # Exact PID/start/command validation through a pidfd closes PID reuse races.
   fd=os.pidfd_open(int(p.name));fresh=(p/'stat').read_text().split()
   if int(fresh[21])!=row['start_ticks']or hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()!=row['cmd_sha256']:raise RuntimeError('Process changed during administrative exit')
   signal.pidfd_send_signal(fd,signal.SIGKILL);os.close(fd);row['action']='FORCED_EXIT_LAST_LEGAL_CHECKPOINT_ONLY'
  rows.append(row)
 except (FileNotFoundError,ProcessLookupError):continue
print(json.dumps(dict(at_unix=time.time(),mode=request['mode'],owned_processes=rows)))
'''


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def fence(plan):
    with Path('/mnt/project/server_dispatch_policy.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);policy=json.loads(POLICY.read_text());now=datetime.now(timezone.utc).isoformat()
        for host in plan['temporary_hosts']:
            policy.setdefault('global_dispatch_prohibitions',{})[host]=dict(new_task_dispatch_allowed=False,utc=now,
                record=str(Path(plan['root'])/'PLAN.json'),reason='User temporary resource window; exit before 2026-11-02 07:00 Asia/Shanghai')
            aliases=set(policy.get('allowed_hosts',{}).get(host,{}).get('aliases',[]));aliases.add(host)
            policy.get('allowed_hosts',{}).pop(host,None);policy['allowed_aliases']=[x for x in policy.get('allowed_aliases',[])if x not in aliases]
        policy['updated_utc']=now;atomic(POLICY,policy)


def inspect(plan,mode,tag):
    results={};root=Path(plan['root'])
    for host in plan['temporary_hosts']:
        request=dict(mode=mode,owned_roots=plan['owned_roots'],owned_modules=plan['owned_modules'])
        program=REMOTE.replace('__REQUEST__',repr(request))
        try:
            value=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6',host,
                '/usr/bin/python3','-c',shlex.quote(program)],text=True,timeout=25)
            results[host]=json.loads(value)
        except Exception as error:results[host]=dict(error_type=type(error).__name__,error=str(error)[:300],physical_exit_verified=False)
    atomic(root/(tag+'.json'),results);return results


def queue_a(plan):
    """Only the existing A IL lineage, on the two unprotected fixed hosts."""
    root=Path(plan['root']);lock=(root/'A_migration_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=Path('/mnt/project/qwen-world-four-frameworks-20261029');run=campaign/'runs/ARCH_A/IL_seed42'
    if(root/'A_MIGRATION_LAUNCH.json').exists():raise FileExistsError('A recovery already dispatched; inspect it instead of duplicating')
    source=Path('/mnt/project/qwen-world-four-frameworks-20261029/runtime/fast_5a2d5b5');ops=plan['ops_source']
    started=time.time();attempt=0
    while time.time()-started<48*3600:
        policy=json.loads(POLICY.read_text());state=json.loads((campaign/'campaign/STATE.json').read_text())
        jobs=[v for v in state['jobs'].values()if v.get('architecture')=='A'and v.get('family')=='A_IL']
        old=max(jobs,key=lambda x:x.get('at',0));receipt_path=Path(old['receipt'])/'RESULT.json'
        if not receipt_path.exists():time.sleep(30);continue
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('status')!='EXITED'or receipt.get('returncode')!=0:
            atomic(root/'A_MIGRATION_QUEUE.json',dict(status='REVIEW_REQUIRED',receipt=str(receipt_path)));return
        latest=json.loads((run/'LATEST.json').read_text());parent=Path(latest['checkpoint']);seal=json.loads((parent/'COMPLETE').read_text())
        if latest['updates']==100000:atomic(root/'A_MIGRATION_QUEUE.json',dict(status='ALREADY_IL100K'));return
        if seal['updates']!=latest['updates']or hashlib.sha256((parent/'files.json').read_bytes()).hexdigest()!=seal['files_sha256']:raise ValueError('A native boundary seal mismatch')
        identity=json.loads((parent/'identity.json').read_text())
        if identity['world']!=8 or identity['qualification']:raise ValueError('Only original full8 formal A lineage')
        for name,digest in identity['source_files'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Scientific source drift')
        peak=0
        for rank in range(8):
            path=run/f'metrics_rank{rank}.jsonl'
            with path.open('rb')as stream:stream.seek(max(0,stream.seek(0,2)-262144));lines=stream.read().splitlines()
            rows=[]
            for line in lines:
                try:rows.append(json.loads(line))
                except ValueError:pass
            peak=max(peak,max(row['peak_allocated']for row in rows[-64:]))
        minimum=int(peak*1.2/1024**2)+512
        route=None
        code="""import sys,json,fcntl
from pathlib import Path
sys.path.insert(0,__SOURCE__)
from tools.qwen_world_compare.job_worker import _admission_sample,_known_owners
from starVLA.rl.qwen_world_postcot.resources import gpu_activity_issue
devices,owners=_admission_sample();ok=True
available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
if available<512*1024**3:ok=False
for i,g in devices.items():
 try:_known_owners(owners,{i:g['uuid']},[i])
 except Exception:ok=False
 if g['free_MiB']<__MINIMUM__ or gpu_activity_issue(g,owners):ok=False
slots=[]
if ok:
 for slot,folder in [('primary','runtime_gpu_leases'),('shared','runtime_gpu_shared_workers')]:
  streams=[];free=True
  for g in devices.values():
   f=(Path('/mnt/project')/folder/(g['uuid']+'.lock')).open('a');streams.append(f)
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:free=False
  for f in streams:f.close()
  if free:slots.append(slot)
print(json.dumps(dict(ok=ok,slots=slots,uuids=[g['uuid']for g in devices.values()])))""".replace('__SOURCE__',repr(ops)).replace('__MINIMUM__',str(minimum))
        for host in ('training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt5'):
            if host not in policy['allowed_hosts']or policy.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed')is False:continue
            try:sample=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6',host,'/usr/bin/python3','-c',shlex.quote(code)],text=True,timeout=20))
            except Exception:continue
            if sample['ok']and sample['slots']:route=dict(host=host,slot=sample['slots'][0],uuids=sample['uuids']);break
        if route is None:
            atomic(root/'A_MIGRATION_QUEUE.json',dict(status='WAITING_FIXED_WHITELIST_CAPACITY',parent=str(parent),minimum_free_MiB=minimum,at_unix=time.time()));time.sleep(60);continue
        attempt+=1;job=campaign/'jobs'/f'A_deadline_native_resume{latest["updates"]:06}_attempt{attempt:03}'
        command=list(receipt['command']);command[command.index('--resume')+1]=str(parent)
        argv=['/usr/bin/python3','-u','-m','tools.qwen_world_compare.job_worker','--receipt-root',str(job),'--gpus','0,1,2,3,4,5,6,7',
            '--expected-gpu-uuids-json',json.dumps(route['uuids']),'--command-json',json.dumps(command),'--cwd',str(source),
            '--share-min-free-mib',str(minimum),'--lease-slot',route['slot']]
        program='import subprocess,os;f=open('+repr(str(root/'A_worker_transport.log'))+',"ab",buffering=0);p=subprocess.Popen('+repr(argv)+',cwd='+repr(ops)+',env=dict(os.environ,PYTHONPATH='+repr(ops)+'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)'
        pid=int(subprocess.check_output(['ssh','-o','BatchMode=yes',route['host'],'/usr/bin/python3','-c',shlex.quote(program)],text=True))
        for _ in range(60):
            if(job/'RUNNING.json').exists()or(job/'RESULT.json').exists():break
            time.sleep(1)
        if(job/'RESULT.json').exists():
            outcome=json.loads((job/'RESULT.json').read_text())
            if outcome.get('status')=='REFUSED_RESOURCE'and outcome.get('child_not_started')and attempt<3:continue
            if outcome.get('returncode')!=0:atomic(root/'A_MIGRATION_QUEUE.json',dict(status='REVIEW_REQUIRED',job=str(job)));return
        if not(job/'RUNNING.json').exists():raise RuntimeError('A worker dispatch ambiguous; never duplicate')
        live=json.loads((job/'RUNNING.json').read_text());handoff=dict(status='DISPATCHED_NATIVE_RECOVERY',job=str(job),parent=str(parent),route=route,worker_pid=pid,at_unix=time.time())
        atomic(root/'A_MIGRATION_LAUNCH.json',handoff)
        # Existing root coordinator is CPU-only; adopt this precise invocation
        # without touching healthy B/C/D children or changing its frozen plan.
        owner=json.loads((campaign/'campaign/OWNER.json').read_text());proc=Path('/proc')/str(owner['pid']);st=(proc/'stat').read_text().split()
        if int(st[21])!=owner['start_ticks']:raise ValueError('Campaign CPU owner changed')
        cpu=[x for x in(proc/'cmdline').read_bytes().decode().split(chr(0))if x];cwd=str((proc/'cwd').resolve())
        if 'tools.qwen_world_compare.campaign'not in cpu:raise ValueError('Foreign CPU owner')
        fd=os.pidfd_open(owner['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
        with(campaign/'campaign/owner.lock').open('a')as cpu_lock:
            limit=time.monotonic()+60
            while True:
                try:fcntl.flock(cpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>limit:raise RuntimeError('CPU owner lock held; no duplicate coordinator')
                    time.sleep(1)
            state=json.loads((campaign/'campaign/STATE.json').read_text());entry=dict(old)
            entry.update(logical='A_IL_deadline_resume'+str(latest['updates']),attempt=attempt,command=command,host=route['host'],
                receipt=str(job),status='RUNNING',at=time.time(),uuids=route['uuids'],result=live)
            state['jobs'][job.name]=entry;atomic(campaign/'campaign/STATE.json',state)
            registry_path=campaign/'campaign/IL_EXECUTION_HANDOFFS.json';registry=json.loads(registry_path.read_text());registry['architectures']['A'].update(host=route['host'],uuids=route['uuids'],lease_slot=route['slot'],share_min_free_mib=minimum);atomic(registry_path,registry)
        log=(root/'A_coordinator_adoption.log').open('ab',buffering=0);new=subprocess.Popen(cpu,cwd=cwd,env=dict(os.environ,PYTHONPATH=cwd),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        handoff['new_campaign_cpu']=new.pid;atomic(root/'A_MIGRATION_LAUNCH.json',handoff);return
    atomic(root/'A_MIGRATION_QUEUE.json',dict(status='CAPACITY_REVIEW_REQUIRED',waited_seconds=time.time()-started))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--queue-a',action='store_true');a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['root'])
    if a.queue_a:queue_a(plan);return
    if set(plan['temporary_hosts'])&set(plan['fixed_whitelist']):raise ValueError('Deadline must not withdraw fixed whitelist')
    if not plan['guard_unix']<plan['force_unix']<plan['deadline_unix']:raise ValueError('Invalid save/exit window')
    lock=(root/'guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='ARMED',pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),plan=str(a.plan),at_unix=time.time(),new_GPU_forwards=0)
    atomic(root/'OWNER.json',state)
    def wait_until(when):
        while time.time()<when:time.sleep(min(30,when-time.time()))
    wait_until(plan['guard_unix']);fence(plan);first=inspect(plan,'stop','GUARD_STOP_REQUESTS');state.update(status='SAVING_AND_MIGRATING_TO_WHITELIST',at_unix=time.time());atomic(root/'OWNER.json',state)
    log=(root/'A_migration_queue.log').open('ab',buffering=0)
    queue=subprocess.Popen(['/usr/bin/python3','-u',str(Path(__file__).resolve()),'--plan',str(a.plan),'--queue-a'],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(root/'A_MIGRATION_QUEUE_OWNER.json',dict(pid=queue.pid,at_unix=time.time(),source=str(Path(__file__).resolve())))
    # The immutable native controllers keep the same optimizer/RNG/chain and
    # route to allowed hosts. Exports/evaluation retain their existing cache.
    # No score-based pruning or restart of terminated experiments is allowed.
    for when,mode,tag in [(plan['guard_unix']+600,'stop','REPEATED_STOP_REQUESTS'),
                          (plan['force_unix'],'audit','BEFORE_HARD_EXIT')]:
        wait_until(when);results=inspect(plan,mode,tag)
    if any(v.get('owned_processes')for v in results.values()):
        results=inspect(plan,'kill','FORCED_EXIT_EVIDENCE')
        state['forced_exit_requires_recovery_review']=True
    wait_until(plan['deadline_unix']-30);final=inspect(plan,'audit','FINAL_AUDIT')
    clean=all('owned_processes'in row and not row['owned_processes']for row in final.values())
    state.update(status='EXIT_VERIFIED'if clean else'HOST_EXIT_REVIEW_REQUIRED',at_unix=time.time(),final_audit=str(root/'FINAL_AUDIT.json'))
    atomic(root/'OWNER.json',state);atomic(root/'COMPLETE.json',state)


if __name__=='__main__':main()
