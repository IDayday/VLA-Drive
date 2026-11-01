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


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['root'])
    if set(plan['temporary_hosts'])&set(plan['fixed_whitelist']):raise ValueError('Deadline must not withdraw fixed whitelist')
    if not plan['guard_unix']<plan['force_unix']<plan['deadline_unix']:raise ValueError('Invalid save/exit window')
    lock=(root/'guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='ARMED',pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),plan=str(a.plan),at_unix=time.time(),new_GPU_forwards=0)
    atomic(root/'OWNER.json',state)
    def wait_until(when):
        while time.time()<when:time.sleep(min(30,when-time.time()))
    wait_until(plan['guard_unix']);fence(plan);first=inspect(plan,'stop','GUARD_STOP_REQUESTS');state.update(status='SAVING_AND_MIGRATING_TO_WHITELIST',at_unix=time.time());atomic(root/'OWNER.json',state)
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
