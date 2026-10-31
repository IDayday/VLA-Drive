"""Phase-report resource review: observe first, backfill, avoid native churn.

The isolated probe imports the qualified operational runtime, never this report
worktree's historical model/training contracts. No model is instantiated here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .health import atomic, alive, read

POLICY = Path('/mnt/project/server_dispatch_policy.json')
OPERATOR = Path('/mnt/project/VLA-Drive-qwen-world-time-owner-adoption-20261029')
CADENCE = Path('/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031')
LEASE_ROOTS = dict(primary=Path('/mnt/project/runtime_gpu_leases'),
                   shared=Path('/mnt/project/runtime_gpu_shared_workers'),
                   research=Path('/mnt/project/runtime_gpu_research_slots'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def permission_signature(policy):
    return fingerprint({k:policy.get(k)for k in ('allowed_hosts','global_dispatch_prohibitions','task_holds','task_authorizations')})


def lease_held(path):
    if not path.exists():return False
    with path.open('r+')as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True


def probe_all():
    # Python imports are process-local: do not mix historical report-tree and
    # actual runtime resource contracts in an already imported package.
    sys.path.insert(0,str(OPERATOR))
    from starVLA.rl.qwen_world_postcot.resources import probe
    p=read(POLICY);hosts=[];now=time.time()
    for host,row in p['allowed_hosts'].items():
        if p.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed')is False:continue
        if row.get('temporary')and not row.get('unbounded')and row.get('expires_unix',0)<=now:continue
        hosts.append(host)
    def get(host):
        try:return host,probe(host),None
        except Exception as error:return host,None,str(error)
    snapshots=[];errors={}
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=max(1,len(hosts)))as pool:rows=list(pool.map(get,hosts))
        snapshots.append({h:d for h,d,e in rows if e is None});errors.update({h:e for h,d,e in rows if e})
        if len(snapshots)==1:time.sleep(1)
    locks={g['uuid']:{name:lease_held(root/(g['uuid']+'.lock'))for name,root in LEASE_ROOTS.items()}
           for sample in snapshots for host in sample.values()for g in host['gpus']}
    return dict(authorization_signature=permission_signature(p),captured_unix=time.time(),
        samples=snapshots,errors=errors,leases=locks,hosts=hosts,
        probe_source=str(OPERATOR/'starVLA/rl/qwen_world_postcot/resources.py'))


def gpu_issue(gpu,owners):
    matching=[o for o in owners if o['uuid']==gpu['uuid']]
    if any(not o.get('visible')for o in matching):return 'owner_not_visible'
    if any(o.get('used_MiB')is None for o in matching):return 'owner_memory_unknown'
    known=sum(o['used_MiB']for o in matching)
    if gpu['total_MiB']-gpu['free_MiB']-known>2048:return 'unexplained_allocation'
    if not matching and gpu['utilization']>10:return 'active_without_owner'
    return None


def decide(snapshot,queue):
    before,after=snapshot['samples'];hosts=[];seen=set();idle=[]
    for host,last in after.items():
        first=before.get(host)
        if not first or [g['uuid']for g in first['gpus']]!=[g['uuid']for g in last['gpus']]:continue
        cpu_ratio=last['load1']/max(1,last['CPU_affinity'])
        row=dict(host=host,free_GiB_min=min(g['free_MiB']for g in last['gpus'])/1024,
            cpu_load_per_allowed_cpu=cpu_ratio,available_RAM_GiB=last['MemAvailable_bytes']/1024**3,
            idle_gpus=[],blocked_gpus=[],registered_outputs=sorted(set(o.get('output')or'unknown/hidden'for o in last['owners'])),
            decision='KEEP_CURRENT_PLACEMENT')
        for a,b in zip(first['gpus'],last['gpus']):
            if b['uuid']in seen:continue
            seen.add(b['uuid'])
            issue=gpu_issue(a,first['owners'])or gpu_issue(b,last['owners'])
            if issue:row['blocked_gpus'].append(dict(index=b['index'],reason=issue));continue
            owned=any(o['uuid']==b['uuid']for s in (first,last)for o in s['owners'])
            leased=any(snapshot['leases'].get(b['uuid'],{}).values())
            if (not owned and not leased and min(a['free_MiB'],b['free_MiB'])>=49152
                    and last.get('worker_python_exists')and cpu_ratio<.9 and last['MemAvailable_bytes']>=128*1024**3):
                row['idle_gpus'].append(b['index']);idle.append(b['uuid'])
        if row['idle_gpus']:row['decision']='BACKFILL_EXISTING_AUTHORIZED_QUEUE_FIRST'
        elif cpu_ratio>=.9:row['decision']='KEEP_NATIVE_JOBS; CPU_BUSY_DESPITE_FREE_VRAM'
        hosts.append(row)
    pending=[j for j in queue.get('jobs',[])if j['status']=='PENDING']
    active=sum(j['status']in('RUNNING','DISPATCHING')for j in queue.get('jobs',[]))
    return dict(hosts=hosts,verified_idle_physical_gpus=idle,pending_models=len(pending),active_models=active,
        existing_eval_model_limit=2,
        wake_existing_queue=bool(idle and pending),native_migrations=[],
        unchanged_reason=('No registered pending checkpoint: retain capacity for next qualified milestone; do not repeat scores or invent jobs.'
            if not pending else 'No capacity proven idle across both snapshots; existing scheduler may use separately verified sharing.'if not idle
            else 'Backfill the existing queue through its owner; preserve active native jobs.'),
        placement_reason='Preserve active placement; first backfill registered pending work when capacity is verified. '
          'A selective native move requires a legal boundary, enough remaining work and measured benefit exceeding three times save/load cost. '
          'Unknown owners, zero utilization and free VRAM alone never authorize eviction or migration.')


def wake_cadence(owner):
    if not alive(owner):return 'NO_LIVE_QUEUE_OWNER'
    proc=Path('/proc')/str(owner['pid'])
    args=proc.joinpath('cmdline').read_bytes().decode().split('\0')
    if 'tools.il_report.navtest_cadence'not in args:return 'UNRELATED_OWNER_REFUSED'
    cgt=next(line.split()[1]for line in proc.joinpath('status').read_text().splitlines()if line.startswith('SigCgt:'))
    if not(int(cgt,16)&(1<<(signal.SIGUSR1-1))):return 'WAKE_HANDLER_NOT_INSTALLED; NO_SIGNAL_SENT'
    os.kill(owner['pid'],signal.SIGUSR1)
    return 'EXISTING_QUEUE_CPU_OWNER_WOKEN; NO_NATIVE_SIGNALS'


def review(root,state):
    root=Path(root);folder=root/'resource_reviews_20261031';folder.mkdir(exist_ok=True)
    queue=read(CADENCE/'STATE.json',{'jobs':[]});policy=read(POLICY)
    terminal=[(r['id'],r['status'])for r in state.get('health',{}).get('rl_runs',[])
              if r['status']in('STOPPED_FUTILITY','COMPLETE','PILOT_COMPLETE_REVIEW_REQUIRED')]
    context=fingerprint([permission_signature(policy),terminal,
                         [(j['method'],j['updates'],j['status'])for j in queue['jobs']]])
    previous=read(folder/'LATEST.json',{})
    if previous.get('context')==context and time.time()-previous.get('observed_unix',0)<180:
        return dict(previous,reused_recent_inventory=True)
    with(folder/'review.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            result=subprocess.run(['/usr/bin/python3','-m','tools.il_report.resource_review','--probe-only'],
                cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,WORLD_POSTCOT_TASK='qwen_world_postcot_20261025'),
                capture_output=True,text=True,timeout=100,check=True)
            inventory=json.loads(result.stdout)
            # A collection can complete during the SSH inventory. Use current
            # queue state for actions; the next cycle will refresh its context.
            queue=read(CADENCE/'STATE.json',{'jobs':[]})
            decision=decide(inventory,queue);actions=[]
            unchanged=inventory['authorization_signature']==permission_signature(read(POLICY))
            if unchanged and decision['wake_existing_queue']:
                signature=fingerprint([context,decision['verified_idle_physical_gpus']])
                last=read(folder/'LAST_QUEUE_ACTION.json',{})
                if signature!=last.get('signature')and time.time()-last.get('at',0)>=600:
                    action=wake_cadence(read(CADENCE/'OWNER.json',{}));actions.append(action)
                    atomic(folder/'LAST_QUEUE_ACTION.json',dict(at=time.time(),signature=signature,action=action))
            receipt=dict(schema='phase_resource_review_v1',context=context,observed_unix=time.time(),
                authorization_still_current=unchanged,decision=decision,actions=actions,
                errors=inventory['errors'],native_jobs_paused=0,native_jobs_migrated=0,
                new_experiments=0,science_or_budgets_changed=False)
            path=folder/('review_'+str(time.time_ns())+'.json')
            atomic(path,dict(receipt,inventory=inventory));receipt['receipt']=str(path)
            atomic(folder/'LATEST.json',receipt)
            atomic(Path('/mnt/project/qwen-world-rl-optimization-20261030/resources/RESOURCE_REVIEW_LATEST.json'),receipt)
            return receipt
        except Exception as error:
            failed=dict(status='RESOURCE_REVIEW_FAILED; NO_ALLOCATION_CHANGED',error=str(error),at=time.time(),
                        previous_receipt=previous.get('receipt'),native_jobs_paused=0)
            atomic(folder/'FAILURE.json',failed);return failed


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--probe-only',action='store_true',required=True)
    parser.parse_args();print(json.dumps(probe_all()))
