"""Native train/evaluation dispatch using current grants and finite job receipts.

Evaluation shares identified workloads, reserves physical GPUs before CPU model
loading, and keeps failed attempts. No unknown process is signalled.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import socket
import subprocess
import sys
import time

from starVLA.rl.qwen_world_postcot.contracts import allowed_hosts, fingerprint, file_sha
from starVLA.rl.qwen_world_postcot.resources import probe, gpu_activity_issue, remote

MODEL='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
TASK='qwen_world_rl_optimization_20261030'
TASK_STORAGE=Path('/mnt/project/qwen-world-postcot-20261025/rl_optimization_20261030')
PROCESSES={}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


@contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);yield


def runs(root):
    return json.loads((Path(root)/'RUNS.json').read_text())['runs']


def update_run(root,arm,**values):
    root=Path(root)
    with locked(root/'RUNS.lock'):
        value=runs(root)
        row=next(x for x in value if x['id']==arm);row.update(values)
        row['updated_unix']=time.time();write(root/'RUNS.json',dict(runs=value))
    return row


def operator_source(root):
    return Path(json.loads((Path(root)/'SOURCE_EVALUATION.json').read_text())['source'])


def start_worker(root,job,*,host,source,command,gpus,uuids=None,lease='research',minimum_free=32768):
    root,job,source=Path(root),Path(job),Path(source)
    if host not in allowed_hosts(admit_long_job=True):
        raise PermissionError('Host withdrawn before dispatch: '+host)
    job.mkdir(parents=True,exist_ok=False)
    operator_record=root/'SOURCE_OPERATOR.json'
    worker_source=Path(json.loads(operator_record.read_text())['source']) if operator_record.exists() else source
    worker_entry=(['/usr/bin/python3',str(worker_source/'tools/qwen_world_research/job_worker.py')]
                  if operator_record.exists() else ['/usr/bin/python3','-m','tools.qwen_world_research.job_worker'])
    argv=[*worker_entry,
          '--receipt-root',str(job),'--gpus',','.join(map(str,gpus)),
          '--share-min-free-mib',str(minimum_free),'--command-json',json.dumps(command),'--cwd',str(source)]
    if uuids is not None:argv+=['--expected-gpu-uuids-json',json.dumps(uuids)]
    if lease=='primary':argv+=['--primary-shared-training']
    elif lease=='research':argv+=['--research-shared-slot']
    elif lease=='evaluation':argv+=['--evaluation-shared-slot']
    elif lease!='secondary':raise ValueError('Unknown lease kind')
    invocation=['env','WORLD_POSTCOT_TASK='+TASK,'PYTHONPATH='+str(source),
                'CUBLAS_WORKSPACE_CONFIG=:4096:8',*argv]
    if host!=socket.gethostname().removesuffix('-worker-0'):
        invocation=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(invocation)]
    log=(job/'transport.log').open('ab',buffering=0)
    process=subprocess.Popen(invocation,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();PROCESSES[str(job)]=process
    write(job/'LAUNCH.json',dict(host=host,source=str(source),worker_source=str(worker_source),command=command,gpus=gpus,
          gpu_uuids=uuids,lease=lease,transport_pid=process.pid,
          transport_start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),
          dispatched_unix=time.time(),receipt=str(job)))
    return str(job)


def _pid_wait(launch):
    """Wait on the exact transport process without repeatedly reading journals."""
    try:
        pid=int(launch['transport_pid'])
        ticks=int(Path(f'/proc/{pid}/stat').read_text().split()[21])
        if ticks!=launch['transport_start_ticks']:return False
        if hasattr(os,'pidfd_open'):fd=os.pidfd_open(pid)
        else:
            # Linux x86_64 pidfd_open; operator interpreter is preferred.
            fd=ctypes.CDLL(None,use_errno=True).syscall(434,pid,0)
            if fd<0:return False
        try:select.select([fd],[],[],1800)
        finally:os.close(fd)
        return True
    except (OSError,KeyError,ValueError):return False


def wait_job(job):
    if isinstance(job,dict):job=job.get('receipt') or job.get('job') or job.get('path')
    job=Path(job)
    while not (job/'RESULT.json').exists():
        process=PROCESSES.get(str(job))
        if process is not None:
            try:process.wait(timeout=1800)
            except subprocess.TimeoutExpired:continue
        elif (job/'LAUNCH.json').exists():
            if _pid_wait(json.loads((job/'LAUNCH.json').read_text())):
                if not (job/'RESULT.json').exists():
                    try:
                        launch=json.loads((job/'LAUNCH.json').read_text())
                        p=Path(f"/proc/{launch['transport_pid']}/stat")
                        if p.exists() and p.read_text().split()[2] not in ('Z','X'):continue
                    except (OSError,KeyError,ValueError):pass
            elif not (job/'RESULT.json').exists():
                time.sleep(2)  # receipt publication after process exit
        else:raise ValueError('No exact job launch receipt: '+str(job))
        if not (job/'RESULT.json').exists():
            # Record the operational absence; no scene receives a fake score.
            value=dict(status='TRANSPORT_EXIT_WITHOUT_RESULT',returncode=76,receipt=str(job),
                       at_unix=time.time(),scientific_result_unknown=True)
            write(job/'TRANSPORT_FAILURE.json',value);return value
    value=json.loads((job/'RESULT.json').read_text());value['receipt']=str(job)
    if value.get('returncode')==75 and value.get('child_not_started') is True:
        wait_transport_exit(job)
    return value


def preferred_training_route(row,target_updates):
    """One optional natural-gate preference; unavailable capacity never waits."""
    fallback=(row.get('fallback_host',row['host']),row.get('fallback_lease',row.get('lease','primary')))
    preferred=row.get('preferred_host');reason='not configured'
    if not preferred:return row['host'],row.get('lease','primary'),None,reason
    previous=int(row.get('stage_budget_updates',0))
    if target_updates<int(row.get('from_target',0)):
        return *fallback,None,'before preferred from_target'
    if previous<=0 or target_updates<=previous or int(row.get('completed_updates',0))<previous:
        return *fallback,None,'previous natural target not complete; use fallback'
    lease=row.get('preferred_lease',row.get('lease','primary'))
    lease_roots=dict(primary='/mnt/project/runtime_gpu_leases',
        secondary='/mnt/project/runtime_gpu_shared_workers',research='/mnt/project/runtime_gpu_research_slots')
    try:
        if preferred not in allowed_hosts(admit_long_job=True) or lease not in lease_roots:
            return *fallback,None,'preferred host/slot not currently authorized'
        inventory=probe(preferred)
        gpus={g['index']:g for g in inventory['gpus']}
        if not all(i in gpus for i in range(8)):
            return *fallback,None,'preferred host has fewer than eight visible GPUs'
        selected=[gpus[i] for i in range(8)];owners=inventory['owners']
        if len({g['uuid']for g in selected})!=8 or any(
                g['free_MiB']<32768 or gpu_activity_issue(g,owners)
                or any(not o.get('visible') for o in owners if o['uuid']==g['uuid']) for g in selected):
            return *fallback,None,'preferred GPU headroom/owner accounting unavailable'
        # Test the exact eight admission slots without blocking or writing.
        # The worker independently rechecks UUIDs/owners/headroom under its locks.
        code="""import fcntl,json,sys
from pathlib import Path
locks=[];free=True
try:
 for uuid in json.loads(sys.argv[2]):
  path=Path(sys.argv[1])/(uuid+'.lock')
  if not path.exists():continue
  lock=path.open('r');locks.append(lock)
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:free=False;break
 print(json.dumps(dict(available=free)))
finally:
 for lock in locks:lock.close()
"""
        slots=json.loads(remote(preferred,['/usr/bin/python3','-c',code,lease_roots[lease],
                                          json.dumps([g['uuid']for g in selected])],timeout=15))
        if slots.get('available') is not True:
            return *fallback,None,'preferred admission slots busy'
        inventory=dict(inventory,gpus=selected)
        return preferred,lease,inventory,'preferred eight-GPU route available at natural boundary'
    except Exception as error:
        return *fallback,None,'preferred unavailable; immediate fallback: '+type(error).__name__+': '+str(error)[:300]


def dispatch_training(root,arm,target_updates,resume=None,source=None):
    root=Path(root);r=next(x for x in runs(root) if x['id']==arm)
    source=Path(source or r['source']);host,lease,inventory,route_reason=preferred_training_route(r,target_updates)
    if host not in allowed_hosts(admit_long_job=True):
        raise PermissionError('Training host is no longer authorized: '+str(host))
    jobs=root/'jobs'/arm;attempt=1
    while (jobs/f'formal_to{target_updates:06}_attempt{attempt:03}').exists():attempt+=1
    command=[MODEL,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','-m',r['module'],
             '--config',r['config'],'--manifest',r['manifest'],'--output',r['root'],
             '--il-checkpoint',str(root/'IL_PARENT'),'--parent-selection',str(root/'PARENT_SELECTION.json'),
             '--data',str(root/'data/NAVTRAIN'),'--stop-after-update',str(target_updates)]
    if resume:command+=['--resume',str(resume)]
    inventory=inventory if inventory is not None else probe(host);uuids=[g['uuid'] for g in inventory['gpus']]
    job=start_worker(root,jobs/f'formal_to{target_updates:06}_attempt{attempt:03}',
        host=host,source=source,command=command,gpus=list(range(8)),uuids=uuids,
        lease=lease,minimum_free=32768)
    route_fields={}
    if r.get('preferred_host'):
        write(Path(job)/'ROUTE.json',dict(host=host,lease=lease,reason=route_reason,
            preferred_host=r['preferred_host'],from_target=r.get('from_target',0),
            fallback_host=r.get('fallback_host',r['host']),target_updates=target_updates,
            gpu_uuids=uuids,minimum_free_MiB=32768,at_unix=time.time()))
        route_fields=dict(host=host,lease=lease,last_route_reason=route_reason,
            fallback_host=r.get('fallback_host',r['host']),
            fallback_lease=r.get('fallback_lease',r.get('lease','primary')))
    update_run(root,arm,status='TRAINING_DISPATCHED',job=job,stage_budget_updates=target_updates,**route_fields)
    return job


def _process_alive(launch):
    try:
        p=Path(f"/proc/{launch['transport_pid']}/stat")
        row=p.read_text().split()
        return row[2] not in ('Z','X') and int(row[21])==launch['transport_start_ticks']
    except (OSError,KeyError,ValueError):return False


def wait_transport_exit(job):
    """A refused RESULT can precede transport teardown; never race its retry."""
    job=Path(job);process=PROCESSES.get(str(job))
    if process is not None:
        while True:
            try:process.wait(timeout=1800);return
            except subprocess.TimeoutExpired:continue
    launch=json.loads((job/'LAUNCH.json').read_text())
    while _process_alive(launch):
        if not _pid_wait(launch):time.sleep(2)  # exact-process fallback only


def _eval_routes(root,*,maximum,rank0_extra=True):
    hosts=allowed_hosts();preference=['training-vlawm-zt2','training-vlawm-zt','training-vla-zt','training-vlawm-tsinghua-zt4',
        'training-vlawm-tsinghua-zt','training-rl-zt2','training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt3']
    def read(host):
        try:return host,probe(host)
        except Exception as e:return host,dict(error=str(e))
    with ThreadPoolExecutor(max_workers=max(1,len(hosts))) as pool:inventory=dict(pool.map(read,hosts))
    claims_path=root/'resources/EVALUATION_CLAIMS.json'
    claims=json.loads(claims_path.read_text()) if claims_path.exists() else {}
    for gpu,row in list(claims.items()):
        job=Path(row['receipt'])
        if (job/'RESULT.json').exists():claims.pop(gpu);continue
        if (job/'LAUNCH.json').exists() and not _process_alive(json.loads((job/'LAUNCH.json').read_text())):
            claims.pop(gpu)
    routes=[];seen=set()
    for host in preference:
        if host not in inventory or 'error' in inventory[host]:continue
        evidence=inventory[host]
        for gpu in evidence['gpus']:
            owners=[o for o in evidence['owners'] if o['uuid']==gpu['uuid']]
            required=32768 if rank0_extra and not routes else 20480
            if (gpu['uuid'] in seen or gpu['uuid'] in claims or gpu['free_MiB']<required
                    or any(not o.get('visible') for o in owners)
                    or gpu_activity_issue(gpu,evidence['owners'])):continue
            routes.append(dict(host=host,gpu=gpu['index'],uuid=gpu['uuid'],minimum_free=required))
            seen.add(gpu['uuid'])
            if len(routes)==maximum:break
        if len(routes)==maximum:break
    write(root/'resources/LATEST_EVALUATION_INVENTORY.json',dict(at_unix=time.time(),hosts=inventory,
          routes=routes,withdrawn_hosts=['training-vlawm-tsinghua-zt2']))
    return routes,claims


def dispatch_evaluation(root,label,parent_checkpoint,rl_checkpoint,cohort,manifest,data,baseline=None):
    # Prevent duplicate CPU export callers from dispatching the same census
    # after both observed that its spec did not exist yet.
    with locked(Path(root)/'evaluation_specs'/f'{label}.dispatch.lock'):
        return _dispatch_evaluation(root,label,parent_checkpoint,rl_checkpoint,cohort,manifest,data,baseline)


def _dispatch_evaluation(root,label,parent_checkpoint,rl_checkpoint,cohort,manifest,data,baseline=None):
    root=Path(root);source=operator_source(root)
    output=TASK_STORAGE/'evaluations'/label;export=TASK_STORAGE/'exports'/label
    spec_path=root/'evaluation_specs'/f'{label}.json'
    if spec_path.exists():
        value=json.loads(spec_path.read_text())
        if any(value[k]!=str(v) for k,v in [('parent_checkpoint',parent_checkpoint),('manifest',manifest),('data',data)]):
            raise ValueError('Evaluation label reused with a different policy/population')
        refused=[]
        for rank,job in enumerate(value['jobs']):
            result=Path(job)/'RESULT.json'
            if result.exists():
                row=json.loads(result.read_text())
                if row.get('returncode')==75 and row.get('child_not_started') is True:
                    wait_transport_exit(job)
                    refused.append((rank,Path(job)))
        if not refused:return value
        with locked(root/'resources/EVALUATION_DISPATCH.lock'):
            routes,claims=_eval_routes(root,maximum=len(refused),rank0_extra=any(rank==0 for rank,_ in refused))
            if len(routes)<len(refused):raise BlockingIOError('Refused pre-child shards retain their completed siblings; waiting for capacity')
            value.setdefault('previous_attempts',[])
            for (rank,old),route in zip(refused,routes):
                launch=json.loads((old/'LAUNCH.json').read_text());attempt=2
                while (old.parent/f'rank{rank:03}_attempt{attempt:03}').exists():attempt+=1
                job=old.parent/f'rank{rank:03}_attempt{attempt:03}'
                receipt=start_worker(root,job,host=route['host'],source=Path(value['source']),command=launch['command'],
                    gpus=[route['gpu']],uuids=[route['uuid']],lease='evaluation',minimum_free=route['minimum_free'])
                value['previous_attempts'].append(dict(rank=rank,receipt=str(old),reason='resource refusal before model child'))
                value['jobs'][rank]=receipt
                claims[route['uuid']]=dict(receipt=receipt,label=label,rank=rank,claimed_unix=time.time())
            write(root/'resources/EVALUATION_CLAIMS.json',claims);write(spec_path,value)
        return value
    output.parent.mkdir(parents=True,exist_ok=True);export.parent.mkdir(parents=True,exist_ok=True)
    # CPU export overlaps other training and evaluation; native bytes and strict
    # deployment mapping remain unchanged.
    command=[MODEL,'-m','tools.qwen_world_research.evaluate','--export-only',
        '--parent-checkpoint',str(parent_checkpoint),'--export',str(export),'--output',str(output)]
    if rl_checkpoint:command+=['--rl-checkpoint',str(rl_checkpoint)]
    logroot=root/'exports'/label;logroot.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(source),WORLD_POSTCOT_TASK=TASK,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1')
    if (export/'COMPLETE').exists():
        # A capacity wake-up must not rehash an already published multi-GB
        # export. Each actual GPU restore still performs native strict checks.
        meta_path=export/'export.json'
        seal=json.loads((export/'COMPLETE').read_text())
        if file_sha(meta_path)!=seal['export_sha256']:
            raise ValueError('Published export metadata changed while waiting for capacity')
        from tools.qwen_world_research.evaluate import verify_export_binding
        verify_export_binding(json.loads(meta_path.read_text()),
                              parent_checkpoint=parent_checkpoint,rl_checkpoint=rl_checkpoint)
    else:
        with (logroot/'export.log').open('ab',buffering=0) as log:
            subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    maximum=8 if cohort=='TRAIN_DIAGNOSTIC' else 16
    with locked(root/'resources/EVALUATION_DISPATCH.lock'):
        routes,claims=_eval_routes(root,maximum=maximum)
        minimum=8 if cohort=='NAVTEST' else 4
        if len(routes)<minimum:
            raise BlockingIOError(f'Need {minimum} evaluation shards before freezing a new census layout; '
                                  f'{len(routes)} currently available; keep exact checkpoint waiting')
        jobs=[]
        for rank,route in enumerate(routes):
            job=root/'jobs'/('EVAL_'+label)/f'rank{rank:03}_attempt001'
            command=[MODEL,'-u','-m','tools.qwen_world_research.evaluate',
                '--parent-checkpoint',str(parent_checkpoint),'--export',str(export),
                '--data',str(data),'--manifest',str(manifest),'--cohort',cohort,
                '--output',str(output),'--rank',str(rank),'--world',str(len(routes)),'--seed','2101']
            if rl_checkpoint:command+=['--rl-checkpoint',str(rl_checkpoint)]
            if rank==0:command+=['--native-parity-output',str(output/'native_export_parity.json')]
            receipt=start_worker(root,job,host=route['host'],source=source,command=command,
                gpus=[route['gpu']],uuids=[route['uuid']],lease='evaluation',minimum_free=route['minimum_free'])
            claims[route['uuid']]=dict(receipt=receipt,label=label,rank=rank,claimed_unix=time.time())
            jobs.append(receipt)
        write(root/'resources/EVALUATION_CLAIMS.json',claims)
        spec=dict(label=label,parent_checkpoint=str(parent_checkpoint),rl_checkpoint=str(rl_checkpoint) if rl_checkpoint else None,
            cohort=cohort,manifest=str(manifest),data=str(data),baseline=str(baseline) if baseline else None,
            output=str(output),export=str(export),jobs=jobs,world=len(routes),source=str(source),
            training_bank=str(root/'banks'/f'{label}.json') if cohort=='TRAIN_DIAGNOSTIC' else None)
        write(spec_path,spec)
    return spec


def finish_evaluation(spec):
    root=Path('/mnt/project/qwen-world-rl-optimization-20261030')
    for job in spec['jobs']:
        result=wait_job(job)
        if result.get('returncode')!=0:
            raise RuntimeError('Evaluation shard did not complete; successful scenes retained: '+str(job))
    command=[MODEL,'-m','tools.qwen_world_research.evaluate','--summarize','--output',spec['output']]
    if spec.get('training_bank'):command+=['--training-bank',spec['training_bank']]
    if spec.get('baseline'):command+=['--baseline',spec['baseline']]
    env=dict(os.environ,PYTHONPATH=spec['source'],WORLD_POSTCOT_TASK=TASK,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1')
    subprocess.run(command,cwd=spec['source'],env=env,check=True,stdout=subprocess.DEVNULL)
    result=json.loads((Path(spec['output'])/'SUMMARY.json').read_text())
    if result.get('status')!='COMPLETE':raise ValueError('Only complete fixed-census evaluation can finish')
    result['evidence']=str(Path(spec['output'])/'SUMMARY.json')
    return result
