"""Reuse qualified leases/owned workers; formal learners occupy disjoint hosts."""
import json
import os
from pathlib import Path
import time
from starVLA.rl.qwen_world_compare.contracts import ROOT,allowed_hosts,file_sha
from starVLA.rl.qwen_world_compare.resources import probe,remote,MODEL_PYTHON,gpu_activity_issue
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json

def leases_available(uuids):
    import fcntl
    locks=[]
    try:
        for root in ('/mnt/project/runtime_gpu_leases','/mnt/project/runtime_gpu_shared_workers'):
            Path(root).mkdir(parents=True,exist_ok=True)
            for uuid in uuids:
                stream=(Path(root)/(uuid+'.lock')).open('a')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:stream.close();return False
                locks.append(stream)
        return True
    finally:
        for stream in locks:stream.close()

def choose_host(*,excluded_uuids=(),minimum=65000,maximum_load_fraction=.85):
    candidates=[]
    for host in allowed_hosts(admit_long_job=True):
        try:
            before=probe(host);time.sleep(.2);after=probe(host)
            if len(after['gpus'])!=8 or not after['worker_python_exists']:continue
            uuids=[g['uuid']for g in after['gpus']]
            if uuids!=[g['uuid']for g in before['gpus']]or set(uuids)&set(excluded_uuids):continue
            if after['owners']or before['owners']:continue
            if any(gpu_activity_issue(g,s['owners'])for s in(before,after)for g in s['gpus']):continue
            if min(g['free_MiB']for g in after['gpus']+before['gpus'])<minimum:continue
            if max(g['utilization']for g in after['gpus']+before['gpus'])>5:continue
            if after['load1']>after['CPU_affinity']*maximum_load_fraction:continue
            if after['MemAvailable_bytes']<250*2**30:continue
            # Observe the same advisory locks used by all other World tasks.
            if not leases_available(uuids):continue
            candidates.append((after['load1']/after['CPU_affinity'],host,after))
        except Exception:continue
    if not candidates:return None
    _,host,record=min(candidates);return host,record

def launch(host,source,command,name,*,gpus='',expected_uuids=None,minimum=None):
    directory=ROOT/'jobs'/name;directory.mkdir(parents=True,exist_ok=True)
    if(directory/'DISPATCH.json').exists():raise ValueError('An invocation may only be dispatched once')
    executor=str(Path(source)/'tools/qwen_world_compare/job_worker.py')
    args=[MODEL_PYTHON,executor,'--receipt-root',str(directory),'--gpus',gpus,'--command-json',json.dumps(command),'--cwd',str(source)]
    if expected_uuids is not None:args+=['--expected-gpu-uuids-json',json.dumps(expected_uuids)]
    if minimum:args+=['--minimum-free-mib',str(minimum),'--lease-slot','primary']
    env=dict(WORLD_COMPARE_ROOT=str(ROOT),WORLD_COMPARE_TASK='qwen_world_compare_20261029',PYTHONPATH=str(source),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    code=('import subprocess,json,os;from pathlib import Path;'
        'p=subprocess.Popen('+repr(args)+',cwd='+repr(str(source))+',stdout=open('+repr(str(directory/'worker.log'))+',"ab"),stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**'+repr(env)+'));'
        'print(json.dumps(dict(worker_pid=p.pid,worker_start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().split()[21]))))')
    started=json.loads(remote(host,[MODEL_PYTHON,'-c',code]))
    receipt=dict(host=host,source=str(source),command=command,invocation=name,dispatched_unix=time.time(),executor=executor,executor_sha256=file_sha(executor),**started)
    atomic_json(directory/'DISPATCH.json',receipt);return directory

def job_state(directory):
    directory=Path(directory)
    for name in ['RESULT.json','REFUSED.json','RUNNING.json','DISPATCH.json']:
        p=directory/name
        if p.exists():
            record=json.loads(p.read_text())
            record.setdefault('status','DISPATCHED' if name=='DISPATCH.json' else 'UNKNOWN')
            return record
    raise ValueError('Missing dispatched job receipt')
