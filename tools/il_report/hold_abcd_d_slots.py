"""CPU-only D reservation, including the previously omitted fourth namespace.

Existing cohost evaluators finish naturally. Locks are acquired as released;
there are no model/process signals or CUDA imports. Release on exact invocation
exit, permission withdrawal, explicit release or guardian termination.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


STOP=False


def halt(*_):
    global STOP
    STOP=True


def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--handoff-state',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'guardian.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(Path(a.source).resolve()))
    from starVLA.rl.qwen_world_compare.contracts import allowed_hosts
    from starVLA.rl.qwen_world_compare.resources import remote
    for sig in(signal.SIGTERM,signal.SIGINT):signal.signal(sig,halt)
    def publish(status,**kw):write(out/'STATE.json',dict(status=status,pid=os.getpid(),at=time.time(),CUDA_contexts=0,model_signals=0,**kw))
    publish('WAITING_FOR_NEW_D_INVOCATION')
    job=None;held=[]
    try:
        while not STOP:
            state=json.loads(Path(a.handoff_state).read_text());row=state.get('architectures',{}).get('D',{})
            if row.get('new_job'):
                job=Path(row['new_job']);break
            if state.get('status')=='FAILED_REVIEW_REQUIRED':publish('HANDOFF_REVIEW_REQUIRED');return
            time.sleep(5)
        if job is None:return
        live=json.loads((job/'RUNNING.json').read_text());host=live['host'];uuids=live['gpu_uuids']
        if live['lease_slot']not in('shared','primary'):raise ValueError('Unknown actual native slot')
        if host not in allowed_hosts():raise PermissionError('D host permission withdrawn')
        slots={'primary':'runtime_gpu_leases','shared':'runtime_gpu_shared_workers',
            'research':'runtime_gpu_research_slots','evaluation':'runtime_gpu_research_evaluation_slots'}
        folders=tuple(folder for slot,folder in slots.items()if slot!=live['lease_slot'])
        wanted=[Path('/mnt/project')/folder/(uuid+'.lock')for folder in folders for uuid in uuids]
        remaining=list(wanted);last_check=0
        while not STOP and not(job/'RESULT.json').exists()and not(out/'RELEASE').exists():
            if host not in allowed_hosts():break
            changed=False
            for path in list(remaining):
                path.parent.mkdir(exist_ok=True);f=path.open('a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:f.close();continue
                held.append(f);remaining.remove(path);changed=True
            if changed or not last_check:
                publish('HELD'if not remaining else'WAITING_FOR_EXISTING_COHOSTS',invocation=str(job),held=len(held),required=len(wanted),
                    namespaces=list(folders),native_slot=live['lease_slot'],all_four_namespaces_accounted=True)
            if time.monotonic()-last_check>=30:
                code="import pathlib,json; p=pathlib.Path('/proc')/str(%d); s=(p/'stat').read_text().split() if p.exists()else[]; print(json.dumps(bool(s and s[2]not in('Z','X')and int(s[21])==%d)))"%(live['worker_pid'],live['worker_start_ticks'])
                if not json.loads(remote(host,['/usr/bin/python3','-c',code],timeout=30)):break
                last_check=time.monotonic()
            time.sleep(2 if remaining else 30)
    finally:
        for f in held:f.close()
        publish('RELEASED',invocation=None if job is None else str(job),released=len(held))


if __name__=='__main__':main()
