"""Finite eight-shard TRAIN support diagnostic owner; no actor updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT,file_sha,allowed_hosts
from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON,probe,gpu_activity_issue
from starVLA.rl.flow_grpo.transactions import atomic_json
from .dispatch import launch,job_state
from .notify import refresh


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--deployment',required=True);a=p.parse_args()
    plan=json.loads(Path(a.deployment).read_text());directory=Path(plan['root']);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parents[2]
    if str(source)!=plan['source'] or not directory.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Wrong fixed support owner')
    state_path=directory/'STATE.json'
    state=json.loads(state_path.read_text())if state_path.exists()else dict(ranks={})
    atomic_json(directory/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        source=str(source),file_sha256=file_sha(__file__),deployment=str(Path(a.deployment).resolve()),started_unix=time.time()))
    for rank in range(8):state['ranks'].setdefault(str(rank),{})
    def run_rank(route):
        rank=route['rank'];node=json.loads(json.dumps(state['ranks'][str(rank)]))
        try:
            return rank_loop(route,node)
        finally:
            atomic_json(directory/f'OWNER_RANK{rank}.json',node)

    def rank_loop(route,node):
        rank=route['rank']
        while True:
            atomic_json(directory/f'OWNER_RANK{rank}.json',node)
            if not node.get('job'):
                if (directory/'STOP_REQUESTED').exists():node['status']='STOPPED_BEFORE_DISPATCH';return
                try:
                    if route['host'] not in allowed_hosts(admit_long_job=True):node['status']='WAITING_AUTHORIZED_RESOURCE';time.sleep(30);continue
                    sample=probe(route['host']);g=sample['gpus'][route['gpu']]
                    if (g['uuid']!=route['uuid'] or g['free_MiB']<route['minimum_free_MiB'] or gpu_activity_issue(g,sample['owners'])
                            or sample['load1']>sample['CPU_affinity']*1.5 or sample['MemAvailable_bytes']<64*2**30):
                        node['status']='WAITING_RESOURCE';time.sleep(30);continue
                    attempt=node.get('attempt',0)+1;name=f'support_probe/rank{rank:02}_attempt{attempt:03}'
                    command=[MODEL_PYTHON,'-u','-m','tools.qwen_world_safety_exploration.probe_support','run',
                        '--manifest',plan['manifest'],'--output',plan['output'],'--rank',str(rank),'--world','8']
                    node.update(attempt=attempt,job=str(ROOT/'jobs'/name),status='DISPATCHING',host=route['host'],gpu=route['gpu'])
                    launch(route['host'],source,command,name,gpus=str(route['gpu']),expected_uuids=[route['uuid']],
                        share_minimum=route['minimum_free_MiB'],lease_slot=route['lease_slot'],worker_source=source)
                except Exception:
                    node.update(status='DISPATCH_REVIEW_REQUIRED',error=traceback.format_exc());return
            receipt=job_state(node['job'])
            if receipt.get('status')=='REFUSED_RESOURCE':
                if receipt.get('returncode')!=75 or not receipt.get('child_not_started'):
                    node.update(status='FAILED_REVIEW_REQUIRED',receipt=receipt);return
                node.setdefault('refused_jobs',[]).append(node.pop('job'));node['status']='RESOURCE_REQUEUE';time.sleep(30);continue
            if 'returncode'not in receipt:node['status']='RUNNING';time.sleep(10);continue
            if receipt['returncode']!=0:node.update(status='FAILED_REVIEW_REQUIRED',receipt=receipt);return
            completed=json.loads((Path(plan['output'])/f'rank{rank}.json').read_text())
            if completed['status']!='COMPLETE' or completed['world']!=8 or len(completed['completed_tokens'])!=39:
                node.update(status='FAILED_REVIEW_REQUIRED',reason='Missing whole fixed shard');return
            node.update(status='COMPLETE',completed_tokens=39);return
    state['status']='RUNNING'
    with ThreadPoolExecutor(max_workers=8)as pool:
        futures=[pool.submit(run_rank,route)for route in plan['routes']]
        while not all(f.done()for f in futures):

            for rank in range(8):
                p=directory/f'OWNER_RANK{rank}.json'
                if p.exists():state['ranks'][str(rank)]=json.loads(p.read_text())
            state['heartbeat_unix']=time.time();atomic_json(state_path,state);time.sleep(5)
        for f in futures:f.result()
    for rank in range(8):state['ranks'][str(rank)]=json.loads((directory/f'OWNER_RANK{rank}.json').read_text())
    if all(n.get('status')=='COMPLETE'for n in state['ranks'].values()):
        from .probe_support import summarize
        result=summarize(plan['output']);state.update(status='COMPLETE',summary=str(Path(plan['output'])/'SUMMARY.json'),count=result['count'])
    else:state['status']='FINISHED_WITH_REVIEW_REQUIRED'
    state['finished_unix']=time.time();atomic_json(state_path,state);refresh(state_path,'Finite TRAIN support diagnosis completed or requires review')

if __name__=='__main__':main()
