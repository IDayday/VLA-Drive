"""Finite local eight-process CPU-only proposal scoring; no GPU reservation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT,require_dispatch,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .notify import refresh


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--owner-name',choices=('STRUCTURED_PROBE','NOISE_PROFILE_CONTROLS'),default='STRUCTURED_PROBE')
    p.add_argument('--module',choices=('probe_structured','probe_noise_profiles'),default='probe_structured');a=p.parse_args()
    require_dispatch();plan=json.loads(Path(a.plan).read_text())
    expected_queries=14976 if a.module=='probe_structured' else 9984
    if plan['CPU_only'] is not True or plan['world']!=8 or plan['new_official_queries']!=expected_queries:
        raise ValueError('Only the finite registered CPU proposal plan')
    directory=ROOT/'owners'/a.owner_name;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'STATE.json').exists():raise ValueError('Existing CPU invocation requires explicit adoption, not duplicate scoring')
    source=Path(__file__).resolve().parents[2]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=str(source))
    state=dict(status='RUNNING',CPU_only=True,actor_updates=0,GPU_requests=0,query_budget=expected_queries,ranks={},source=str(source),started_unix=time.time())
    atomic_json(directory/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=__file__,source_sha256=file_sha(__file__)))
    children=[]
    for rank in range(8):
        command=[os.sys.executable,'-u','-m','tools.qwen_world_safety_exploration.'+a.module,'run','--manifest',plan['manifest'],'--output',plan['output'],'--rank',str(rank),'--world','8','--cpu-threads','2']
        with (directory/f'rank{rank}.log').open('ab',buffering=0)as log:
            child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['ranks'][str(rank)]=dict(status='STARTED',pid=child.pid,start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),command=command)
        children.append(child);atomic_json(directory/'STATE.json',state)
    while True:
        live=False
        for rank,child in enumerate(children):
            code=child.poll();node=state['ranks'][str(rank)]
            if code is None:live=True;node['status']='RUNNING'
            else:node.update(status='EXITED',returncode=code)
            receipt=Path(plan['output'])/f'rank{rank}.json'
            if receipt.exists():
                r=json.loads(receipt.read_text());node.update(scientific_status=r['status'],completed_scenes=len(r['completed_tokens']))
        state['heartbeat_unix']=time.time();atomic_json(directory/'STATE.json',state)
        if not live:break
        time.sleep(5)
    if any(n.get('returncode')!=0 for n in state['ranks'].values()):
        state['status']='FINISHED_WITH_REVIEW_REQUIRED'
    else:
        from importlib import import_module
        summarize=import_module('tools.qwen_world_safety_exploration.'+a.module).summarize
        result=summarize(plan['output']);state.update(status='COMPLETE',summary=str(Path(plan['output'])/'SUMMARY.json'),summary_sha256=result['sha256'])
    state['finished_unix']=time.time();atomic_json(directory/'STATE.json',state);refresh(directory/'STATE.json','Structured TRAIN proposal diagnosis completed or requires review')

if __name__=='__main__':main()
