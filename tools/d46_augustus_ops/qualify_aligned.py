"""One bounded, explicit qualifier -> fresh formal handoff; filesystem events.

Starts only the two registered independently initialized routes. Does not
restart a failed child or an ambiguous launch. Existing GPU jobs stay intact.
"""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from .owner import read,review_native,head
from .event_wait import wait_for_events
from tools.d44_native_rl.launch import launch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha

ROOT=Path('/mnt/project/d46-priority-rl-20261103')
DIRECTORY=ROOT/'operations/AUGUSTUS_ALIGNED_20261103'
WORKER=ROOT/'runtime/operations_augustus_aligned_v5'
MONITOR=ROOT/'runtime/operations_augustus_aligned_v6'
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SCI='09514510afa29e7fdbce834caffd93f3825eea24'


def guard_new_root(output,receipts):
    if Path(output).exists()and any(Path(output).iterdir())and not receipts:
        raise ValueError('Nonempty formal root without durable existing launch receipts')


def qualification(r,prefix):
    out=Path(r['output']);first=read(DIRECTORY/(prefix+'_QUALIFICATION1_PROOF.json'))
    if first.get('status')!='PASS'or first['updates']!=1 or first['next_inner']!=1:raise ValueError('Actual pending-inner q1 proof required')
    proof=review_native(r,out,2,True)
    if proof['phase']!='READY'or proof['next_inner']!=0 or proof['scheduler_updates']!=2:raise ValueError('Own native q2 incomplete')
    ranks=[]
    for rank in range(r['world']):
        x=read(out/f'result_rank{rank}.json',{})
        if(x.get('updates')!=2 or x.get('qualification')is not True or x.get('status')!='QUALIFICATION_STOPPED'
            or x.get('next_inner')!=0 or x.get('frozen_before')!=x.get('frozen_after')
            or x.get('frozen_before')!=first['rank_results'][rank]['frozen_before']):raise ValueError('Actual frozen/Adam/resume guard failed')
        ranks.append(dict(rank=rank,result_file_sha256=file_sha(out/f'result_rank{rank}.json'),frozen_before=x['frozen_before']))
    if proof['allrank_RNG_sha256']!=first['allrank_RNG_sha256']:raise ValueError('Pending M2 resume unexpectedly consumed RNG')
    proof.update(rank_results=ranks,private_and_global_RNG_unchanged_on_pending_reuse=True)
    atomic_json(DIRECTORY/(prefix+'_QUALIFICATION2_PROOF.json'),proof)
    return dict(status='PASS',q1_file=str(DIRECTORY/(prefix+'_QUALIFICATION1_PROOF.json')),q2_file=str(DIRECTORY/(prefix+'_QUALIFICATION2_PROOF.json')),
        q1_sha256=file_sha(DIRECTORY/(prefix+'_QUALIFICATION1_PROOF.json')),q2_sha256=file_sha(DIRECTORY/(prefix+'_QUALIFICATION2_PROOF.json')),
        same_source=SCI,actual_world=r['world'],allrank_native_B1_parity_zero=True)


def formal_command(r,node):
    return ['/usr/bin/env','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=eth0',PYTHON,'-m','torch.distributed.run',
        '--nnodes='+str(len(r['hosts'])),'--nproc_per_node=8','--node_rank='+str(node),'--master_addr='+r['master_addr'],
        '--master_port='+str(29911 if 'FULL'in r['experiment']else 29912),'--rdzv_conf=timeout=180','--max_restarts=0',
        '-m','tools.d46_augustus.train','--config',r['config'],'--output',r['formal_output']]


def formal(r):
    path=DIRECTORY/'FORMAL_LAUNCHES.json';all_receipts=read(path,{'receipts':[]})
    existing=[x for x in all_receipts['receipts']if r['formal_output']in x['command']]
    guard_new_root(r['formal_output'],existing)
    for node,host in enumerate(r['hosts']):
        job=ROOT/'jobs/AUGUSTUS_ALIGNED_20261103'/r['experiment']/'formal'/f'node{node}'
        command=formal_command(r,node);found=read(job/'LAUNCH.json')
        if found:
            if found['command']!=command or found['host']!=host or found['source_sha']!=SCI:raise ValueError('Mismatched existing formal receipt')
            if not any(x['job']==str(job)for x in all_receipts['receipts']):all_receipts['receipts'].append(found);atomic_json(path,all_receipts)
            continue
        job.mkdir(parents=True,exist_ok=True)
        if(job/'LAUNCH_INTENT.json').exists()or any((job/name).exists()for name in('RUNNING.json','RESULT.json')):
            raise ValueError('Ambiguous durable launch; never guess/repeat Popen')
        atomic_json(job/'LAUNCH_INTENT.json',dict(host=host,command=command,source=r['source'],source_sha=SCI,at=time.time(),fresh_IL46_Adam0=True))
        receipt=launch(host,r['source'],job,command,list(range(8)),share_min_free_mib=62000 if 'FULL'in r['experiment']else 45000,
            worker_source=WORKER,task_root=ROOT,task='d46_priority_rl_20261103')
        all_receipts['receipts'].append(receipt);atomic_json(path,all_receipts)


def register_evaluation(routes):
    old=read(ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103/STATE.json')['routes']
    combined=dict(old)
    for exp,r in routes.items():
        combined[exp]=dict(experiment=exp,phase='formal',formal_output=r['formal_output'],route=r,stages={'formal':{}},status='FORMAL_REGISTERED')
    atomic_json(DIRECTORY/'EVALUATION_ROUTES.json',dict(schema='d46_augustus_qualified_evaluation_routes_v2',routes=combined))


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with(DIRECTORY/'qualification_handoff.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(DIRECTORY/'PLAN.json');routes=plan['routes']
        if len(routes)!=2 or set(r['experiment']for r in routes)!={'AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED'}or plan['maximum_simultaneous_routes']!=2:raise ValueError('Only two explicitly registered routes')
        for r in routes:
            if head(r['source'])!=SCI:raise ValueError('Scientific source changed')
            r.update(formal_output=str(ROOT/'runs'/(r['experiment']+'_v2')),worker_source=str(WORKER),config_sha256=file_sha(r['config']))
        atomic_json(DIRECTORY/'PLAN.json',plan)
        status=read(DIRECTORY/'HANDOFF_STATE.json',dict(status='WAITING_REAL_QUALIFICATION',routes={}))
        proofs=read(DIRECTORY/'QUALIFICATION_PROOFS.json',dict(status='PARTIAL'))
        while True:
            paths=[]
            for r in routes:
                exp=r['experiment'];prefix='FULL'if 'FULL'in exp else'LORA';state=status['routes'].setdefault(exp,{'phase':'WAITING_QUALIFICATION'})
                jobs=[ROOT/'jobs/AUGUSTUS_ALIGNED_20261103'/exp/'qualification_resume_shared'/f'node{n}'for n in range(len(r['hosts']))]
                paths+=jobs
                if state['phase']=='FORMAL_DISPATCHED':continue
                try:
                    results=[read(j/'RESULT.json')for j in jobs]
                    if any(x and x.get('returncode')!=0 for x in results):raise ValueError('Actual qualification child failure; preserve for operator review')
                    if not all(results):continue
                    if exp not in proofs:proofs[exp]=qualification(r,prefix)
                    proofs['status']='PASS'if all(x['experiment']in proofs for x in routes)else'PARTIAL';atomic_json(DIRECTORY/'QUALIFICATION_PROOFS.json',proofs)
                    formal(r);state.update(phase='FORMAL_DISPATCHED',at=time.time(),output=r['formal_output'])
                    register_evaluation({x['experiment']:x for x in routes if status['routes'].get(x['experiment'],{}).get('phase')=='FORMAL_DISPATCHED'})
                except Exception:
                    state.update(phase='REVIEW_REQUIRED',error=traceback.format_exc());status['status']='REVIEW_REQUIRED';atomic_json(DIRECTORY/'HANDOFF_STATE.json',status);raise
            atomic_json(DIRECTORY/'HANDOFF_STATE.json',status)
            if all(s['phase']=='FORMAL_DISPATCHED'for s in status['routes'].values()):break
            wait_for_events(paths,1800)
        command=[PYTHON,'-u','-m','tools.d46_augustus_ops.aligned_owner'];env=dict(os.environ,PYTHONPATH=str(MONITOR),CUDA_VISIBLE_DEVICES='')
        existing=read(DIRECTORY/'MONITOR_DAEMON_LAUNCH.json')
        if existing:
            process=Path('/proc',str(existing['pid']))
            if process.exists()and int((process/'stat').read_text().split()[21])==existing['start_ticks']:
                status.update(status='FORMAL_BOTH_DISPATCHED',at=time.time());atomic_json(DIRECTORY/'HANDOFF_STATE.json',status);return
            raise ValueError('Previous monitor receipt needs explicit review, never duplicate/restart implicitly')
        log=open(DIRECTORY/'aligned_monitor.log','ab',buffering=0)
        child=subprocess.Popen(command,cwd=MONITOR,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(DIRECTORY/'MONITOR_DAEMON_LAUNCH.json',dict(pid=child.pid,start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),
            command=command,source=str(MONITOR),at=time.time()))
        status.update(status='FORMAL_BOTH_DISPATCHED',at=time.time());atomic_json(DIRECTORY/'HANDOFF_STATE.json',status)


if __name__=='__main__':main()
