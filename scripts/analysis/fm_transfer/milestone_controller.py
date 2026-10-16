"""Finite registered SFT milestone queue; never launch/restart a trainer.

One owner, immutable evaluation source, existing dispatch receipts and current
host admission. A failed worker stops the queue for inspection; it is never
silently duplicated or retried. Both grids use the registered single seed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from starVLA.rl.fm_transfer.contracts import ROOT, TASK, CONFIGS, require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
BASELINE_RECEIPT_SHA='985f724ea736ba16cde846424b1f08d0f8f3bd5e4a582f430279b9da0c8c05d1'


def worker_allowed(host):
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if (host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions',{}) or
            host not in policy['task_authorizations'][TASK]['allowed_hosts']):
        raise PermissionError('Evaluation host withdrawn: '+host)


def remote(host, args):
    worker_allowed(host)
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
        shlex.join(args)],text=True,timeout=60)


def live_workers(host, receipts):
    # PID reuse cannot make a dead evaluator look alive. These are read-only
    # checks; never signal another owner or clear GPU memory.
    expected=[dict(pid=r['pid'],start_ticks=r['start_ticks']) for r in receipts]
    program="""import json,pathlib,sys
result=[]
for row in json.loads(sys.argv[1]):
 p=pathlib.Path('/proc')/str(row['pid'])
 try:
  fields=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  alive=int(fields[21])==row['start_ticks'] and fields[2]!='Z' and 'fm_transfer.evaluate_policy' in cmd
 except FileNotFoundError:alive=False
 result.append(alive)
print(json.dumps(result))"""
    return json.loads(remote(host,[PYTHON,'-c',program,json.dumps(expected)]))


def jobs(root):
    for variant,cfg in CONFIGS.items():
        for update in cfg['milestones']:
            for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
                yield dict(variant=variant,update=update,cohort=cohort,
                    checkpoint=str(root/'runs'/f'{variant}_SFT_seed42_v1'/'checkpoints'/f'update{update:06d}'),
                    export=str(root/'exports'/variant/f'SFT_update{update:06d}'),
                    output=str(root/'evaluations'/f'{variant}_SFT_update{update:06d}_{cohort}'))


def initial_A_reused(root, job):
    if job['variant']!='A_DDP_TIME' or job['update']!=0:return False
    path=root/'qualification/A_DDP_TIME/initial_baseline_reuse.json'
    proof=json.loads(path.read_text())
    if file_sha(path)!=BASELINE_RECEIPT_SHA or proof['status']!='PASS':
        raise ValueError('A initial baseline reuse must retain its actual equality evidence')
    reuse=proof['reuse'][job['cohort']]
    if file_sha(Path(reuse['path'])/'manifest.json')!=reuse['manifest_sha256']:
        raise ValueError('Reused initial baseline changed')
    return True


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation-source',required=True)
    p.add_argument('--dispatch-source',required=True);p.add_argument('--host',required=True)
    p.add_argument('--root',default=str(ROOT));p.add_argument('--once',action='store_true');a=p.parse_args()
    require_dispatch();worker_allowed(a.host)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller must be CPU-only')
    root=Path(a.root);control=root/'milestone_controller';control.mkdir(parents=True,exist_ok=True)
    lock=(control/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evaluation=Path(a.evaluation_source).resolve();dispatcher=Path(a.dispatch_source).resolve()
    for source in (evaluation,dispatcher):
        if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
            raise ValueError('Incomplete source snapshot')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    identity=dict(schema='fm_transfer_finite_SFT_evaluation_queue_v1',host=a.host,
        evaluation_source=str(evaluation),dispatch_source=str(dispatcher),
        jobs=list(jobs(root)),script_sha256=file_sha(__file__))
    path=control/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Changed queue; use explicit migration')
    atomic_json(path,identity)
    invocation=dict(start_unix=time.time(),pid=os.getpid(),
        start_ticks=int((Path('/proc')/str(os.getpid())/'stat').read_text().split()[21]))
    atomic_json(control/'owner.json',invocation)
    try:
        while True:
            require_dispatch();worker_allowed(a.host);ready=[];active=[];complete=[];pending=[]
            for job in identity['jobs']:
                if initial_A_reused(root,job):complete.append(dict(**job,reused=True));continue
                output=Path(job['output']);receipts=[json.loads(q.read_text()) for q in sorted(output.glob('dispatches/*.json'))]
                if receipts:
                    if len(receipts)!=8 or {r['rank'] for r in receipts}!=set(range(8)):
                        raise ValueError('Partial dispatch needs explicit recovery, never duplicate all workers: '+str(output))
                    if any(r['host']!=a.host or r['source']!=str(evaluation) for r in receipts):
                        raise ValueError('Foreign source/host in evaluation receipts')
                    statuses=[json.loads(q.read_text()) if q.exists() else None for q in (output/f'rank{i}.json' for i in range(8))]
                    if all(r and r['status']=='ASSIGNED_COMPLETE' for r in statuses):
                        analysis=root/'analysis'/(output.name+'.json')
                        if not analysis.exists():
                            require_dispatch()
                            subprocess.run([PYTHON,'-m','scripts.analysis.simwam_recipe.analyze_evaluation',
                                '--evaluation',str(output),'--output',str(analysis)],cwd=evaluation,env=env,check=True)
                        complete.append(job);continue
                    alive=live_workers(a.host,receipts)
                    if any(not alive[i] and not (statuses[i] and statuses[i]['status']=='ASSIGNED_COMPLETE') for i in range(8)):
                        raise RuntimeError('Evaluation worker failed; inspect preserved evidence: '+str(output))
                    active.append(job);continue
                if (Path(job['checkpoint'])/'COMPLETE').exists():ready.append(job)
                else:pending.append(job)
            status=dict(**invocation,status='RUNNING',complete=complete,active=active,ready=ready,pending=pending,
                observed_unix=time.time())
            if not active and ready:
                job=ready[0];export=Path(job['export'])
                # Export only a sealed model; this does not load/modify a trainer.
                if not export.exists():
                    require_dispatch()
                    subprocess.run([PYTHON,'-m','scripts.analysis.fm_transfer.export_policy','--checkpoint',job['checkpoint'],
                        '--output',str(export)],cwd=evaluation,env=env,check=True)
                free=json.loads(remote(a.host,[PYTHON,'-c',"import subprocess,json; print(json.dumps([int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]))"]))
                if len(free)==8 and min(free)>=24000:
                    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
                        'PYTHONPATH='+str(dispatcher),PYTHON,'-m','scripts.analysis.fm_transfer.dispatch_evaluation',
                        '--source',str(evaluation),'--export',str(export),'--output',job['output'],'--cohort',job['cohort']]
                    response=remote(a.host,command)
                    atomic_json(control/(Path(job['output']).name+'.dispatch.json'),json.loads(response))
                    status['last_dispatched']=job
                else:status['status']='WAITING_FOR_AUTHORIZED_MEMORY_HEADROOM'
            elif not active and pending:status['status']='WAITING_FOR_SEALED_SFT_MILESTONE'
            elif not active and not ready and not pending:status['status']='ALL_REGISTERED_SFT_EVALUATIONS_COMPLETE'
            atomic_json(control/'status.json',status)
            if a.once or status['status']=='ALL_REGISTERED_SFT_EVALUATIONS_COMPLETE':break
            time.sleep(30)
    except BaseException:
        import traceback
        atomic_json(control/'failure.json',dict(**invocation,end_unix=time.time(),error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
