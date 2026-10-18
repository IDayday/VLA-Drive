"""Finite native IL milestone evaluation queue; never duplicate/restart trainers.

All currently suitable unique GPUs participate. Model and seed remain fixed;
host/shard placement is execution metadata. A failed invocation stops dispatch
for diagnosis rather than dropping scenes or automatically changing protocols.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from starVLA.rl.qwen_sourceflow.contracts import ROOT,require_dispatch,file_sha
from starVLA.rl.qwen_sourceflow.resources import MODEL_PYTHON,available,launch,operator_alive
from starVLA.rl.qwen_sourceflow.loading import inspect_il
from starVLA.rl.flow_grpo.transactions import atomic_json


def jobs(root,il_run):
    for update in (20000,40000,60000,80000,100000):
        yield dict(name=f'IL_update{update:06d}_DEV',update=update,
            checkpoint=str(il_run/'checkpoints'/f'update{update:06d}'),
            output=str(root/'evaluations'/f'IL_update{update:06d}_DEV'),
            summary=str(root/'analysis'/f'IL_update{update:06d}_DEV.json'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',required=True);p.add_argument('--root',default=str(ROOT));p.add_argument('--il-run',required=True);p.add_argument('--control-dir',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    require_dispatch();source=Path(a.source).resolve();root=Path(a.root);control=Path(a.control_dir);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Queue owner must be CPU-only')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Evaluation snapshot must be immutable')
    identity=dict(schema='native_IL_five_DEV_queue_v1',source=str(source),source_sha=sha,jobs=list(jobs(root,Path(a.il_run))))
    if (control/'identity.json').exists() and json.loads((control/'identity.json').read_text())!=identity:raise ValueError('Changed queue identity')
    atomic_json(control/'identity.json',identity)
    atomic_json(control/'owner.json',dict(pid=os.getpid(),start_unix=time.time()))
    try:
        while True:
            require_dispatch();active=[];pending=[];ready=[];complete=[];used=set()
            for job in identity['jobs']:
                statefile=control/(job['name']+'.json')
                if Path(job['summary']).exists():
                    summary=json.loads(Path(job['summary']).read_text())
                    if summary['status']!='COMPLETE' or summary['qualification'] or not summary['full_population']:raise ValueError('Invalid formal DEV summary')
                    complete.append(job);continue
                if statefile.exists():
                    state=json.loads(statefile.read_text());done=True
                    for item in state['workers']:
                        result=Path(item['receipt_dir'])/'result.json'
                        if not result.exists():
                            operator=Path(item['receipt_dir'])/'operator.json'
                            if not operator.exists() or not operator_alive(item,json.loads(operator.read_text())):
                                raise RuntimeError('Partial dispatch or lost worker: '+str(item['receipt_dir']))
                            done=False;used.update(item['gpu_uuids']);continue
                        if json.loads(result.read_text())['status']!='PROCESS_COMPLETE':raise RuntimeError('Worker failed; inspect '+str(result))
                    if done:
                        subprocess.run([MODEL_PYTHON,'-m','scripts.analysis.qwen_sourceflow.summarize_results','--evaluation',job['output'],'--output',job['summary']],cwd=source,check=True)
                        complete.append(job)
                    else:active.append(job)
                    continue
                if (Path(job['checkpoint'])/'COMPLETE').exists():
                    inspect_il(job['checkpoint']);ready.append(job)
                else:pending.append(job)
            if ready:
                routes,inventory=available(used,min_free_MiB=32000)
                atomic_json(control/'last_admission.json',dict(unix=time.time(),inventory=inventory,routes=routes))
                # IL milestones are separated by 20k updates; one complete
                # population uses all available cards, then the next queued one.
                if routes:
                    job=ready.pop(0);workers=[]
                    for rank,route in enumerate(routes):
                        receipt=control/'jobs'/job['name']/f'rank{rank:03d}';receipt.mkdir(parents=True,exist_ok=False)
                        command=[MODEL_PYTHON,'-u','-m','scripts.analysis.qwen_sourceflow.evaluate_policy',
                            '--checkpoint',job['checkpoint'],'--output',job['output'],'--cohort','DEV',
                            '--data',str(root/'data/DEV_v2'),'--rank',str(rank),'--world',str(len(routes))]
                        spec=dict(source=str(source),source_sha=sha,host=route['host'],gpus=[route['gpu']],gpu_uuids=[route['uuid']],
                            min_free_MiB=32000,module='scripts.analysis.qwen_sourceflow.evaluate_policy',command=command,receipt_dir=str(receipt))
                        atomic_json(receipt/'spec.json',spec);workers.append(spec)
                    state=dict(job=job,workers=workers,admitted_unix=time.time(),status='DISPATCHING')
                    # Save intent for every worker before any launch. A partial
                    # dispatch cannot be mistaken for an untouched job on restart.
                    atomic_json(control/(job['name']+'.json'),state)
                    for spec in workers:
                        receipt=Path(spec['receipt_dir']);record=launch(receipt/'spec.json',source);atomic_json(receipt/'operator.json',record)
                    state['status']='RUNNING';atomic_json(control/(job['name']+'.json'),state);active.append(job)
            status=dict(observed_unix=time.time(),complete=complete,active=active,pending=pending,ready=ready,
                status='ALL_REGISTERED_DEV_COMPLETE' if len(complete)==5 else 'RUNNING' if active else 'WAITING_FOR_SEALED_MILESTONE')
            atomic_json(control/'status.json',status)
            if len(complete)==5:
                selection=root/'frozen_IL_parent.json'
                if not selection.exists():
                    subprocess.run([MODEL_PYTHON,'-m','scripts.analysis.qwen_sourceflow.freeze_il','--summaries',*[job['summary'] for job in identity['jobs']],'--output',str(selection)],cwd=source,check=True)
                atomic_json(control/'READY_FOR_RL.json',dict(selection=str(selection),selection_sha256=file_sha(selection),
                    next='Run selected-parent actual source-GRPO qualification/profile, then the two registered 50k lines; qualification weights excluded'))
                break
            if a.once or (control/'STOP_AFTER_TICK').exists():break
            time.sleep(60 if active else 300)
    except BaseException:
        atomic_json(control/'failure.json',dict(time_unix=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
