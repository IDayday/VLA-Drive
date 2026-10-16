"""Finite registered SFT/RL milestone queue; never launch/restart a trainer.

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
    if host == require_dispatch():
        # The controller's own authorized host may not have a self-SSH alias.
        # A neutral cwd keeps the explicit dispatch PYTHONPATH authoritative,
        # exactly as on a remote host without a checkout in its login directory.
        return subprocess.check_output(args,cwd='/',text=True,timeout=60)
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


def jobs(root,include_rl=False):
    for variant,cfg in CONFIGS.items():
        for update in cfg['milestones']:
            for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
                yield dict(variant=variant,stage='SFT',update=update,cohort=cohort,
                    checkpoint=str(root/'runs'/f'{variant}_SFT_seed42_v1'/'checkpoints'/f'update{update:06d}'),
                    export=str(root/'exports'/variant/f'SFT_update{update:06d}'),
                    output=str(root/'evaluations'/f'{variant}_SFT_update{update:06d}_{cohort}'))
    if include_rl:
        # RL step0 is exactly the own complete SFT and is already evaluated.
        # No separate root/sample selection is introduced for the RL baseline.
        for variant in CONFIGS:
            for update in (5000,10000,15000,20000):
                for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
                    yield dict(variant=variant,stage='RL',update=update,cohort=cohort,
                        checkpoint=str(root/'runs'/f'{variant}_R_SOURCE_seed42_v1'/'checkpoints'/f'update{update:06d}_inner0'),
                        export=str(root/'exports'/variant/f'RL_update{update:06d}'),
                        output=str(root/'evaluations'/f'{variant}_RL_update{update:06d}_{cohort}'))


def reference_for(root,job):
    if job['stage']=='SFT' and job['update']==0:return None
    if job['stage']=='SFT' and job['variant']=='A_DDP_TIME':
        proof=json.loads((root/'qualification/A_DDP_TIME/initial_baseline_reuse.json').read_text())
        return proof['reuse'][job['cohort']]['path']
    update=CONFIGS[job['variant']]['max_updates'] if job['stage']=='RL' else 0
    return str(root/'evaluations'/f"{job['variant']}_SFT_update{update:06d}_{job['cohort']}")


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


def dispatch_plan(ready,active,hosts):
    """Keep one existing eight-shard job per host, never change a job's identity."""
    if len(set(hosts))!=len(hosts):raise ValueError('Duplicate evaluation host')
    occupied={job['host'] for job in active}
    outputs=[job['output'] for job in ready+active]
    if len(set(outputs))!=len(outputs):raise ValueError('A milestone is already active or duplicated')
    return list(zip(ready,[host for host in hosts if host not in occupied]))


def resource_probe(host):
    program="""import json,subprocess
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
gpus=[dict(uuid=a.strip(),used=int(b),free=int(c)) for a,b,c in (line.split(',') for line in rows.splitlines())]
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
owners=[dict(uuid=u.strip(),visible=(Path('/proc')/p.strip()/'cmdline').exists()) for p,u in (line.split(',') for line in rows.splitlines())]
print(json.dumps(dict(gpus=gpus,owners=owners)))"""
    return json.loads(remote(host,[PYTHON,'-c',program]))


def capacity_available(first,second,expected):
    if (len(first['gpus'])!=8 or len(second['gpus'])!=8 or
            [g['uuid'] for g in first['gpus']]!=expected or [g['uuid'] for g in second['gpus']]!=expected):
        raise ValueError('Host physical GPU identity changed')
    for sample in (first,second):
        for gpu in sample['gpus']:
            owners=[o for o in sample['owners'] if o['uuid']==gpu['uuid']]
            if gpu['free']<24000 or (gpu['used']>0 and not owners) or any(not o['visible'] for o in owners):return False
    return True


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation-source',required=True)
    p.add_argument('--dispatch-source',required=True);p.add_argument('--host',required=True)
    p.add_argument('--additional-hosts',nargs='*',default=[],help='Independent registered milestones may use these authorized hosts')
    p.add_argument('--root',default=str(ROOT));p.add_argument('--once',action='store_true')
    p.add_argument('--include-rl',action='store_true');p.add_argument('--control-dir');a=p.parse_args()
    hosts=[a.host]+a.additional_hosts
    if len(set(hosts))!=len(hosts):raise ValueError('Duplicate evaluation host')
    require_dispatch()
    for host in hosts:worker_allowed(host)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller must be CPU-only')
    root=Path(a.root);control=Path(a.control_dir) if a.control_dir else root/'milestone_controller';control.mkdir(parents=True,exist_ok=True)
    lock=(control/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evaluation=Path(a.evaluation_source).resolve();dispatcher=Path(a.dispatch_source).resolve()
    for source in (evaluation,dispatcher):
        if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
            raise ValueError('Incomplete source snapshot')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    analysis_source=Path(__file__).resolve().parents[3]
    resources={host:resource_probe(host) for host in hosts}
    host_uuids={host:[g['uuid'] for g in value['gpus']] for host,value in resources.items()}
    all_uuids=[u for values in host_uuids.values() for u in values]
    if any(len(values)!=8 for values in host_uuids.values()) or len(set(all_uuids))!=len(all_uuids):
        raise ValueError('Hosts must have eight distinct physical GPUs each')
    identity=dict(schema='fm_transfer_finite_SFT_RL_evaluation_queue_v3',host=a.host,hosts=hosts,
        host_GPU_uuids=host_uuids,
        evaluation_source=str(evaluation),dispatch_source=str(dispatcher),
        analysis_source=str(analysis_source),jobs=list(jobs(root,a.include_rl)),script_sha256=file_sha(__file__))
    path=control/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Changed queue; use explicit migration')
    atomic_json(path,identity)
    invocation=dict(start_unix=time.time(),pid=os.getpid(),
        start_ticks=int((Path('/proc')/str(os.getpid())/'stat').read_text().split()[21]))
    atomic_json(control/'owner.json',invocation)
    try:
        while True:
            require_dispatch()
            for host in hosts:worker_allowed(host)
            ready=[];active=[];complete=[];pending=[]
            for job in identity['jobs']:
                if initial_A_reused(root,job):complete.append(dict(**job,reused=True));continue
                output=Path(job['output']);receipts=[json.loads(q.read_text()) for q in sorted(output.glob('dispatches/*.json'))]
                if receipts:
                    if len(receipts)!=8 or {r['rank'] for r in receipts}!=set(range(8)):
                        raise ValueError('Partial dispatch needs explicit recovery, never duplicate all workers: '+str(output))
                    receipt_hosts={r['host'] for r in receipts}
                    if len(receipt_hosts)!=1 or any(r['source']!=str(evaluation) for r in receipts):
                        raise ValueError('Foreign source/mixed host in evaluation receipts')
                    receipt_host=next(iter(receipt_hosts))
                    statuses=[json.loads(q.read_text()) if q.exists() else None for q in (output/f'rank{i}.json' for i in range(8))]
                    if all(r and r['status']=='ASSIGNED_COMPLETE' for r in statuses):
                        analysis=root/'analysis'/(output.name+'.json')
                        if not analysis.exists():
                            require_dispatch()
                            subprocess.run([PYTHON,'-m','scripts.analysis.simwam_recipe.analyze_evaluation',
                                '--evaluation',str(output),'--output',str(analysis)],cwd=evaluation,env=env,check=True)
                        reference=reference_for(root,job);paired=root/'analysis'/(output.name+'_paired_initial.json' if job['stage']=='SFT' else output.name+'_paired_own_SFT.json')
                        if reference and not paired.exists():
                            require_dispatch()
                            analysis_env=dict(env);analysis_env['PYTHONPATH']=str(analysis_source)
                            subprocess.run([PYTHON,'-m','scripts.analysis.fm_transfer.analyze_evaluation',
                                '--evaluation',str(output),'--reference',reference,'--output',str(paired)],
                                cwd=analysis_source,env=analysis_env,check=True)
                        complete.append(job);continue
                    alive=live_workers(receipt_host,receipts)
                    if any(not alive[i] and not (statuses[i] and statuses[i]['status']=='ASSIGNED_COMPLETE') for i in range(8)):
                        raise RuntimeError('Evaluation worker failed; inspect preserved evidence: '+str(output))
                    active.append(dict(**job,host=receipt_host));continue
                if (Path(job['checkpoint'])/'COMPLETE').exists():ready.append(job)
                else:pending.append(job)
            status=dict(**invocation,status='RUNNING',complete=complete,active=active,ready=ready,pending=pending,
                observed_unix=time.time())
            planned=dispatch_plan(ready,active,hosts)
            for job,host in planned:
                export=Path(job['export'])
                # Export only a sealed model; this does not load/modify a trainer.
                if not export.exists():
                    require_dispatch()
                    subprocess.run([PYTHON,'-m','scripts.analysis.fm_transfer.export_policy','--checkpoint',job['checkpoint'],
                        '--output',str(export)],cwd=evaluation,env=env,check=True)
                first=resource_probe(host);time.sleep(2);second=resource_probe(host)
                if capacity_available(first,second,host_uuids[host]):
                    atomic_json(control/(Path(job['output']).name+'.admission.json'),dict(host=host,time=time.time(),first=first,second=second))
                    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
                        'PYTHONPATH='+str(dispatcher),PYTHON,'-m','scripts.analysis.fm_transfer.dispatch_evaluation',
                        '--source',str(evaluation),'--export',str(export),'--output',job['output'],'--cohort',job['cohort']]
                    response=remote(host,command)
                    atomic_json(control/(Path(job['output']).name+'.dispatch.json'),json.loads(response))
                    status.setdefault('launched_this_tick',[]).append(dict(**job,host=host))
                else:status['status']='WAITING_FOR_AUTHORIZED_MEMORY_HEADROOM'
            if not active and not ready and pending:status['status']='WAITING_FOR_SEALED_REGISTERED_MILESTONE'
            elif not active and not ready and not pending:status['status']='ALL_REGISTERED_EVALUATIONS_COMPLETE'
            atomic_json(control/'status.json',status)
            if (control/'STOP_AFTER_TICK').exists():
                status.update(status='STOPPED_CONTROLLER_ONLY',active_workers_preserved=True)
                atomic_json(control/'status.json',status);break
            if a.once or status['status']=='ALL_REGISTERED_EVALUATIONS_COMPLETE':break
            time.sleep(30)
    except BaseException:
        import traceback
        atomic_json(control/'failure.json',dict(**invocation,end_unix=time.time(),error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
