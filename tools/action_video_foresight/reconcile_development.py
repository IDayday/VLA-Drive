"""Recover failed SSH scoring locally without changing any trainer or old evidence.

Only complete, registered FP32 development exports are eligible. The original
immutable scoring/ego code and audited CPU environment are reused. Failed old
states remain intact; verified recovery results are independent sidecars.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.local_interaction_mask_v2.score_async import python_tree_digest


def read(path):
    return json.loads(Path(path).read_text())


def expected_identity(plan, arm, update):
    return {'plan':plan['identity'],'arm':arm,'update':update,
        'evaluation_source':plan['training_source_sha'],
        'purpose':'registered complete development; not Navtest'}


def complete_export(plan, arm, update, bank):
    manifest=read(bank/'identity.json')
    cp=manifest['checkpoint'];protocol=manifest['protocol']
    # The inherited checkpoint field "arm" records the C1 visual candidate,
    # whereas the formal registration records the S0--S4 experimental arm.
    root=Path(plan['campaign_root']);run_id=plan['runs'][arm]['run_id']
    registration_path=root/'registrations'/(run_id+'.json')
    registration=read(registration_path);training=read(root/'students'/run_id/'identity.json')
    if (training['identity']!=identity_hash({k:v for k,v in training.items() if k!='identity'})
        or training['registration_sha256']!=file_sha256(registration_path)
        or registration['arm']!=arm or registration['run_id']!=run_id
        or registration['training_source_sha']!=plan['training_source_sha']
        or cp['run_identity']!=training['identity']):
        raise ValueError('Export is not bound to the registered experimental arm')
    if (cp['completed']!=update or cp['arm']!=training['candidate']['name'] or cp['scope']!='formal'
        or cp['training_source_sha']!=plan['training_source_sha']
        or manifest['current_identity']['split']!='dev' or manifest['limit']
        or manifest['world_size']!=len(plan['runs'][arm]['gpus'])):
        raise ValueError('Foreign or partial development export')
    if (protocol['precision']!='FP32' or protocol['tf32']
        or protocol['future_conditioning'] or protocol['scorer'] is not None
        or protocol['candidates_per_scene']!=1 or protocol['steps']!=10
        or not protocol['auxiliary_heads_removed']):
        raise ValueError('Changed current-only deployment protocol')
    index=read(Path(plan['dev_data'])/'index.json')
    if len(index)!=plan['dev_scenes'] or identity_hash(index)!=manifest['current_identity']['index_sha256']:
        raise ValueError('Development population changed')
    signature=identity_hash(manifest)
    for shard in range(manifest['world_size']):
        path=bank/f'shard_{shard}.json'
        if not path.exists():return False
        state=read(path)
        if state['identity_sha256']!=signature:raise ValueError('Export shard identity changed')
        if state['status']!='complete':return False
        if state['completed']!=len(index[shard::manifest['world_size']]):
            raise ValueError('Incomplete exported population')
    return True


def validate_results(plan, score, ego, evaluator):
    if any(not value['valid'] or value['failed'] or value['scenes']!=plan['dev_scenes']
           for value in (score,ego)):
        raise ValueError('Incomplete result; do not filter failed rows')
    if score['evaluator_identity']!=evaluator:
        raise ValueError('Canonical scoring runtime differs from audited complete reference')
    values=[score['PDMS'],*score['metrics'].values(),
        *(ego['groups']['all'][key] for key in ('ADE','FDE','yaw_MAE_rad'))]
    if not all(math.isfinite(value) for value in values):raise ValueError('Nonfinite result')


def recover(plan,arm,update,source,evaluator,workers,sidecars):
    root=Path(plan['campaign_root']);label=plan['runs'][arm]['run_id']+f'_dev{update}_seed42'
    original=root/'evaluations'/(label+'_state.json')
    state_path=sidecars/(label+'_state.json')
    identity=expected_identity(plan,arm,update)
    old=read(original)
    if old['identity']!=identity or old['status']!='FAILED' or 'CalledProcessError(255' not in old.get('error',''):
        raise ValueError('Only identified SSH transport failures are recovered')
    if not complete_export(plan,arm,update,root/'evaluations'/label):
        raise ValueError('CPU recovery requires complete immutable prediction shards')
    record={'identity':identity,'status':'RUNNING','transport_source':source,
        'original_failed_state_SHA256':file_sha256(original),'started_unix':time.time(),
        'recovery_scope':'CPU scoring only; no GPU inference or optimizer updates'}
    atomic_json(state_path,record)
    try:
        attempt=sidecars/label/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=False)
        scores=root/'scores_reconciled_development_v1'/label;scores.parent.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        if not (scores/'summary.json').exists():
            command=[plan['scoring_python'],'-u','-m','tools.foresight.score_pdms',
                '--devkit',plan['devkit'],'--metric-index',plan['metric_index'],
                '--current-index',str(Path(plan['dev_data'])/'index.json'),
                '--predictions',str(root/'evaluations'/label),'--output',str(scores),
                '--campaign-root',str(root),'--run-id',label+'_local_recovery_'+str(time.time_ns()),
                '--workers',str(workers)]
            if scores.exists():command.append('--resume')
            with (attempt/'score.log').open('x') as stream:
                subprocess.run(command,cwd=plan['source_worktree'],env=env,check=True,
                    stdout=stream,stderr=subprocess.STDOUT)
        ego=attempt/'ego'
        with (attempt/'ego.log').open('x') as stream:
            subprocess.run([plan['scoring_python'],'-m','tools.foresight.evaluate_ego',
                '--predictions',str(root/'evaluations'/label),'--current-root',plan['dev_data'],
                '--output',str(ego)],cwd=plan['source_worktree'],env=env,check=True,
                stdout=stream,stderr=subprocess.STDOUT)
        score=read(scores/'summary.json');fit=read(ego/'summary.json')
        validate_results(plan,score,fit,evaluator)
        record.update(status='COMPLETE',scores=str(scores/'summary.json'),ego=str(ego/'summary.json'),
            PDMS_points=100*score['PDMS'],scenes=score['scenes'],failed=score['failed'],ended_unix=time.time())
        atomic_json(state_path,record)
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),ended_unix=time.time());atomic_json(state_path,record)
        raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--reference-score',required=True)
    parser.add_argument('--workers',type=int,default=16);parser.add_argument('--slots',type=int,default=2)
    parser.add_argument('--interval',type=int,default=60);parser.add_argument('--once',action='store_true')
    parser.add_argument('--recovery-directory',default='reconciled_development')
    args=parser.parse_args()
    if not 1<=args.workers<=16 or not 1<=args.slots<=2 or args.interval<10:
        raise ValueError('Bounded CPU-only allocation required')
    if not args.recovery_directory.startswith('reconciled_development') or Path(args.recovery_directory).name!=args.recovery_directory:
        raise ValueError('Use a versioned recovery directory inside this campaign')
    plan=read(args.plan)
    if plan['identity']!=identity_hash({k:v for k,v in plan.items() if k!='identity'}):raise ValueError('Frozen plan changed')
    if socket.gethostname()!=plan['runs']['S4']['hostname']:raise ValueError('Run on audited canonical CPU host')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze recovery source')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['source_worktree'],text=True).strip()!=plan['training_source_sha']
        or subprocess.check_output(['git','status','--porcelain'],cwd=plan['source_worktree']).strip()):
        raise ValueError('Original scoring/training source changed')
    gate=read(plan['scoring_protocol']);reference=read(args.reference_score)
    if not gate['passed'] or file_sha256(plan['metric_index'])!=gate['metric_index_sha256']:
        raise ValueError('Audited cache identity changed')
    if python_tree_digest(Path(plan['devkit'])/'navsim')!=gate['navsim_tree_sha256']:
        raise ValueError('Official evaluation implementation changed')
    if not reference['valid'] or reference['failed'] or reference['scenes']!=plan['dev_scenes']:
        raise ValueError('Complete canonical reference required')
    evaluator=reference['evaluator_identity']
    versions=json.loads(subprocess.check_output([plan['scoring_python'],'-c',
        "import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ('numpy','scipy','shapely')}))"],text=True))
    if versions!=evaluator['runtime_versions']:raise ValueError('Canonical CPU versions changed')
    root=Path(plan['campaign_root']);out=root/args.recovery_directory;out.mkdir(exist_ok=True)
    lock=(out/'observer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract={'plan':plan['identity'],'source':source,'reference_SHA256':file_sha256(args.reference_score),
        'workers':args.workers,'slots':args.slots}
    if (out/'status.json').exists() and read(out/'status.json')['contract']!=contract:
        raise ValueError('Recovery identity changed')
    with metered_run(root,'development_local_reconciliation_'+str(time.time_ns()),0,
        {'kind':'official_CPU_development_recovery','real_optimizer_updates':0}) as (meter,_,save):
        while True:
            pending=[];completed=0
            for arm,spec in sorted(plan['runs'].items()):
                for update in plan['evaluation_updates']:
                    label=spec['run_id']+f'_dev{update}_seed42'
                    original=root/'evaluations'/(label+'_state.json');sidecar=out/(label+'_state.json')
                    if not original.exists():continue
                    old=read(original)
                    if old['identity']!=expected_identity(plan,arm,update):raise ValueError('Foreign original evaluation')
                    if old['status']=='COMPLETE':
                        validate_results(plan,read(old['scores']),read(old['ego']),evaluator)
                        completed+=1;continue
                    if sidecar.exists() and read(sidecar)['status']=='COMPLETE':
                        saved=read(sidecar)
                        if saved['identity']!=expected_identity(plan,arm,update) or saved['original_failed_state_SHA256']!=file_sha256(original):
                            raise ValueError('Recovery provenance changed')
                        validate_results(plan,read(saved['scores']),read(saved['ego']),evaluator)
                        completed+=1;continue
                    if old['status']=='FAILED' and 'CalledProcessError(255' in old.get('error',''):
                        pending.append((arm,update))
            atomic_json(out/'status.json',{'contract':contract,'status':'RUNNING','pid':os.getpid(),
                'completed_registered_evaluations':completed,'pending_transport_recoveries':pending,
                'updated_unix':time.time()})
            with ThreadPoolExecutor(max_workers=args.slots) as executor:
                futures=[executor.submit(recover,plan,arm,update,source,evaluator,args.workers,out) for arm,update in pending]
                for future in futures:
                    try:future.result()
                    except BaseException as error:
                        state=read(out/'status.json');state.update(status='FAILED',error=repr(error),updated_unix=time.time())
                        atomic_json(out/'status.json',state);raise
            save()
            if args.once or completed==len(plan['runs'])*len(plan['evaluation_updates']) or (out/'STOP_REQUESTED').exists():
                state=read(out/'status.json');state.update(status='COMPLETE' if not args.once and not pending else 'PAUSED')
                atomic_json(out/'status.json',state);return
            time.sleep(args.interval)


if __name__=='__main__':main()
