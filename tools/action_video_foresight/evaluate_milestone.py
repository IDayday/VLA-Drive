"""Current-only FP32 development export and the already audited canonical v1 score."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash


def read(path):
    return json.loads(Path(path).read_text())


def checked_ego_summary(output, scenes):
    value=read(Path(output)/'summary.json')
    if not value['valid'] or value['failed'] or value['scenes']!=scenes:
        raise RuntimeError('Incomplete ego evaluation; keep every failed row')
    if any(not math.isfinite(value['groups']['all'][key]) for key in ('ADE','FDE','yaw_MAE_rad')):
        raise RuntimeError('Nonfinite ego evaluation')
    return value


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--update',type=int,required=True)
    args=parser.parse_args()
    plan=read(args.plan)
    if plan['identity']!=identity_hash({k:v for k,v in plan.items() if k!='identity'}):
        raise ValueError('Frozen campaign changed')
    if args.update not in plan['evaluation_updates']:
        raise ValueError('Unregistered development checkpoint')
    gate=read(plan['scoring_protocol'])
    if not gate['passed'] or file_sha256(plan['metric_index'])!=gate['metric_index_sha256']:
        raise ValueError('Audited full-precision v1 cache required')
    from tools.local_interaction_mask_v2.score_async import python_tree_digest
    if python_tree_digest(Path(plan['devkit'])/'navsim')!=gate['navsim_tree_sha256']:
        raise ValueError('Official v1 implementation changed')
    root=Path(plan['campaign_root']);spec=plan['runs'][args.arm]
    label=spec['run_id']+'_dev'+str(args.update)+'_seed42'
    path=root/'evaluations'/label;path.parent.mkdir(exist_ok=True)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    # A single export wave per host avoids overlapping milestone model loads.
    leases=root/'evaluation_leases';leases.mkdir(exist_ok=True)
    lock=(leases/(spec['hostname']+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    state=root/'evaluations'/(label+'_state.json')
    identity={'plan':plan['identity'],'arm':args.arm,'update':args.update,
              'evaluation_source':plan['training_source_sha'],'purpose':'registered complete development; not Navtest'}
    if state.exists() and read(state).get('status')=='COMPLETE':
        if read(state)['identity']!=identity:
            raise ValueError('Existing evaluation belongs to another model/protocol')
        return
    atomic_json(state,{'identity':identity,'status':'RUNNING','pid':os.getpid()})
    try:
        stamp=str(time.time_ns())
        command=[sys.executable,'-u','-m','tools.full_foresight.export_development',
                 '--training-run',str(root/'students'/spec['run_id']),
                 '--checkpoint-tag',f'milestone_{args.update:06d}',
                 '--current-root',plan['dev_data'],'--output',str(path),
                 '--campaign-root',str(root),'--run-id',label+'_export_'+stamp,
                 '--gpus',str(len(spec['gpus'])),'--sampling-seed','42',
                 '--campaign-gpu-hours','1000000000']
        with (logs/(label+'_export_'+stamp+'.log')).open('x') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        scores=root/'scores'/label;scores.parent.mkdir(exist_ok=True)
        if not (scores/'summary.json').exists():
            cpu=[plan['scoring_python'],'-u','-m','tools.foresight.score_pdms',
                 '--devkit',plan['devkit'],'--metric-index',plan['metric_index'],
                 '--current-index',str(Path(plan['dev_data'])/'index.json'),
                 '--predictions',str(path),'--output',str(scores),
                 '--campaign-root',str(root),'--run-id',label+'_score_'+stamp,'--workers','16']
            if scores.exists():
                cpu.append('--resume')
            remote='cd '+shlex.quote(plan['source_worktree'])+' && '+shlex.join(
                ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',*cpu])
            with (logs/(label+'_score_'+stamp+'.log')).open('x') as stream:
                subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                                plan['canonical_score_host'],remote],
                               check=True,stdout=stream,stderr=subprocess.STDOUT)
        summary=read(scores/'summary.json')
        if not summary['valid'] or summary['failed'] or summary['scenes']!=plan['dev_scenes']:
            raise RuntimeError('Development evaluation incomplete; failed rows remain in the denominator')
        if not math.isfinite(summary['PDMS']) or any(not math.isfinite(v) for v in summary['metrics'].values()):
            raise RuntimeError('Nonfinite official development result')
        ego=root/'evaluations'/(label+'_ego')
        if not (ego/'summary.json').exists():
            subprocess.run([sys.executable,'-m','tools.foresight.evaluate_ego',
                            '--predictions',str(path),'--current-root',plan['dev_data'],
                            '--output',str(ego)],check=True)
        checked_ego_summary(ego,plan['dev_scenes'])
        atomic_json(state,{'identity':identity,'status':'COMPLETE',
            'scores':str(scores/'summary.json'),'ego':str(ego/'summary.json'),'PDMS_points':100*summary['PDMS'],
            'scenes':summary['scenes'],'failed':summary['failed']})
    except BaseException as error:
        atomic_json(state,{'identity':identity,'status':'FAILED','error':repr(error)})
        raise


if __name__=='__main__':
    main()
