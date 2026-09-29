"""Finite registered main run; evaluate fixed milestones and retain exact resume state."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('registration','training-worktree','campaign-root','local-root','qwen','sources',
                'run-id','master-addr','dev-data','devkit','metric-index','scoring-python'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--resume-controller',action='store_true')
    a=p.parse_args();root=Path(a.campaign_root);reg=read(a.registration);work=Path(a.training_worktree)
    if subprocess.check_output(['git','status','--porcelain'],cwd=work).strip() or subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()!=reg['training_source_sha']:
        raise ValueError('Training source must be the registered clean checkout')
    if a.run_id not in reg['runs']:raise ValueError('Unregistered main run')
    out=root/'priority_queues'/(a.run_id+'.json');out.parent.mkdir(exist_ok=True)
    lock=out.with_suffix('.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract={'registration_sha256':file_sha256(a.registration),'training_source':reg['training_source_sha']}
    if out.exists():
        state=read(out)
        if not a.resume_controller or state['contract']!=contract:raise ValueError('Existing controller needs unchanged explicit resume')
    else:state={'contract':contract,'events':[]}
    state.update(status='RUNNING',pid=__import__('os').getpid(),controller_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    def event(value):state['events'].append({'time':time.time(),**value});atomic_json(out,state)
    def stopped():return (root/'STOP_REQUESTED').exists() or charged_gpu_hours(root)>=reg['gpu_hours_cap']
    def execute(label,command,cwd):
        log=root/'logs'/(label+'_'+str(time.time_ns())+'.log')
        with log.open('x') as stream:proc=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
        event({'event':'START','label':label,'pid':proc.pid,'command':command,'log':str(log)})
        while proc.poll() is None:time.sleep(5)
        event({'event':'EXIT','label':label,'code':proc.returncode})
        if proc.returncode:raise RuntimeError(label+' failed; see '+str(log))
    run=root/'students'/a.run_id;spec=reg['runs'][a.run_id]
    try:
        for endpoint in reg['development_updates']:
            if stopped():state['status']='PAUSED';atomic_json(out,state);return
            status=read(run/'status.json') if (run/'status.json').exists() else None
            if not status or status['completed']<endpoint:
                if status and status['status']!='PAUSED':raise ValueError('Refuse to duplicate active/failed run')
                command=[sys.executable,'-m','tools.full_foresight.distributed_job','--campaign-root',str(root),
                    '--local-root',a.local_root,'--qwen',a.qwen,'--sources',a.sources,'--run-id',a.run_id,
                    '--candidate',spec['candidate'],'--nodes',str(reg['nodes']),'--master-addr',a.master_addr,
                    '--master-port',str(a.master_port),'--scope','formal','--updates',str(reg['updates']),
                    '--stop-after',str(endpoint),'--max-seconds','216000','--registration',a.registration,
                    '--campaign-gpu-hours',str(reg['gpu_hours_cap'])]
                if status:command.append('--resume')
                execute('train_'+a.run_id+'_to'+str(endpoint),command,work)
                status=read(run/'status.json')
                if status['completed']!=endpoint or status['status'] not in ('PAUSED','COMPLETE') or stopped():
                    state['status']='PAUSED';atomic_json(out,state);return
            label=a.run_id+'_dev'+str(endpoint)+'_seed42';export=root/'evaluations'/label
            if not all((export/f'shard_{rank}.json').exists() and read(export/f'shard_{rank}.json')['status']=='complete' for rank in range(8)):
                stamp=str(time.time_ns());allocation=root/'allocations'/(label+'_'+stamp+'.json')
                execute(label,[sys.executable,'-m','tools.foresight.run_allocated','--gpus','0,1,2,3,4,5,6,7',
                    '--worktree',str(Path.cwd()),'--record',str(allocation),'--pressure-script','/mnt/project/gpu_stress.py',
                    '--pressure-python','/root/miniconda3/envs/navsim/bin/python','--',sys.executable,
                    '-m','tools.full_foresight.export_development','--training-run',str(run),
                    '--checkpoint-tag',f'milestone_{endpoint:06d}','--current-root',a.dev_data,
                    '--output',str(export),'--campaign-root',str(root),'--run-id',label+'_'+stamp,
                    '--gpus','8','--sampling-seed','42','--campaign-gpu-hours',str(reg['gpu_hours_cap'])],Path.cwd())
            gate=root/'scoring_protocol_v1.json'
            if not gate.exists() or not read(gate).get('passed'):raise ValueError('Audited official v1 score protocol required')
            scores=root/'scores'/label
            if not (scores/'summary.json').exists():
                cmd=[a.scoring_python,'-m','tools.foresight.score_pdms','--devkit',a.devkit,'--metric-index',a.metric_index,
                    '--current-index',str(Path(a.dev_data)/'index.json'),'--predictions',str(export),'--output',str(scores),
                    '--campaign-root',str(root),'--run-id',label+'_score_'+str(time.time_ns()),'--workers','16']
                if scores.exists():cmd.append('--resume')
                import os
                with (root/'logs'/(label+'_score.log')).open('a') as stream:
                    proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                event({'event':'CPU_SCORE','pid':proc.pid,'command':cmd})
            ego=root/'evaluations'/(label+'_ego.json')
            if not ego.exists():execute(label+'_ego',[sys.executable,'-m','tools.foresight.evaluate_ego','--predictions',str(export),'--current-root',a.dev_data,'--output',str(ego)],Path.cwd())
        state['status']='MAIN_TRAINING_COMPLETE_EVALUATION_TRACKED_SEPARATELY';atomic_json(out,state)
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));atomic_json(out,state);raise


if __name__=='__main__':main()
