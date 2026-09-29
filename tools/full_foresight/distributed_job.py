"""Bounded one/two-host student allocation using only authorized local/vla-zt2 GPUs.

One torchrun agent per host; a single global training ledger is owned by rank0.
The original pressure allocator verifies ownership and restores reservations.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('campaign-root','local-root','qwen','sources','run-id','candidate','master-addr'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--nodes',type=int,choices=(1,2),default=2)
    p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--scope',choices=('startup','profile','formal'),required=True)
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--registration')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--campaign-gpu-hours',type=float,default=6000)
    a = p.parse_args();root=Path(a.campaign_root);work=Path.cwd();local=Path(a.local_root)
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Immutable clean source required')
    if (root/'STOP_REQUESTED').exists():raise RuntimeError('Explicit campaign stop pending')
    if charged_gpu_hours(root)>=a.campaign_gpu_hours:raise RuntimeError('Campaign budget exhausted')
    if a.scope=='formal':
        reg=json.loads(Path(a.registration).read_text())
        warmup=reg['warmup'];save_every=reg['save_every'];milestones=reg['milestones'];horizon=reg['updates']
    else:warmup=5000;save_every=100;milestones=[0];horizon=100000
    attempt=a.run_id+'_'+str(time.time_ns());out=root/'distributed_jobs'/(attempt+'.json');out.parent.mkdir(exist_ok=True)
    env={'NCCL_SOCKET_IFNAME':'eth0','GLOO_SOCKET_IFNAME':'eth0','NCCL_IB_DISABLE':'1',
         'NCCL_DEBUG':'INFO','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1','OMP_NUM_THREADS':'2',
         'OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    record={'status':'STARTING','attempt':attempt,'run_id':a.run_id,'arguments':vars(a),
            'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'started_unix':time.time(),'network_environment':env,'jobs':[]}
    processes=[]
    for node in range(a.nodes):
        allocation=root/'allocations'/(attempt+f'_node{node}.json')
        cmd=[sys.executable,'-m','tools.full_foresight.run_student','--candidate',a.candidate,
             '--campaign-root',str(root),'--data',str(local/'student_train_v1'),'--image-root',str(local/'images'),
             '--targets',str(local/'targets'),'--index',str(root/'dino_index_v1'),
             '--interaction-root',str(local/'interaction_train_v1'),'--calibration',str(root/'four_loss_calibration_v1.json'),
             '--teacher-verification',str(root/'teacher_reuse_verification.json'),'--qwen',a.qwen,'--sources',a.sources,
             '--run-id',a.run_id,'--gpus',str(8*a.nodes),'--nodes',str(a.nodes),'--node-rank',str(node),
             '--master-addr',a.master_addr,'--master-port',str(a.master_port),'--micro-batch',str(32//(8*a.nodes)),
             '--scope',a.scope,'--updates',str(a.updates),'--schedule-updates',str(horizon),'--warmup',str(warmup),
             '--max-seconds',str(a.max_seconds),'--campaign-gpu-hours',str(a.campaign_gpu_hours),
             '--limit',str(a.limit),'--stop-after',str(a.stop_after),'--save-every',str(save_every),
             '--milestones',','.join(map(str,milestones)),'--deterministic']
        if a.registration:cmd+=['--registration',a.registration]
        if a.resume:cmd+=['--resume','--acknowledge-stop']
        wrapped=['timeout','--signal=TERM','--kill-after=120s',str(a.max_seconds+900)+'s',
                 sys.executable,'-m','tools.foresight.run_allocated','--gpus','0,1,2,3,4,5,6,7',
                 '--worktree',str(work),'--record',str(allocation),'--pressure-script','/mnt/project/gpu_stress.py',
                 '--pressure-python','/root/miniconda3/envs/navsim/bin/python','--',*cmd]
        launch=['env',*[f'{k}={v}' for k,v in env.items()],*wrapped]
        if node:
            launch=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','training-vla-zt2',
                    'cd '+shlex.quote(str(work))+' && '+shlex.join(launch)]
        log=root/'logs'/(attempt+f'_node{node}.log')
        with log.open('xb') as stream:proc=subprocess.Popen(launch,cwd=work,stdout=stream,stderr=subprocess.STDOUT)
        processes.append(proc);record['jobs'].append({'node':node,'host':'local' if node==0 else 'training-vla-zt2',
            'pid':proc.pid,'command':launch,'log':str(log),'allocation':str(allocation)})
    record['status']='RUNNING';atomic_json(out,record)
    while any(proc.poll() is None for proc in processes):
        failed=any(proc.poll() not in (None,0) for proc in processes)
        if failed or (root/'STOP_REQUESTED').exists() or charged_gpu_hours(root)>=a.campaign_gpu_hours:
            run=root/'students'/a.run_id
            if run.exists():(run/'STOP_REQUESTED').write_text('Owned distributed allocation stop/failure/budget; save optimizer boundary\n')
        time.sleep(5)
    record.update(status='COMPLETE' if all(p.returncode==0 for p in processes) else 'FAILED',
                  exit_codes=[p.returncode for p in processes],ended_unix=time.time())
    status=root/'students'/a.run_id/'status.json'
    if status.exists():record['student_status']=json.loads(status.read_text())['status']
    # The trainer starts its global16GPU meter after rendezvous. Account for
    # allocation/release time outside that meter once per eight-GPU host.
    meters=[json.loads(p.read_text()) for p in (root/'runs').glob(a.run_id+'_attempt_*/status.json')]
    matched=[m for m in meters if m['start_unix']>=record['started_unix']]
    for job in record['jobs']:
        path=Path(job['allocation'])
        if not path.exists():continue
        allocation=json.loads(path.read_text());start=allocation['started_unix'];end=allocation.get('ended_unix',record['ended_unix'])
        intervals=[(start,end)] if not matched else [(start,min(end,matched[0]['start_unix'])),(max(start,matched[0].get('end_unix',end)),end)]
        for i,(begin,finish) in enumerate(intervals):
            if finish<=begin:continue
            meter=root/'runs'/(attempt+f'_node{job["node"]}_overhead{i}');meter.mkdir(exist_ok=False)
            atomic_json(meter/'status.json',{'run_id':meter.name,'status':'COMPLETE','kind':'distributed_allocation_overhead',
                'start_unix':begin,'end_unix':finish,'gpu_count':8,'gpu_hours':8*(finish-begin)/3600,
                'real_optimizer_updates':0,'host':job['host'],'run_id_parent':a.run_id})
    atomic_json(out,record);print(json.dumps({'record':str(out),'status':record['status'],'exit_codes':record['exit_codes']}))
    if record['status']=='FAILED':raise RuntimeError('Distributed job failed; node logs and checkpoints preserved')


if __name__=='__main__':main()
