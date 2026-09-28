"""Random-initialized GT-only teacher; atomic full state and bounded allocations."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import numpy as np
import torch
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.modules.trajectory_mae.masking import TeacherMaskScheduler
from starVLA.model.modules.trajectory_mae.losses import reconstruction_loss
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from .teacher_runtime import TeacherDataset,inputs,fixed_queries,evaluate_teacher


def rng_state(scheduler):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None,'mask':scheduler.state_dict()}


def restore_rng(state,scheduler):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])
    scheduler.load_state_dict(state['mask'])


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('data','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--max-updates',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--save-every',type=int,default=200);p.add_argument('--eval-epochs',default='0,1,2,4,8,16,24,30')
    p.add_argument('--max-seconds',type=float,default=24*3600);p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--acknowledge-stop',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--deterministic',action='store_true')
    p.add_argument('--campaign-gpu-hours',type=float,default=6000.)
    a=p.parse_args()
    if min(a.batch,a.epochs,a.save_every,a.max_seconds,a.lr)<=0 or min(a.limit,a.max_updates,a.stop_after)<0:raise ValueError('Illegal trainer configuration')
    attempt='attempt_'+str(time.time_ns())
    with metered_run(a.campaign_root,a.run_id+'_'+attempt,int(a.device.startswith('cuda')),{'kind':'gt_mae_teacher'}) as (record,_,save_meter):
        run(a,record,save_meter)


def run(a,record,save_meter):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'RUN.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any(x.name!='RUN.lock' for x in out.iterdir()) and not a.resume:raise FileExistsError('Nonempty run requires explicit resume')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Commit and lock source before training')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dataset=TeacherDataset(a.data,limit=a.limit)
    if a.limit:
        evaluation=dataset;eval_scope='TRAIN_SMALL_FIT'
    else:
        evaluation=TeacherDataset(a.data,'dev');eval_scope='FIXED_LOG_DISJOINT_DEV'
        if {r['log'] for r in dataset.index}&{r['log'] for r in evaluation.index}:raise ValueError('Teacher train/dev leakage')
    steps_per_epoch=math.ceil(len(dataset)/a.batch);updates=a.max_updates or steps_per_epoch*a.epochs
    config={k:v for k,v in vars(a).items() if k not in ('resume','acknowledge_stop','stop_after','max_seconds')}
    identity={'schema':'gt_mae_training_v1','source_sha':source,'data_identity':dataset.identity['identity'],
              'train_index_hash':dataset.index_hash,'eval_index_hash':evaluation.index_hash,'config':config,
              'steps_per_epoch':steps_per_epoch,'updates':updates,'model':{'dim':512,'layers':6,'decoder_layers':2,'heads':8,'steps':8,'xy_scale':20.}}
    identity['identity']=identity_hash(identity)
    if a.resume:
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Resume source/data/config mismatch')
        if not a.acknowledge_stop:raise ValueError('Explicit stop acknowledgment required')
        if (out/'STOP_REQUESTED').exists():(out/'STOP_REQUESTED').rename(out/('STOP_ACKNOWLEDGED_'+str(time.time_ns())))
    else:atomic_json(out/'identity.json',identity)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    if a.deterministic:
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(a.device);model=TrajectoryMAE(**identity['model']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    warmup=min(500,max(1,updates//20))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/warmup)*(.05+.95*.5*(1+math.cos(math.pi*min(step,updates)/updates))))
    roles=TeacherMaskScheduler(a.seed+91);epoch=offset=completed=exposure=0
    if a.resume:
        payload=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        if payload['identity']!=identity['identity']:raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);schedule.load_state_dict(payload['scheduler'])
        epoch,offset,completed,exposure=[payload[k] for k in ('epoch','offset','completed','exposure')]
        # RNG byte tensors must be restored from CPU even when model maps to GPU.
        state=payload['rng'];state['torch']=state['torch'].cpu();state['mask']['rng']=state['mask']['rng'].cpu()
        if state['cuda'] is not None:state['cuda']=state['cuda'].cpu()
        restore_rng(state,roles)
    stop=[False]
    def handler(*_):stop[0]=True
    signal.signal(signal.SIGINT,handler);signal.signal(signal.SIGTERM,handler)
    milestones={int(x) for x in a.eval_epochs.split(',') if x};evaluated=set()
    if (out/'evaluated.json').exists():evaluated=set(json.loads((out/'evaluated.json').read_text()))
    query_path=out/'evaluation_queries.json'
    queries=json.loads(query_path.read_text()) if query_path.exists() else fixed_queries(evaluation)
    if not query_path.exists():atomic_json(query_path,queries)
    begin=time.time();initial_completed=completed
    from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
    def save_checkpoint(tag,immutable=False):
        payload={'identity':identity['identity'],'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),
                 'rng':rng_state(roles),'epoch':epoch,'offset':offset,'completed':completed,'exposure':exposure}
        path=out/(tag+'.pt')
        if immutable and path.exists():raise FileExistsError('Milestone must not be overwritten')
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
        atomic_json(out/'status.json',{'status':record['status'],'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,
            'roles':dict(roles.counts),'source_sha':source,'identity':identity['identity'],'last_checkpoint':path.name,'checkpoint_sha256':file_sha256(path)})
    def evaluate(mark):
        if mark in evaluated:return
        save_checkpoint('milestone_'+str(mark).zfill(3),immutable=True)
        rows,summary=evaluate_teacher(model,evaluation,queries,device)
        summary.update(scope=eval_scope,completed=completed,epoch=epoch)
        atomic_json(out/('metrics_'+str(mark)+'.json'),summary)
        with (out/('queries_'+str(mark)+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
        evaluated.add(mark);atomic_json(out/'evaluated.json',sorted(evaluated))
    if completed==0 and 0 in milestones:evaluate(0)
    while completed<updates:
        if stop[0] or (out/'STOP_REQUESTED').exists() or time.time()-begin>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours or (a.stop_after and completed>=a.stop_after):
            record['status']='PAUSED';save_checkpoint('latest');save_meter();break
        batches=epoch_batches(len(dataset),a.batch,a.seed,epoch)
        if offset==len(batches):
            epoch+=1;offset=0
            if epoch in milestones:evaluate(epoch)
            continue
        ids=batches[offset];batch=dataset.batch(ids,device);target,visible=roles.sample(batch['active'],batch['point_valid'])
        model.train();optimizer.zero_grad(set_to_none=True);start=time.time()
        prediction=model(inputs(batch,target,visible))['xy'];rows=torch.arange(len(ids),device=device)
        loss,count=reconstruction_loss(prediction,batch['future'][rows,target],batch['point_valid'][rows,target])
        if not torch.isfinite(loss):raise FloatingPointError('Invalid teacher loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        probe=model.reconstruct[-1].weight.detach().clone();optimizer.step();schedule.step()
        change=float((model.reconstruct[-1].weight-probe).abs().max())
        if count>0 and change==0:raise RuntimeError('Teacher optimizer made no measured update')
        completed+=1;offset+=1;exposure+=len(ids)
        log={'update':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'loss':float(loss),'valid_xy_coordinates':int(count),
             'grad_norm':float(norm),'parameter_change':change,'lr':optimizer.param_groups[0]['lr'],'seconds':time.time()-start,
             'roles':dict(roles.counts),'peak_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0}
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        record.update(real_optimizer_updates=completed-initial_completed,total_run_updates=completed,inference_scenes=0,sample_presentations=exposure);save_meter()
        if completed%a.save_every==0:save_checkpoint('latest')
    else:
        if offset==len(epoch_batches(len(dataset),a.batch,a.seed,epoch)):epoch+=1;offset=0
        if epoch in milestones:evaluate(epoch)
        record['status']='COMPLETE';save_checkpoint('latest');save_meter()

if __name__=='__main__':main()
