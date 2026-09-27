"""Matched two-forward ALL/MASK training of the sole joint trajectory decoder."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,flow_loss_sums,normalized_loss
from starVLA.model.modules.joint_scene.masks import role_completion_mask
from tools.joint_local_scene_v3.data import AnnotatedCorpus
from tools.joint_local_scene_v3.runtime import batch_scenes,evaluate,summarize
from tools.local_interaction_mask_v2.foundation import rng_state,restore_rng
from tools.local_interaction_mask_v2.train_foundation import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train','holdout','config','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=['all','mask'],required=True);p.add_argument('--updates',type=int,required=True)
    p.add_argument('--schedule-updates',type=int,required=True);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=200);p.add_argument('--eval-train',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    if a.updates>a.schedule_updates or a.batch<1:raise ValueError('Invalid finite schedule')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists() and not a.resume:raise FileExistsError('Use a new run or explicit resume')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    cfg=json.loads(Path(a.config).read_text());train=AnnotatedCorpus(a.train,a.limit);holdout=AnnotatedCorpus(a.holdout)
    model=JointSceneFlow(**cfg['model']).cuda()
    # Visual projection is reserved for the visual stage; no false claim of training it here.
    model.condition.requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['initial_lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/cfg['warmup_steps'],1)*(.1+.9*.5*(1+math.cos(math.pi*min(s,a.schedule_updates)/a.schedule_updates))))
    noise_rng=torch.Generator(device='cuda').manual_seed(a.seed+101);time_rng=torch.Generator(device='cuda').manual_seed(a.seed+201)
    mask_rng=torch.Generator().manual_seed(a.seed+301)
    identity={'schema_version':3,'stage':'structured_current_GT_state_mechanism_not_camera_planning',
        'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'arguments':{k:v for k,v in vars(a).items() if k not in ('resume','stop_after','output','updates')},'config':cfg,
        'train_identity':train.manifest['identity_sha256'],'holdout_identity':holdout.manifest['identity_sha256'],
        'trainable_parameters':sum(p.numel() for p in parameters),'all_parameters':sum(p.numel() for p in model.parameters()),
        'forwards_per_batch':2,'only_executed_ego':'joint sample slot0','initialization_seed':a.seed}
    step=epoch=offset=presentations=0;supervised={};forward_count=0
    if a.resume:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Mechanism resume identity differs')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step,epoch,offset,presentations,forward_count=[saved[k] for k in ('step','epoch','offset','presentations','forward_count')]
        supervised=saved['supervised'];restore_rng(saved['rng'])
        noise_rng.set_state(saved['noise_rng']);time_rng.set_state(saved['time_rng']);mask_rng.set_state(saved['mask_rng'])
    atomic_json(out/'manifest.json',identity)
    def save():
        torch.save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'forward_count':forward_count,'supervised':supervised,
            'rng':rng_state(),'noise_rng':noise_rng.get_state(),'time_rng':time_rng.get_state(),'mask_rng':mask_rng.get_state()},out/'checkpoint.tmp')
        (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    def assess(tag):
        for name,corpus,limit in [('holdout',holdout,None)]+([('train',train,min(64,len(train)))] if a.eval_train else []):
            rows=evaluate(model,corpus,cfg['sampling_seed'],cfg['sampling_steps'],batch=8,limit=limit)
            with (out/f'{name}_{tag}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');writer.writeheader();writer.writerows(rows)
            atomic_json(out/f'{name}_{tag}.json',summarize(rows))
    if not a.resume:assess('0')
    model.train();paused=False
    while step<a.updates:
        order=np.random.default_rng(np.random.SeedSequence([a.seed,epoch])).permutation(len(train)).tolist()
        ids=order[offset:offset+a.batch];scenes=[train[i] for i in ids]
        start=time.monotonic();g,y,valid=batch_scenes(scenes);optimizer.zero_grad(set_to_none=True)
        losses=[];count_parts=[];role_stats={};loss_values=[]
        for pass_index in range(2):
            hidden=g.active_actor_mask
            if pass_index==1 and a.mode=='mask':
                hidden,statistics=role_completion_mask(valid,g.active_actor_mask,mask_rng)
                role_stats={k:(int(v.sum()) if v.dtype==torch.bool else v.tolist()) for k,v in statistics.items()}
            noise=torch.randn(y.shape,device='cuda',generator=noise_rng)
            tau=torch.rand(len(scenes),device='cuda',generator=time_rng)
            sums,counts=flow_loss_sums(model,y,valid,hidden,noise,tau,g)
            loss=normalized_loss(sums,counts)
            if not torch.isfinite(loss):raise FloatingPointError('Joint flow loss is not finite')
            # Same two forwards/backwards for each arm; no role-conditioned path feeds another planner.
            (loss*.5).backward();loss_values.append(float(loss.detach()));count_parts.append(counts)
            for k,v in counts.items():
                name=('all_hidden' if pass_index==0 or a.mode=='all' else 'role')+'_'+k
                supervised[name]=supervised.get(name,0)+v
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();scheduler.step()
        step+=1;offset+=len(scenes);presentations+=len(scenes);forward_count+=2
        if offset==len(train):epoch+=1;offset=0
        torch.cuda.synchronize()
        row={'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'forward_scene_presentations':presentations*2,
            'forward_count':forward_count,'loss':sum(loss_values)/2,'forward_losses':loss_values,'valid_coordinates_per_forward':count_parts,
            'supervised_coordinates_cumulative':supervised,'role_tasks':role_stats,'gradient_before_clip':norm,
            'seconds':time.monotonic()-start,'samples_per_second':len(scenes)/(time.monotonic()-start),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'lr':optimizer.param_groups[0]['lr']}
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if step%20==0:print(json.dumps(row),flush=True)
        atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
        paused=(out/'STOP_REQUESTED').exists() or (a.stop_after is not None and step>=a.stop_after)
        if step%a.eval_every==0 or step==a.updates:assess(str(step))
        if step%a.eval_every==0 or step==a.updates or paused:save()
        if paused:break
    save();atomic_json(out/'status.json',{'status':'paused' if paused else 'complete','step':step,'epochs':epoch,'presentations':presentations,'forward_count':forward_count,'supervised_coordinates':supervised})


if __name__=='__main__':main()
