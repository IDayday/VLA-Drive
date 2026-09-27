"""Longer paired training with bounded CPU cache, exact exposure and guarded resume."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow,actor_mask,training_loss_sums
from tools.joint_world.train_graph import load_samples,current_batch,evaluate
from tools.structured_world.runtime import seed_all
from tools.structured_world_v1p1.budget import start,record


class Corpus:
    def __init__(self, cache, targets, data_root, resident=256):
        self.cache,self.targets,self.data_root=cache,targets,data_root
        self.manifest=json.loads((Path(cache)/'manifest.json').read_text())
        self.records=self.manifest['records'];self.resident=resident;self.memo=OrderedDict()
        if not 1<=resident<=512:raise ValueError('Bounded resident cache required')
        if len({r['token'] for r in self.records})!=len(self.records):raise ValueError('Duplicate scenes')
        h=hashlib.sha256()
        for r in self.records:
            token=r['token']
            for path in [Path(targets)/'targets'/(token+'.pt'),Path(data_root)/'meta/train'/(token+'.pkl')]:
                h.update(token.encode()+hashlib.sha256(path.read_bytes()).digest())
        self.target_fingerprint=h.hexdigest()

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        if index not in self.memo:
            samples,_,_=load_samples(self.cache,self.targets,self.data_root,8,records=[self.records[index]],manifest_override=self.manifest)
            self.memo[index]=samples[0]
            if len(self.memo)>self.resident:self.memo.popitem(last=False)
        self.memo.move_to_end(index)
        return self.memo[index]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','targets','data-root','config','output','ledger','run-id']:p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--all-hidden-probability',type=float,default=.5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resident',type=int,default=256);p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if not 1<=a.epochs<=8 or not 1<=a.batch<=16:raise ValueError('Bounded corpus phase supports at most8 passes,batch16')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=bool(a.resume));cfg=json.loads(Path(a.config).read_text())
    seed_all(a.seed);ds=Corpus(a.cache,a.targets,a.data_root,a.resident)
    if not 1<=len(ds)<=8192:raise ValueError('Corpus cap8192 scenes')
    total_presentations=len(ds)*a.epochs;max_steps=math.ceil(total_presentations/a.batch)
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    identity={'arguments':vars(a),'code_sha':code,'config':cfg,'cache_identity':ds.manifest['identity_sha256'],
              'label_fingerprint':ds.target_fingerprint,'unique_scenes':len(ds),'planned_presentations':total_presentations,
              'max_steps':max_steps,'schedule':'linear warmup min(200,10%) then cosine to10% initial lr1e-4',
              'pdms_training':False,'frozen_upstream':True}
    saved=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    if saved is not None:
        for key in ['code_sha','config','cache_identity','label_fingerprint','planned_presentations','max_steps']:
            if saved['identity'][key]!=identity[key]:raise ValueError('Resume identity mismatch: '+key)
        for key in ['batch','all_hidden_probability','seed','output','run_id']:
            if saved['identity']['arguments'][key]!=vars(a)[key]:raise ValueError('Resume setting mismatch: '+key)
        d=json.loads(Path(a.ledger).read_text());r=next(r for r in d['runs'] if r['id']==a.run_id)
        if r['status']=='running' or r['optimizer_steps']!=saved['step'] or saved['step']>=max_steps:
            raise ValueError('Resume needs terminal exactly-accounted incomplete phase')
    start(a.ledger,a.run_id,max_steps,identity,resume=bool(a.resume));completed=saved['step'] if saved else 0
    try:
        first=ds[0];model=JointTrajectoryFlow(first['cache']['context'].shape[-1],**{k:v for k,v in cfg.items() if k in ['dim','heads','layers','scale_m','trajectory_mode','agent_scale_m']}).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        warmup=min(200,max(1,max_steps//10))
        def factor(step):
            if step<warmup:return (step+1)/warmup
            progress=(step-warmup)/max(1,max_steps-warmup)
            return .1+.9*.5*(1+math.cos(math.pi*min(progress,1.)))
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,factor)
        order=list(range(len(ds)));random.shuffle(order);position=0;presentations=0;seen=set()
        fixed_diagnostics=[ds[i] for i in range(min(64,len(ds)))]
        if saved:
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
            order=saved['order'];position=saved['position'];presentations=saved['presentations'];seen=set(saved['seen'])
            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        else:evaluate(model,fixed_diagnostics,out,0)
        milestones={math.ceil(len(ds)*e/a.batch) for e in [1,2,4,6,8] if e<=a.epochs}|{max_steps}
        (out/'manifest.json').write_text(json.dumps(identity,indent=2))
        for step in range(completed+1,max_steps+1):
            if not record(a.ledger,a.run_id,completed):raise RuntimeError('Combined experiment budget exhausted')
            batch=[]
            for _ in range(min(a.batch,total_presentations-presentations)):
                if position==len(order):random.shuffle(order);position=0
                batch.append(ds[order[position]]);position+=1
            current=current_batch(batch);xy=torch.cat([s['xy'] for s in batch]).cuda();valid=torch.cat([s['valid'] for s in batch]).cuda()
            hidden=actor_mask(len(batch),xy.shape[1],'cuda',all_hidden_probability=a.all_hidden_probability)
            opt.zero_grad(set_to_none=True)
            sums,counts=training_loss_sums(model,xy,valid,hidden,torch.randn_like(xy),torch.rand(len(batch),device='cuda'),**current)
            loss=sum(sums[k]/max(counts[k],1) for k in sums)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite corpus loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();schedule.step();completed=step
            presentations+=len(batch);seen.update(s['cache']['token'] for s in batch)
            row={'step':step,'loss':float(loss.detach()),'task_loss':{k:float(sums[k].detach())/max(counts[k],1) for k in sums},
                 'supervised_coordinates':counts,'lr':opt.param_groups[0]['lr'],'gradient_before_clip':norm,
                 'unique_scenes_seen':len(seen),'sample_presentations':presentations,'effective_epochs':presentations/len(ds),
                 'global_batch':len(batch),'resident_samples':len(ds.memo),'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            record(a.ledger,a.run_id,step)
            if step%100==0:print(json.dumps(row),flush=True)
            if step in milestones or step%250==0 or step==a.stop_after:
                if step in milestones:evaluate(model,fixed_diagnostics,out,step)
                payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'step':step,'identity':identity,
                         'order':order,'position':position,'presentations':presentations,'seen':sorted(seen),
                         'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
                temp=out/'checkpoint.tmp';torch.save(payload,temp);temp.replace(out/f'checkpoint_{step}.pt')
            if a.stop_after and step>=a.stop_after:break
        record(a.ledger,a.run_id,completed,'complete' if completed==max_steps else 'paused')
    except BaseException:record(a.ledger,a.run_id,completed,'failed');raise


if __name__=='__main__':main()
