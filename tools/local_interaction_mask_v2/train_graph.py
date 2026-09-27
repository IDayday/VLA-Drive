"""Matched local ALL/MASK training; single seed, independent noise and exact resume."""
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
from tools.local_interaction_mask_v2.graph_runtime import LocalCorpus,batch_current,evaluate_graph,summarize,cached_worker_init,task_supervision_statistics
from tools.local_interaction_mask_v2.foundation import rng_state,restore_rng,collate
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow,training_loss_sums
from starVLA.model.modules.joint_world.local_masks import task_masks,stable_noise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','targets','meta-root','holdout','config','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['all','mask'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--schedule-epochs',type=int,default=32)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    if a.batch<1 or a.epochs>a.schedule_epochs:raise ValueError('Invalid finite training plan')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists() and not a.resume:raise FileExistsError('Use new run or explicit resume')
    cached_worker_init(0)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    corpus=LocalCorpus(a.cache,a.targets,a.meta_root)
    spec=json.loads(Path(a.holdout).read_text());holdout=LocalCorpus(spec['cache'],spec['targets'],spec['meta_root'])
    cfg=json.loads(Path(a.config).read_text());dim=corpus[0]['current']['context'].shape[-1]
    model=JointTrajectoryFlow(dim,**{k:v for k,v in cfg.items() if k in ('dim','heads','layers','steps','scale_m','trajectory_mode','agent_scale_m','edge_feature_dim')}).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    horizon=math.ceil(len(corpus)/a.batch)*a.schedule_epochs
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/200,1)*(.1+.9*.5*(1+math.cos(math.pi*min(s,horizon)/horizon))))
    mask_rng=torch.Generator().manual_seed(a.seed+201);time_rng=torch.Generator().manual_seed(a.seed+401)
    identity={'schema_version':2,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'arguments':{k:v for k,v in vars(a).items() if k not in ('output','resume','stop_after','epochs')},'config':cfg,
        'cache_identity':corpus.manifest['identity_sha256'],'labels':corpus.label_fingerprint,
        'holdout_identity':holdout.manifest['identity_sha256'],'holdout_labels':holdout.label_fingerprint,
        'main_seed_count':1,'targets_in_input':False,'noise_seed':a.seed+301,'noise_sample_index':'epoch',
        'normalization':'separate ego/agents global valid hidden coordinate sums','no_old_step_cap':True}
    step=epoch=offset=presentations=0;seen=set()
    if a.resume:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Graph resume identity differs')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        step,epoch,offset,presentations=[saved[k] for k in ('step','epoch','offset','presentations')]
        seen=set(saved['seen']);restore_rng(saved['rng']);mask_rng.set_state(saved['mask_rng']);time_rng.set_state(saved['time_rng'])
    atomic_json(out/'manifest.json',identity)
    def evaluate(tag):
        rows=evaluate_graph(model,holdout,cfg.get('sampling_seed',2037));keys=sorted(set().union(*(r.keys() for r in rows)))
        with (out/f'holdout_{tag}.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
        atomic_json(out/f'holdout_{tag}.json',summarize(rows))
    def save():
        payload={'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                 'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'seen':sorted(seen),
                 'rng':rng_state(),'mask_rng':mask_rng.get_state(),'time_rng':time_rng.get_state(),
                 'resume_boundary':'optimizer step; deterministic epoch order, source-based noise, same cache/config'}
        torch.save(payload,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    if not a.resume:evaluate('0')
    paused=False
    while epoch<a.epochs:
        order=np.random.default_rng(np.random.SeedSequence([a.seed,epoch])).permutation(len(corpus)).tolist()
        batches=[order[i:i+a.batch] for i in range(offset,len(order),a.batch)]
        loader=torch.utils.data.DataLoader(corpus,batch_sampler=batches,num_workers=a.workers,collate_fn=collate,
            worker_init_fn=cached_worker_init,prefetch_factor=1 if a.workers else None,
            multiprocessing_context='spawn' if a.workers else None,generator=torch.Generator().manual_seed(a.seed+epoch))
        for samples in loader:
            start=time.monotonic();current=batch_current(samples);graph=current['local_graph']
            xy=torch.cat([s['xy'] for s in samples]).cuda();valid=torch.cat([s['valid'] for s in samples]).cuda()
            hidden,tasks=task_masks(graph,mask_rng,a.mode)
            noise=stable_noise([s['token'] for s in samples],graph.source_slot_ids,model.steps,a.seed+301,[epoch]*len(samples),device='cuda')
            times=torch.rand(len(samples),generator=time_rng).cuda();opt.zero_grad(set_to_none=True)
            sums,counts=training_loss_sums(model,xy,valid,hidden,noise,times,**current)
            loss=sum(sums[k]/max(counts[k],1) for k in sums)
            if not torch.isfinite(loss):raise FloatingPointError('Local graph nonfinite loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();schedule.step()
            step+=1;presentations+=len(samples);offset+=len(samples);seen.update(s['token'] for s in samples)
            selection=valid&hidden[:,:,None]
            row={'step':step,'epoch':epoch,'presentations':presentations,'unique_scenes':len(seen),'effective_epochs':presentations/len(corpus),
                 'loss':float(loss.detach()),'loss_components':{k:float(sums[k].detach())/max(counts[k],1) for k in sums},'supervised_coordinates':counts,
                 'task_nominal':torch.bincount(tasks['nominal'],minlength=3).tolist(),'task_actual':torch.bincount(tasks['actual'],minlength=3).tolist(),
                 'fallbacks':int(tasks['fallback'].sum()),'tasks_without_valid_hidden_target':int((~selection.flatten(1).any(-1)).sum()),
                 'supervision_by_actual_task':task_supervision_statistics(graph,hidden,valid,tasks),
                 'eligible_neighbors':int(graph.predictable_actor_mask[:,1:].sum()),'matched_neighbors':sum(s['mapping']['selected_with_accepted_association'] for s in samples),
                 'hidden_actors':int(hidden.sum()),'lr':opt.param_groups[0]['lr'],'gradient_before_clip':norm,'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%100==0:print(json.dumps(row),flush=True)
            end_epoch=offset==len(corpus)
            if end_epoch:epoch+=1;offset=0
            atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
            paused=((a.stop_after is not None and step>=a.stop_after) or (out/'STOP_REQUESTED').exists()) and epoch<a.epochs
            if end_epoch and epoch in (1,2,4,8,12,16,24,32):evaluate(str(epoch))
            if end_epoch or paused:save()
            if end_epoch or paused:break
        if paused:break
    save();atomic_json(out/'status.json',{'status':'paused' if paused else 'complete','step':step,'epochs':epoch,'presentations':presentations})


if __name__=='__main__':main()
