"""Matched downstream representation probes: train bridge with original frozen DiT FM loss."""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from tools.joint_world.planner_runtime import (CachedCurrentPlanner, ego_action_target,
    file_sha256, graph_noise, load_original_head, predict)
from tools.joint_world.train_corpus import Corpus
from tools.joint_world.train_graph import current_batch
from tools.structured_world.runtime import seed_all
from tools.structured_world_v1p1.budget import start, record
from tools.structured_world_v1p1.reaudit_metrics import write_csv


@torch.no_grad()
def evaluate(head, planner, samples, out, step, act_norm, labels):
    from infer import deal_action_1225
    planner.eval(); rows=[]; banks=out/f'predictions_{step}';banks.mkdir(exist_ok=False)
    for sample in samples:
        token=sample['cache']['token'];normalized,joint=predict(head,planner,sample)
        trajectory=deal_action_1225(normalized,act_norm=act_norm)[0]
        target=deal_action_1225(labels(token)[None].numpy(),act_norm=act_norm)[0]
        error=np.linalg.norm(trajectory[:,:2]-target[:,:2],axis=-1)
        if not np.isfinite(trajectory).all():raise FloatingPointError('Invalid DiT trajectory')
        rows.append({'token':token,'status':'ok','ego_ADE':float(error.mean()),'ego_FDE':float(error[-1])})
        np.savez(banks/(token+'.npz'),trajectory=trajectory,normalized_actions=normalized[0],joint_xy=joint[0])
    summary={'step':step,'scenes':len(rows),'failed':0,'ego_ADE':float(np.mean([r['ego_ADE'] for r in rows])),
             'ego_FDE':float(np.mean([r['ego_FDE'] for r in rows])),'gate':float(planner.adapter.gate),
             'scope':'Actual original DiT output; imitation errors, not PDMS','candidates':1,'sampling_steps':10}
    write_csv(out/f'eval_{step}.csv',rows);(out/f'eval_{step}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True);planner.train();planner.graph.eval()


def main():
    # Third-party dataset/decode imports can consume Python RNG. Finish them before
    # seeding or restoring the sampler; never defer them until the first resumed batch.
    from infer import deal_action_1225
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','targets','data-root','base-checkpoint','graph-checkpoint','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    p.add_argument('--bev-index');p.add_argument('--bev-tasks',action='store_true')
    p.add_argument('--bev-task-weight',type=float,default=.1)
    a=p.parse_args()
    if not 1<=a.epochs<=4 or not 1<=a.batch<=16:raise ValueError('Bridge probe bounded to4passes,batch16')
    if a.bev_tasks and not a.bev_index:raise ValueError('BEV tasks require current BEV features')
    if not 0<a.bev_task_weight<=1:raise ValueError('BEV auxiliary weight must be in(0,1]')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=bool(a.resume))
    seed_all(a.seed);ds=Corpus(a.cache,a.targets,a.data_root)
    bev_store=None
    if a.bev_index:
        from tools.joint_world.bev_cache import BEVFeatureStore,batch_bev
        bev_store=BEVFeatureStore(a.bev_index)
        if set(bev_store.records)!={r['token'] for r in ds.records}:raise ValueError('BEV/current corpus token mismatch')
    if not 1<=len(ds)<=8192:raise ValueError('Scene cap8192')
    trained=torch.load(a.graph_checkpoint,map_location='cpu',weights_only=False);cfg=trained['identity']['config']
    source=json.loads((Path(trained['identity']['arguments']['cache'])/'manifest.json').read_text())
    for key in ['world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract']:
        if ds.manifest['identity'][key]!=source['identity'][key]:raise ValueError('Incompatible graph current features: '+key)
    presentations_total=len(ds)*a.epochs;steps=math.ceil(presentations_total/a.batch)
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'cache_identity':ds.manifest['identity_sha256'],'label_fingerprint':ds.target_fingerprint,
              'graph_checkpoint_sha256':file_sha256(a.graph_checkpoint),'graph_config':cfg,
              'original_checkpoint_sha256':ds.manifest['identity']['baseline_checkpoint_sha256'],
              'planned_steps':steps,'planned_presentations':presentations_total,'trainable':'graph_to_world and gated action attention only',
              'frozen':'Qwen,vision,world Reader/heads,joint graph,original DiT','pdms_training':False,
              'graph_condition':'all-hidden current-only rollout, independent noise seed2037',
              'deterministic_algorithms':True,
              'purpose':'Matched frozen-representation transfer probe; joint fine-tuning is a separate phase'}
    identity.update(bev_enabled=bool(bev_store),bev_identity=bev_store.identity_sha256 if bev_store else None,
                    bev_task_supervision=a.bev_tasks,bev_task_weight=a.bev_task_weight)
    if a.bev_tasks:identity['bev_tasks']=['current_occupancy','tracked_displacement','pair_separation','all_hidden_joint_flow']
    if bev_store:
        identity.update(trainable='graph_to_world,action attention,BEV encoder/fusion; auxiliary heads receive updates only with tasks enabled',
                        frozen='Qwen,vision,world Reader/heads,pretrained BEV provider,joint graph parameters,original DiT',
                        purpose='Matched BEV semantic/motion task-on/task-off planning transfer with differentiable all-hidden rollout')
    saved=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    if saved:
        for key in ['code_sha','cache_identity','label_fingerprint','graph_checkpoint_sha256','graph_config','planned_steps','planned_presentations','bev_identity','bev_task_supervision','bev_task_weight']:
            if identity[key]!=saved['identity'][key]:raise ValueError('Resume identity mismatch: '+key)
        for key in ['batch','seed','output','run_id','base_checkpoint']:
            if vars(a)[key]!=saved['identity']['arguments'][key]:raise ValueError('Resume argument mismatch: '+key)
        ledger=json.loads(Path(a.ledger).read_text());run=next(r for r in ledger['runs'] if r['id']==a.run_id)
        if run['status']=='running' or run['optimizer_steps']!=saved['step'] or saved['step']>=steps:
            raise ValueError('Resume requires terminal accounted incomplete run')
    start(a.ledger,a.run_id,steps,identity,resume=bool(saved));completed=saved['step'] if saved else 0
    try:
        head,base_config=load_original_head(a.base_checkpoint,identity['original_checkpoint_sha256'])
        # Fixed seed for new bridge independent of model-loading initialization draws.
        seed_all(a.seed);first=ds[0];model=CachedCurrentPlanner(first['cache']['context'].shape[-1],cfg,bev_enabled=bool(bev_store)).cuda()
        model.graph.load_state_dict(trained['model'],strict=True);model.graph.requires_grad_(False).eval()
        params=[v for v in model.parameters() if v.requires_grad]
        identity['trainable_parameters']=sum(v.numel() for v in params)
        identity['frozen_DiT_parameters']=sum(v.numel() for v in head.parameters())
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        warmup=min(100,max(1,steps//10))
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warmup if s<warmup else .1+.9*.5*(1+math.cos(math.pi*min((s-warmup)/max(1,steps-warmup),1.))))
        order=list(range(len(ds)));random.shuffle(order);position=0;presentations=0;seen=set()
        act_norm=int(base_config.datasets.vla_data.act_norm)
        labels=lru_cache(maxsize=8192)(lambda token:ego_action_target(token,a.data_root,act_norm))
        aux_generator=torch.Generator(device='cuda').manual_seed(a.seed+10000) if bev_store else None
        def with_bev(sample):
            return dict(sample,bev=bev_store[sample['cache']['token']]) if bev_store else sample
        diagnostics=[with_bev(ds[i]) for i in range(min(64,len(ds)))]
        if saved:
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
            order=saved['order'];position=saved['position'];presentations=saved['presentations'];seen=set(saved['seen'])
            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
            if aux_generator is not None:aux_generator.set_state(saved['rng']['bev_aux'])
        else:evaluate(head,model,diagnostics,out,0,act_norm,labels)
        (out/'manifest.json').write_text(json.dumps(identity,indent=2))
        milestones={math.ceil(len(ds)*epoch/a.batch) for epoch in [1,2,4] if epoch<=a.epochs}|{steps}
        for step in range(completed+1,steps+1):
            if not record(a.ledger,a.run_id,completed):raise RuntimeError('Budget exhausted')
            batch=[]
            for _ in range(min(a.batch,presentations_total-presentations)):
                if position==len(order):random.shuffle(order);position=0
                batch.append(with_bev(ds[order[position]]));position+=1
            current=current_batch(batch);native=torch.cat([s['cache']['native_actions'] for s in batch]).cuda()
            actions=torch.stack([labels(s['cache']['token']) for s in batch]).cuda()
            noise=graph_noise(len(batch),current['actor_features'].shape[1],model.graph.steps,native.device,cfg.get('sampling_seed',2037))
            opt.zero_grad(set_to_none=True)
            bev_aux=None
            if bev_store:
                bev=batch_bev(batch)
                current,bev_aux=model.fuse_bev(current,bev['features'],bev['coordinates'],bev['observation_support'])
                # Frozen graph parameters still propagate ego gradients to trainable BEV inputs.
                conditions,_,_=model.rollout_condition(native,current,noise)
            else:conditions,_=model.condition_from_frozen_graph(native,current,noise)
            # Frozen weights do NOT mean no_grad: the ego loss must reach the bridge.
            ego_loss=head(conditions,actions,None);loss=ego_loss;task_values={};task_counts={}
            if a.bev_tasks:
                from starVLA.model.modules.structured_world.contracts import WorldTargets
                from starVLA.model.modules.joint_world.bev_tasks import bev_loss_sums,interaction_loss_sum
                targets=[WorldTargets(**{k:v.cuda() if torch.is_tensor(v) else v for k,v in vars(s['target']).items()}) for s in batch]
                sums,task_counts=bev_loss_sums(bev_aux,targets,bev['coordinates'],bev['observation_support'])
                sums['interaction'],task_counts['interaction']=interaction_loss_sum(bev_aux['pair_separation'],torch.cat([s['xy'] for s in batch]).cuda(),torch.cat([s['valid'] for s in batch]).cuda())
                # Train the BEV-to-graph gate before the outer planning gate opens.
                # Auxiliary hidden states never replace generated planner conditions.
                from starVLA.model.modules.joint_world.flow import training_loss_sums
                xy=torch.cat([s['xy'] for s in batch]).cuda();valid=torch.cat([s['valid'] for s in batch]).cuda()
                gs,gc=training_loss_sums(model.graph,xy,valid,torch.ones_like(valid[:,:,0]),
                    torch.randn(xy.shape,device=xy.device,dtype=xy.dtype,generator=aux_generator),
                    torch.rand(len(batch),device='cuda',generator=aux_generator),**current)
                for key in gs:sums['graph_'+key]=gs[key];task_counts['graph_'+key]=gc[key]
                loss=loss+a.bev_task_weight*sum(sums[k]/max(task_counts[k],1) for k in sums)
                task_values={k:float(sums[k].detach())/max(task_counts[k],1) for k in sums}
            if not torch.isfinite(loss):raise FloatingPointError('Planner FM loss nonfinite')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            gate_gradient=float(model.adapter.gate.grad)
            bev_gradients={}
            if bev_store:
                for name in ['bev_encoder','bev_fusion','interaction_head']:
                    values=[v.grad.detach().square().sum() for v in getattr(model,name).parameters() if v.grad is not None]
                    bev_gradients[name]=float(torch.stack(values).sum().sqrt()) if values else 0.
                bev_gradients['fusion_gate_gradient']=float(model.bev_fusion.gate.grad) if model.bev_fusion.gate.grad is not None else 0.
            opt.step();schedule.step();completed=step
            presentations+=len(batch);seen.update(s['cache']['token'] for s in batch)
            row={'step':step,'ego_FM_loss':float(ego_loss.detach()),'total_loss':float(loss.detach()),'bev_task_loss':task_values,'bev_task_counts':task_counts,
                 'bev_gradients_after_clip':bev_gradients,'bev_gate':float(model.bev_fusion.gate) if bev_store else None,'gradient_before_clip':norm,'gate_gradient':gate_gradient,
                 'gate':float(model.adapter.gate),'lr':opt.param_groups[0]['lr'],'presentations':presentations,
                 'effective_epochs':presentations/len(ds),'unique_scenes':len(seen),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            record(a.ledger,a.run_id,step)
            if step%100==0:print(json.dumps(row),flush=True)
            if step in milestones or step%250==0 or step==a.stop_after:
                if step in milestones:evaluate(head,model,diagnostics,out,step,act_norm,labels)
                payload={'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                         'step':step,'order':order,'position':position,'presentations':presentations,'seen':sorted(seen),
                         'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
                                'bev_aux':aux_generator.get_state() if aux_generator is not None else None}}
                temp=out/'checkpoint.tmp';torch.save(payload,temp);temp.replace(out/f'checkpoint_{step}.pt')
            if a.stop_after and step>=a.stop_after:break
        record(a.ledger,a.run_id,completed,'complete' if completed==steps else 'paused')
    except BaseException:record(a.ledger,a.run_id,completed,'failed');raise


if __name__=='__main__':main()
