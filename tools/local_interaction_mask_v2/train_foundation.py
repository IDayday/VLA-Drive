"""Single-seed public Qwen + fresh original DiT/current Reader training, real DDP."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from tools.local_interaction_mask_v2.foundation import (FoundationDataset,FoundationObjective,collate,epoch_batches,create_world,
    set_train_mode,module_state,restore_modules,rng_state,restore_rng,modules,to_device)
from starVLA.model.modules.structured_world.metrics import diagnostics


def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('public-qwen','config','provenance','tokens','meta-root','observations','targets','output'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--visual-cache');p.add_argument('--holdout');p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--initial-epochs',type=int);p.add_argument('--schedule-epochs',type=int,default=16);p.add_argument('--global-batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark',action='store_true');p.add_argument('--check-empty-rank',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();rank=int(os.environ.get('RANK',0));count=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if count>1:dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tokens=json.loads(Path(a.tokens).read_text())
    if a.limit:tokens=tokens[:a.limit]
    if len(set(tokens))!=len(tokens):raise ValueError('Repeated training scene')
    if a.benchmark and (not a.limit or a.limit>1024 or a.epochs>2):raise ValueError('Throughput benchmark must be small and never a research result')
    if a.check_empty_rank and not a.limit:raise ValueError('Artificial empty annotation rank is a diagnostic only')
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Byte identities of every current metadata/calibration and label used by this run.
    datahash=hashlib.sha256()
    for token in tokens:
        for path in (Path(a.meta_root)/(token+'.pkl'),Path(a.observations)/(token+'.npz'),Path(a.targets)/'targets'/(token+'.pt')):
            datahash.update((token+digest(path)).encode())
    settings={k:v for k,v in vars(a).items() if k not in ('resume','stop_after','output')}
    identity={'schema_version':2,'code_sha':code,'arguments':settings,'world_size':count,'data_sha256':datahash.hexdigest(),
        'tokens_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),'public_provenance':json.loads(Path(a.provenance).read_text()),
        'private_driving_weights_loaded':False,'exact_resume_boundary':'optimizer step, same code/data/topology/precision, explicit epoch offset and all rank RNG',
        'trainable':'fresh original DiT, history projector, post-Qwen Reader and current bbox/classification heads; Qwen frozen; unused motion head frozen'}
    if not a.resume and (out/'checkpoint.pt').exists():raise ValueError('Refuse to overwrite existing training run')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed(a.seed)
    world=create_world(a.public_qwen,OmegaConf.load(a.config),identity['public_provenance'],a.seed,a.visual_cache,code)
    objective=FoundationObjective(world)
    params=[p for p in objective.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    horizon=math.ceil(len(tokens)/a.global_batch)*a.schedule_epochs
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/100,1)*(.1+.9*.5*(1+math.cos(math.pi*min(s,horizon)/horizon))))
    step=epoch=offset=presentations=0
    dataset=FoundationDataset(tokens,a.meta_root,a.observations,a.targets)
    if count>1:objective=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    torch.manual_seed(a.seed+1000+rank);torch.cuda.manual_seed(a.seed+1000+rank)
    if a.resume:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Foundation resume source/config/data identity differs')
        restore_modules(world,saved['modules']);opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        step,epoch,offset,presentations=[saved[k] for k in ('step','epoch','offset','presentations')]
        restore_rng(saved['rng_by_rank'][rank]);del saved
    if rank==0:
        atomic_json(out/'manifest.json',identity)
        atomic_json(out/'trainable.json',{'parameters':sum(p.numel() for p in params),'names':[n for n,p in world.named_parameters() if p.requires_grad]})
        world.baseline.qwen_vl_interface.processor.save_pretrained(out/'tokenizer')
    holdout=None
    if a.holdout:
        spec=json.loads(Path(a.holdout).read_text());holdout=FoundationDataset(json.loads(Path(spec['tokens']).read_text()),spec['meta_root'],spec['observations'],spec['targets'])
    def evaluate(tag):
        if holdout is None:return
        world.eval();rows=[]
        with torch.no_grad():
            for i in range(rank,len(holdout),count):
                s=holdout[i];t=to_device(s['target'],'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):c,pred=world.encode_conditions([s['example']])
                seed=int.from_bytes(hashlib.sha256(('20260926:'+s['example']['token']).encode()).digest()[:4],'little')
                generator=torch.Generator(device='cuda').manual_seed(seed)
                noise=torch.randn(1,8,4,device='cuda',dtype=c.dtype,generator=generator)
                action=world.baseline.action_model.predict_action(c.float(),initial_noise=noise.float())
                predicted=action[0,:,:2].float()*action.new_tensor([8.805105,2.277741])+action.new_tensor([10.172484,.360762])
                target=s['ego'][:,:2].cuda()*action.new_tensor([8.805105,2.277741])+action.new_tensor([10.172484,.360762])
                row={'token':s['example']['token'],'status':'ok','ego_ADE_m':float((predicted-target).norm(dim=-1).mean()),'ego_FDE_m':float((predicted[-1]-target[-1]).norm())}
                row.update(diagnostics({k:v[0] for k,v in pred.items()},t,include_legacy=False));rows.append(row)
        gathered=[None]*count if rank==0 else None
        if count>1:dist.gather_object(rows,gathered,dst=0)
        elif rank==0:gathered=[rows]
        if rank==0:
            rows=sum(gathered,[])
            atomic_json(out/f'holdout_{tag}.json',{'rows':rows,'count':len(rows),'failed':0,'ego_ADE_m':float(np.mean([r['ego_ADE_m'] for r in rows])),
                'class_filtered_recall':sum(r.get('class_filtered_tp',0) for r in rows)/max(sum(r['gt_targets'] for r in rows),1)})
        set_train_mode(world)
    def save():
        if a.benchmark:return
        states=[None]*count if rank==0 else None
        state=rng_state()
        if count>1:dist.gather_object(state,states,dst=0)
        elif rank==0:states=[state]
        if rank==0:
            payload={'identity':identity,'public_origin':world.baseline.public_origin,'modules':module_state(world),'optimizer':opt.state_dict(),
                'scheduler':schedule.state_dict(),'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'rng_by_rank':states}
            torch.save(payload,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
            atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
        if count>1:dist.barrier()
    if not a.resume:evaluate('0')
    started=time.monotonic();paused=False
    while epoch<a.epochs:
        batches=epoch_batches(len(tokens),a.global_batch,rank,count,a.seed,epoch,offset)
        generator=torch.Generator().manual_seed(a.seed+epoch) # worker launch does not consume training RNG
        loader=torch.utils.data.DataLoader(dataset,batch_sampler=batches,collate_fn=collate,num_workers=a.workers,
            multiprocessing_context='spawn' if a.workers else None,generator=generator)
        for samples in loader:
            opt.zero_grad(set_to_none=True);t0=time.monotonic()
            loss,numerators,denominators=objective(samples,empty_targets=a.check_empty_rank and rank==count-1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite foundation loss')
            loss.backward()
            gradients={name:float(sum((p.grad.detach().float().square().sum() for p in m.parameters() if p.grad is not None),torch.zeros((),device='cuda')).sqrt()) for name,m in modules(world).items()}
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step();schedule.step()
            step+=1;added=min(a.global_batch,len(tokens)-offset);offset+=added;presentations+=added
            if offset==len(tokens):epoch+=1;offset=0;end_epoch=True
            else:end_epoch=False
            row={'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'effective_epochs':presentations/len(tokens),
                'global_batch_actual':added,'loss_ego_cls_box':(numerators/denominators.clamp_min(1)).cpu().tolist(),
                'global_denominators':denominators.cpu().tolist(),'gradients':gradients,'gradient_before_clip':norm,
                'lr':opt.param_groups[0]['lr'],'seconds':time.monotonic()-t0,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            if rank==0:
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
                if step%10==0 or step<3:print(json.dumps(row),flush=True)
            flag=torch.tensor(int((a.stop_after is not None and step>=a.stop_after) or (out/'STOP_REQUESTED').exists()),device='cuda')
            if count>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            paused=bool(flag) and epoch<a.epochs
            if end_epoch:
                evaluate(str(epoch))
                if a.initial_epochs and epoch==a.initial_epochs:
                    if a.holdout is None:raise ValueError('Foundation extension needs declared holdout')
                    extend=torch.zeros((),device='cuda',dtype=torch.int)
                    if rank==0:
                        earlier=json.loads((out/f'holdout_{epoch-2}.json').read_text());latest=json.loads((out/f'holdout_{epoch}.json').read_text())
                        changes={k:abs(latest[k]-earlier[k])/max(abs(earlier[k]),1e-6) for k in ('ego_ADE_m','class_filtered_recall')}
                        extend.fill_(int(any(v>.02 for v in changes.values())))
                        atomic_json(out/'extension_decision.json',{'initial_epochs':epoch,'maximum_epochs':a.epochs,'relative_changes':changes,'extend':bool(extend),'PDMS_consulted':False})
                    if count>1:dist.broadcast(extend,0)
                    if not bool(extend):
                        save()
                        if rank==0:atomic_json(out/'status.json',{'status':'complete','step':step,'epochs_completed':epoch,'presentations':presentations,'stopped_by_preregistered_holdout_rule':True})
                        if count>1:dist.destroy_process_group()
                        return
            if end_epoch or paused:save()
            if paused or end_epoch:break
        if paused:break
    save()
    if rank==0:atomic_json(out/'status.json',{'status':'paused' if paused else 'complete','step':step,'epochs_completed':epoch,'presentations':presentations,'training_seconds':time.monotonic()-started})
    if count>1:dist.destroy_process_group()


if __name__=='__main__':main()
