"""Bounded single-GPU frozen-representation experiments, no VLA optimizer.

Reuses the frozen-W probe's epoch/offset recipe, existing RNG capture/restore,
and the exact structured-world final_trajectory_loss. No distributed scheduler.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, identity, read, source_sha
from tools.foresight.student_state import capture_rng, restore_rng, learning_rate
from starVLA.dataloader.decision_local import FrozenConsequenceData, inputs_queries_targets, oracle_relations
from starVLA.model.modules.decision_local.consequence_head import ConsequenceQueryHead
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations
from starVLA.model.modules.decision_local.consequence_losses import consequence_loss
from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
from starVLA.model.modules.structured_world.losses import final_trajectory_loss
from starVLA.dataloader.foresight_dataset import decode_ego

KINDS = ('E_NONE','E_CURRENT_GT','E_CV_GT','E_FUTURE_GT','E_MASK_ONLY',
         'P_ABS','P_DIFF','P_QUERY_ONLY','R_CURRENT_PRED','R_FUTURE_ABS','R_FUTURE_DIFF')


def build_model(kind, seed, dim=512):
    if kind not in KINDS: raise ValueError('Unsupported experiment')
    # Isolate initialization from scene order and anchor/proposal RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ConsequenceQueryHead(dim=dim, query_only=kind=='P_QUERY_ONLY') if kind.startswith('P_') else ResidualRefiner(dim=dim)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def forward(model, kind, batch, predictor=None):
    inputs, queries, targets = inputs_queries_targets(batch)
    if kind.startswith('P_'):
        raw = model(inputs, queries)
        result = consequence_loss(raw.float(), targets.relations, targets.valid, difference_weight=1. if kind=='P_DIFF' else 0.)
        losses = {'total': result['loss'], 'absolute': result['absolute'], 'difference': result['difference'], **result['channels']}
        return raw, losses, targets
    if kind in ('E_CURRENT_GT','E_CV_GT','E_FUTURE_GT','E_MASK_ONLY'):
        output = model.oracle_probe(inputs, queries, oracle_relations(batch, kind))
    elif kind == 'E_NONE':
        output = model.camera_only(inputs, queries)
    else:
        if predictor is None: raise ValueError('R requires its own normally predicted frozen relations')
        with torch.no_grad(): relations = predicted_relations(predictor(inputs, queries).float())
        if kind == 'R_CURRENT_PRED':
            relations = torch.cat((relations[..., :3], relations[..., 1:3]), -1)
        output = model.camera_only(inputs, queries, relations)
    loss, _ = final_trajectory_loss(output.float(), targets.ego.float(), targets.future_valid)
    return output, {'total':loss, 'trajectory':loss}, targets


def load_predictor(path, expected_kind):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity']['kind'] != expected_kind: raise ValueError('Wrong predictor arm')
    model = build_model(expected_kind, payload['identity']['seed'], payload['identity']['dim'])
    model.load_state_dict(payload['model'], strict=True); model.eval().requires_grad_(False)
    return model.cuda(), payload


@torch.no_grad()
def evaluate(model, kind, data, microbatch, predictor=None):
    model.eval(); metrics = []; n = len(data.index)
    for start in range(0, n, microbatch):
        batch = data.batch(list(range(start,min(n,start+microbatch))), 'cuda', evaluation=True)
        output, losses, target = forward(model, kind, batch, predictor)
        row = {'scenes':len(batch['ego']), **{k:float(v) for k,v in losses.items()}}
        if not kind.startswith('P_'):
            p, gt = decode_ego(output.float()), decode_ego(target.ego.float())
            distance = (p[...,:2]-gt[...,:2]).norm(dim=-1)
            row.update(ADE=float(distance.mean()), FDE=float(distance[:,-1].mean()),
                residual_m=float((p[...,:2]-batch['q0'][...,:2]).norm(dim=-1).mean()))
        metrics.append(row)
    model.train()
    return {k:sum(r[k]*r['scenes'] for r in metrics)/n for k in metrics[0] if k!='scenes'} | {'scenes':n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('features','labels','output'): p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--dev-features',type=Path); p.add_argument('--dev-labels',type=Path)
    p.add_argument('--kind',choices=KINDS,required=True); p.add_argument('--predictor',type=Path)
    p.add_argument('--scope',choices=('smoke','diagnostic','formal'),required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch',type=int,default=128); p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--dim',type=int,default=512)
    p.add_argument('--max-updates',type=int,default=0); p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--resume',action='store_true'); p.add_argument('--save-every',type=int,default=200)
    a = p.parse_args()
    if min(a.epochs,a.batch,a.microbatch,a.max_seconds)<1: raise ValueError('Finite positive budget required')
    if a.scope=='formal' and (a.epochs not in (24,32) or a.batch!=128 or a.dim!=512):
        raise ValueError('Formal common recipe cannot silently change')
    resources = check_host(); start = time.monotonic()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed+1000); torch.cuda.manual_seed_all(a.seed+1000); np.random.seed(a.seed+1000); random.seed(a.seed+1000)
    train = FrozenConsequenceData(a.features,a.labels)
    dev = FrozenConsequenceData(a.dev_features,a.dev_labels) if a.dev_features else None
    if a.scope=='formal' and (len(train.index)!=101592 or dev is None or len(dev.index)!=1696):
        raise ValueError('Full common population required')
    if dev and {r['log'] for r in train.index}&{r['log'] for r in dev.index}: raise ValueError('Train/dev log leakage')
    model = build_model(a.kind,a.seed,a.dim); initial_hash = tensor_hash(model.state_dict()); model=model.cuda().train()
    predictor=payload=None
    if a.kind.startswith('R_'):
        if a.predictor is None: raise ValueError('Frozen corresponding predictor required')
        predictor,payload=load_predictor(a.predictor,'P_DIFF' if a.kind=='R_FUTURE_DIFF' else 'P_ABS')
        if payload['identity']['features']!=train.feature_identity['identity'] or payload['identity']['labels']!=train.label_identity['identity']:
            raise ValueError('Predictor trained on foreign anchor or labels')
        if a.scope=='formal' and (payload['epoch']!=24 or payload['offset']!=0): raise ValueError('Use common completed predictor endpoint')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001)
    generators={'shuffle':torch.Generator().manual_seed(a.seed+13000)}
    updates_per_epoch=math.ceil(len(train.index)/a.batch); horizon=updates_per_epoch*a.epochs
    # Explicit scheduler state included; same cosine horizon for every arm.
    warmup=min(updates_per_epoch,max(0,horizon-1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: learning_rate(min(step,horizon-1),a.lr,a.lr*.1,warmup,horizon)/a.lr)
    source_files=[*sorted((ROOT/'starVLA/model/modules/decision_local').glob('*.py')),
                  ROOT/'starVLA/dataloader/decision_local.py',Path(__file__)]
    run_identity={'schema':'decision_local_small_head_v1','source_sha':source_sha(ROOT),
        'source_files':{str(f.relative_to(ROOT)):digest(f) for f in source_files},'kind':a.kind,'seed':a.seed,
        'scope':a.scope,'features':train.feature_identity['identity'],'labels':train.label_identity['identity'],
        'dev_features':dev.feature_identity['identity'] if dev else None,'dev_labels':dev.label_identity['identity'] if dev else None,
        'epochs':a.epochs,'batch':a.batch,'microbatch':a.microbatch,'lr':a.lr,'dim':a.dim,'weight_decay':.001,
        'scheduler':'one epoch linear warmup then cosine to0.1LR','clip':1.,'compute_precision':'FP32','TF32':False,
        'initial_model_sha256':initial_hash,'predictor_sha256':digest(a.predictor) if a.predictor else None,
        'trainable_parameters':sum(x.numel() for x in model.parameters()),'frozen_base_optimizer_updates':0}
    a.output.mkdir(parents=True,exist_ok=True); saved_path=a.output/'latest.pt'
    completed=epoch=offset=exposure=0; order=torch.randperm(len(train.index),generator=generators['shuffle'])
    previous_gpu_hours=0.
    if a.resume:
        s=torch.load(saved_path,map_location='cpu',weights_only=False)
        if s['identity']!=run_identity: raise ValueError('Exact resume identity mismatch')
        model.load_state_dict(s['model'],strict=True); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        restore_rng(s['rng'],generators)
        completed,epoch,offset,exposure,order=[s[k] for k in ('completed','epoch','offset','exposure','order')]
        previous_gpu_hours=s['GPU_hours']
    elif (a.output/'identity.json').exists(): raise FileExistsError('Use explicit --resume')
    else: atomic_json(a.output/'identity.json',run_identity)
    def checkpoint(name='latest.pt'):
        torch.cuda.synchronize()
        s={'identity':run_identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'rng':capture_rng(generators),'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'order':order,
           'GPU_hours':previous_gpu_hours+(time.monotonic()-start)/3600.}
        tmp=a.output/(name+'.tmp'); torch.save(s,tmp); tmp.replace(a.output/name)
        atomic_json(a.output/'STATUS.json',{'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,
            'GPU_hours':s['GPU_hours'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'resource_check':resources})
    if completed==0: checkpoint()
    timings=[]; last=time.monotonic()
    while epoch<a.epochs:
        if (a.output/'STOP_REQUESTED').exists() or time.monotonic()-start>=a.max_seconds or (a.max_updates and completed>=a.max_updates):
            checkpoint(); atomic_json(a.output/'PAUSED.json',{'reason':'finite session/step budget or explicit STOP','completed':completed}); return
        ids=order[offset:offset+a.batch].tolist(); optimizer.zero_grad(set_to_none=True); losses_sum={}
        for sub in range(0,len(ids),a.microbatch):
            selected=ids[sub:sub+a.microbatch]; batch=train.batch(selected,'cuda',epoch=epoch)
            output,losses,targets=forward(model,a.kind,batch,predictor)
            weight=len(selected)/len(ids); (losses['total']*weight).backward()
            for k,v in losses.items(): losses_sum[k]=losses_sum.get(k,0)+v.detach()*weight
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(losses_sum['total']): raise FloatingPointError('Nonfinite loss; optimizer not stepped')
        optimizer.step(); scheduler.step(); completed+=1; offset+=len(ids); exposure+=len(ids)
        # One packed read/synchronization per optimizer boundary, including work
        # already submitted. Timings include input IO and complete backward.
        names=sorted(losses_sum); values=torch.stack([losses_sum[k] for k in names]+[norm]).tolist()
        now=time.monotonic(); timings.append(now-last); last=now
        row={'update':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'seconds':timings[-1],
             'lr_next':optimizer.param_groups[0]['lr'],'grad_norm':values[-1],**dict(zip(names,values[:-1]))}
        with (a.output/'train.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if completed%20==0 or completed==1: print(json.dumps(row),flush=True)
        if offset==len(order):
            epoch+=1; offset=0; order=torch.randperm(len(train.index),generator=generators['shuffle'])
            if epoch in (6,12,18,24,32) or epoch==a.epochs:
                checkpoint('epoch_%02d.pt'%epoch)
                if dev:
                    atomic_json(a.output/('dev_epoch_%02d.json'%epoch),evaluate(model,a.kind,dev,a.microbatch,predictor))
                checkpoint(); last=time.monotonic()
        if completed%a.save_every==0: checkpoint(); last=time.monotonic()
    checkpoint()
    atomic_json(a.output/'COMPLETE.json',{'epoch':epoch,'completed':completed,'exposure':exposure,
        'full_training_population':len(train.index)==101592,'median_update_s':float(np.median(timings[5:] or timings)),
        'p90_update_s':float(np.percentile(timings[5:] or timings,90)), 'GPU_hours':previous_gpu_hours+(time.monotonic()-start)/3600.})


if __name__=='__main__': main()
