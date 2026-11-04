"""Actual TRAIN gradient comparison at the same saved policy/behavior batch.

No optimizer is instantiated or stepped. No new rewards, NAVTEST inputs or
formal training run. Group-std/batch-std gradients use identical trajectories,
policy weights, source probabilities, reference and valid-scene weights.
"""
import argparse,json,os,time
from pathlib import Path
import torch
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.safety_credit import raw_credit
from starVLA.rl.d46_augustus.actor import AugustusActor
from starVLA.rl.d46_augustus.runtime import AugustusExtension
from starVLA.rl.d46_augustus.deployment import inspect
from starVLA.rl.d46_augustus.kernel import policy_loss
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import require_dispatch,fingerprint,file_sha


def cosine(a,b):
    den=float(a.norm()*b.norm())
    return float(torch.dot(a,b))/den if den>0 else None


def gradient(loss,params,retain):
    values=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros_like(p)if g is None else g).detach().float().reshape(-1)for p,g in zip(params,values)])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    host=require_dispatch();configure_numerics();torch.set_num_threads(2);torch.cuda.set_device(0)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if(out/'SUMMARY.json').exists():raise FileExistsError('Preserve previous diagnostic')
    cp=Path(a.checkpoint);identity,seal,files=inspect(cp);cfg=identity['recipe'];updates=seal['updates']
    if(identity['experiment_id']!='AUG4_LORA_ALIGNED'or updates!=200 or seal['next_inner']!=0):raise ValueError('One exact qualified LoRA200 READY diagnostic')
    stream=DataStream(cfg,cfg['global_scenes']);device=torch.device('cuda',0)
    torch.manual_seed(42);model,info=load(cfg['parent_checkpoint'],device,binding=cfg['parent_registration'])
    actor=AugustusActor(model,source_config(cfg),cfg)
    conditions=DConditions(model,stream.data,info,group=cfg['group'],capacity=cfg['condition_cache_scenes']);object.__setattr__(actor,'conditions',conditions)
    ext=AugustusExtension();ext.cfg=cfg;ext.configure_actor(actor)
    common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True);named=dict(actor.named_parameters())
    with torch.no_grad():
        for name,value in common['adapters'].items():named[name].copy_(value.to(device))
    del common
    params=tuple(p for p in actor.parameters()if p.requires_grad);names=[n for n,p in actor.named_parameters()if p.requires_grad]
    offset=0;roles={}
    for name,param in zip(names,params):
        role='text_lora'if'.language_model.'in name else'action_time_interfaces'
        roles.setdefault(role,[]).append((offset,offset+param.numel()));offset+=param.numel()
    sums={};reference=torch.zeros(offset,device=device);rows=[];tokens=set();start=time.monotonic()
    collection=updates//cfg['inner_updates']+1
    paths=sorted(cp.parents[1].joinpath('behavior').glob(f'*/collection{collection:06}/rank*.pt'))
    if len(paths)!=identity['world']:raise ValueError('Complete same-behavior next collection required')
    materials=[]
    for path in paths:
        batch=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if batch['behavior_update']!=updates:raise ValueError('Not this exact saved behavior policy')
        for r in batch['scenes']:materials.append((r,batch['global_std']))
    if len(materials)!=64:raise ValueError('All64 distinct TRAIN scenes required')
    for index,(saved,std)in enumerate(materials):
        require_dispatch();token=saved['token']
        if token not in stream.data.records or token in tokens or saved['padding']:raise ValueError('Illegal/duplicate/padded diagnostic scene')
        tokens.add(token);r={k:v.to(device)if torch.is_tensor(v)else v for k,v in saved.items()};r['expect_fresh']=True
        raw,feasible=raw_credit(r['factors'][None]);mask=feasible[0];category='all_feasible'if mask.all()else'mixed_safety'if mask.any()else'all_infeasible'
        alternate=(raw[0]/(std+1e-8)).clamp(-5,5).float()
        _,metrics,parts=actor(r,return_components=True);logr=parts['logratio'];zero=torch.zeros_like(logr)
        pg_group,_=policy_loss(logr,r['advantages'],zero,beta=0.)
        pg_batch,_=policy_loss(logr,alternate,zero,beta=0.)
        gg=gradient(pg_group,params,True)/64;gb=gradient(pg_batch,params,True)/64;gr=gradient(parts['reference'],params,False)/64
        if not all(torch.isfinite(g).all()for g in(gg,gb,gr)):raise FloatingPointError('Nonfinite diagnostic gradient')
        own=sums.setdefault(category,{'group':torch.zeros_like(gg),'batch':torch.zeros_like(gb)})
        own['group']+=gg;own['batch']+=gb;reference+=gr
        row=dict(index=index,token=token,category=category,fresh_ratio_error=float(metrics['fresh_ratio_error']),group_gradient_norm=float(gg.norm()),batch_gradient_norm=float(gb.norm()),weighted_reference_gradient_norm=float(gr.norm()))
        rows.append(row);atomic_json(out/f'scene{index:03}.json',row)
        del parts,logr,pg_group,pg_batch,gg,gb,gr,r
    final={}
    for mode in('group','batch'):
        total=sum((x[mode]for x in sums.values()),torch.zeros_like(reference));safe=sums.get('all_feasible',{}).get(mode);mixed=sums.get('mixed_safety',{}).get(mode)
        final[mode]=dict(PG_norm=float(total.norm()),reference_norm=float(reference.norm()),PG_reference_cosine=cosine(total,reference),total_loss_gradient_norm=float((total+reference).norm()),
            categories={k:dict(norm=float(v[mode].norm()),projection_on_total=float(torch.dot(v[mode],total)/total.square().sum()))for k,v in sums.items()},
            safe_mixed_cosine=cosine(safe,mixed)if safe is not None and mixed is not None else None,
            roles={role:dict(safe_mixed_cosine=cosine(torch.cat([safe[l:h]for l,h in spans]),torch.cat([mixed[l:h]for l,h in spans])))for role,spans in roles.items()}if safe is not None and mixed is not None else {})
    tg=sum((x['group']for x in sums.values()),torch.zeros_like(reference));tb=sum((x['batch']for x in sums.values()),torch.zeros_like(reference))
    result=dict(schema='d46_actual_gradient_normalization_diagnostic_v1',status='COMPLETE',host=host,checkpoint=str(cp),policy_updates=updates,TRAIN_scenes=len(tokens),
        fresh_ratio_max=max(r['fresh_ratio_error']for r in rows),normalizations=final,PG_group_batch_cosine=cosine(tg,tb),total_group_batch_cosine=cosine(tg+reference,tb+reference),
        optimizer_steps=0,new_PDM_queries=0,NAVTEST_used=False,seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        source_file_sha256=file_sha(__file__),limits='All64 scenes of one actual fresh collection at LoRA200. Raw aggregate parameter gradients before Adam/clipping; not all training, a causal proof of PDMS change, or a new algorithm score.')
    result['sha256']=fingerprint(result);atomic_json(out/'SUMMARY.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
