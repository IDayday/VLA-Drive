"""Bounded real Qwen/DiT/world training, with exact single-process step-boundary resume."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import yaml
from runtime import load_baseline,load_dataset,load_world_batch,seed_all
from budget import reserve,record
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def gradient_group(name):
    if name.startswith('baseline.action_model.'):return 'action'
    if '.visual.' in name:return 'vision'
    if name in ['reader.queries','reader.types']:return 'world_tokens'
    for prefix,group in [('heads.box.','bbox_head'),('heads.motion.','motion_head'),('heads.classifier.','classification_head'),('heads.shared.','world_shared_head'),('adapter.gate','adapter_gate'),('adapter.','adapter_attention')]:
        if name.startswith(prefix):return group
    return name.split('.')[0]


def checkpoint(policy,optimizer,scheduler,step,order,position,args):
    # Frozen original parameters are restored from the strict-audited base checkpoint.
    names={n for n,p in policy.named_parameters() if p.requires_grad}
    names.update(n for n in policy.state_dict() if not n.startswith('baseline.'))
    return {'schema_version':1,'delta':{n:v.detach().cpu() for n,v in policy.state_dict().items() if n in names},
            'delta_keys':sorted(names),'world_config':policy.world_config,'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'step':step,'order':order,'position':position,
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            'arguments':vars(args),'resume_boundary':'single-process, optimizer step, no prefetch/no accumulation; bitwise updates require --deterministic/math attention and identical topology',
            'base_sha256':args.base_sha256,'manifest_sha256':__import__('hashlib').sha256(Path(args.manifest).read_bytes()).hexdigest(),
            'target_audit_sha256':__import__('hashlib').sha256((Path(args.target_cache)/'audit.json').read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','vlm','data-root','manifest','target-cache','config','output','ledger','run-id','base-sha256']:p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--limit',type=int,default=64);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--action-lr',type=float,default=1e-5);p.add_argument('--resume');p.add_argument('--initialize-world');p.add_argument('--provider-checkpoint');p.add_argument('--save-every',type=int,default=100);p.add_argument('--stop-after',type=int);p.add_argument('--deterministic',action='store_true')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    if a.limit>8192 or a.steps>1000:raise ValueError('Pilot per-run budget limit exceeded')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not a.resume:reserve(a.ledger,a.run_id,a.steps,vars(a))
    seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False)
    policy=StructuredWorldPolicy(agent.model,cfg).cuda()
    if not policy.world_enabled:
        policy.reader.requires_grad_(False);policy.heads.requires_grad_(False)
    if a.initialize_world:
        init=torch.load(a.initialize_world,map_location='cpu',weights_only=False)['delta']
        for name,module in [('reader',policy.reader),('heads',policy.heads)]:
            module.load_state_dict({k[len(name)+1:]:v for k,v in init.items() if k.startswith(name+'.')},strict=True)
    if policy.provider is not None and cfg.get('freeze_provider',True):
        if not a.provider_checkpoint and not a.resume:raise ValueError('Frozen BEV requires a trained provider checkpoint')
        if a.provider_checkpoint:
            init=torch.load(a.provider_checkpoint,map_location='cpu',weights_only=False)['delta']
            policy.provider.load_state_dict({k[len('provider.'):]:v for k,v in init.items() if k.startswith('provider.')},strict=True)
    if cfg.get('train_action',False):policy.baseline.action_model.requires_grad_(True)
    if cfg.get('vision_trainable',False):policy.baseline.qwen_vl_interface.model.model.visual.requires_grad_(True)
    if policy.provider is not None and cfg.get('freeze_provider',True):policy.provider.requires_grad_(False)
    # Keep baseline dropout behavior identical across all continued-training variants.
    policy.train();policy.baseline.eval()
    if cfg.get('train_action',False):policy.baseline.action_model.train()
    agent.model.qwen_vl_interface.processor.save_pretrained(out/'processor')
    from omegaconf import OmegaConf
    OmegaConf.save(agent.model_config,out/'base_config.yaml')
    ds=load_dataset(agent,a.manifest,a.data_root,a.limit)
    params=[p for p in policy.parameters() if p.requires_grad]
    action_params=[p for n,p in policy.named_parameters() if p.requires_grad and n.startswith('baseline.action_model.')]
    world_params=[p for n,p in policy.named_parameters() if p.requires_grad and not n.startswith('baseline.action_model.')]
    optimizer=torch.optim.AdamW([{'params':world_params,'lr':a.lr},{'params':action_params,'lr':a.action_lr}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    order=list(range(len(ds)));random.shuffle(order);position=0;start=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved['base_sha256']!=a.base_sha256 or saved['world_config']!=cfg:raise ValueError('Resume identity mismatch')
        for key in ['manifest','target_cache','batch_size','limit','seed','lr','action_lr','steps']:
            if saved['arguments'][key]!=getattr(a,key):raise ValueError(f'Resume run contract mismatch: {key}')
        import hashlib
        if saved['manifest_sha256']!=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest():raise ValueError('Changed sample manifest')
        if saved['target_audit_sha256']!=hashlib.sha256((Path(a.target_cache)/'audit.json').read_bytes()).hexdigest():raise ValueError('Changed target contract')
        expected=checkpoint(policy,optimizer,scheduler,0,[],0,a)['delta_keys']
        if saved['delta_keys']!=expected or set(saved['delta'])!=set(expected):raise ValueError('Resume delta keys mismatch')
        state=policy.state_dict()
        for key,value in saved['delta'].items():
            if state[key].shape!=value.shape:raise ValueError(f'Resume shape mismatch: {key}')
            state[key].copy_(value.to(state[key].device))
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start=saved['step'];order=saved['order'];position=saved['position']
        random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    initial_probe={n:p.detach().flatten()[:8].cpu().clone() for n,p in policy.named_parameters() if p.requires_grad}
    counts={'total':sum(p.numel() for p in policy.parameters()),'trainable':sum(p.numel() for p in params)}
    (out/'parameters.json').write_text(json.dumps(counts,indent=2))
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    for step in range(start,a.steps):
        examples=[]
        for _ in range(a.batch_size):
            if position==len(order):random.shuffle(order);position=0
            examples.append(ds[order[position]]);position+=1
        inputs,targets=load_world_batch(examples,a.target_cache)
        optimizer.zero_grad(set_to_none=True)
        result=policy(examples,inputs,targets)
        if not torch.isfinite(result['loss']):raise FloatingPointError(f'Nonfinite loss at {step}')
        result['loss'].backward()
        grad={}
        for name,param in policy.named_parameters():
            if param.grad is not None:
                group=gradient_group(name)
                grad[group]=grad.get(group,0.)+float(param.grad.float().square().sum())
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();record(a.ledger,a.run_id,step+1)
        row={'step':step+1,'loss':float(result['loss'].detach()),'losses':{k:float(v.detach()) for k,v in result['losses'].items()},
             'gradient_norms':{k:v**.5 for k,v in grad.items()},'peak_memory_bytes':torch.cuda.max_memory_allocated(),
             'elapsed_seconds':time.perf_counter()-begin,'tokens':[e['token'] for e in examples]}
        with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if (step+1)%10==0:print(json.dumps(row),flush=True)
        if (step+1)%a.save_every==0 or step+1==a.steps or step+1==a.stop_after:
            payload=checkpoint(policy,optimizer,scheduler,step+1,order,position,a)
            tmp=out/'checkpoint.tmp';torch.save(payload,tmp);tmp.replace(out/'checkpoint.pt')
        if step+1==a.stop_after:
            return
    updates={}
    for name,param in policy.named_parameters():
        if name in initial_probe:
            group=gradient_group(name);entry=updates.setdefault(group,{'optimizer_parameter_tensors':0,'changed_probe_tensors':0})
            entry['optimizer_parameter_tensors']+=1
            entry['changed_probe_tensors']+=int(not torch.equal(initial_probe[name],param.detach().flatten()[:8].cpu()))
    (out/'parameter_updates.json').write_text(json.dumps(updates,indent=2))
    record(a.ledger,a.run_id,a.steps,'complete')

if __name__=='__main__':main()
