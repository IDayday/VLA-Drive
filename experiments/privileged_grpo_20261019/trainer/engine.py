"""Scope-safe continuous Flow-GRPO update shared by real runs and CPU tests."""
import hashlib,itertools,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/grpo_smoke_v1'))
from grpo_math import transition,log_prob,advantage,objective
SCOPES={'joint':('qwen_vl_interface.','action_model.','action_input_model.'),'vlm_only':('qwen_vl_interface.',),'dit_only':('action_model.',)}
def arm_scope(arm):return 'vlm_only' if arm.endswith('_VLM') else 'dit_only' if arm.endswith('_DiT') else 'joint'
def input_arm(arm):return arm.split('_')[0]
def seed(*parts):return int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:8],16)
def order(tokens,training_seed,epoch):return sorted(tokens,key=lambda t:(seed('order',training_seed,epoch,t),t))
def scope(model,name):
    inventory=[]
    for n,p in model.named_parameters():
        if not n.startswith(SCOPES['joint']):raise ValueError('Unclassified parameter '+n)
        p.requires_grad_(n.startswith(SCOPES[name]));inventory.append({'name':n,'shape':list(p.shape),'numel':p.numel(),'trainable':p.requires_grad})
    model.eval();return inventory

def optimizer(model,cfg):
    groups=[]
    for label,prefix,lr in [('vlm','qwen_vl_interface.',cfg['vlm_lr']),('action','action_model.',cfg['action_lr']),('history','action_input_model.',cfg['action_lr'])]:
        params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:groups.append({'params':params,'lr':lr,'name':label})
    assert sum(len(g['params']) for g in groups)==sum(p.requires_grad for p in model.parameters())
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)

def state_hash(model,which='frozen'):
    h=hashlib.sha256()
    entries=[(n,p) for n,p in model.named_parameters() if which=='all' or not p.requires_grad]+list(model.named_buffers())
    for n,t in entries:
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        flat=t.detach().reshape(-1)
        for block in flat.split(1024*1024):h.update(block.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def sample_indices(size,device):
    count=min(17,size)
    return torch.tensor([i*(size-1)//max(count-1,1) for i in range(count)],device=device,dtype=torch.long)

def witnesses(model):
    # Small deterministic samples per tensor for inexpensive change diagnostics.
    result={}
    for n,p in model.named_parameters():
        if p.requires_grad and p.numel():
            flat=p.detach().reshape(-1);idx=sample_indices(flat.numel(),flat.device);result[n]=flat[idx].float().cpu().clone()
    return result

def changes(before,after):
    return {group:max((float((after[n]-v).abs().max()) for n,v in before.items() if n.startswith(prefix)),default=0.) for group,prefix in [('vlm','qwen_vl_interface.'),('action','action_model.'),('history','action_input_model.')]}

def sampling(model,cond,velocity,noise,eta):
    x=noise[0];trans=[]
    for step in range(10):
        mean,std=transition(x,velocity(model,cond,x,step),step,eta=eta);nxt=mean+std*noise[step+1]
        lp=log_prob(nxt,mean,std)
        if not torch.isfinite(nxt).all() or not torch.isfinite(lp).all():raise FloatingPointError('Invalid flow sample')
        trans.append({'step':step,'x':x.detach(),'next':nxt.detach(),'old_logp':lp.detach()});x=nxt
    return x,trans

def make_noise(device,value,g=8):
    gen=torch.Generator(device=device).manual_seed(value)
    return [torch.randn((g,8,4),device=device,dtype=torch.float32,generator=gen) for _ in range(11)]
def noise_hash(noise):return hashlib.sha256(torch.stack(noise).cpu().contiguous().numpy().tobytes()).hexdigest()

def backward_scene(model,reference,condition,velocity,noise,reward_fn,cfg,divisor):
    """Leaf-condition bridge bounds memory without detaching VLM gradients."""
    with torch.no_grad():
        cond=condition(model);ref_cond=condition(reference);raw,trans=sampling(model,cond,velocity,noise,cfg['eta'])
        ref_means=[transition(t['x'],velocity(reference,ref_cond,t['x'],t['step']),t['step'],eta=cfg['eta'])[0] for t in trans]
    rewards,record=reward_fn(raw)
    rewards=torch.as_tensor(rewards,device=cond.device,dtype=torch.float32)
    if rewards.shape!=(8,) or not torch.isfinite(rewards).all() or torch.any((rewards<0)|(rewards>1)):raise ValueError('Invalid PDMS rewards')
    adv,std_reward=advantage(rewards)
    if std_reward<1e-6:adv=torch.zeros_like(adv)
    graph=checkpoint(lambda:condition(model),use_reentrant=False)
    if not graph.requires_grad:raise RuntimeError('Trainable condition graph detached')
    if not torch.allclose(graph.detach(),cond,rtol=1e-5,atol=1e-5):raise RuntimeError('Recomputed condition differs')
    leaf=graph.detach().requires_grad_();losses=[];kls=[];lp_error=0.
    for t,ref_mean in zip(trans,ref_means):
        mean,std=transition(t['x'],velocity(model,leaf,t['x'],t['step']),t['step'],eta=cfg['eta']);lp=log_prob(t['next'],mean,std)
        error=float((lp.detach()-t['old_logp']).abs().max());lp_error=max(lp_error,error)
        if error>=1e-3:raise RuntimeError('On-policy log-probability replay failed')
        loss,ratio,kl=objective(lp,t['old_logp'],adv,mean,ref_mean,std,clip=cfg['clip'],beta=cfg['beta'])
        if not torch.isfinite(loss) or not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite GRPO objective')
        (loss/(10*divisor)).backward();losses.append(float(loss.detach()));kls.extend(kl.detach().cpu().tolist())
    if leaf.grad is None:raise RuntimeError('No gradient through frozen/trainable action head')
    graph.backward(leaf.grad)
    return dict(record,loss_mean=float(np.mean(losses)),kl_mean=float(np.mean(kls)),old_new_logprob_max_abs_error=lp_error,reward_std=float(std_reward),noise_sha256=noise_hash(noise))

def step(model,opt,clip):
    stats={}
    for label,prefix in [('vlm','qwen_vl_interface.'),('action','action_model.'),('history','action_input_model.')]:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite gradient '+label)
        stats[label]=float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
    for n,p in model.named_parameters():
        if not p.requires_grad and p.grad is not None:raise RuntimeError('Frozen parameter has gradient '+n)
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip,error_if_nonfinite=True));opt.step()
    return {'gradient_norms':stats,'pre_clip_gradient_norm':norm}

def exploration(trajectories,metrics,base_pdms):
    p=np.asarray(trajectories);r=np.asarray([m['PDMS'] for m in metrics]);passed=np.array([m['PDMS']>=.8 and all(m[k]==1 for k in ['NC','DAC','TTC']) for m in metrics]);repair=np.array([m['PDMS']>base_pdms and all(m[k]==1 for k in ['NC','DAC','TTC']) for m in metrics])
    i,j=np.triu_indices(8,1);diff=np.linalg.norm(p[i,:,:2]-p[j,:,:2],axis=-1);yaw=p[i,-1,2]-p[j,-1,2]
    curves={}
    for k in [1,2,4,8]:
        idx=np.array(list(itertools.combinations(range(8),k)));curves[str(k)]={'mean_best':float(r[idx].max(1).mean()),'pass_at_k':float(passed[idx].any(1).mean()),'repair_at_k':float(repair[idx].any(1).mean())}
    return {'pdms_mean':float(r.mean()),'pdms_std':float(r.std()),'oracle8':float(r.max()),'all_zero':bool(np.all(r==0)),'all_one':bool(np.all(r==1)),'constant_other':bool(r.std()<1e-6 and not(np.all(r==0) or np.all(r==1))),'effective_group':bool(r.std()>=1e-6),'repair_fraction':float(repair.mean()),'curves':curves,'pairwise_ADE_m':float(diff.mean()),'pairwise_FDE_m':float(diff[:,-1].mean()),'pairwise_heading_rad':float(np.abs(np.arctan2(np.sin(yaw),np.cos(yaw))).mean())}
