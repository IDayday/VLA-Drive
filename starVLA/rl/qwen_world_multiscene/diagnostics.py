"""Single-device baseline versus packed math/timing, without Adam or scoring."""
import time
import torch
from .objective import source_loss,source_probabilities


def _sync(actor):
    if actor.device.type=='cuda':torch.cuda.synchronize(actor.device)


def _stats(left,right):
    aa=bb=ab=dd=0.;maximum=0.
    for name,x in left.items():
        y=right[name];x=x.double();y=y.double();d=x-y
        aa+=float(x.square().sum());bb+=float(y.square().sum());ab+=float((x*y).sum());dd+=float(d.square().sum())
        maximum=max(maximum,float(d.abs().max()))
    return dict(sequential_l2=aa**.5,packed_l2=bb**.5,relative_l2_error=dd**.5/max(aa**.5,1e-30),
        cosine=ab/max((aa*bb)**.5,1e-30),max_abs_error=maximum)


def compare_execution(actor,samples,packed,*,loss,weights=None):
    """No equivalence tolerance is silently relaxed; record the actual guard."""
    n=len(samples);weights=[1/n]*n if weights is None else list(weights)
    sequential=[];refs=[]
    with torch.no_grad():
        for r in samples:
            for use_ref,target in ((False,sequential),(True,refs)):
                p,t=actor.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'],return_terms=True,use_ref=use_ref)
                target.append((p,t))
        current,_=source_probabilities(actor,samples,packed)
        reference,_=source_probabilities(actor,samples,packed,use_ref=True)
    named={k:v for k,v in actor.named_parameters()if v.requires_grad}
    actor.zero_grad(set_to_none=True);_sync(actor)
    if actor.device.type=='cuda':torch.cuda.reset_peak_memory_stats(actor.device)
    tick=time.monotonic();base=0.;seqmetrics=[]
    for weight,r in zip(weights,samples):
        value,metric=loss._policy_loss(actor,r);(float(weight)*value).backward();base+=float(weight)*float(value.detach());seqmetrics.append(metric)
    _sync(actor);sequential_seconds=time.monotonic()-tick
    sequential_peak=torch.cuda.max_memory_allocated(actor.device)if actor.device.type=='cuda'else None
    missing=[name for name,p in named.items()if p.grad is None]
    if missing:raise ValueError('Sequential missing gradients: '+repr(missing))
    grad={name:p.grad.detach().cpu().clone()for name,p in named.items()}
    actor.zero_grad(set_to_none=True);_sync(actor)
    if actor.device.type=='cuda':torch.cuda.reset_peak_memory_stats(actor.device)
    tick=time.monotonic();new,metrics,probs=source_loss(actor,samples,packed,loss=loss,weights=weights);new.backward();_sync(actor)
    packed_seconds=time.monotonic()-tick
    packed_peak=torch.cuda.max_memory_allocated(actor.device)if actor.device.type=='cuda'else None
    missing=[name for name,p in named.items()if p.grad is None]
    if missing:raise ValueError('Packed missing gradients: '+repr(missing))
    now={name:p.grad.detach().cpu()for name,p in named.items()}
    if any(not torch.isfinite(x).all()for x in now.values()):raise FloatingPointError('Nonfinite packed gradient')
    result=dict(loss_sequential=base,loss_packed=float(new.detach()),loss_absolute_error=abs(base-float(new.detach())),
        current_logprob_max_abs=float((current[0]-torch.cat([x[0]for x in sequential])).abs().max()),
        reference_logprob_max_abs=float((reference[0]-torch.cat([x[0]for x in refs])).abs().max()),
        current_mean_max_abs=float((current[1]['mean']-torch.cat([x[1]['mean']for x in sequential])).abs().max()),
        reference_mean_max_abs=float((reference[1]['mean']-torch.cat([x[1]['mean']for x in refs])).abs().max()),
        fresh_ratio_max_abs=float(((current[0]-torch.cat([r['old_logp']for r in samples])).exp()-1).abs().max()),
        gradients=_stats(grad,now),gradient_scopes={prefix:_stats({k:v for k,v in grad.items()if k.startswith(prefix)},
            {k:v for k,v in now.items()if k.startswith(prefix)})for prefix in ('action_expert.','scene_provider.','predictive_provider.')},
        per_scene_policy_losses=[float(m['policy_loss'])for m in metrics],
        sequential_seconds=sequential_seconds,packed_seconds=packed_seconds,
        sequential_peak_GPU_bytes=sequential_peak,packed_peak_GPU_bytes=packed_peak,
        trainable_numel=sum(p.numel()for p in named.values()),scene_count=n,group_size=packed['groups'],
        weights=weights,scoring_queries=0,optimizer_steps=0,
        interpretation='Single-device source PG+IL math/throughput only; no DDP, auxiliary, rollout, scoring or B1 deployment speed claim')
    result['fresh_ratio_within_1e_4']=result['fresh_ratio_max_abs']<=1e-4
    return result
