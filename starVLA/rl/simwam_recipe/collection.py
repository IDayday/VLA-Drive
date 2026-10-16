"""Fresh actual source chains and fixed on-policy update material."""
import copy
import time
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from .contracts import fingerprint,registered_stochastic_steps
from .source_advantages import advantages


@torch.no_grad()
def collect(actor,condition,token,log,service,generator,*,act_norm,variant,version,progress=None):
    cond={'context':condition.expand(8,-1,-1).contiguous()}
    actor._noise_audit=[]
    begin=time.monotonic()
    chain,times,deltas,indices=actor.sample_action_chain(cond,8,generator=generator)
    if chain.shape!=(8,11,8,4) or indices.tolist()!=registered_stochastic_steps(variant) or not torch.isfinite(chain).all():
        raise ValueError('Invalid source chain; no replacement candidates allowed')
    torch.cuda.synchronize();rollout_seconds=time.monotonic()-begin
    if progress:progress('generated',rollout_seconds,8)
    requested=decode_request(chain[:,-1].cpu().numpy(),act_norm=act_norm)
    begin=time.monotonic();records=service.score((token,log,requested));scoring_seconds=time.monotonic()-begin
    if progress:progress('scored',scoring_seconds,8)
    key='train_reward_source' if variant=='R_SOURCE' else 'canonical_PDMS'
    rewards=torch.tensor([[r['readouts'][key] for r in records]],device=actor.device,dtype=torch.float32)
    begin=time.monotonic()
    old_logp=actor.action_chain_logprobs(cond,chain,times,deltas,stoch_idx=indices).detach()
    old=copy.deepcopy(actor._probability_audit['current'])
    torch.cuda.synchronize();old_probability_seconds=time.monotonic()-begin
    local=dict(cond_bg=cond,chain=chain.detach(),timesteps=times,deltas=deltas,stoch_idx=indices,
               old_logp=old_logp,rewards=rewards,old_probability=old,records=records,token=token,log=log,
               behavior_version=version,noise_audit=copy.deepcopy(actor._noise_audit),
               seconds=dict(rollout=rollout_seconds,scoring_wait=scoring_seconds,old_probability=old_probability_seconds,
                            scoring_cpu=sum(r['scoring_seconds'] for r in records)),
               protocol_sha256=service.protocol['sha256'])
    return local


def finalize_advantages(rollout):
    """Called only after every rank has successfully generated and scored."""
    stats=advantages(rollout['rewards'])
    rollout.update({k:stats[k] for k in ('advantages','adv_mask','global_std','raw_group_std')})
    rollout['advantage_statistics']={k:v for k,v in stats.items() if k not in ('advantages','adv_mask','global_std','raw_group_std')}
    return rollout


def exploration_summary(rollout):
    metrics=[r['metrics'] for r in rollout['records']]
    scores=np.array([m['score'] for m in metrics]);ep=np.array([m['ego_progress'] for m in metrics])
    safe=np.array([all(m[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')) for m in metrics])
    safety_diff=any(len({m[k] for m in metrics})>1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
    other=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort','driving_direction_compliance']
    progress_only=np.ptp(scores)>0 and all(len({m[k] for m in metrics})==1 for k in other)
    positive=rollout['advantages'].cpu().numpy()>0
    best_safe=float(scores[safe].max()) if safe.any() else None
    return dict(canonical_random_mean=float(scores.mean()),train_reward_mean=float(rollout['rewards'].mean()),
                NC=float(np.mean([m['no_at_fault_collisions'] for m in metrics])),
                DAC=float(np.mean([m['drivable_area_compliance'] for m in metrics])),
                TTC=float(np.mean([m['time_to_collision_within_bound'] for m in metrics])),EP=float(ep.mean()),
                all_full=bool(np.all(scores==1)),all_zero=bool(np.all(scores==0)),tied=bool(np.ptp(scores)==0),
                safety_differing=bool(safety_diff),progress_only=bool(progress_only),safe_fraction=float(safe.mean()),
                finite_best_safe=best_safe,positive_unsafe_with_safe_alternative=int((positive&~safe).sum()) if safe.any() else 0,
                positive_safe=int((positive&safe).sum()),positive_still_unsafe=int((positive&~safe).sum()),
                zero_PG=not bool(rollout['adv_mask'].any()),global_reward_std=float(rollout['global_std']))
