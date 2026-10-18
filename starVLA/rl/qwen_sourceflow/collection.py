"""Same fresh source chain, scoring and probability lifecycle; native decode only."""
import copy
import time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


@torch.no_grad()
def collect(actor, condition, token, log, service, generator, *, variant, version, progress=None):
    actor._noise_audit=[]
    begin=time.monotonic()
    chain,times,deltas,indices=actor.sample_action_chain(condition,8,generator=generator)
    if chain.shape!=(8,11,8,3) or indices.tolist()!=[7,8,9] or not torch.isfinite(chain).all():
        raise ValueError('Invalid native source chain; no replacement candidates')
    torch.cuda.synchronize();rollout_seconds=time.monotonic()-begin
    if progress:progress('generated',rollout_seconds,8)
    requested=ActionCodec.denorm_odo(chain[:,-1]).cpu().numpy()
    begin=time.monotonic();records=service.score((token,log,requested));scoring_seconds=time.monotonic()-begin
    if progress:progress('scored',scoring_seconds,8)
    key='train_reward_source' if variant=='R_SOURCE' else 'canonical_PDMS'
    rewards=torch.tensor([[r['readouts'][key] for r in records]],device=actor.device,dtype=torch.float32)
    begin=time.monotonic()
    old_logp=actor.action_chain_logprobs(condition,chain,times,deltas,stoch_idx=indices).detach()
    old=copy.deepcopy(actor._probability_audit['current'])
    sampling_check=None
    if actor.record_sampling_terms:
        if len(actor.sampling_terms)!=3:raise ValueError('Actual source stochastic sampling window changed')
        for i,(mean,std) in enumerate(actor.sampling_terms):
            torch.testing.assert_close(mean,old['mean'][:,i],rtol=0,atol=0)
            torch.testing.assert_close(torch.tensor(std,device=old['std'].device,dtype=torch.float32),old['std'][i],rtol=0,atol=0)
        sampling_check=dict(status='PASS',actual_sample_vs_recompute_mean_std='exact',steps=[7,8,9])
    torch.cuda.synchronize();probability_seconds=time.monotonic()-begin
    return dict(cond_bg=condition,chain=chain.detach(),timesteps=times,deltas=deltas,stoch_idx=indices,
        old_logp=old_logp,rewards=rewards,old_probability=old,records=records,token=token,log=log,
        behavior_version=version,noise_audit=copy.deepcopy(actor._noise_audit),qualification_sampling_check=sampling_check,
        seconds=dict(rollout=rollout_seconds,scoring_wait=scoring_seconds,old_probability=probability_seconds,
            scoring_cpu=sum(r['scoring_seconds'] for r in records)),protocol_sha256=service.protocol['sha256'])
