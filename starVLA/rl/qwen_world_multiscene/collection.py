"""One actual S*G source rollout, split into S separately scored G groups."""
import copy,time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest
from starVLA.rl.qwen_world_multiscene.conditions import PackedConditions


def slice_audit(a,sl):
    result={k:(v[sl].detach().clone()if torch.is_tensor(v)and v.ndim>1 else v.detach().clone()if torch.is_tensor(v)else v)for k,v in a.items()}
    result['source_clamp_hit_rate']=float((result['raw_gaussian_logprob']!=result['source_clamped_logprob']).float().mean())
    return result


@torch.no_grad()
def collect_packed(actor,conditions,descriptors,tokens,logs,service,generator,*,version,pack_index,valid_masks=None):
    g=descriptors[0]['repeats'];n=len(tokens)
    raw=PackedConditions(conditions,selected_layers=actor.scene_provider.selected_layers).pack(descriptors,actor.device,valid_masks=valid_masks)
    before=generator.get_state().clone();actor._noise_audit=[];previous=actor.record_sampling_terms
    actor.record_sampling_terms=True
    tick=time.monotonic()
    try:
        chain,times,deltas,steps=actor.sample_action_chain(raw,8,generator=generator)
        sampling=copy.deepcopy(actor.sampling_terms)
    finally:actor.record_sampling_terms=previous
    if actor.device.type=='cuda':torch.cuda.synchronize(actor.device)
    rollout_seconds=time.monotonic()-tick;after=generator.get_state().clone();noise=copy.deepcopy(actor._noise_audit)
    tick=time.monotonic();old=actor.action_chain_logprobs(raw,chain,times,deltas,stoch_idx=steps).detach()
    audit=copy.deepcopy(actor._probability_audit['current'])
    if actor.device.type=='cuda':torch.cuda.synchronize(actor.device)
    probability_seconds=time.monotonic()-tick
    if chain.shape!=(n*g,11,8,3)or steps.tolist()!=[7,8,9]or len(sampling)!=3:raise ValueError('Packed source W3 shape changed')
    errors=[dict(step=int(steps[i]),mean_max_abs=float((mean-audit['mean'][:,i]).abs().max()),
        std_abs=float((audit['std'][i]-torch.tensor(std,device=actor.device,dtype=torch.float32)).abs()))for i,(mean,std)in enumerate(sampling)]
    if any(x['mean_max_abs']or x['std_abs']for x in errors):raise ValueError('Packed sampler/replay mean or std mismatch: '+repr(errors))
    leaves=[];scoring_seconds=0.
    for i,(token,log,descriptor)in enumerate(zip(tokens,logs,descriptors)):
        sl=slice(i*g,(i+1)*g);requested=ActionCodec.denorm_odo(chain[sl,-1]).float().cpu().numpy()
        tick=time.monotonic();records=service.score((token,log,requested));elapsed=time.monotonic()-tick;scoring_seconds+=elapsed
        if len(records)!=g:raise ValueError('Packed collection must score each original G16 group independently')
        rewards=torch.tensor([[r['readouts']['canonical_PDMS']for r in records]],device=actor.device,dtype=torch.float32)
        leaves.append(dict(cond_bg=copy.deepcopy(descriptor),chain=chain[sl].detach(),timesteps=times,deltas=deltas,stoch_idx=steps,
            old_logp=old[sl].detach(),old_probability=slice_audit(audit,sl),rewards=rewards,records=records,
            token=token,log=log,behavior_version=copy.deepcopy(version),group_size=g,
            packed_position=dict(pack_index=pack_index,scene_ordinal=i,pack_tokens=list(tokens),scenes=n),
            qualification_sampling_check=dict(status='PASS',actual_sample_vs_recompute_mean_std='exact_required',steps=errors),
            qualification_sample_terms=[(mean[sl].detach().clone(),std)for mean,std in sampling],
            noise_audit=dict(scope='shared genuine packed draw; do not count once per scene',pack_index=pack_index),
            seconds=dict(rollout=rollout_seconds/n,old_probability=probability_seconds/n,scoring_wait=elapsed,
                scoring_cpu=sum(r['scoring_seconds']for r in records)),protocol_sha256=service.protocol['sha256']))
    material=dict(pack_index=pack_index,tokens=list(tokens),scenes=n,G=g,steps=[7,8,9],
        generator_before=before,generator_after=after,noise_audit=noise,
        sampled_mean=torch.stack([x[0]for x in sampling],1).detach(),sampled_std=audit['std'].detach().clone(),
        old_logp=old.detach(),chain_sha256=digest(chain),old_probability_sha256=digest({k:v for k,v in audit.items()if torch.is_tensor(v)}),
        seconds=dict(rollout=rollout_seconds,old_probability=probability_seconds,scoring_wait=scoring_seconds))
    return leaves,material
