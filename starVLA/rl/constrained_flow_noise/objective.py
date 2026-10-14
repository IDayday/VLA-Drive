"""Scene-normalized full-chain proximal loss and unfiltered SFT retention."""
import torch
from starVLA.rl.flow_grpo.math import clipped_surrogate, conditional_kl
from starVLA.rl.flow_grpo.temporal_noise import whiten
from .transition import gaussian_kl_parts, gaussian_entropy
from .contracts import seed_for


def reference_kl(current, reference, spec):
    mean,std=current['mean'],current['std']
    ref_mean,ref_std=reference['mean'].detach(),reference['std'].detach()
    if spec.variant == 'N_FLOW_NOISE':
        mean_kl,variance_kl=gaussian_kl_parts(mean,std,ref_mean,ref_std)
    else:
        mean_kl=.5*(whiten(mean-ref_mean,spec.correlation)/ref_std).square()
        total=conditional_kl(mean,std,ref_mean,ref_std,temporal_correlation=spec.correlation)
        variance_kl=total-mean_kl
    return mean_kl,variance_kl


def replay_loss(actor, raw, target, seed):
    head=actor.policy.action_model
    repeats=int(head.full_config.framework.action_model.get('flow_train_repeats',
                       head.full_config.framework.action_model.repeated_diffusion_steps))
    target=target.detach().float().repeat(repeats,1,1)
    raw=raw.detach().repeat(repeats,1,1)
    # Separate fresh FM RNG. Candidate root/innovations are never the FM coupling.
    with torch.random.fork_rng(devices=[raw.device.index] if raw.is_cuda else []):
        torch.random.default_generator.manual_seed(int(seed))
        if raw.is_cuda:torch.cuda.default_generators[raw.device.index].manual_seed(int(seed))
        noise=torch.randn_like(target)
        time=head.sample_time(len(target),target.device,target.dtype)
    # Original SFT training uses its configured DiT dropout (0.2). PG remains
    # deterministic eval. Isolate dropout from both FM draws and rollout RNG,
    # and restore the exact module modes even if a forward raises.
    modes={module:module.training for module in head.modules()}
    try:
        head.train(True)
        with torch.random.fork_rng(devices=[raw.device.index] if raw.is_cuda else []):
            dropout_seed=seed_for('cfn_GT_replay_dropout',42,int(seed))
            torch.random.default_generator.manual_seed(dropout_seed)
            if raw.is_cuda:torch.cuda.default_generators[raw.device.index].manual_seed(dropout_seed)
            with torch.autocast(raw.device.type,dtype=torch.float32,enabled=raw.is_cuda):
                return head(raw,target,explicit_randomness={'noise':noise,'t':time})
    finally:
        for module,mode in modes.items():module.training=mode


def actor_loss(actor, *, pg, retention, replay, pg_factor, retention_factor,
               kl_coefficient, replay_seed, clip=.02, chunk=None):
    # Padding reuses a real saved group for a well-formed graph, with exactly
    # zero PG/reference weight. It never changes sample or cost denominators.
    current=actor.saved_statistics(pg['raw'],pg['chain'],chunk=chunk)
    old=pg['old_elementwise_logp'].detach().mean((-2,-1))
    loss,ratio=clipped_surrogate(current['mean_transition_logp'],old,pg['advantage'][...,None],clip)
    pg_loss=loss.mean()*pg_factor
    if pg['scene_key'] == retention['scene_key']:
        ref_current=current
    else:
        ref_current=actor.saved_statistics(retention['raw'],retention['chain'],chunk=chunk)
    mean_kl,var_kl=reference_kl(ref_current,retention['reference'],actor.spec)
    mean_kl,var_kl=mean_kl.mean(),var_kl.mean()
    retention_loss=(mean_kl+var_kl)*retention_factor
    replay_value=replay_loss(actor,replay['raw'],replay['target'],replay_seed)
    total=pg_loss+kl_coefficient*retention_loss+.1*replay_value
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite actor objective; stop batch, never replace its value')
    diagnostics={'PG':pg_loss.detach(),'reference_mean_KL':mean_kl.detach(),
                 'reference_variance_KL':var_kl.detach(),'reference_KL_weighted':retention_loss.detach(),
                 'GT_replay':replay_value.detach(),'loss':total.detach(),
                 'ratio_mean':ratio.detach().mean(),'ratio_max_abs_from_one':(ratio.detach()-1).abs().max(),
                 'clip_fraction':((ratio.detach()<1-clip)|(ratio.detach()>1+clip)).float().mean(),
                 'joint_transition_logratio_abs_max':((current['joint_transition_logp']-pg['old_elementwise_logp'].detach().sum((-2,-1)))).detach().abs().max(),
                 'sigma_mean':current['std'].detach().mean()}
    if actor.spec.variant == 'N_FLOW_NOISE':
        diagnostics['entropy_per_dimension']=gaussian_entropy(current['std']).detach().mean()
    return total,diagnostics
