"""Explicit D adaptation of CFM-energy ratio; NOT a trajectory likelihood."""
import torch


def velocity_energy(state, prediction, target, sigma):
    if (sigma<=0).any():raise ValueError('Positive sigma required')
    velocity=(state.float()-prediction.float())/sigma.float()[:,None,None]
    return (velocity-target.detach().float()).square().mean((-2,-1))


def awm_loss(energy, advantage, reference_prediction, prediction, reference_coefficient=1.):
    # Conservative positive-only adaptation, NOT the full signed AWM theorem.
    weight=advantage.detach().float().clamp(0,5)
    positive=(weight*energy).mean()
    keep=(prediction.float()-reference_prediction.detach().float()).square().mean()
    return positive+reference_coefficient*keep


def mc_predictions_energy(policy,prefix,states,sigmas,targets,*,fuse=False):
    """Original per-MC clean-x energies, optionally one MC*G denoiser call.

    No random draws, new sigma weighting or MC reduction. Shared candidates
    use the exact same scene CP via zero-index narrow+expand, never 64 physical
    copies of long KV. Trainable prefixes are rebuilt by the caller each pass.
    """
    if not fuse:
        predictions=[];energies=[]
        for state,sigma,target in zip(states,sigmas,targets):
            with torch.autocast(states.device.type,dtype=torch.bfloat16):pred=policy.denoise_x0(state,sigma,prefix)
            predictions.append(pred);energies.append(velocity_energy(state,pred,target,sigma))
        return torch.stack(energies),torch.stack(predictions)
    from .fast_prefix import fast_index_select
    if states.ndim!=4 or states.shape[-2:]!=(8,3)or sigmas.shape!=states.shape[:2]or targets.shape!=states.shape:
        raise ValueError('Fused MC shape must be [MC,G,8,3] and sigma[MC,G]')
    mc,g=states.shape[:2];base=getattr(prefix,'tensors',prefix)
    if mc!=4 or not g or len(base.valid)!=g:raise ValueError('Only original MC4 same-scene G prefix')
    if g>1 and(any(x.stride(0)!=0 for pair in base.layer_kv for x in pair)or
        any(x.stride(0)!=0 for x in(base.valid,base.next_positions,base.reasoning))):
        raise ValueError('MC fusion requires shared single-scene stride0 prefix')
    selected=fast_index_select(prefix,torch.zeros(mc*g,device=states.device,dtype=torch.long))
    flat=states.flatten(0,1);sigma=sigmas.flatten(0,1);target=targets.flatten(0,1)
    with torch.autocast(states.device.type,dtype=torch.bfloat16):pred=policy.denoise_x0(flat,sigma,selected)
    return velocity_energy(flat,pred,target,sigma).reshape(mc,g),pred.reshape(mc,g,8,3)
