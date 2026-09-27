"""Label-aware auxiliary task construction is permitted in training only."""
import torch


def role_completion_mask(feature_valid,active,generator,neighbor_role=None):
    if feature_valid.ndim!=4 or feature_valid.shape[:2]!=active.shape:raise ValueError('Mask contract mismatch')
    eligible=feature_valid[...,:2].any((-1,-2))&active
    b,a=active.shape
    if not eligible[:,0].all():raise ValueError('Role task requires a valid ego target')
    if neighbor_role is None:
        # Exactly balanced roles when possible, random batch permutation; no complete-future requirement.
        neighbor_role=(torch.arange(b)%2==1)[torch.randperm(b,generator=generator)].to(active.device)
    hidden=torch.zeros_like(active);chosen=torch.zeros(b,device=active.device,dtype=torch.long)
    fallback=torch.zeros(b,device=active.device,dtype=torch.bool)
    for i in range(b):
        candidates=torch.where(eligible[i,1:])[0]+1
        if bool(neighbor_role[i]) and len(candidates):
            chosen[i]=candidates[int(torch.randint(len(candidates),(1,),generator=generator))]
        else:fallback[i]=bool(neighbor_role[i])
        hidden[i,chosen[i]]=True
    valid_hidden=feature_valid&hidden[:,:,None,None]
    if not valid_hidden.flatten(1).any(-1).all():raise RuntimeError('Role task unexpectedly empty')
    return hidden,{'neighbor_requested':neighbor_role,'chosen_actor':chosen,'ego_only_fallback':fallback,'valid_hidden_coordinates':valid_hidden.sum((-1,-2,-3))}
