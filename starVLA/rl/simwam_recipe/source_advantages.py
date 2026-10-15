"""Source group centering / global RAW reward std, with explicit FP64 centering."""
import torch
import torch.distributed as dist


@torch.no_grad()
def advantages(rewards, *, all_rewards=None, eps=1e-8):
    if rewards.ndim!=2 or not torch.isfinite(rewards).all():raise ValueError('Rewards must be finite [scene,candidate]')
    values=rewards.detach().double()
    if all_rewards is None:
        if dist.is_available() and dist.is_initialized():
            gathered=[torch.empty_like(values) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered,values)
            all_rewards=torch.cat(gathered,dim=0)
        else:all_rewards=values
    all_rewards=all_rewards.detach().double()
    std=all_rewards.std(unbiased=False)
    # Translation by the first candidate removes rounded constant-value residuals.
    offsets=values-values[:,:1];centered=offsets-offsets.mean(dim=1,keepdim=True)
    group_std=offsets.std(dim=1,unbiased=False,keepdim=True)
    a=centered/(std+eps)
    mask=(group_std>eps).expand_as(values).reshape(-1).float()
    return dict(advantages=a.reshape(-1).float().detach(),adv_mask=mask.detach(),
                global_std=std.detach(),raw_group_std=group_std.detach(),
                all_full=int((values==1).all(dim=1).sum()),all_zero=int((values==0).all(dim=1).sum()),
                tied=int((group_std<=eps).sum()),
                effective_PG_candidate_fraction=float(mask.mean()),
                normalization='group centered / population std of unfiltered global RAW rewards',
                reduction='source local effective mean, then equal mean across ranks')
