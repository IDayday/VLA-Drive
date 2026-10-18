"""FP32, per-scene task means with exact DDP effective-count scaling."""
import torch
from torch import distributed as dist
from torch.nn import functional as F


def scene_mean(error, valid, global_valid_scenes=None):
    if error.shape != valid.shape:
        valid = torch.broadcast_to(valid,error.shape)
    valid = valid.bool()
    # Unknown NaNs cannot contaminate a valid reduction.
    clean = torch.where(valid,error.float(),torch.zeros_like(error,dtype=torch.float32))
    counts = valid.flatten(1).sum(1)
    means = clean.flatten(1).sum(1)/counts.clamp_min(1)
    active = counts > 0
    local_count = active.sum()
    world = dist.get_world_size() if dist.is_initialized() else 1
    if global_valid_scenes is None:
        denominator = local_count.detach().clone()
        if world>1:dist.all_reduce(denominator)
    else:
        denominator = torch.as_tensor(global_valid_scenes,device=error.device)
    # Caller must supply the full optimizer-batch denominator when accumulating.
    return (means*active).sum()*world/denominator.clamp_min(1), local_count.detach()


def feature_loss(prediction,target,valid,mean,std,global_valid_scenes=None,cosine_weight=.01):
    target=target.detach().float()
    std=torch.as_tensor(std,device=prediction.device,dtype=torch.float32).clamp_min(1e-6)
    mean=torch.as_tensor(mean,device=prediction.device,dtype=torch.float32)
    p,t=(prediction.float()-mean)/std,(target-mean)/std
    mse,count=scene_mean((p-t).square(),valid,global_valid_scenes)
    # Feature-channel-last contract for this loss, including sampled BEV features.
    mask=torch.broadcast_to(valid,p.shape).all(-1)
    cos,_=scene_mean(1-F.cosine_similarity(p,t,dim=-1,eps=1e-6),mask,global_valid_scenes)
    return mse+cosine_weight*cos,count


def occupancy_loss(logits,labels,valid,global_valid_scenes=None,positive_weight=1.):
    labels=labels.detach()
    valid=valid.bool() & (labels>=0)
    if logits.shape == labels.shape:
        loss=F.binary_cross_entropy_with_logits(logits.float(),torch.where(valid,labels,0).float(),
              reduction="none",pos_weight=logits.new_tensor(positive_weight,dtype=torch.float32))
    elif logits.shape[1]==2 and logits.shape[:1]+logits.shape[2:]==labels.shape:
        loss=F.cross_entropy(logits.float(),torch.where(valid,labels,0).long(),reduction="none")
    else:raise ValueError("Occupancy logits/label layout mismatch")
    return scene_mean(loss,valid,global_valid_scenes)
