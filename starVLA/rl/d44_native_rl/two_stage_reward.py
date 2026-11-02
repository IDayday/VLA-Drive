"""TRAIN-only endpoint shaping; labels never enter current observation inputs."""
import torch


@torch.no_grad()
def endpoint_quality(ground_truth,requested,temperature=3.):
    gt=torch.as_tensor(ground_truth,device=requested.device,dtype=torch.float32)
    if requested.shape[-2:]!=(8,3)or not torch.isfinite(requested).all():raise ValueError('Malformed actual policy requests')
    if gt.shape!=(8,3)or not torch.isfinite(gt).all():
        return requested.new_zeros(len(requested)),requested.new_zeros(())
    # A locked TRAIN-only downweighting heuristic for stationary/very short
    # recorded motion; no claim that it is an official NAVSIM progress metric.
    points=torch.cat((gt.new_zeros(1,2),gt[:,:2]))
    distance=(points[1:]-points[:-1]).norm(dim=-1).sum()
    confidence=(distance/3.).clamp(0,1)
    score=torch.exp(-(requested[:,-1,:2]-gt[-1,:2]).norm(dim=-1)/temperature)
    return score,confidence
