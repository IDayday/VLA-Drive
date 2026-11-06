"""Released AUG Stage-I action reward/normalization in native D coordinates.

The official split-advantage driver uses population std, a hard std<1e-6
zero-signal gate, and no epsilon in the surviving denominator. It is not the
generic verl sample-std implementation. Canonical scorer records stay intact.
"""
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def reverse_travel(physical):
    if physical.shape[-2:]!=(8,3) or not torch.isfinite(physical).all():
        raise ValueError('Expected finite physical trajectories [...,8,3]')
    x=physical.double()[...,0]
    dx=torch.diff(x,dim=-1,prepend=torch.zeros_like(x[...,:1]))
    return torch.where(dx<-.01,-dx,torch.zeros_like(dx)).sum(-1)


def action_reward(scene):
    score=torch.tensor([r['metrics']['score']for r in scene['records']],
                       dtype=torch.float64,device=scene['chain'].device)
    physical=ActionCodec.denorm_odo(scene['chain'][:,-1].detach())
    # stage1.yaml: planning=5, pdms=1, all other planning coefficients=0,
    # reverse_gate=True. No GT exemption on this boolean gate in the release.
    return 5*score*(reverse_travel(physical)==0)


def normalize_centered(raw):
    std=raw.double().std(-1,unbiased=False,keepdim=True)
    return torch.where(std>=1e-6,raw.double()/std.clamp_min(1e-6),0.).float()
