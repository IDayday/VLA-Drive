"""Explicit conversion of TRAIN exploration scales into the unchanged D codec.

The upstream policy uses per-pose q01/q99 normalization. D46 uses fixed global
absolute min/max. Equal numeric `a` therefore does not mean equal physical
exploration. This optional calibration changes covariance, never IL units.
It is NOT enabled in the running frozen AUG3/AUG4 recipes.
"""
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def codec_jacobian():
    zero=torch.zeros(1,8,3,dtype=torch.float64)
    return (ActionCodec.denorm_odo(zero+1)-ActionCodec.denorm_odo(zero))[0]


def calibrated_scale(q01,q99,*,relative_scale=(.05,.05,.02),physical_floor=(.02,.02,.002)):
    """Return a[8,3], physical multiplier[8,3], in m/m/rad respectively.

    Desired physical multiplier = max(a_relative*(q99-q01)/2, floor).
    D multiplier = desired / D codec Jacobian. Both sampling mean/variance and
    probability re-scoring must use this SAME complete matrix. No smoothing,
    truncation or G16 subdivision. Quantiles must come from legal TRAIN only.
    """
    low=torch.as_tensor(q01,dtype=torch.float64);high=torch.as_tensor(q99,dtype=torch.float64)
    relative=torch.as_tensor(relative_scale,dtype=torch.float64);floor=torch.as_tensor(physical_floor,dtype=torch.float64)
    if(low.shape!=(8,3)or high.shape!=low.shape or relative.shape!=(3,)or floor.shape!=(3,)
        or not all(torch.isfinite(x).all()for x in(low,high,relative,floor))
        or(high<=low).any()or(relative<=0).any()or(floor<=0).any()):raise ValueError('Finite positive per-pose TRAIN ranges and covariance floor required')
    desired=torch.maximum(relative*(high-low)/2,floor)
    return (desired/codec_jacobian()).float(),desired
