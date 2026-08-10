"""Interpretable MVP planning losses."""

from typing import Dict

import torch
from torch.nn import functional as F

from .trajectory_codec import TrajectoryCodec


def trajectory_refinement_losses(
    final_action: torch.Tensor,
    target_action: torch.Tensor,
    delta_physical: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return scalar plan and delta regularization losses.

    ``final_action`` and ``target_action`` are ``[B,M,H,4]`` (target may use
    M=1); ``delta_physical`` is ``[B,M,H,3]``.
    """

    if final_action.ndim != 4 or final_action.shape[-1] != 4:
        raise ValueError("final_action must have shape [B,M,H,4]")
    if target_action.ndim == 3:
        target_action = target_action[:, None]
    if target_action.shape[0] != final_action.shape[0] or target_action.shape[-2:] != final_action.shape[-2:]:
        raise ValueError("target_action must match B,H,4")
    if target_action.shape[1] == 1 and final_action.shape[1] > 1:
        target_action = target_action.expand(-1, final_action.shape[1], -1, -1)
    if target_action.shape != final_action.shape or delta_physical.shape != (*final_action.shape[:-1], 3):
        raise ValueError("final, target and delta shapes are inconsistent")
    codec = TrajectoryCodec()
    final_physical = codec.decode_action(final_action)
    target_physical = codec.decode_action(target_action)
    if not isinstance(final_physical, torch.Tensor) or not isinstance(target_physical, torch.Tensor):
        raise TypeError("loss expects torch tensors")
    xy_loss = F.smooth_l1_loss(final_physical[..., :2], target_physical[..., :2])
    heading_loss = (1.0 - torch.cos(final_physical[..., 2] - target_physical[..., 2])).mean()
    return {
        "plan": xy_loss + heading_loss,
        "delta_reg": delta_physical.abs().mean(),
    }
