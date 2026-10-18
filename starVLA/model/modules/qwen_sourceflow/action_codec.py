# Copyright (c) 2026 The SimWAM Authors. MIT; see vendor/LICENSE.SimWAM.
# Exact source absolute trajectory functions. No z-score or sin/cos conversion.
import torch

class ActionCodec:
    horizon = 8
    action_dim = 3
    interval = 0.5

    @staticmethod
    def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

    @staticmethod
    def denorm_odo(normalized_trajectory: torch.Tensor) -> torch.Tensor:
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)
