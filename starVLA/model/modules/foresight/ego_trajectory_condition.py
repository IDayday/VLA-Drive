"""Training-only GT action tokens, separate from the execution action encoder."""
import torch
from torch import nn


class EgoTrajectoryConditionEncoder(nn.Module):
    def __init__(self, dim=512, xy_scale=20.):
        super().__init__()
        if xy_scale <= 0:
            raise ValueError('Positive physical coordinate scale required')
        self.xy_scale = float(xy_scale)
        self.point = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer('physical_times_s', torch.arange(1, 9).float() * .5, persistent=True)

    def forward(self, gt_xy_sincos):
        if gt_xy_sincos.ndim != 3 or gt_xy_sincos.shape[1:] != (8, 4) or not torch.isfinite(gt_xy_sincos).all():
            raise ValueError('GT physical ego xy/sin/cos at 0.5--4 seconds required')
        gt = gt_xy_sincos.detach()
        gt = torch.cat((gt[..., :2] / self.xy_scale, gt[..., 2:]), -1)
        t = self.physical_times_s / 4.
        time = torch.stack((t, (torch.pi * t).sin(), (torch.pi * t).cos()), -1)
        dtype = self.point[0].weight.dtype
        return self.point(gt.to(dtype)) + self.time(time.to(dtype))[None]
