"""Independent deterministic visualization decoder; never called by the actor."""
import torch
from torch import nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.block(x)


class DINORGBDecoder(nn.Module):
    def __init__(self, feature_dim=1024, channels=(256, 128, 96, 64, 32, 16)):
        super().__init__()
        if len(channels) != 6 or any(c % 8 for c in channels):
            raise ValueError('Five2x upsample stages and GroupNorm-compatible widths required')
        self.feature_dim = feature_dim
        self.input = nn.Sequential(nn.Conv2d(feature_dim, channels[0], 1), ResidualConv(channels[0]))
        self.stages = nn.ModuleList([nn.Sequential(nn.Conv2d(left, right, 3, padding=1),
            ResidualConv(right)) for left, right in zip(channels[:-1], channels[1:])])
        self.output = nn.Conv2d(channels[-1], 3, 1)

    def forward(self, features):
        if features.shape[-3:] != (9, 16, self.feature_dim):
            raise ValueError('RGB decoder only accepts registered9x16 teacher/student feature grids')
        leading = features.shape[:-3]
        x = features.reshape(-1, 9, 16, self.feature_dim).permute(0, 3, 1, 2)
        x = self.input(x)
        for stage in self.stages:
            x = stage(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        image = self.output(x).sigmoid()
        return image.reshape(*leading, 3, 288, 512)


def rgb_reconstruction_loss(prediction, target, valid_views):
    if prediction.shape != target.shape or prediction.shape[-3:] != (3, 288, 512):
        raise ValueError('RGB target must be the exact288x512 teacher preprocessing image')
    if valid_views.dtype != torch.bool or valid_views.shape != prediction.shape[:-3]:
        raise ValueError('Explicit valid-view mask required')
    if not torch.isfinite(target[valid_views]).all() or not torch.isfinite(prediction[valid_views]).all():
        raise FloatingPointError('Invalid valid RGB reconstruction')
    mask = valid_views[..., None, None, None].expand_as(prediction)
    error = torch.where(mask, prediction.float(), 0.) - torch.where(mask, target.detach().float(), 0.)
    return error.abs().sum() / mask.sum().clamp_min(1)
