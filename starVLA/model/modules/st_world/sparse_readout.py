"""Sparse physical-time/view/position readout over the complete W memory.

Reuses the original CrossReadout and its state-dict naming. Neither the number
of output queries nor the teacher grid is tied to the number of W tokens.
The coordinate/time query formulation is a project adaptation.
"""
import torch
from torch import nn
from starVLA.model.modules.foresight.future_latent_head import CrossReadout


class SparseFeatureReadout(nn.Module):
    def __init__(self, hidden_dim, feature_dim, *, views=3, dim=512, layers=2, heads=8,
                 allowed_times_s=(0., .5, 1., 2., 4.)):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (hidden_dim, feature_dim, views, dim, layers, heads)) or dim % heads:
            raise ValueError('Positive, head-divisible readout dimensions required')
        allowed = torch.as_tensor(allowed_times_s, dtype=torch.float32)
        if allowed.ndim != 1 or not len(allowed) or not torch.isfinite(allowed).all() or (allowed < 0).any():
            raise ValueError('Explicit finite nonnegative physical-time table required')
        if len(torch.unique(allowed)) != len(allowed):
            raise ValueError('Physical times must not be duplicated')
        self.hidden_dim, self.feature_dim, self.views = hidden_dim, feature_dim, views
        self.register_buffer('allowed_times_s', allowed, persistent=False)
        self.project = nn.Linear(hidden_dim, dim)
        self.time = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.view = nn.Embedding(views, dim)
        self.spatial = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossReadout(dim, heads) for _ in range(layers))
        self.output = nn.Linear(dim, feature_dim)

    def forward(self, world, view_ids, xy, physical_times_s, *, memory_valid=None):
        if world.ndim != 3 or world.shape[-1] != self.hidden_dim or min(world.shape[:2]) < 1 or not world.is_floating_point():
            raise ValueError('Expected nonempty floating W[B,Nw,hidden]')
        batch = len(world)
        if view_ids.dtype != torch.long or view_ids.ndim != 2 or view_ids.shape[0] != batch or not view_ids.shape[1]:
            raise ValueError('Expected nonempty B,Q integer view IDs')
        if ((view_ids < 0) | (view_ids >= self.views)).any():
            raise ValueError('Unknown view identity')
        if xy.shape != (*view_ids.shape, 2) or not xy.is_floating_point() or not torch.isfinite(xy).all() or (xy.abs() > 1).any():
            raise ValueError('Expected finite normalized B,Q,2 image coordinates')
        if physical_times_s.shape != view_ids.shape or not physical_times_s.is_floating_point():
            raise ValueError('Expected B,Q physical seconds')
        allowed = self.allowed_times_s.to(physical_times_s.device)
        if not (physical_times_s[..., None] == allowed).any(-1).all():
            raise ValueError('Unregistered physical time requested')
        if any(x.device != world.device for x in (view_ids, xy, physical_times_s)):
            raise ValueError('All readout inputs must share the W device')
        if memory_valid is not None:
            if memory_valid.dtype != torch.bool or memory_valid.shape != world.shape[:2] or memory_valid.device != world.device:
                raise ValueError('Expected Boolean W-memory mask')
            if not memory_valid.any(-1).all():
                raise ValueError('A scene must contain at least one valid W token')
            if not torch.isfinite(world[memory_valid]).all():
                raise ValueError('Nonfinite valid W')
            world = torch.where(memory_valid[..., None], world, 0.)
        elif not torch.isfinite(world).all():
            raise ValueError('Nonfinite W')
        memory = self.project(world)  # exactly once for all times/views/positions
        # Match the mature DINOFeatureHead boundary: encode physical time in
        # FP32 first, then cast the complete encoding to the memory dtype.
        h = physical_times_s.float() / 4.
        temporal = self.time(torch.stack((h, torch.sin(torch.pi * h), torch.cos(torch.pi * h)), -1).to(memory.dtype))
        query = self.spatial(xy.to(memory.dtype)) + temporal + self.view(view_ids)
        for block in self.blocks:
            query = block(query, memory, memory_valid)
        return self.output(query)
