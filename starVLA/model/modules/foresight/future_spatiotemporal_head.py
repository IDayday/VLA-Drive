"""Direct W/action memory readout; auxiliary results never write back to W."""
import torch
from torch import nn
from .future_latent_head import CrossReadout
from .ego_trajectory_condition import EgoTrajectoryConditionEncoder


class FutureSpatiotemporalHead(nn.Module):
    def __init__(self, hidden_dim, feature_dim, time_intervals_s, dim=512, layers=2,
                 action_condition='none', use_world=True):
        super().__init__()
        if action_condition not in ('none', 'gt_ego') or not (use_world or action_condition == 'gt_ego'):
            raise ValueError('At least one explicitly declared conditioning source required')
        spans = torch.as_tensor(time_intervals_s, dtype=torch.float32)
        if spans.ndim != 2 or spans.shape[1] != 2 or not torch.isfinite(spans).all() or (spans[:, 0] > spans[:, 1]).any():
            raise ValueError('Native temporal intervals required')
        self.register_buffer('time_intervals_s', spans, persistent=True)
        self.action_condition, self.use_world = action_condition, use_world
        self.project = nn.Linear(hidden_dim, dim)
        self.action_encoder = EgoTrajectoryConditionEncoder(dim)
        self.memory_type = nn.Embedding(2, dim)
        self.time = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.view = nn.Embedding(3, dim)
        self.spatial = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossReadout(dim, 8) for _ in range(layers))
        self.output = nn.Linear(dim, feature_dim)
        self.action_encoder.requires_grad_(action_condition == 'gt_ego')
        self.project.requires_grad_(use_world)

    def forward(self, world, grid, *, gt_action=None):
        if world.ndim != 3 or not torch.isfinite(world).all():
            raise ValueError('Invalid current W')
        if (gt_action is not None) != (self.action_condition == 'gt_ego'):
            raise ValueError('Absent condition is not a zero trajectory; obey declared mode')
        height, width = grid
        if min(height, width) < 1:
            raise ValueError('Invalid native spatial grid')
        memory = []
        if self.use_world:
            memory.append(self.project(world) + self.memory_type.weight[0])
        if gt_action is not None:
            if len(gt_action) != len(world):
                raise ValueError('Action/scene batch mismatch')
            memory.append(self.action_encoder(gt_action) + self.memory_type.weight[1])
        memory = torch.cat(memory, 1)
        yy, xx = torch.meshgrid((torch.arange(height, device=world.device) + .5) / height,
                               (torch.arange(width, device=world.device) + .5) / width, indexing='ij')
        xy = torch.stack((2*xx-1, 2*yy-1), -1).reshape(-1, 2).to(memory.dtype)
        span = self.time_intervals_s.to(device=world.device, dtype=memory.dtype) / 4.
        mid = span.mean(-1)
        temporal = self.time(torch.stack((span[:, 0], span[:, 1], mid.sin(), mid.cos()), -1))
        query = (self.view(torch.arange(3, device=world.device))[None, :, None, None]
                 + temporal[None, None, :, None] + self.spatial(xy)[None, None, None])
        query = query.expand(len(world), -1, -1, -1, -1).flatten(1, 3)
        for block in self.blocks:
            query = block(query, memory)
        return self.output(query).reshape(len(world), 3, len(span), height, width, -1)


def normalized_clip_loss(prediction, target, valid, *, eps=1e-5, global_count=None):
    """Mean channel -> space/time -> valid view -> valid scene; DDP global mean."""
    from torch import distributed as dist
    from torch.nn import functional as F
    if prediction.shape != target.shape or prediction.ndim != 6 or valid.shape != target.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('Expected B,V,T,H,W,C and a Boolean token mask')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite valid clip feature')
    p = torch.where(valid[..., None], prediction.float(), 0.)
    t = torch.where(valid[..., None], target.detach().float(), 0.)
    p = F.layer_norm(p, (p.shape[-1],), eps=eps)
    t = F.layer_norm(t, (t.shape[-1],), eps=eps)
    error = (p-t).square().mean(-1)
    token_count = valid.sum((2, 3, 4)); view_valid = token_count > 0
    view_error = (error*valid).sum((2, 3, 4)) / token_count.clamp_min(1)
    scene_valid = view_valid.any(-1)
    scene_error = (view_error*view_valid).sum(-1) / view_valid.sum(-1).clamp_min(1)
    count = int(scene_valid.sum()) if global_count is None else int(global_count)
    world = dist.get_world_size() if dist.is_initialized() else 1
    scale = world if global_count is not None else 1
    return (scene_error*scene_valid).sum() * scale / max(count, 1), count
