"""Direct W/action memory readout; auxiliary results never write back to W."""
import torch
from torch import nn
from .future_latent_head import CrossReadout
from .ego_trajectory_condition import EgoTrajectoryConditionEncoder


class FutureSpatiotemporalHead(nn.Module):
    def __init__(self, hidden_dim, feature_dim, time_intervals_s, dim=512, layers=2,
                 action_condition='none', use_world=True, action_injection='memory_only',
                 action_query_scale=1.0):
        super().__init__()
        if action_condition not in ('none', 'gt_ego') or not (use_world or action_condition == 'gt_ego'):
            raise ValueError('At least one explicitly declared conditioning source required')
        spans = torch.as_tensor(time_intervals_s, dtype=torch.float32)
        if spans.ndim != 2 or spans.shape[1] != 2 or not torch.isfinite(spans).all() or (spans[:, 0] > spans[:, 1]).any():
            raise ValueError('Native temporal intervals required')
        self.register_buffer('time_intervals_s', spans, persistent=True)
        self.action_condition, self.use_world = action_condition, use_world
        import math
        if action_injection not in ('memory_only', 'memory_and_query') or not math.isfinite(action_query_scale):
            raise ValueError('Invalid action injection/query scale')
        if action_injection != 'memory_only' and action_condition != 'gt_ego':
            raise ValueError('Query injection needs GT action tokens')
        self.action_injection, self.action_query_scale = action_injection, float(action_query_scale)
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
        self.capture_conditioning = False

    def aligned_action_tokens(self, tokens):
        """Average ENCODED vectors in each declared interval, never yaw angles."""
        times = self.action_encoder.physical_times_s
        groups = []
        for lo, hi in self.time_intervals_s:
            selected = (times >= lo - 1e-6) & (times <= hi + 1e-6)
            if not selected.any() or not torch.isclose(times[selected][0], lo, atol=1e-6) or not torch.isclose(times[selected][-1], hi, atol=1e-6):
                raise ValueError('Future interval has no matching physical action endpoints')
            groups.append(tokens[:, selected].mean(1))
        return torch.stack(groups, 1)

    def forward(self, world, grid, *, gt_action=None):
        if (gt_action is not None) != (self.action_condition == 'gt_ego'):
            raise ValueError('Absent condition is not a zero trajectory; obey declared mode')
        if self.use_world:
            if world is None or world.ndim != 3 or not torch.isfinite(world).all():
                raise ValueError('Invalid current W')
            batch, device = len(world), world.device
        else:
            if world is not None:
                raise ValueError('Action-only control has no W input')
            batch, device = len(gt_action), gt_action.device
        height, width = grid
        if min(height, width) < 1:
            raise ValueError('Invalid native spatial grid')
        memory = []
        action_tokens = None
        if self.use_world:
            memory.append(self.project(world) + self.memory_type.weight[0])
        if gt_action is not None:
            if len(gt_action) != batch:
                raise ValueError('Action/scene batch mismatch')
            action_tokens = self.action_encoder(gt_action)
            memory.append(action_tokens + self.memory_type.weight[1])
        memory = torch.cat(memory, 1)
        yy, xx = torch.meshgrid((torch.arange(height, device=device) + .5) / height,
                               (torch.arange(width, device=device) + .5) / width, indexing='ij')
        xy = torch.stack((2*xx-1, 2*yy-1), -1).reshape(-1, 2).to(memory.dtype)
        span = self.time_intervals_s.to(device=device, dtype=memory.dtype) / 4.
        mid = span.mean(-1)
        temporal = self.time(torch.stack((span[:, 0], span[:, 1], mid.sin(), mid.cos()), -1))
        query = (self.view(torch.arange(3, device=device))[None, :, None, None]
                 + temporal[None, None, :, None] + self.spatial(xy)[None, None, None])
        # Skip the addition at zero scale for exact legacy numerical regression.
        if self.action_injection == 'memory_and_query' and self.action_query_scale != 0:
            query = query + self.action_query_scale * self.aligned_action_tokens(action_tokens)[:, None, :, None]
        query = query.expand(batch, -1, -1, -1, -1).flatten(1, 3)
        if self.capture_conditioning:
            self.conditioning_diagnostics = {'query_norm':float(query.detach().float().norm(dim=-1).mean()),
                'memory_norm':float(memory.detach().float().norm(dim=-1).mean()),
                'action_query_norm':float(self.aligned_action_tokens(action_tokens).detach().float().norm(dim=-1).mean()) if action_tokens is not None else None,
                'world_tokens':world.shape[1] if self.use_world else 0,'action_tokens':8 if action_tokens is not None else 0,'layers':[]}
        for block in self.blocks:
            if self.capture_conditioning:
                self._record_group_attention(block, query, memory, (batch, 3, len(span), height*width))
            query = block(query, memory)
        return self.output(query).reshape(batch, 3, len(span), height, width, -1)

    @torch.no_grad()
    def _record_group_attention(self, block, query, memory, layout):
        """Diagnostic only, matching torch2.5 MHA q/k projection and softmax."""
        import math
        from torch.nn import functional as F
        attn = block.attn
        if not attn._qkv_same_embed_dim or attn.add_zero_attn or attn.bias_k is not None:
            raise ValueError('Unsupported diagnostic MHA layout')
        qw, kw, _ = attn.in_proj_weight.detach().chunk(3)
        biases = attn.in_proj_bias.detach().chunk(3) if attn.in_proj_bias is not None else (None, None, None)
        q = F.linear(block.qnorm(query).float(), qw.float(), None if biases[0] is None else biases[0].float())
        k = F.linear(block.mnorm(memory).float(), kw.float(), None if biases[1] is None else biases[1].float())
        b, v, t, p = layout;heads=attn.num_heads;dim=q.shape[-1]//heads
        q=q.reshape(b,-1,heads,dim).transpose(1,2);k=k.reshape(b,-1,heads,dim).transpose(1,2)
        weights=(q @ k.transpose(-1,-2) / math.sqrt(dim)).softmax(-1)
        split = self.conditioning_diagnostics['world_tokens']
        rows={}
        for name,lo,hi in [('world',0,split),('action',split,memory.shape[1])]:
            n=hi-lo
            if not n:continue
            mass=weights[...,lo:hi].sum(-1).reshape(b,heads,v,t,p).mean((0,2,4))
            rows[name]={'token_count':n,'total_mass_by_head_time':mass.cpu().tolist(),
                        'per_token_mass_by_head_time':(mass/n).cpu().tolist()}
        self.conditioning_diagnostics['layers'].append(rows)


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
