"""Actor-masked conditional flow; deployment starts with every future hidden."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def actor_mask(batch, actors, device, generator=None, all_hidden_probability=.5):
    """True hides a complete trajectory; sampling never reads GT count/validity."""
    if actors < 1 or not 0 <= all_hidden_probability <= 1:
        raise ValueError('Invalid actor mask configuration')
    hidden = torch.rand(batch, actors, device=device, generator=generator) < .5
    all_hidden = torch.rand(batch, device=device, generator=generator) < all_hidden_probability
    hidden[all_hidden] = True
    # Every scene has at least one reconstruction target, irrespective of GT.
    empty = ~hidden.any(-1)
    selected = torch.randint(actors, (batch,), device=device, generator=generator)
    hidden[empty, selected[empty]] = True
    return hidden


class GraphBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temporal = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.actor = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.context = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.relative_bias = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, heads))
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, centres, context):
        b, a, t, d = x.shape
        h = self.norms[0](x).reshape(b * a, t, d)
        x = x + self.temporal(h, h, h, need_weights=False)[0].reshape(b, a, t, d)
        h = self.norms[1](x).transpose(1, 2).reshape(b * t, a, d)
        delta = centres[:, :, None] - centres[:, None, :]
        bias = self.relative_bias(delta).permute(0, 3, 1, 2)
        bias = bias[:, None].expand(b, t, self.heads, a, a).reshape(b * t * self.heads, a, a)
        x = x + self.actor(h, h, h, attn_mask=bias, need_weights=False)[0].reshape(b, t, a, d).transpose(1, 2)
        h = self.norms[2](x).reshape(b, a * t, d)
        x = x + self.context(h, context, context, need_weights=False)[0].reshape(b, a, t, d)
        return x + self.mlp(self.norms[3](x))


class JointTrajectoryFlow(nn.Module):
    """XY in ego(t0), internally divided by scale_m; slot0 is ego, remaining exchangeable."""
    def __init__(self, condition_dim, dim=256, heads=8, layers=3, steps=8, scale_m=20.):
        super().__init__()
        if dim % heads or scale_m <= 0 or steps < 1:
            raise ValueError('Invalid graph dimensions')
        self.steps, self.scale_m = steps, scale_m
        self.state = nn.Linear(2, dim)
        self.known = nn.Linear(3, dim)  # masked-clean XY and known indicator
        self.current = nn.Linear(3, dim)  # predicted centre and existence, never GT
        self.condition = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, dim))
        self.role = nn.Embedding(2, dim)
        self.time = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('horizon', torch.arange(1, steps + 1, dtype=torch.float32) / steps)
        self.horizon_project = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([GraphBlock(dim, heads) for _ in range(layers)])
        self.velocity = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, noisy, time, actor_features, context, current_xy, existence,
                known_xy=None, known_mask=None):
        b, a, t, xy = noisy.shape
        if t != self.steps or xy != 2 or actor_features.shape[:2] != (b, a):
            raise ValueError('Joint trajectory shape mismatch')
        if current_xy.shape != (b, a, 2) or existence.shape != (b, a):
            raise ValueError('Current actor shape mismatch')
        if (known_xy is None) != (known_mask is None):
            raise ValueError('Known trajectory and mask must be paired')
        if known_xy is None:
            known_xy = torch.zeros_like(noisy)
            known_mask = torch.zeros_like(noisy[..., 0], dtype=torch.bool)
        if known_xy.shape != noisy.shape or known_mask.shape != noisy.shape[:-1]:
            raise ValueError('Known trajectory shape mismatch')
        # Values outside visible context cannot influence anything (even NaNs).
        visible = torch.where(known_mask[..., None], known_xy, torch.zeros_like(known_xy))
        clean = torch.cat([visible / self.scale_m, known_mask[..., None].to(noisy.dtype)], -1)
        roles = torch.ones(a, device=noisy.device, dtype=torch.long)
        roles[0] = 0
        current = torch.cat([current_xy / self.scale_m, existence[..., None]], -1)
        te = torch.stack([time, torch.sin(time * math.pi), torch.cos(time * math.pi)], -1)
        x = (self.state(noisy) + self.known(clean) + self.condition(actor_features)[:, :, None]
             + self.current(current)[:, :, None] + self.role(roles)[None, :, None]
             + self.time(te)[:, None, None] + self.horizon_project(self.horizon[:, None])[None, None])
        memory = self.condition(context)
        for block in self.blocks:
            x = block(x, current_xy / self.scale_m, memory)
        return self.velocity(x), x.mean(2)

    def sample(self, noise, actor_features, context, current_xy, existence, sampling_steps=10):
        """Only current observations/model features and noise; no target/context-future arguments."""
        if sampling_steps < 1:
            raise ValueError('Positive sampling steps required')
        x = noise
        for step in range(sampling_steps):
            time = x.new_full((x.shape[0],), step / sampling_steps)
            velocity, features = self(x, time, actor_features, context, current_xy, existence)
            x = x + velocity / sampling_steps
        # Recompute representation at the sampled endpoint for train/inference agreement.
        _, features = self(x, x.new_ones(x.shape[0]), actor_features, context, current_xy, existence)
        return x * self.scale_m, features


def training_loss_sums(model, target_xy, valid, hidden, noise, time,
                       actor_features, context, current_xy, existence):
    """Conditional imputation supervision; returned features must NOT condition a planner."""
    if hidden.shape != valid.shape[:2] or noise.shape != target_xy.shape:
        raise ValueError('Training masks/noise shape mismatch')
    clean = torch.where(valid[..., None], target_xy / model.scale_m, noise)
    mixed = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * clean
    visible = ~hidden[..., None] & valid
    mixed = torch.where(visible[..., None], clean, mixed)
    velocity, _ = model(mixed, time, actor_features, context, current_xy, existence,
                        target_xy, visible)
    expected = clean - noise
    error = (velocity - expected).square().sum(-1)
    selected = valid & hidden[..., None]
    # Separate ego and agent normalizers so many cars do not dilute the ego signal.
    return {'ego': error[:, 0][selected[:, 0]].sum(), 'agents': error[:, 1:][selected[:, 1:]].sum()}, {
        'ego': int(selected[:, 0].sum()) * 2, 'agents': int(selected[:, 1:].sum()) * 2}
