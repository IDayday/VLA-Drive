"""Actor-masked conditional flow; deployment starts with every future hidden."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def actor_mask(batch, actors, device, generator=None, all_hidden_probability=.5, partial_mode='bernoulli'):
    """True hides a complete trajectory; sampling never reads GT count/validity."""
    if actors < 1 or not 0 <= all_hidden_probability <= 1:
        raise ValueError('Invalid actor mask configuration')
    if partial_mode not in ('bernoulli','single_actor'):raise ValueError('Unknown partial actor mask mode')
    hidden = torch.rand(batch, actors, device=device, generator=generator) < .5
    all_hidden = torch.rand(batch, device=device, generator=generator) < all_hidden_probability
    hidden[all_hidden] = True
    # Every scene has at least one reconstruction target, irrespective of GT.
    empty = ~hidden.any(-1)
    selected = torch.randint(actors, (batch,), device=device, generator=generator)
    if partial_mode=='single_actor':
        hidden.zero_();hidden.scatter_(1,selected[:,None],True);hidden[all_hidden]=True
        return hidden
    hidden[empty, selected[empty]] = True
    return hidden


class GraphBlock(nn.Module):
    def __init__(self, dim, heads, edge_feature_dim=0):
        super().__init__()
        self.heads = heads
        self.temporal = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.actor = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.context = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.relative_bias = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, heads))
        self.edge_bias = nn.Linear(edge_feature_dim, heads, bias=False) if edge_feature_dim else None
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, centres, context, local_graph=None, context_mask=None):
        b, a, t, d = x.shape
        active = local_graph.active_actor_mask if local_graph is not None else None
        def clean(value):
            return value if active is None else torch.where(active[:,:,None,None], value, torch.zeros_like(value))
        x = clean(x)
        if active is not None:
            centres=torch.where(active[...,None],centres,torch.zeros_like(centres))
        h = self.norms[0](x).reshape(b * a, t, d)
        x = clean(x + self.temporal(h, h, h, need_weights=False)[0].reshape(b, a, t, d))
        h = clean(self.norms[1](x)).transpose(1, 2).reshape(b * t, a, d)
        delta = centres[:, :, None] - centres[:, None, :]
        bias = self.relative_bias(delta).permute(0, 3, 1, 2)
        if local_graph is not None:
            allowed=local_graph.edge_mask & active[:,:,None] & active[:,None,:]
            if self.edge_bias is not None:
                edge=torch.where(allowed[...,None],local_graph.edge_features,torch.zeros_like(local_graph.edge_features))
                bias=bias+self.edge_bias(edge).permute(0,3,1,2)
            # Invalid queries have only a zero self KV, never an all-inf softmax row.
            safe=allowed | ((~active)[:,:,None] & torch.eye(a,device=x.device,dtype=torch.bool)[None])
            bias=bias.masked_fill(~safe[:,None],float('-inf'))
        bias = bias[:, None].expand(b, t, self.heads, a, a).reshape(b * t * self.heads, a, a)
        x = clean(x + self.actor(h, h, h, attn_mask=bias, need_weights=False)[0].reshape(b, t, a, d).transpose(1, 2))
        h = self.norms[2](x).reshape(b, a * t, d)
        x = clean(x + self.context(h, context, context, key_padding_mask=None if context_mask is None else ~context_mask, need_weights=False)[0].reshape(b, a, t, d))
        return clean(x + self.mlp(self.norms[3](x)))


class JointTrajectoryFlow(nn.Module):
    """XY in ego(t0), internally divided by scale_m; slot0 is ego, remaining exchangeable."""
    def __init__(self, condition_dim, dim=256, heads=8, layers=3, steps=8, scale_m=20.,
                 trajectory_mode='absolute', agent_scale_m=None, edge_feature_dim=0):
        super().__init__()
        if dim % heads or scale_m <= 0 or steps < 1:
            raise ValueError('Invalid graph dimensions')
        self.steps, self.scale_m = steps, scale_m
        if trajectory_mode not in ('absolute', 'current_residual'):
            raise ValueError('Unknown trajectory coordinates')
        self.trajectory_mode = trajectory_mode
        self.agent_scale_m = scale_m if agent_scale_m is None else agent_scale_m
        if self.agent_scale_m <= 0:
            raise ValueError('Agent scale must be positive')
        self.state = nn.Linear(2, dim)
        self.known = nn.Linear(3, dim)  # masked-clean XY and known indicator
        self.current = nn.Linear(3, dim)  # predicted centre and existence, never GT
        self.condition = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, dim))
        self.role = nn.Embedding(2, dim)
        self.time = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('horizon', torch.arange(1, steps + 1, dtype=torch.float32) / steps)
        self.horizon_project = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([GraphBlock(dim, heads, edge_feature_dim) for _ in range(layers)])
        self.velocity = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def _trajectory_coordinates(self, current_xy):
        # Predicted centres, never target centres; detached to isolate motion gradients.
        anchor = current_xy.detach()[:, :, None]
        if self.trajectory_mode == 'absolute':
            anchor = torch.zeros_like(anchor)
        scale = current_xy.new_full((1, current_xy.shape[1], 1, 1), self.agent_scale_m)
        scale[:, 0] = self.scale_m
        return anchor, scale

    def encode_trajectories(self, xy_metres, current_xy):
        anchor, scale = self._trajectory_coordinates(current_xy)
        return (xy_metres - anchor) / scale

    def decode_trajectories(self, normalized, current_xy):
        anchor, scale = self._trajectory_coordinates(current_xy)
        return normalized * scale + anchor

    def forward(self, noisy, time, actor_features, context, current_xy, existence,
                known_xy=None, known_mask=None, local_graph=None, context_mask=None):
        b, a, t, xy = noisy.shape
        if t != self.steps or xy != 2 or actor_features.shape[:2] != (b, a):
            raise ValueError('Joint trajectory shape mismatch')
        if current_xy.shape != (b, a, 2) or existence.shape != (b, a):
            raise ValueError('Current actor shape mismatch')
        active=None
        if local_graph is not None:
            active=local_graph.active_actor_mask
            if active.shape!=(b,a):raise ValueError('Local graph shape mismatch')
            # Sanitize before linear layers, arithmetic and normalization, not afterward.
            noisy=torch.where(active[:,:,None,None],noisy,torch.zeros_like(noisy))
            actor_features=torch.where(active[:,:,None],actor_features,torch.zeros_like(actor_features))
            current_xy=torch.where(active[:,:,None],current_xy,torch.zeros_like(current_xy))
            existence=torch.where(active,existence,torch.zeros_like(existence))
        if context_mask is not None:
            if context_mask.shape!=context.shape[:2] or not context_mask.any(-1).all():raise ValueError('Invalid current context mask')
            context=torch.where(context_mask[...,None],context,torch.zeros_like(context))
        if (known_xy is None) != (known_mask is None):
            raise ValueError('Known trajectory and mask must be paired')
        if known_xy is None:
            known_xy = torch.zeros_like(noisy)
            known_mask = torch.zeros_like(noisy[..., 0], dtype=torch.bool)
        if known_xy.shape != noisy.shape or known_mask.shape != noisy.shape[:-1]:
            raise ValueError('Known trajectory shape mismatch')
        if local_graph is not None:
            permitted=local_graph.trajectory_condition_mask[:,:,None]
            if (known_mask & ~permitted).any():raise ValueError('Ineligible trajectory supplied as known future')
        # Values outside visible context cannot influence anything (even NaNs).
        visible = torch.where(known_mask[..., None], known_xy, torch.zeros_like(known_xy))
        encoded = self.encode_trajectories(visible, current_xy)
        encoded = torch.where(known_mask[..., None], encoded, torch.zeros_like(encoded))
        clean = torch.cat([encoded, known_mask[..., None].to(noisy.dtype)], -1)
        roles = torch.ones(a, device=noisy.device, dtype=torch.long)
        roles[0] = 0
        current = torch.cat([current_xy / self.scale_m, existence[..., None]], -1)
        te = torch.stack([time, torch.sin(time * math.pi), torch.cos(time * math.pi)], -1)
        x = (self.state(noisy) + self.known(clean) + self.condition(actor_features)[:, :, None]
             + self.current(current)[:, :, None] + self.role(roles)[None, :, None]
             + self.time(te)[:, None, None] + self.horizon_project(self.horizon[:, None])[None, None])
        memory = self.condition(context)
        for block in self.blocks:
            x = block(x, current_xy / self.scale_m, memory, local_graph, context_mask)
        velocity=self.velocity(x)
        if active is not None:
            velocity=torch.where(active[:,:,None,None],velocity,torch.zeros_like(velocity))
            x=torch.where(active[:,:,None,None],x,torch.zeros_like(x))
        return velocity, x.mean(2)

    def sample(self, noise, actor_features, context, current_xy, existence, sampling_steps=10,
               local_graph=None, context_mask=None):
        """Only current observations/model features and noise; no target/context-future arguments."""
        return self.sample_conditional(noise,actor_features,context,current_xy,existence,
                                       sampling_steps=sampling_steps,local_graph=local_graph,context_mask=context_mask)

    def sample_conditional(self, noise, actor_features, context, current_xy, existence,
                           known_xy=None, known_mask=None, sampling_steps=10, local_graph=None,
                           context_mask=None, return_path=False):
        """Formal conditional sampler. Known coordinates are clamped at every step.

        GT-conditioned features are for imputation diagnostics/training only, never
        deployment planner conditions. sample() remains the current-only entry.
        """
        if sampling_steps < 1:
            raise ValueError('Positive sampling steps required')
        if (known_xy is None)!=(known_mask is None):raise ValueError('Known xy/mask must be paired')
        if known_xy is None:
            known_xy=torch.zeros_like(noise);known_mask=torch.zeros_like(noise[...,0],dtype=torch.bool)
        if known_xy.shape!=noise.shape or known_mask.shape!=noise.shape[:-1]:raise ValueError('Invalid conditional shape')
        active=local_graph.active_actor_mask if local_graph is not None else torch.ones(noise.shape[:2],device=noise.device,dtype=torch.bool)
        if local_graph is not None and (known_mask & ~local_graph.trajectory_condition_mask[:,:,None]).any():
            raise ValueError('Known context outside eligible local actors')
        centres=torch.where(active[...,None],current_xy,torch.zeros_like(current_xy))
        visible=torch.where(known_mask[...,None],known_xy,torch.zeros_like(known_xy))
        clean=self.encode_trajectories(visible,centres)
        x=torch.where(known_mask[...,None],clean,noise)
        x=torch.where(active[:,:,None,None],x,torch.zeros_like(x))
        path=[]
        def decode(value):
            xy=self.decode_trajectories(value,centres)
            xy=torch.where(known_mask[...,None],visible,xy)
            return torch.where(active[:,:,None,None],xy,torch.zeros_like(xy))
        if return_path:path.append(decode(x))
        for step in range(sampling_steps):
            time = x.new_full((x.shape[0],), step / sampling_steps)
            velocity, features = self(x, time, actor_features, context, centres, existence,
                                      visible,known_mask,local_graph,context_mask)
            x = x + velocity / sampling_steps
            x=torch.where(known_mask[...,None],clean,x)
            x=torch.where(active[:,:,None,None],x,torch.zeros_like(x))
            if return_path:path.append(decode(x))
        # Recompute representation at the sampled endpoint for train/inference agreement.
        _, features = self(x, x.new_ones(x.shape[0]), actor_features, context, centres, existence,
                           visible,known_mask,local_graph,context_mask)
        result=(decode(x),features)
        return (*result,path) if return_path else result


def training_loss_sums(model, target_xy, valid, hidden, noise, time,
                       actor_features, context, current_xy, existence, local_graph=None, context_mask=None):
    """Conditional imputation supervision; returned features must NOT condition a planner."""
    if hidden.shape != valid.shape[:2] or noise.shape != target_xy.shape:
        raise ValueError('Training masks/noise shape mismatch')
    if local_graph is not None:
        valid=valid & (local_graph.active_actor_mask & local_graph.predictable_actor_mask)[:,:,None]
        current_xy=torch.where(local_graph.active_actor_mask[...,None],current_xy,torch.zeros_like(current_xy))
        noise=torch.where(local_graph.active_actor_mask[:,:,None,None],noise,torch.zeros_like(noise))
    safe_target=torch.where(valid[...,None],target_xy,torch.zeros_like(target_xy))
    clean = torch.where(valid[..., None], model.encode_trajectories(safe_target, current_xy), noise)
    mixed = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * clean
    visible = ~hidden[..., None] & valid
    mixed = torch.where(visible[..., None], clean, mixed)
    velocity, _ = model(mixed, time, actor_features, context, current_xy, existence,
                        safe_target, visible,local_graph,context_mask)
    expected = clean - noise
    error = (velocity - expected).square().sum(-1)
    selected = valid & hidden[..., None]
    # Separate ego and agent normalizers so many cars do not dilute the ego signal.
    return {'ego': error[:, 0][selected[:, 0]].sum(), 'agents': error[:, 1:][selected[:, 1:]].sum()}, {
        'ego': int(selected[:, 0].sum()) * 2, 'agents': int(selected[:, 1:].sum()) * 2}
