"""One shared deterministic latent readout; queries contain no future content."""
import torch
from torch import nn


class CrossReadout(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.qnorm, self.mnorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffnorm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, queries, memory, memory_valid=None):
        memory = self.mnorm(memory)
        out = queries + self.attn(self.qnorm(queries), memory, memory,
              key_padding_mask=None if memory_valid is None else ~memory_valid,
              need_weights=False)[0]
        return out + self.ff(self.ffnorm(out))


class FutureLatentHead(nn.Module):
    def __init__(self, hidden_dim, latent_channels, dim=512, layers=2, heads=8):
        super().__init__()
        self.project = nn.Linear(hidden_dim, dim)
        self.horizon = nn.Embedding(3, dim)
        self.view = nn.Embedding(3, dim)
        self.spatial = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossReadout(dim, heads) for _ in range(layers))
        self.output = nn.Linear(dim, latent_channels)

    def forward(self, world, horizon, grid):
        if world.ndim != 3 or not torch.isfinite(world).all(): raise ValueError('Invalid W')
        if horizon.shape != (len(world),) or horizon.dtype != torch.long or ((horizon < 0)|(horizon > 2)).any():
            raise ValueError('Horizon IDs must index fixed1/2/4seconds')
        height, width = grid
        if height < 1 or width < 1: raise ValueError('Invalid VAE spatial grid')
        memory = self.project(world)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=world.device),
                               torch.linspace(-1, 1, width, device=world.device), indexing='ij')
        xy = torch.stack((xx, yy), -1).reshape(-1, 2).to(memory.dtype)
        q = (self.spatial(xy)[None, None] + self.horizon(horizon)[:, None, None]
             + self.view(torch.arange(3, device=world.device))[None, :, None])
        q = q.flatten(1, 2)
        for block in self.blocks: q = block(q, memory)
        return self.output(q).reshape(len(world), 3, height, width, -1).permute(0,1,4,2,3)
