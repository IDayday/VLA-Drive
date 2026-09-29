"""One W-only spatial readout for physical horizons 0, 1, 2 and 4 seconds."""
import torch
from torch import nn
from .future_latent_head import CrossReadout


class DINOFeatureHead(nn.Module):
    def __init__(self, hidden_dim, feature_dim, dim=512, layers=2, heads=8):
        super().__init__()
        self.project = nn.Linear(hidden_dim, dim)
        self.time = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.view = nn.Embedding(3, dim)
        self.spatial = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossReadout(dim, heads) for _ in range(layers))
        self.output = nn.Linear(dim, feature_dim)

    def forward(self, world, horizon_s, grid):
        if world.ndim != 3 or not torch.isfinite(world).all():
            raise ValueError('Invalid shared W')
        horizon_s = torch.as_tensor(horizon_s, device=world.device, dtype=torch.float32)
        allowed = horizon_s.new_tensor([0., 1., 2., 4.])
        if horizon_s.shape != (len(world),) or not (horizon_s[:, None] == allowed).any(-1).all():
            raise ValueError('Expected physical horizon seconds: 0,1,2,4')
        height, width = grid
        if min(height, width) < 1:
            raise ValueError('Invalid rectangular patch grid')
        memory = self.project(world)
        yy, xx = torch.meshgrid((torch.arange(height, device=world.device) + .5) / height,
                               (torch.arange(width, device=world.device) + .5) / width, indexing='ij')
        xy = torch.stack((xx * 2 - 1, yy * 2 - 1), -1).reshape(-1, 2).to(memory.dtype)
        h = horizon_s / 4
        time = self.time(torch.stack((h, torch.sin(torch.pi*h), torch.cos(torch.pi*h)), -1).to(memory.dtype))
        query = (self.spatial(xy)[None, None] + time[:, None, None]
                 + self.view(torch.arange(3, device=world.device))[None, :, None]).flatten(1, 2)
        for block in self.blocks:
            query = block(query, memory)
        return self.output(query).reshape(len(world), 3, height, width, -1).permute(0, 1, 4, 2, 3)
