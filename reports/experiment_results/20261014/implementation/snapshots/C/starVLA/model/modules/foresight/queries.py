"""Current-only spatial/view query identity, never derived from target tensors."""
import torch
from torch import nn


class SpatialViewEncoding(nn.Module):
    views = ('cam_f0', 'cam_l0', 'cam_r0')

    def __init__(self, hidden, grid_hw):
        super().__init__()
        h, w = grid_hw
        if min(h, w, hidden) < 1:
            raise ValueError('Invalid spatial query geometry')
        self.grid_hw = (h, w)
        self.view = nn.Embedding(3, hidden)
        self.position = nn.Linear(2, hidden, bias=False)
        nn.init.normal_(self.view.weight, std=.02)
        nn.init.normal_(self.position.weight, std=.02)
        yy, xx = torch.meshgrid((torch.arange(h)+.5)/h*2-1,
                               (torch.arange(w)+.5)/w*2-1, indexing='ij')
        self.register_buffer('positions', torch.stack((xx, yy), -1).reshape(-1, 2), persistent=True)
        self.register_buffer('view_ids', torch.arange(3), persistent=True)

    def forward(self, queries):
        h, w = self.grid_hw
        if queries.shape != (3*h*w, self.view.embedding_dim):
            raise ValueError('Expected view→row→column query layout')
        position = self.position(self.positions.to(dtype=self.position.weight.dtype))
        offsets = self.view(self.view_ids)[:, None] + position[None]
        return queries + offsets.reshape_as(queries)
