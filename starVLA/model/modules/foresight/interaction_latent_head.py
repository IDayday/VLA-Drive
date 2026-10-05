import torch
from torch import nn
from .future_latent_head import CrossReadout


class InteractionLatentHead(nn.Module):
    def __init__(self, hidden_dim, dim=512, steps=8, layers=2, heads=8):
        super().__init__()
        self.steps = steps
        self.project = nn.Linear(hidden_dim, dim)
        self.time = nn.Embedding(steps, dim)
        self.blocks = nn.ModuleList(CrossReadout(dim, heads) for _ in range(layers))
        self.output = nn.Linear(dim, dim)

    def forward(self, memory):
        if memory.ndim != 3 or memory.shape[1] == 0 or not torch.isfinite(memory).all():
            raise ValueError('Invalid current interaction memory')
        q = self.time(torch.arange(self.steps, device=memory.device))[None].expand(len(memory), -1, -1)
        memory = self.project(memory)
        for block in self.blocks: q = block(q, memory)
        return self.output(q)
