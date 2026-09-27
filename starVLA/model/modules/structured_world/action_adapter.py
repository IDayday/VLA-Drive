import torch
from torch import nn


class WorldToActionAdapter(nn.Module):
    def __init__(self, hidden_dim, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim,heads,batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, action, world, world_mask=None):
        if world_mask is not None:
            if world_mask.shape!=world.shape[:2] or not world_mask.any(-1).all():raise ValueError('Invalid planner memory mask')
            world=torch.where(world_mask[...,None],world,torch.zeros_like(world))
        residual, _ = self.attention(self.norm(action),world,world,key_padding_mask=None if world_mask is None else ~world_mask,need_weights=False)
        return action + self.gate.tanh() * residual
