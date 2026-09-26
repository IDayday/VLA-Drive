import torch
from torch import nn


class WorldToActionAdapter(nn.Module):
    def __init__(self, hidden_dim, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim,heads,batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, action, world):
        residual, _ = self.attention(self.norm(action),world,world,need_weights=False)
        return action + self.gate.tanh() * residual
