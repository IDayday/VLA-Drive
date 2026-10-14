"""Matched absolute/difference models use this identical current-token readout."""
import torch
from torch import nn
from starVLA.model.modules.foresight.future_latent_head import CrossReadout
from .consequence_contract import query_features


class ConsequenceQueryHead(nn.Module):
    def __init__(self, hidden_dim=2048, dim=512, layers=2, query_only=False, clip_m=10.):
        super().__init__()
        self.query_only, self.clip_m = query_only, float(clip_m)
        self.memory = nn.Linear(hidden_dim, dim)
        self.query = nn.Sequential(nn.Linear(14, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossReadout(dim, 8) for _ in range(layers))
        self.output = nn.Linear(dim, 5)
        # An independent query-only model has the same layers. No W/H_A is read;
        # its memory is the physical query encoding, never a removed-head test.

    def forward(self, inputs, queries):
        x = self.query(query_features(inputs, queries)).flatten(1, 2)
        if self.query_only:
            memory = x
        else:
            memory = self.memory(torch.cat((inputs.W, inputs.H_A), 1).float())
        for block in self.blocks:
            x = block(x, memory)
        raw = self.output(x).reshape(*queries.poses.shape[:-1], 5)
        # Continuous heads live in the registered clipped physical scale. The
        # sigmoid channels remain uncalibrated overlap scores, not risk GT.
        return torch.stack((self.clip_m*raw[..., 0].tanh(), self.clip_m*raw[..., 1].tanh(), raw[..., 2],
                            self.clip_m*raw[..., 3].tanh(), raw[..., 4]), -1)
