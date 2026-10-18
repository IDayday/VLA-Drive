import torch
from torch import nn


class TemporalProjector(nn.Module):
    def __init__(self, hidden_size, contract, seed=4203):
        super().__init__()
        self.contract = contract
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Linear(hidden_size, contract.hidden_size)

    def forward(self, hidden):
        if hidden.shape[1] != self.contract.num_queries:
            raise ValueError("Temporal hidden must match the native decoder input count")
        # Native Decoder2D.project remains inside the frozen teacher decoder.
        return self.project(hidden)
