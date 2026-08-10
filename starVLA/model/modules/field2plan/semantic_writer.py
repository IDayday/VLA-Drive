"""Optional global semantic token writer for the MVP."""

import torch
from torch import nn
from torch.nn import functional as F


class SemanticWriter(nn.Module):
    """Pool language/action-query features ``[B,L,Cin]`` to ``[B,K,Cout]``."""

    def __init__(self, input_dim: int, output_dim: int, num_tokens: int) -> None:
        super().__init__()
        if min(input_dim, output_dim, num_tokens) <= 0:
            raise ValueError("semantic dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.input_dim:
            raise ValueError("hidden_states must have shape [B,L,input_dim]")
        pooled = F.adaptive_avg_pool1d(
            hidden_states.transpose(1, 2), self.num_tokens
        ).transpose(1, 2)
        return self.projection(pooled)
