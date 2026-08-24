"""Parameter-free scene summary for GP-SQ3D-Mix."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskedSceneSummary(nn.Module):
    def __init__(self, hidden_dim: int = 2048, action_query_count: int = 8) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_query_count = int(action_query_count)

    def forward(
        self,
        last_hidden: Tensor,
        attention_mask: Tensor,
        action_positions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if last_hidden.ndim != 3 or last_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"last_hidden must be [B,L,{self.hidden_dim}]")
        batch, length, _ = last_hidden.shape
        if attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must be [B,L]")
        if action_positions.shape != (batch, self.action_query_count):
            raise ValueError(
                f"action_positions must be [B,{self.action_query_count}]"
            )
        if action_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("action_positions must use an integer dtype")
        if (action_positions < 0).any() or (action_positions >= length).any():
            raise ValueError("action_positions contains an out-of-range index")
        scene_mask = attention_mask.bool().clone()
        scene_mask.scatter_(1, action_positions.long(), False)
        normalized = F.layer_norm(
            last_hidden.float(), normalized_shape=(self.hidden_dim,)
        )
        weights = scene_mask.unsqueeze(-1).to(normalized.dtype)
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary = (normalized * weights).sum(dim=1, keepdim=True) / counts
        diagnostics = {
            "gp_sq3dmix/scene_summary_norm": summary.detach().norm(dim=-1).mean(),
            "gp_sq3dmix/scene_memory_token_count": scene_mask.detach().sum(dim=1).float().mean(),
            "gp_sq3dmix/scene_summary_batch_variance": summary.detach().var(
                dim=0, unbiased=False
            ).mean(),
        }
        return summary, diagnostics
