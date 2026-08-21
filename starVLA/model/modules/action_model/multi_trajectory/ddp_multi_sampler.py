"""Multi-candidate wrapper around the unchanged DDP action-head sampler."""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import torch
from torch import nn


class DDPMultiSampler(nn.Module):
    """Batch-expand existing Qwen hidden states and call ``predict_action`` once."""

    def __init__(self, action_head: nn.Module, num_candidates: int = 64):
        super().__init__()
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        # Deliberately bypass nn.Module registration.  The action head remains
        # owned by the original framework, so this wrapper adds no duplicate
        # state-dict namespace or optimizer parameters.
        object.__setattr__(self, "_action_head", action_head)
        self.num_candidates = int(num_candidates)

    @property
    def action_head(self) -> nn.Module:
        return object.__getattribute__(self, "_action_head")

    @staticmethod
    def _rng_context(seed: Optional[int]):
        if seed is None:
            return nullcontext()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        return torch.random.fork_rng(devices=devices, enabled=True)

    def sample(
        self,
        vl_embs_list: List[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if not vl_embs_list:
            raise ValueError("vl_embs_list must contain at least one hidden-state tensor")
        if any(hidden.ndim != 3 for hidden in vl_embs_list):
            raise ValueError("every Qwen hidden state must have shape [B, L, D]")
        batch_size = vl_embs_list[0].shape[0]
        if any(hidden.shape[0] != batch_size for hidden in vl_embs_list):
            raise ValueError("Qwen hidden-state batch dimensions do not match")
        if state is not None and state.shape[0] != batch_size:
            raise ValueError("state batch dimension does not match Qwen hidden states")

        candidates = self.num_candidates if num_candidates is None else int(num_candidates)
        if candidates <= 0:
            raise ValueError("num_candidates must be positive")

        expanded_vl = [
            hidden.repeat_interleave(candidates, dim=0) for hidden in vl_embs_list
        ]
        expanded_state = (
            state.repeat_interleave(candidates, dim=0) if state is not None else None
        )

        with self._rng_context(seed):
            if seed is not None:
                torch.manual_seed(seed)
            # DDP is a frozen proposal generator.  This also prevents proposal
            # scorer losses from ever reaching Qwen/DiT through this wrapper.
            with torch.no_grad():
                flat_trajectories = self.action_head.predict_action(
                    expanded_vl, expanded_state
                )

        if flat_trajectories.ndim != 3:
            raise ValueError(
                "action_head.predict_action must return [B*K, horizon, action_dim]"
            )
        expected_flat_batch = batch_size * candidates
        if flat_trajectories.shape[0] != expected_flat_batch:
            raise ValueError(
                f"action head returned batch {flat_trajectories.shape[0]}, "
                f"expected {expected_flat_batch}"
            )
        return flat_trajectories.reshape(
            batch_size,
            candidates,
            flat_trajectories.shape[-2],
            flat_trajectories.shape[-1],
        )

    def forward(
        self,
        vl_embs_list: List[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        return self.sample(vl_embs_list, state, num_candidates, seed)
