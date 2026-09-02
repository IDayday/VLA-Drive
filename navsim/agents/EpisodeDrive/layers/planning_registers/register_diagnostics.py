"""Numerically stable diagnostics for planning-register collapse."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_register_diagnostics(registers: torch.Tensor) -> Dict[str, torch.Tensor]:
    if registers.ndim != 3:
        raise ValueError(
            f"Planning registers must have shape [B,R,D], got {tuple(registers.shape)}"
        )
    registers = registers.detach()
    _, register_count, _ = registers.shape
    centered = registers.float() - registers.float().mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    squared = singular_values.square()
    probabilities = squared / squared.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    ).mean()

    normalized = F.normalize(registers.float(), dim=-1, eps=1e-12)
    similarities = normalized @ normalized.transpose(-1, -2)
    if register_count > 1:
        mask = ~torch.eye(
            register_count, dtype=torch.bool, device=registers.device
        )
        pairwise_cosine = similarities[:, mask].mean()
    else:
        pairwise_cosine = similarities.new_ones(())
    return {
        "register_effective_rank": effective_rank,
        "register_mean_pairwise_cosine": pairwise_cosine,
        "register_std": registers.float().std(unbiased=False),
    }
