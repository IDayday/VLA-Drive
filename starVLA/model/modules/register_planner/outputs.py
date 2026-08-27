"""Typed outputs for the deterministic Register trajectory planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import Tensor


@dataclass
class RegisterGeneratorOutput:
    """All proposal stages produced by one Register decoder forward."""

    proposals: Tensor
    proposal_list: List[Tensor]
    final_tokens: Tensor
    token_list: List[Tensor]
    sanitization_metrics: Optional[Dict[str, Tensor]] = None


@dataclass
class RegisterLossOutput:
    """Winner-take-all generator loss and non-loss diversity diagnostics."""

    loss: Tensor
    winner_index: Tensor
    metrics: Dict[str, Tensor]
    stage_losses: List[Tensor]


@dataclass
class RegisterPlannerOutput:
    """Integrated inference result in physical and normalized coordinates."""

    trajectory_navsim_8: Tensor
    normalized_actions: Tensor
    selected_index: Tensor
    selected_source: Tensor
    all_proposals: Optional[Tensor] = None
    drivor_score: Optional[Tensor] = None
    suprim_score: Optional[Tensor] = None
