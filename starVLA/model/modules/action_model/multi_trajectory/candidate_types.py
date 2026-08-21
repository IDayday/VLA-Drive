"""Typed candidate and output contracts shared by DDP-DRS modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class DynamicScorerOutput:
    sub_scores: Dict[str, torch.Tensor]
    aggregate_score: torch.Tensor
    topk_indices: torch.Tensor
    topk_trajectories: torch.Tensor
    score_states: torch.Tensor


@dataclass
class CandidateMetadata:
    """Candidate provenance; source 0 is static and source 1 is dynamic."""

    source: torch.Tensor
    source_index: torch.Tensor
    dynamic_candidate_id: Optional[torch.Tensor]

    def validate(self, expected_shape: Optional[torch.Size] = None) -> None:
        if self.source.shape != self.source_index.shape:
            raise ValueError("candidate source and source_index shapes differ")
        if expected_shape is not None and self.source.shape != expected_shape:
            raise ValueError(
                f"candidate metadata shape {tuple(self.source.shape)} does not match "
                f"{tuple(expected_shape)}"
            )
        if self.dynamic_candidate_id is not None:
            if self.dynamic_candidate_id.shape != self.source.shape:
                raise ValueError("dynamic_candidate_id shape does not match source")
            dynamic = self.source == 1
            if dynamic.any() and (self.dynamic_candidate_id[dynamic] < 0).any():
                raise ValueError("dynamic candidates are missing dynamic_candidate_id")
        if not torch.all((self.source == 0) | (self.source == 1)):
            raise ValueError("candidate source must be 0 (static) or 1 (dynamic)")
        if (self.source_index < 0).any():
            raise ValueError("candidate source_index must be non-negative")

    def gather(self, indices: torch.Tensor) -> "CandidateMetadata":
        if indices.ndim != 2 or indices.shape[0] != self.source.shape[0]:
            raise ValueError("metadata gather indices must have shape [B, K]")
        gathered_dynamic_id = (
            None
            if self.dynamic_candidate_id is None
            else torch.gather(self.dynamic_candidate_id, 1, indices)
        )
        result = CandidateMetadata(
            source=torch.gather(self.source, 1, indices),
            source_index=torch.gather(self.source_index, 1, indices),
            dynamic_candidate_id=gathered_dynamic_id,
        )
        result.validate(indices.shape)
        return result


@dataclass
class JointSelectorOutput:
    selected_trajectory_40: torch.Tensor
    selected_trajectory_8: torch.Tensor
    selected_source: torch.Tensor
    selected_source_index: torch.Tensor
    coarse_scores: Dict[str, torch.Tensor]
    fine_scores: Dict[str, object]
    top256_indices: torch.Tensor
    top256_metadata: CandidateMetadata


@dataclass
class PlannerDiagnostics:
    # These two fields are deliberately never populated by inference.  They
    # are reserved for an external, offline evaluator when that mode permits.
    single_ddp_score: Optional[torch.Tensor] = None
    dynamic_oracle_score: Optional[torch.Tensor] = None
    drivor_selected_index: Optional[torch.Tensor] = None
    drivor_sub_scores: Optional[Dict[str, torch.Tensor]] = None
    drivor_aggregate_score: Optional[torch.Tensor] = None
    dynamic_top16_indices: Optional[torch.Tensor] = None
    suprim_top256_indices: Optional[torch.Tensor] = None
    final_candidate_source: Optional[torch.Tensor] = None
    final_candidate_source_index: Optional[torch.Tensor] = None
    dynamic_selected_ratio: Optional[torch.Tensor] = None
    dynamic_enter_top256_ratio: Optional[torch.Tensor] = None
    latency_qwen: Optional[float] = None
    latency_ddp_sampling: Optional[float] = None
    latency_scene_compressor: Optional[float] = None
    latency_drivor_scorer: Optional[float] = None
    latency_suprim_coarse: Optional[float] = None
    latency_suprim_refinement: Optional[float] = None
    latency_total_inference: Optional[float] = None
    global_scene_tokens_bytes: Optional[int] = None
    dense_scene_memory_bytes: Optional[int] = None
    peak_memory: Optional[int] = None
