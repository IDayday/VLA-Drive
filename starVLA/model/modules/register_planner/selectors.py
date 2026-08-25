"""Adapters that keep staged Register64 selectors independent of generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.planning.types import CandidateMetadata
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DynamicScorerOutput,
)
from starVLA.model.modules.trajectory_scorer.drivesuprim_joint_scorer import (
    DriveSuprimCoarseOutput,
    DriveSuprimCoarseScorer,
    DriveSuprimFineOutput,
    DriveSuprimFineRefiner,
)


def dynamic_coarse_output(
    drivor_output: DynamicScorerOutput,
    *,
    codec: Optional[TrajectoryCodec] = None,
) -> DriveSuprimCoarseOutput:
    """Expose DrivoR Top-M as the complete DriveSuprim fine candidate set."""

    codec = codec or TrajectoryCodec()
    trajectories_40 = codec.upsample_8_to_40(
        drivor_output.topm_trajectories_8
    )
    batch_size, candidate_count = drivor_output.topm_indices.shape
    device = drivor_output.topm_indices.device
    local_indices = torch.arange(
        candidate_count, device=device, dtype=torch.long
    )[None].expand(batch_size, -1)
    metadata = CandidateMetadata(
        source=torch.ones_like(local_indices),
        source_index=drivor_output.topm_indices,
        absolute_index=drivor_output.topm_indices,
    )
    metadata.validate(batch_size, candidate_count)
    topm_logits = {
        name: torch.gather(value, 1, drivor_output.topm_indices)
        for name, value in drivor_output.metric_logits.items()
    }
    topm_score = torch.gather(
        drivor_output.aggregate_score, 1, drivor_output.topm_indices
    )
    return DriveSuprimCoarseOutput(
        metric_logits=topm_logits,
        aggregate_score=topm_score,
        joint_candidates_40=trajectories_40,
        candidate_states=drivor_output.topm_candidate_states,
        metadata=metadata,
        topk_indices=local_indices,
        topk_metric_logits=topm_logits,
        topk_trajectories_40=trajectories_40,
        topk_candidate_states=drivor_output.topm_candidate_states,
        topk_metadata=metadata,
    )


class DynamicDriveSuprimSelector(nn.Module):
    """Fine-only DriveSuprim selector over exactly DrivoR Top-M candidates."""

    def __init__(self, fine_refiner: DriveSuprimFineRefiner) -> None:
        super().__init__()
        self.fine_refiner = fine_refiner
        self.codec = TrajectoryCodec()

    def forward(
        self,
        drivor_output: DynamicScorerOutput,
        scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> DriveSuprimFineOutput:
        coarse = dynamic_coarse_output(drivor_output, codec=self.codec)
        return self.fine_refiner(
            coarse, scene_memory, memory_key_padding_mask
        )


@dataclass
class HybridSelectorOutput:
    coarse: DriveSuprimCoarseOutput
    fine: DriveSuprimFineOutput


class HybridDriveSuprimSelector(nn.Module):
    """Shared-head static + DrivoR-Top-M dynamic DriveSuprim selector."""

    def __init__(
        self,
        coarse_scorer: DriveSuprimCoarseScorer,
        fine_refiner: DriveSuprimFineRefiner,
    ) -> None:
        super().__init__()
        self.coarse_scorer = coarse_scorer
        self.fine_refiner = fine_refiner
        self.codec = TrajectoryCodec()

    def forward(
        self,
        drivor_output: DynamicScorerOutput,
        global_scene_tokens: Tensor,
        ego_state: Tensor,
        fine_scene_memory: Tensor,
        fine_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> HybridSelectorOutput:
        dynamic_40 = self.codec.upsample_8_to_40(
            drivor_output.topm_trajectories_8
        )
        coarse = self.coarse_scorer(
            global_scene_tokens,
            ego_state,
            dynamic_trajectories_40=dynamic_40,
            dynamic_candidate_ids=drivor_output.topm_indices,
        )
        fine = self.fine_refiner(
            coarse, fine_scene_memory, fine_memory_key_padding_mask
        )
        return HybridSelectorOutput(coarse=coarse, fine=fine)
