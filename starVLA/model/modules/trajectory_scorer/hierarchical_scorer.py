# SPDX-License-Identifier: Apache-2.0
# DrivoR portions are adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a.  DriveSuprim portions are
# adapted from William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02.  This wrapper contributes only
# unified target/index routing and the joint-training interface.

"""One hierarchical scorer joining DrivoR preselection and DriveSuprim."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec

from .drivor_dynamic_scorer import DrivoRDynamicScorer, DynamicScorerOutput
from .drivesuprim_joint_scorer import (
    DriveSuprimCoarseOutput,
    DriveSuprimCoarseScorer,
    DriveSuprimFineOutput,
    DriveSuprimFineRefiner,
)
from .losses import (
    DRIVOR_METRICS,
    SUPRIM_METRICS,
    DriveSuprimMetricLoss,
    DrivoRMetricLoss,
    gather_metric_targets,
)


def _require_targets(
    values: Mapping[str, Tensor], names: tuple[str, ...], label: str
) -> Dict[str, Tensor]:
    missing = set(names).difference(values)
    if missing:
        raise KeyError(f"{label} targets are missing {sorted(missing)}")
    return {name: values[name] for name in names}


def _concat_targets(
    static_targets: Mapping[str, Tensor], dynamic_targets: Mapping[str, Tensor]
) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for name in SUPRIM_METRICS:
        static = static_targets[name]
        dynamic = dynamic_targets[name].to(device=static.device, dtype=static.dtype)
        if static.ndim != 2 or dynamic.ndim != 2 or static.shape[0] != dynamic.shape[0]:
            raise ValueError(f"joint metric target {name} must contain aligned [B,N] tensors")
        result[name] = torch.cat((static, dynamic), dim=1)
    return result


class HierarchicalDrivoRSuprimScorer(nn.Module):
    """A single scorer model containing DrivoR, coarse, and fine modules.

    Dynamic proposal geometry is always detached.  Scene gradients are allowed
    by default and can be disabled at this wrapper boundary for the documented
    ablation.  No scorer tensor is consumed by the Flow proposal generator.
    """

    def __init__(
        self,
        dynamic_prescorer: Optional[DrivoRDynamicScorer],
        joint_coarse_scorer: Optional[DriveSuprimCoarseScorer],
        joint_fine_refiner: Optional[DriveSuprimFineRefiner],
        *,
        detach_scene_for_scorer: bool = False,
        sigma: float = 0.5,
        use_refinement_imitation: bool = True,
        fine_memory_source: str = "dense_qwen_memory",
    ) -> None:
        super().__init__()
        self.dynamic_prescorer = dynamic_prescorer
        self.joint_coarse_scorer = joint_coarse_scorer
        self.joint_fine_refiner = joint_fine_refiner
        self.detach_scene_for_scorer = bool(detach_scene_for_scorer)
        self.use_refinement_imitation = bool(use_refinement_imitation)
        if fine_memory_source == "dense_scene_memory":
            fine_memory_source = "dense_qwen_memory"
        if fine_memory_source not in {"dense_qwen_memory", "global_scene_tokens"}:
            raise ValueError(
                "fine_memory_source must be 'dense_scene_memory' or "
                "'global_scene_tokens'"
            )
        self.fine_memory_source = fine_memory_source
        self.codec = TrajectoryCodec()
        self.drivor_loss_fn = DrivoRMetricLoss()
        self.suprim_loss_fn = DriveSuprimMetricLoss(sigma=sigma)

    def _scene(
        self, global_scene_tokens: Tensor, dense_scene_memory: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.detach_scene_for_scorer:
            return global_scene_tokens.detach(), dense_scene_memory.detach()
        return global_scene_tokens, dense_scene_memory

    def _suprim_from_coarse(
        self,
        coarse: DriveSuprimCoarseOutput,
        dense_scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        joint_targets: Mapping[str, Tensor],
        gt_trajectory_8: Tensor,
    ) -> tuple[DriveSuprimFineOutput, Tensor, Tensor, Dict[str, Tensor]]:
        if self.joint_fine_refiner is None:
            raise RuntimeError("DriveSuprim fine refinement is disabled")
        coarse_loss, coarse_details = self.suprim_loss_fn.one_layer(
            coarse.metric_logits,
            joint_targets,
            coarse.joint_candidates_40,
            gt_trajectory_8,
            use_imitation=True,
        )
        topk_targets = gather_metric_targets(joint_targets, coarse.topk_indices)
        fine = self.joint_fine_refiner(
            coarse, dense_scene_memory, memory_key_padding_mask
        )
        fine_loss, fine_details = self.suprim_loss_fn.refinement(
            fine.layer_metric_logits,
            topk_targets,
            coarse.topk_trajectories_40,
            gt_trajectory_8,
            use_imitation=self.use_refinement_imitation,
        )
        details = {
            **{f"suprim_coarse_{name}": value for name, value in coarse_details.items()},
            **{f"suprim_fine_{name}": value for name, value in fine_details.items()},
        }
        return fine, coarse_loss, fine_loss, details

    def _fine_memory(
        self,
        global_scene_tokens: Tensor,
        dense_scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.fine_memory_source == "global_scene_tokens":
            return global_scene_tokens, None
        return dense_scene_memory, memory_key_padding_mask

    def _selected_trajectory_8(
        self,
        fine: DriveSuprimFineOutput,
        dynamic_proposals_8: Optional[Tensor],
    ) -> Tensor:
        selected = self.codec.downsample_40_to_8(fine.selected_trajectory_40).clone()
        dynamic = fine.selected_source == 1
        if dynamic.any():
            if dynamic_proposals_8 is None:
                raise RuntimeError("dynamic selection is missing original 8-point proposals")
            rows = torch.arange(
                selected.shape[0], device=selected.device, dtype=torch.long
            )[dynamic]
            ids = fine.selected_source_index[dynamic]
            if (ids < 0).any() or (ids >= dynamic_proposals_8.shape[1]).any():
                raise RuntimeError("dynamic candidate source metadata is out of range")
            selected[dynamic] = dynamic_proposals_8.detach().to(
                device=selected.device, dtype=selected.dtype
            )[rows, ids]
        return selected

    def forward_dynamic_only(
        self,
        *,
        dynamic_proposals_8: Tensor,
        dynamic_targets: Mapping[str, Tensor],
        global_scene_tokens: Tensor,
        ego_state: Tensor,
        dynamic_topm: int = 32,
    ) -> Dict[str, Any]:
        """Train/evaluate the B3 DrivoR-only hierarchy and select its Top-1."""

        if self.dynamic_prescorer is None:
            raise RuntimeError("DrivoR dynamic scorer is disabled")
        scene_tokens = (
            global_scene_tokens.detach()
            if self.detach_scene_for_scorer
            else global_scene_tokens
        )
        targets = _require_targets(dynamic_targets, DRIVOR_METRICS, "dynamic")
        dynamic = self.dynamic_prescorer(
            dynamic_proposals_8,
            scene_tokens,
            ego_state,
            topm=dynamic_topm,
        )
        drivor_loss, details = self.drivor_loss_fn(dynamic.metric_logits, targets)
        selected_8 = dynamic.topm_trajectories_8[:, 0]
        selected_id = dynamic.topm_indices[:, 0]
        zero = drivor_loss.new_zeros(())
        return {
            "losses": {
                "drivor": drivor_loss,
                "suprim_coarse": zero,
                "suprim_fine": zero,
            },
            "outputs": {
                "selected_trajectory_8": selected_8,
                "selected_trajectory_40": self.codec.upsample_8_to_40(
                    selected_8[:, None]
                )[:, 0],
                "selected_absolute_index": selected_id,
                "selected_source": torch.ones_like(selected_id),
                "dynamic_topm_indices": dynamic.topm_indices,
                "coarse_topk_indices": None,
            },
            "metrics": {
                **{f"drivor_{name}": value for name, value in details.items()},
                "drivor_score_mean": dynamic.aggregate_score.mean(),
                "drivor_score_std": dynamic.aggregate_score.std(unbiased=False),
            },
        }

    def predict_dynamic_only(
        self,
        *,
        dynamic_proposals_8: Tensor,
        global_scene_tokens: Tensor,
        ego_state: Tensor,
        dynamic_topm: int = 32,
    ) -> Dict[str, Tensor]:
        """B3 inference: DrivoR ranks all dynamics and its Top-1 is final."""

        if self.dynamic_prescorer is None:
            raise RuntimeError("DrivoR dynamic scorer is disabled")
        scene_tokens = (
            global_scene_tokens.detach()
            if self.detach_scene_for_scorer
            else global_scene_tokens
        )
        dynamic = self.dynamic_prescorer(
            dynamic_proposals_8,
            scene_tokens,
            ego_state,
            topm=dynamic_topm,
        )
        selected_8 = dynamic.topm_trajectories_8[:, 0]
        selected_id = dynamic.topm_indices[:, 0]
        return {
            "selected_trajectory_8": selected_8,
            "selected_trajectory_40": self.codec.upsample_8_to_40(
                selected_8[:, None]
            )[:, 0],
            "selected_absolute_index": selected_id,
            "selected_source": torch.ones_like(selected_id),
            "dynamic_topm_indices": dynamic.topm_indices,
            "coarse_topk_indices": None,
            "drivor_aggregate_score": dynamic.aggregate_score,
        }

    def forward_static_only(
        self,
        *,
        global_scene_tokens: Tensor,
        dense_scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        ego_state: Tensor,
        gt_trajectory_8: Tensor,
        static_targets: Mapping[str, Tensor],
    ) -> Dict[str, Any]:
        """Train DriveSuprim using only the static vocabulary curriculum phase."""

        if self.joint_coarse_scorer is None or self.joint_fine_refiner is None:
            raise RuntimeError("DriveSuprim joint scorer is disabled")

        global_scene_tokens, dense_scene_memory = self._scene(
            global_scene_tokens, dense_scene_memory
        )
        static_targets = _require_targets(static_targets, SUPRIM_METRICS, "static")
        coarse = self.joint_coarse_scorer(global_scene_tokens, ego_state)
        for name, target in static_targets.items():
            expected = (coarse.aggregate_score.shape[0], coarse.aggregate_score.shape[1])
            if tuple(target.shape) != expected:
                raise ValueError(
                    f"static target {name} shape is {tuple(target.shape)}, expected {expected}"
                )
        fine_memory, fine_mask = self._fine_memory(
            global_scene_tokens, dense_scene_memory, memory_key_padding_mask
        )
        fine, coarse_loss, fine_loss, details = self._suprim_from_coarse(
            coarse,
            fine_memory,
            fine_mask,
            static_targets,
            gt_trajectory_8,
        )
        zero = coarse_loss.new_zeros(())
        selected_8 = self._selected_trajectory_8(fine, None)
        metrics: Dict[str, Tensor] = {
            **details,
            "coarse_topk_dynamic_ratio": zero,
            "final_selected_dynamic_ratio": zero,
        }
        return {
            "losses": {
                "drivor": zero,
                "suprim_coarse": coarse_loss,
                "suprim_fine": fine_loss,
            },
            "outputs": {
                "selected_trajectory_40": fine.selected_trajectory_40,
                "selected_trajectory_8": selected_8,
                "selected_absolute_index": fine.selected_absolute_index,
                "selected_source": fine.selected_source,
                "dynamic_topm_indices": None,
                "coarse_topk_indices": coarse.topk_indices,
            },
            "metrics": metrics,
        }

    def forward_full(
        self,
        *,
        dynamic_proposals_8: Tensor,
        dynamic_targets: Mapping[str, Tensor],
        global_scene_tokens: Tensor,
        dense_scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        ego_state: Tensor,
        gt_trajectory_8: Tensor,
        static_targets: Mapping[str, Tensor],
        dynamic_topm: int = 32,
    ) -> Dict[str, Any]:
        """Train the full 64 -> Top-M -> unified Top-K hierarchy."""

        if (
            self.dynamic_prescorer is None
            or self.joint_coarse_scorer is None
            or self.joint_fine_refiner is None
        ):
            raise RuntimeError("full hierarchy requires DrivoR and DriveSuprim")

        global_scene_tokens, dense_scene_memory = self._scene(
            global_scene_tokens, dense_scene_memory
        )
        drivor_targets = _require_targets(dynamic_targets, DRIVOR_METRICS, "dynamic")
        suprim_dynamic_targets = _require_targets(
            dynamic_targets, SUPRIM_METRICS, "dynamic"
        )
        static_targets = _require_targets(static_targets, SUPRIM_METRICS, "static")
        dynamic: DynamicScorerOutput = self.dynamic_prescorer(
            dynamic_proposals_8,
            global_scene_tokens,
            ego_state,
            topm=dynamic_topm,
        )
        drivor_loss, drivor_details = self.drivor_loss_fn(
            dynamic.metric_logits, drivor_targets
        )
        topm_dynamic_targets = gather_metric_targets(
            suprim_dynamic_targets, dynamic.topm_indices
        )
        dynamic_40 = self.codec.upsample_8_to_40(dynamic.topm_trajectories_8)
        coarse = self.joint_coarse_scorer(
            global_scene_tokens,
            ego_state,
            dynamic_trajectories_40=dynamic_40,
            dynamic_candidate_ids=dynamic.topm_indices,
        )
        joint_targets = _concat_targets(static_targets, topm_dynamic_targets)
        fine_memory, fine_mask = self._fine_memory(
            global_scene_tokens, dense_scene_memory, memory_key_padding_mask
        )
        fine, coarse_loss, fine_loss, details = self._suprim_from_coarse(
            coarse,
            fine_memory,
            fine_mask,
            joint_targets,
            gt_trajectory_8,
        )
        selected_8 = self._selected_trajectory_8(fine, dynamic_proposals_8)
        topk_dynamic_ratio = coarse.topk_metadata.source.float().mean()
        final_dynamic_ratio = fine.selected_source.float().mean()
        metrics: Dict[str, Tensor] = {
            **{f"drivor_{name}": value for name, value in drivor_details.items()},
            **details,
            "drivor_score_mean": dynamic.aggregate_score.mean(),
            "drivor_score_std": dynamic.aggregate_score.std(unbiased=False),
            "coarse_topk_dynamic_ratio": topk_dynamic_ratio,
            "final_selected_dynamic_ratio": final_dynamic_ratio,
        }
        if "aggregate_score" in dynamic_targets:
            aggregate_target = dynamic_targets["aggregate_score"]
            if tuple(aggregate_target.shape) != tuple(dynamic.aggregate_score.shape):
                raise ValueError("dynamic aggregate_score target has the wrong shape")
            metrics["dynamic_oracle_score"] = aggregate_target.max(dim=1).values.mean()
            selected_dynamic_score = torch.gather(
                aggregate_target.to(dynamic.topm_indices.device),
                1,
                dynamic.topm_indices,
            )
            metrics["dynamic_selected_score"] = selected_dynamic_score.mean()
        return {
            "losses": {
                "drivor": drivor_loss,
                "suprim_coarse": coarse_loss,
                "suprim_fine": fine_loss,
            },
            "outputs": {
                "selected_trajectory_40": fine.selected_trajectory_40,
                "selected_trajectory_8": selected_8,
                "selected_absolute_index": fine.selected_absolute_index,
                "selected_source": fine.selected_source,
                "dynamic_topm_indices": dynamic.topm_indices,
                "coarse_topk_indices": coarse.topk_indices,
            },
            "metrics": metrics,
        }

    def predict(
        self,
        *,
        dynamic_proposals_8: Tensor,
        global_scene_tokens: Tensor,
        dense_scene_memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        ego_state: Tensor,
        dynamic_topm: int = 32,
    ) -> Dict[str, Tensor]:
        """Learned-only inference; no metric evaluator or target cache is read."""

        if (
            self.dynamic_prescorer is None
            or self.joint_coarse_scorer is None
            or self.joint_fine_refiner is None
        ):
            raise RuntimeError("full hierarchy requires DrivoR and DriveSuprim")

        global_scene_tokens, dense_scene_memory = self._scene(
            global_scene_tokens, dense_scene_memory
        )
        dynamic = self.dynamic_prescorer(
            dynamic_proposals_8,
            global_scene_tokens,
            ego_state,
            topm=dynamic_topm,
        )
        dynamic_40 = self.codec.upsample_8_to_40(dynamic.topm_trajectories_8)
        coarse = self.joint_coarse_scorer(
            global_scene_tokens,
            ego_state,
            dynamic_trajectories_40=dynamic_40,
            dynamic_candidate_ids=dynamic.topm_indices,
        )
        fine_memory, fine_mask = self._fine_memory(
            global_scene_tokens, dense_scene_memory, memory_key_padding_mask
        )
        fine = self.joint_fine_refiner(coarse, fine_memory, fine_mask)
        selected_8 = self._selected_trajectory_8(fine, dynamic_proposals_8)
        return {
            "selected_trajectory_40": fine.selected_trajectory_40,
            "selected_trajectory_8": selected_8,
            "selected_absolute_index": fine.selected_absolute_index,
            "selected_source": fine.selected_source,
            "dynamic_topm_indices": dynamic.topm_indices,
            "coarse_topk_indices": coarse.topk_indices,
            "drivor_aggregate_score": dynamic.aggregate_score,
            "coarse_aggregate_score": coarse.aggregate_score,
            "fine_aggregate_score": fine.aggregate_score,
        }
