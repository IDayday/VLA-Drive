"""Stable multi-objective alignment of VLM queries to cached VGGT targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F


@dataclass
class VGGTAlignmentOutput:
    """Alignment result containing a scalar loss and detached diagnostics."""

    loss: torch.Tensor
    losses: Dict[str, torch.Tensor]
    metrics: Dict[str, torch.Tensor]
    projected_queries: torch.Tensor


class VGGTQueryAligner(nn.Module):
    """Align student queries ``[B,Q,H]`` with teacher targets ``[B,Q,D]``."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        special_query_count: int,
        cosine_weight: float = 1.0,
        smooth_l1_weight: float = 0.1,
        relational_weight: float = 0.05,
        scene_relation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.student_norm = nn.LayerNorm(student_dim)
        self.student_projection = nn.Linear(student_dim, teacher_dim)
        if student_dim == teacher_dim:
            nn.init.eye_(self.student_projection.weight)
            nn.init.zeros_(self.student_projection.bias)
        self.special_query_count = int(special_query_count)
        self.cosine_weight = float(cosine_weight)
        self.smooth_l1_weight = float(smooth_l1_weight)
        self.relational_weight = float(relational_weight)
        self.scene_relation_weight = float(scene_relation_weight)

    def project_student(self, student_queries: torch.Tensor) -> torch.Tensor:
        """Return the exact normalized/projection memory consumed by planning."""

        assert student_queries.ndim == 3, "student_queries must be [B,Q,H]"
        compute_dtype = self.student_norm.weight.dtype
        normalized = self.student_norm(student_queries.to(dtype=compute_dtype))
        return self.student_projection(normalized)

    @staticmethod
    def _gather_scenes(value: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Gather variable local batches while retaining local student gradients."""

        if not dist.is_available() or not dist.is_initialized():
            return value, 0
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_size = torch.tensor([value.shape[0]], device=value.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        sizes_int = [int(item.item()) for item in sizes]
        maximum = max(sizes_int)
        if value.shape[0] < maximum:
            padding = value.new_zeros(maximum - value.shape[0], value.shape[1])
            padded = torch.cat((value.detach(), padding), dim=0)
        else:
            padded = value.detach()
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        parts = []
        for owner, (part, size) in enumerate(zip(gathered, sizes_int)):
            parts.append(value if owner == rank else part[:size])
        return torch.cat(parts, dim=0), sum(sizes_int[:rank])

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert values.shape == mask.shape
        count = mask.sum().clamp_min(1)
        return (values * mask.to(values.dtype)).sum() / count

    def forward(
        self,
        student_queries: torch.Tensor,
        teacher_features: torch.Tensor,
        valid_mask: torch.Tensor,
        slot_mean: torch.Tensor | None = None,
        slot_scale: torch.Tensor | None = None,
    ) -> VGGTAlignmentOutput:
        assert student_queries.ndim == 3, "student_queries must be [B,Q,H]"
        assert teacher_features.ndim == 3, "teacher_features must be [B,Q,D]"
        assert student_queries.shape[:2] == teacher_features.shape[:2]
        assert valid_mask.shape == student_queries.shape[:2]
        assert valid_mask.dtype == torch.bool
        assert valid_mask.any(), "VGGT alignment batch contains no valid target"

        # Keep similarity/retrieval losses in FP32 even when DeepSpeed stores
        # the learned projection in BF16. Planner consumers cast this returned
        # memory to their own parameter dtype at the module boundary.
        student = self.project_student(student_queries).float()
        teacher = F.layer_norm(teacher_features.float(), (teacher_features.shape[-1],))
        assert student.shape == teacher.shape
        student_unit = F.normalize(student, dim=-1, eps=1e-6)
        teacher_unit = F.normalize(teacher, dim=-1, eps=1e-6)

        cosine_per_query = 1.0 - (student_unit * teacher_unit).sum(dim=-1)
        cosine_loss = self._masked_mean(cosine_per_query, valid_mask)
        smooth_per_query = F.smooth_l1_loss(student, teacher, reduction="none").mean(dim=-1)
        smooth_loss = self._masked_mean(smooth_per_query, valid_mask)

        relation_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        student_relation = student_unit @ student_unit.transpose(1, 2)
        teacher_relation = teacher_unit @ teacher_unit.transpose(1, 2)
        relation_error = F.smooth_l1_loss(
            student_relation, teacher_relation, reduction="none"
        )
        relational_loss = self._masked_mean(relation_error, relation_mask)

        special_mask = valid_mask[:, : self.special_query_count]
        spatial_mask = valid_mask[:, self.special_query_count :]
        base_per_query = (
            self.cosine_weight * cosine_per_query
            + self.smooth_l1_weight * smooth_per_query
        )
        zero = cosine_loss.detach() * 0.0
        global_loss = (
            self._masked_mean(base_per_query[:, : self.special_query_count], special_mask)
            if special_mask.any()
            else zero
        )
        spatial_loss = (
            self._masked_mean(base_per_query[:, self.special_query_count :], spatial_mask)
            if spatial_mask.any()
            else zero
        )
        cosine_similarity = 1.0 - cosine_per_query
        student_scene = F.normalize(
            (student_unit * valid_mask.unsqueeze(-1)).sum(1)
            / valid_mask.sum(1, keepdim=True).clamp_min(1),
            dim=-1,
        )
        teacher_scene = F.normalize(
            (teacher_unit * valid_mask.unsqueeze(-1)).sum(1)
            / valid_mask.sum(1, keepdim=True).clamp_min(1),
            dim=-1,
        )
        global_student_scene, positive_offset = self._gather_scenes(student_scene)
        global_teacher_scene, _ = self._gather_scenes(teacher_scene.detach())
        student_relation_scene = student_scene @ global_student_scene.transpose(0, 1)
        teacher_relation_scene = teacher_scene @ global_teacher_scene.transpose(0, 1)
        scene_relation_loss = F.smooth_l1_loss(
            student_relation_scene, teacher_relation_scene.detach()
        )
        retrieval_similarity = student_scene @ global_teacher_scene.transpose(0, 1)
        expected = torch.arange(student.shape[0], device=student.device) + positive_offset
        top_count = min(5, global_teacher_scene.shape[0])
        retrieval_top = retrieval_similarity.topk(top_count, dim=1).indices
        retrieval_top1 = retrieval_top[:, 0].eq(expected).float().mean()
        retrieval_top5 = retrieval_top.eq(expected.unsqueeze(1)).any(dim=1).float().mean()
        student_scene_variance = global_student_scene.detach().float().var(
            dim=0, unbiased=False
        ).mean()
        teacher_scene_variance = global_teacher_scene.float().var(
            dim=0, unbiased=False
        ).mean()
        total = (
            global_loss
            + spatial_loss
            + self.relational_weight * relational_loss
            + self.scene_relation_weight * scene_relation_loss
        )
        template_metrics = {}
        if slot_mean is not None:
            assert slot_mean.shape == teacher.shape[1:]
            template = slot_mean.to(device=student.device, dtype=torch.float32)
            template_unit = F.normalize(template, dim=-1, eps=1e-6)
            template_cosine = (student_unit * template_unit.unsqueeze(0)).sum(-1)
            if student.shape[0] > 1:
                shuffled_teacher = teacher_unit.roll(shifts=1, dims=0)
                shuffled_cosine = (student_unit * shuffled_teacher).sum(-1)
            else:
                shuffled_cosine = cosine_similarity.detach()
            scale = (
                slot_scale.to(device=student.device, dtype=torch.float32)
                if slot_scale is not None
                else torch.ones(template.shape[0], device=student.device)
            )
            assert scale.shape == template.shape[:1]
            student_residual = F.normalize(
                (student - template.unsqueeze(0)) / scale.clamp_min(1e-6)[None, :, None],
                dim=-1,
                eps=1e-6,
            )
            teacher_residual = F.normalize(
                (teacher - template.unsqueeze(0)) / scale.clamp_min(1e-6)[None, :, None],
                dim=-1,
                eps=1e-6,
            )
            residual_cosine = (student_residual * teacher_residual).sum(-1)
            template_metrics = {
                "cosine_slot_mean": self._masked_mean(template_cosine, valid_mask).detach(),
                "cosine_shuffled": self._masked_mean(shuffled_cosine, valid_mask).detach(),
                "correct_minus_slot_mean": (
                    self._masked_mean(cosine_similarity - template_cosine, valid_mask).detach()
                ),
                "correct_minus_shuffled": (
                    self._masked_mean(cosine_similarity - shuffled_cosine, valid_mask).detach()
                ),
                "scene_residual_cosine": self._masked_mean(
                    residual_cosine, valid_mask
                ).detach(),
            }
        metrics = {
            "cosine_all": self._masked_mean(cosine_similarity, valid_mask).detach(),
            "cosine_special": (
                self._masked_mean(cosine_similarity[:, : self.special_query_count], special_mask).detach()
                if special_mask.any()
                else zero
            ),
            "cosine_spatial": (
                self._masked_mean(cosine_similarity[:, self.special_query_count :], spatial_mask).detach()
                if spatial_mask.any()
                else zero
            ),
            "student_std": student.detach().std(),
            "teacher_std": teacher.detach().std(),
            "valid_ratio": valid_mask.float().mean().detach(),
            "distributed_retrieval_top1": retrieval_top1.detach(),
            "distributed_retrieval_top5": retrieval_top5.detach(),
            "scene_relation_loss": scene_relation_loss.detach(),
            "student_teacher_scene_variance_ratio": (
                student_scene_variance / teacher_scene_variance.clamp_min(1e-8)
            ).detach(),
            **template_metrics,
        }
        return VGGTAlignmentOutput(
            loss=total,
            losses={
                "cosine": cosine_loss,
                "smooth_l1": smooth_loss,
                "relational": relational_loss,
                "global": global_loss,
                "spatial": spatial_loss,
                "scene_relation": scene_relation_loss,
            },
            metrics=metrics,
            projected_queries=student,
        )
