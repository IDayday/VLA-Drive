# SPDX-License-Identifier: Apache-2.0
# DrivoR losses/formula adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a,
# navsim/agents/drivoR/layers/losses/drivor_loss.py and
# navsim/agents/drivoR/score_module/compute_navsim_score.py.
# DriveSuprim losses/formula adapted from William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02,
# navsim/agents/drivesuprim/drivesuprim_loss_fn.py and drivesuprim_model.py.
# Project adaptations: named metric dictionaries, stable log operations,
# non-mutating targets, dynamic joint pools, and explicit per-layer losses.

"""Official DrivoR/DriveSuprim score aggregation and supervised losses."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec


DRIVOR_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
)

SUPRIM_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
)

SUPRIM_PDM_LOSS_WEIGHTS = {
    "no_at_fault_collisions": 3.0,
    "drivable_area_compliance": 3.0,
    "time_to_collision_within_bound": 4.0,
    "ego_progress": 2.0,
    "driving_direction_compliance": 1.0,
    "lane_keeping": 2.0,
    "traffic_light_compliance": 3.0,
    "history_comfort": 1.0,
}


def aggregate_drivor_score(
    metric_logits: Mapping[str, Tensor],
    *,
    noc: float = 1.0,
    dac: float = 1.0,
    ddc: float = 0.0,
    ttc: float = 5.0,
    ep: float = 5.0,
    comfort: float = 2.0,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the donor DrivoR aggregate in float32 for stability."""

    missing = set(DRIVOR_METRICS).difference(metric_logits)
    if missing:
        raise KeyError(f"missing DrivoR logits: {sorted(missing)}")
    logits = {name: metric_logits[name].float() for name in DRIVOR_METRICS}
    weighted = (
        ttc * torch.sigmoid(logits["time_to_collision_within_bound"])
        + ep * torch.sigmoid(logits["ego_progress"])
        + comfort * torch.sigmoid(logits["comfort"])
    )
    return (
        noc * F.logsigmoid(logits["no_at_fault_collisions"])
        + dac * F.logsigmoid(logits["drivable_area_compliance"])
        + ddc * F.logsigmoid(logits["driving_direction_compliance"])
        + torch.log(weighted.clamp_min(eps))
    )


def aggregate_drivesuprim_score(
    metric_logits: Mapping[str, Tensor],
    *,
    include_imitation: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the official DriveSuprim coarse/fine aggregate in float32."""

    missing = set(SUPRIM_METRICS).difference(metric_logits)
    if missing:
        raise KeyError(f"missing DriveSuprim logits: {sorted(missing)}")
    logits = {name: metric_logits[name].float() for name in SUPRIM_METRICS}
    weighted = (
        5.0 * torch.sigmoid(logits["time_to_collision_within_bound"])
        + 5.0 * torch.sigmoid(logits["ego_progress"])
        + 2.0 * torch.sigmoid(logits["lane_keeping"])
        + torch.sigmoid(logits["history_comfort"])
    )
    score = (
        0.1 * F.logsigmoid(logits["traffic_light_compliance"])
        + 0.5 * F.logsigmoid(logits["no_at_fault_collisions"])
        + 0.5 * F.logsigmoid(logits["drivable_area_compliance"])
        + 0.3 * F.logsigmoid(logits["driving_direction_compliance"])
        + 6.0 * torch.log(weighted.clamp_min(eps))
    )
    if include_imitation:
        if "imi" not in metric_logits:
            raise KeyError("DriveSuprim imitation logits are missing")
        score = score + 0.02 * F.log_softmax(metric_logits["imi"].float(), dim=-1)
    return score


def _three_to_two_classes(target: Tensor) -> Tensor:
    return torch.where(target == 0.5, torch.zeros_like(target), target)


class DrivoRMetricLoss(nn.Module):
    """Six equally weighted DrivoR BCE losses with donor target masking."""

    def forward(
        self,
        metric_logits: Mapping[str, Tensor],
        metric_targets: Mapping[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        missing = set(DRIVOR_METRICS).difference(metric_logits) | set(
            DRIVOR_METRICS
        ).difference(metric_targets)
        if missing:
            raise KeyError(f"missing DrivoR loss fields: {sorted(missing)}")
        reference = metric_logits[DRIVOR_METRICS[0]]
        losses: Dict[str, Tensor] = {}
        for name in DRIVOR_METRICS:
            target = metric_targets[name].to(
                device=reference.device, dtype=torch.float32
            )
            if target.shape != metric_logits[name].shape:
                raise ValueError(
                    f"DrivoR target {name} shape {tuple(target.shape)} does not "
                    f"match logits {tuple(metric_logits[name].shape)}"
                )
            if name in {"no_at_fault_collisions", "driving_direction_compliance"}:
                target = _three_to_two_classes(target)
            if name == "time_to_collision_within_bound":
                valid = target != 2.0
                elementwise = F.binary_cross_entropy_with_logits(
                    metric_logits[name].float(), target, reduction="none"
                )
                loss = (elementwise * valid.to(elementwise.dtype)).sum() / valid.sum().clamp_min(1)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    metric_logits[name].float(), target
                )
            losses[name] = loss
        return torch.stack(tuple(losses.values())).sum(), losses


class PDMSValueLoss(nn.Module):
    """PDMS-direct value learning plus DrivoR sub-score supervision.

    The aggregate BCE follows DriveVLA-M0's score head.  Listwise soft-target
    distillation and hard-pair ordering make the optimization match the actual
    argmax/regret objective instead of relying on six independently calibrated
    BCE heads alone.  Sub-score BCE remains active to preserve the structured
    DrivoR/CLOVER value representation and Pareto targets.
    """

    def __init__(
        self,
        *,
        submetric_weight: float = 1.0,
        aggregate_weight: float = 1.0,
        listwise_weight: float = 1.0,
        pairwise_weight: float = 0.5,
        listwise_temperature: float = 0.1,
        pairwise_temperature: float = 0.1,
        pairwise_margin: float = 0.05,
    ) -> None:
        super().__init__()
        weights = (
            submetric_weight,
            aggregate_weight,
            listwise_weight,
            pairwise_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("PDMS value-loss weights must be non-negative")
        if listwise_temperature <= 0 or pairwise_temperature <= 0:
            raise ValueError("ranking temperatures must be positive")
        if pairwise_margin < 0:
            raise ValueError("pairwise_margin must be non-negative")
        self.submetric_weight = float(submetric_weight)
        self.aggregate_weight = float(aggregate_weight)
        self.listwise_weight = float(listwise_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.listwise_temperature = float(listwise_temperature)
        self.pairwise_temperature = float(pairwise_temperature)
        self.pairwise_margin = float(pairwise_margin)
        self.submetric_loss = DrivoRMetricLoss()

    def _pairwise(self, prediction: Tensor, target: Tensor) -> Tensor:
        candidate_count = prediction.shape[1]
        if candidate_count < 2:
            return prediction.new_zeros(())
        rows, cols = torch.triu_indices(
            candidate_count,
            candidate_count,
            offset=1,
            device=prediction.device,
        )
        target_delta = target[:, rows] - target[:, cols]
        prediction_delta = prediction[:, rows] - prediction[:, cols]
        valid = target_delta.abs() >= self.pairwise_margin
        signed = target_delta.sign()
        elementwise = F.softplus(
            -signed * prediction_delta / self.pairwise_temperature
        )
        return (elementwise * valid.to(elementwise.dtype)).sum() / valid.sum().clamp_min(1)

    def forward(
        self,
        metric_logits: Mapping[str, Tensor],
        aggregate_logit: Tensor | None,
        metric_targets: Mapping[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if aggregate_logit is None:
            raise ValueError("PDMSValueLoss requires the scalar aggregate head")
        if "aggregate_score" not in metric_targets:
            raise KeyError("PDMS value targets require aggregate_score")
        true_score = metric_targets["aggregate_score"].to(
            device=aggregate_logit.device, dtype=torch.float32
        )
        if true_score.shape != aggregate_logit.shape:
            raise ValueError("aggregate target and logit shapes differ")
        if ((true_score < 0) | (true_score > 1)).any():
            raise ValueError("PDMS aggregate targets must lie in [0,1]")

        submetric, submetric_components = self.submetric_loss(
            metric_logits, metric_targets
        )
        aggregate = F.binary_cross_entropy_with_logits(
            aggregate_logit.float(), true_score
        )
        target_distribution = F.softmax(
            true_score.float() / self.listwise_temperature, dim=1
        )
        listwise = -(
            target_distribution
            * F.log_softmax(
                aggregate_logit.float() / self.listwise_temperature, dim=1
            )
        ).sum(dim=1).mean()
        pairwise = self._pairwise(aggregate_logit.float(), true_score.float())
        total = (
            self.submetric_weight * submetric
            + self.aggregate_weight * aggregate
            + self.listwise_weight * listwise
            + self.pairwise_weight * pairwise
        )
        details = {
            "submetric": submetric,
            "aggregate": aggregate,
            "listwise": listwise,
            "pairwise": pairwise,
            **{
                f"submetric_{name}": value
                for name, value in submetric_components.items()
            },
        }
        return total, details


def imitation_distribution_loss(
    imitation_logits: Tensor,
    candidate_trajectories_40: Tensor,
    gt_trajectory_8: Tensor,
    *,
    sigma: float = 0.5,
    codec: TrajectoryCodec | None = None,
) -> Tensor:
    """DriveSuprim soft target cross entropy over physical pose candidates."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    codec = codec or TrajectoryCodec()
    candidates_8 = codec.downsample_40_to_8(candidate_trajectories_40)
    if gt_trajectory_8.ndim != 3 or tuple(gt_trajectory_8.shape[-2:]) != (8, 3):
        raise ValueError("gt_trajectory_8 must have shape [B,8,3]")
    if candidates_8.shape[:2] != imitation_logits.shape:
        raise ValueError("candidate and imitation-logit dimensions differ")
    target = gt_trajectory_8[:, None].to(
        device=candidates_8.device, dtype=candidates_8.dtype
    )
    distance_logits = -((candidates_8 - target) ** 2).sum(dim=(-2, -1)) / sigma
    target_distribution = F.softmax(distance_logits, dim=-1)
    return -(target_distribution * F.log_softmax(imitation_logits, dim=-1)).sum(-1).mean()


def _suprim_pdm_loss(
    metric_logits: Mapping[str, Tensor],
    metric_targets: Mapping[str, Tensor],
    weights: Mapping[str, float],
) -> Tuple[Tensor, Dict[str, Tensor]]:
    missing = set(SUPRIM_METRICS).difference(metric_logits) | set(
        SUPRIM_METRICS
    ).difference(metric_targets)
    if missing:
        raise KeyError(f"missing DriveSuprim loss fields: {sorted(missing)}")
    reference = metric_logits[SUPRIM_METRICS[0]]
    losses: Dict[str, Tensor] = {}
    for name in SUPRIM_METRICS:
        target = metric_targets[name].to(
            device=reference.device, dtype=reference.dtype
        )
        if target.shape != metric_logits[name].shape:
            raise ValueError(
                f"DriveSuprim target {name} shape {tuple(target.shape)} does not "
                f"match logits {tuple(metric_logits[name].shape)}"
            )
        if name in {"no_at_fault_collisions", "driving_direction_compliance"}:
            target = _three_to_two_classes(target)
        losses[name] = weights[name] * F.binary_cross_entropy_with_logits(
            metric_logits[name], target
        )
    return torch.stack(tuple(losses.values())).sum(), losses


class DriveSuprimMetricLoss(nn.Module):
    """Coarse plus independently supervised intermediate refinement losses."""

    def __init__(
        self,
        sigma: float = 0.5,
        imitation_weight: float = 1.0,
        pdm_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.imitation_weight = imitation_weight
        self.pdm_weights = dict(pdm_weights or SUPRIM_PDM_LOSS_WEIGHTS)
        if set(self.pdm_weights) != set(SUPRIM_METRICS):
            raise ValueError("PDM weights must cover all eight DriveSuprim metrics")
        self.codec = TrajectoryCodec()

    def one_layer(
        self,
        metric_logits: Mapping[str, Tensor],
        metric_targets: Mapping[str, Tensor],
        candidate_trajectories_40: Tensor,
        gt_trajectory_8: Tensor,
        *,
        use_imitation: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        total, components = _suprim_pdm_loss(
            metric_logits, metric_targets, self.pdm_weights
        )
        components = dict(components)
        if use_imitation:
            if "imi" not in metric_logits:
                raise KeyError("DriveSuprim logits are missing 'imi'")
            imitation = self.imitation_weight * imitation_distribution_loss(
                metric_logits["imi"],
                candidate_trajectories_40,
                gt_trajectory_8,
                sigma=self.sigma,
                codec=self.codec,
            )
            components["imi"] = imitation
            total = total + imitation
        return total, components

    def refinement(
        self,
        layer_logits: Sequence[Mapping[str, Tensor]],
        metric_targets: Mapping[str, Tensor],
        candidate_trajectories_40: Tensor,
        gt_trajectory_8: Tensor,
        *,
        use_imitation: bool = True,
        layer_weights: Sequence[float] | None = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if not layer_logits:
            raise ValueError("fine refinement requires at least one layer output")
        weights = list(layer_weights or [1.0] * len(layer_logits))
        if len(weights) != len(layer_logits):
            raise ValueError("refinement layer_weights length differs from layer count")
        total = gt_trajectory_8.new_zeros(())
        details: Dict[str, Tensor] = {}
        for index, (logits, weight) in enumerate(zip(layer_logits, weights)):
            layer_loss, _ = self.one_layer(
                logits,
                metric_targets,
                candidate_trajectories_40,
                gt_trajectory_8,
                use_imitation=use_imitation,
            )
            details[f"layer_{index}"] = layer_loss
            total = total + float(weight) * layer_loss
        return total, details


def gather_metric_targets(
    targets: Mapping[str, Tensor], indices: Tensor
) -> Dict[str, Tensor]:
    """Gather aligned ``[B,N]`` named targets using ``[B,K]`` indices."""

    if indices.ndim != 2 or indices.dtype != torch.long:
        raise TypeError("indices must be a [B,K] long tensor")
    gathered: Dict[str, Tensor] = {}
    for name, target in targets.items():
        if target.ndim != 2 or target.shape[0] != indices.shape[0]:
            raise ValueError(f"metric target {name} must have shape [B,N]")
        gathered[name] = torch.gather(target, 1, indices)
    return gathered
