"""Full-list, top-aware objectives for Direct Scorer V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


FULL_CANDIDATE_COUNT = 256
BCE_FACTOR_INDICES = (0, 1, 2, 4, 5)
OBJECTIVES = ("O0", "O1", "O2", "O3")


@dataclass(frozen=True)
class TopAwareLossConfig:
    objective: str = "O3"
    factor_weight: float = 0.5
    score_weight: float = 0.5
    listwise_weight: float = 1.0
    top_pair_weight: float = 0.5
    safety_weight: float = 0.25
    target_temperature: float = 0.05
    prediction_temperature: float = 1.0

    def validate(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective: {self.objective}")
        values = (
            self.factor_weight,
            self.score_weight,
            self.listwise_weight,
            self.top_pair_weight,
            self.safety_weight,
        )
        if any(value < 0 for value in values):
            raise ValueError("loss weights must be non-negative")
        if self.target_temperature <= 0 or self.prediction_temperature <= 0:
            raise ValueError("ListNet temperatures must be positive")

    def active_weights(self) -> Mapping[str, float]:
        self.validate()
        if self.objective == "O0":
            return {
                "factor": 1.0,
                "score": 0.0,
                "listwise": 0.0,
                "top_pair": 0.25,
                "safety": 0.0,
            }
        return {
            "factor": self.factor_weight,
            "score": self.score_weight,
            "listwise": self.listwise_weight if self.objective in {"O2", "O3"} else 0.0,
            "top_pair": self.top_pair_weight if self.objective == "O3" else 0.0,
            "safety": self.safety_weight if self.objective == "O3" else 0.0,
        }

    def as_dict(self) -> Mapping[str, object]:
        return asdict(self)


def hard_safe_targets(factor_labels: Tensor) -> Tensor:
    """NC, DAC, DDC and TTC must all be positive; DDC=0.5 is safe."""

    if factor_labels.ndim != 3 or factor_labels.shape[-1] != 6:
        raise ValueError("six-factor labels must be [B,K,6]")
    required = factor_labels[..., (0, 1, 2, 4)]
    return (required > 0.0).all(dim=-1).to(factor_labels.dtype)


def factor_loss(factor_logits: Tensor, factor_labels: Tensor) -> Tensor:
    if factor_logits.shape != factor_labels.shape or factor_logits.shape[-1] != 6:
        raise ValueError("factor logits and labels must share [B,K,6]")
    indices = torch.as_tensor(BCE_FACTOR_INDICES, device=factor_logits.device)
    bce = F.binary_cross_entropy_with_logits(
        factor_logits.index_select(-1, indices),
        factor_labels.index_select(-1, indices),
    )
    ep = F.smooth_l1_loss(factor_logits[..., 3].sigmoid(), factor_labels[..., 3])
    return bce + ep


def full_listnet_loss(
    utility_logit: Tensor,
    true_score: Tensor,
    *,
    target_temperature: float,
    prediction_temperature: float,
) -> Tensor:
    """ListNet over all fixed 256 candidates; subsets are rejected."""

    if utility_logit.shape != true_score.shape or utility_logit.ndim != 2:
        raise ValueError("utility logits and true scores must share [B,K]")
    if utility_logit.shape[1] != FULL_CANDIDATE_COUNT:
        raise ValueError(
            f"ListNet requires all {FULL_CANDIDATE_COUNT} candidates, "
            f"got {utility_logit.shape[1]}"
        )
    target_probability = F.softmax(true_score / float(target_temperature), dim=-1)
    predicted_log_probability = F.log_softmax(
        utility_logit / float(prediction_temperature), dim=-1
    )
    return -(target_probability * predicted_log_probability).sum(dim=-1).mean()


def _dcg_discount(rank: Tensor) -> Tensor:
    return 1.0 / torch.log2(rank.to(torch.float32) + 2.0)


def build_top_pair_schedule(
    true_score: Tensor,
    factor_labels: Tensor,
) -> Mapping[str, Tensor]:
    """Build deterministic top-vs-mid, top-vs-unsafe and near-top pairs.

    The schedule depends only on the complete labels, so every representation
    receives exactly the same pairs.
    """

    if true_score.ndim != 2 or true_score.shape[1] != FULL_CANDIDATE_COUNT:
        raise ValueError("top-pair schedule requires true_score [B,256]")
    if factor_labels.shape != true_score.shape + (6,):
        raise ValueError("top-pair schedule requires factor_labels [B,256,6]")
    safe = hard_safe_targets(factor_labels).to(torch.bool)
    rows: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    max_pairs = 0
    for batch_index in range(true_score.shape[0]):
        scores = true_score[batch_index]
        order = torch.argsort(scores, descending=True, stable=True)
        inverse_rank = torch.empty_like(order)
        inverse_rank[order] = torch.arange(
            FULL_CANDIDATE_COUNT, device=order.device, dtype=order.dtype
        )
        top = order[:32]
        mid = order[32:96]

        top_left = top[:, None].expand(32, 64).reshape(-1)
        mid_right = mid[None, :].expand(32, 64).reshape(-1)
        pair_left = [top_left]
        pair_right = [mid_right]
        pair_type = [torch.zeros_like(top_left)]

        unsafe = torch.nonzero(~safe[batch_index], as_tuple=False).flatten()
        unsafe = unsafe[~torch.isin(unsafe, top)]
        if len(unsafe):
            # Every top candidate is compared against up to 32 unsafe entries,
            # chosen deterministically by their true rank.
            unsafe = unsafe[torch.argsort(inverse_rank[unsafe], stable=True)][:32]
            unsafe_left = top[:, None].expand(32, len(unsafe)).reshape(-1)
            unsafe_right = unsafe[None, :].expand(32, len(unsafe)).reshape(-1)
            pair_left.append(unsafe_left)
            pair_right.append(unsafe_right)
            pair_type.append(torch.ones_like(unsafe_left))

        adjacent_left = top[:-1]
        adjacent_right = top[1:]
        adjacent_gap = scores[adjacent_left] - scores[adjacent_right]
        near = adjacent_gap <= 0.05
        if not near.any():
            near = torch.ones_like(adjacent_gap, dtype=torch.bool)
        pair_left.append(adjacent_left[near])
        pair_right.append(adjacent_right[near])
        pair_type.append(torch.full_like(adjacent_left[near], 2))

        left = torch.cat(pair_left)
        right = torch.cat(pair_right)
        types = torch.cat(pair_type)
        gaps = scores[left] - scores[right]
        non_tie = gaps > 1.0e-8
        left = left[non_tie]
        right = right[non_tie]
        types = types[non_tie]
        gaps = gaps[non_tie]
        rank_weight = _dcg_discount(inverse_rank[left]) + _dcg_discount(
            inverse_rank[right]
        )
        weights = gaps.clamp_min(0.01) * rank_weight
        weights = weights * torch.where(types == 1, 1.5, 1.0)
        rows.append((left, right, weights, types))
        max_pairs = max(max_pairs, len(left))

    left_batch = torch.zeros(
        len(rows), max_pairs, dtype=torch.long, device=true_score.device
    )
    right_batch = torch.zeros_like(left_batch)
    weight_batch = torch.zeros(
        len(rows), max_pairs, dtype=true_score.dtype, device=true_score.device
    )
    type_batch = torch.full_like(left_batch, -1)
    mask_batch = torch.zeros(
        len(rows), max_pairs, dtype=torch.bool, device=true_score.device
    )
    for index, (left, right, weights, types) in enumerate(rows):
        count = len(left)
        left_batch[index, :count] = left
        right_batch[index, :count] = right
        weight_batch[index, :count] = weights
        type_batch[index, :count] = types
        mask_batch[index, :count] = True
    return {
        "left": left_batch,
        "right": right_batch,
        "weight": weight_batch,
        "type": type_batch,
        "mask": mask_batch,
    }


def top_heavy_pairwise_loss(
    utility_logit: Tensor,
    schedule: Mapping[str, Tensor],
) -> Tensor:
    """Rank with unbounded utility-logit differences, never factor products."""

    if utility_logit.ndim != 2 or utility_logit.shape[1] != FULL_CANDIDATE_COUNT:
        raise ValueError("top-heavy pairwise loss requires utility_logit [B,256]")
    left = schedule["left"]
    right = schedule["right"]
    mask = schedule["mask"].to(torch.bool)
    weights = schedule["weight"]
    if left.shape != right.shape or left.shape != mask.shape or left.shape != weights.shape:
        raise ValueError("invalid top-pair schedule shapes")
    delta = utility_logit.gather(1, left) - utility_logit.gather(1, right)
    losses = F.softplus(-delta)
    normalizer = weights[mask].sum().clamp_min(1.0e-12)
    return (losses[mask] * weights[mask]).sum() / normalizer


def legacy_bounded_pairwise_loss(
    factor_score: Tensor,
    true_score: Tensor,
    schedule: Mapping[str, Tensor],
) -> Tensor:
    """O0-only reproduction of the obsolete bounded pairwise formulation."""

    if factor_score.shape != true_score.shape:
        raise ValueError("legacy factor/true score shapes differ")
    left = schedule["left"]
    right = schedule["right"]
    mask = schedule["mask"].to(torch.bool)
    predicted_delta = factor_score.gather(1, left) - factor_score.gather(1, right)
    return F.binary_cross_entropy_with_logits(
        predicted_delta[mask], torch.ones_like(predicted_delta[mask])
    )


def top_aware_direct_loss(
    outputs: Mapping[str, Tensor],
    factor_labels: Tensor,
    true_score: Tensor,
    config: TopAwareLossConfig,
) -> Mapping[str, Tensor]:
    """Compute O0--O3 while enforcing complete fixed-bank supervision."""

    config.validate()
    required = {
        "factor_logits",
        "factor_score",
        "utility_logit",
        "utility_score",
        "hard_safety_logit",
    }
    missing = required - set(outputs)
    if missing:
        raise ValueError(f"V3 outputs missing objective fields: {sorted(missing)}")
    if true_score.ndim != 2 or true_score.shape[1] != FULL_CANDIDATE_COUNT:
        raise ValueError("V3 training always requires all 256 candidates")
    if factor_labels.shape != true_score.shape + (6,):
        raise ValueError("factor labels must be [B,256,6]")
    if outputs["utility_logit"].shape != true_score.shape:
        raise ValueError("utility logits must cover the complete candidate list")

    factor = factor_loss(outputs["factor_logits"], factor_labels)
    score = F.smooth_l1_loss(outputs["utility_score"], true_score)
    schedule = build_top_pair_schedule(true_score, factor_labels)
    zero = outputs["utility_logit"].sum() * 0.0
    if config.objective == "O0":
        pair = legacy_bounded_pairwise_loss(
            outputs["factor_score"], true_score, schedule
        )
        listwise = zero
        safety = zero
    else:
        listwise = (
            full_listnet_loss(
                outputs["utility_logit"],
                true_score,
                target_temperature=config.target_temperature,
                prediction_temperature=config.prediction_temperature,
            )
            if config.objective in {"O2", "O3"}
            else zero
        )
        pair = (
            top_heavy_pairwise_loss(outputs["utility_logit"], schedule)
            if config.objective == "O3"
            else zero
        )
        safety = (
            F.binary_cross_entropy_with_logits(
                outputs["hard_safety_logit"], hard_safe_targets(factor_labels)
            )
            if config.objective == "O3"
            else zero
        )
    weights = config.active_weights()
    total = (
        weights["factor"] * factor
        + weights["score"] * score
        + weights["listwise"] * listwise
        + weights["top_pair"] * pair
        + weights["safety"] * safety
    )
    return {
        "total": total,
        "factor": factor,
        "score": score,
        "listwise": listwise,
        "top_pair": pair,
        "safety": safety,
    }
