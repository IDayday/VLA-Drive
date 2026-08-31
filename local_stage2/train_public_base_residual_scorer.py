"""Train a lightweight fine-ranker on frozen public-Base inference tensors."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Must be set before CUDA creates a cuBLAS handle.  The scorer experiments are
# small enough that reproducibility is more valuable than fast SDP kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from local_stage2.public_base_residual_scorer import (
    FACTOR_KEYS,
    PublicBaseResidualRanker,
    ResidualScorerConfig,
    build_residual_artifact,
)


LABEL_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


@dataclass
class CacheTensorSet:
    tokens: List[str]
    log_names: List[str]
    proposals: torch.Tensor
    base_scores: torch.Tensor
    factor_logits: torch.Tensor
    candidate_features: torch.Tensor
    scene_features: torch.Tensor
    ego_features: torch.Tensor
    target_factors: torch.Tensor

    def __len__(self) -> int:
        return len(self.tokens)


def _atomic_json_dump(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_official_logs(repo_root: Path) -> Tuple[set[str], set[str]]:
    path = (
        repo_root
        / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
    )
    config = OmegaConf.load(path)
    train_logs = {str(value) for value in config.train_logs}
    val_logs = {str(value) for value in config.val_logs}
    overlap = train_logs.intersection(val_logs)
    if overlap:
        raise RuntimeError(f"Official train/validation logs overlap: {sorted(overlap)[:5]}")
    return train_logs, val_logs


def _source_complete(source_root: Path) -> bool:
    shard_dirs = sorted(source_root.glob("*_shard_*-of-*"))
    return bool(shard_dirs) and all(
        (directory / "manifest.json").is_file() for directory in shard_dirs
    )


def load_tensor_cache(
    source_root: Path,
    label_root: Path,
    allowed_logs: set[str],
    *,
    max_scenes: int = 0,
) -> CacheTensorSet:
    tokens: List[str] = []
    log_names: List[str] = []
    proposals: List[torch.Tensor] = []
    base_scores: List[torch.Tensor] = []
    factor_logits: List[torch.Tensor] = []
    candidate_features: List[torch.Tensor] = []
    scene_features: List[torch.Tensor] = []
    ego_features: List[torch.Tensor] = []
    target_factors: List[torch.Tensor] = []

    for source_path in sorted(source_root.glob("*_shard_*-of-*/chunk_*.pt")):
        label_path = label_root / source_path.relative_to(source_root)
        if not label_path.is_file():
            continue
        source = torch.load(source_path, map_location="cpu")
        labels = torch.load(label_path, map_location="cpu")
        source_tokens = [str(value) for value in source["tokens"]]
        label_tokens = [str(value) for value in labels["tokens"]]
        if source_tokens != label_tokens:
            raise RuntimeError(f"Source/label token order mismatch: {source_path}")
        if tuple(source["factor_keys"]) != FACTOR_KEYS:
            raise RuntimeError(f"Unexpected source factor schema: {source_path}")
        if tuple(labels["target_factor_keys"]) != LABEL_FACTOR_KEYS:
            raise RuntimeError(f"Unexpected target factor schema: {label_path}")
        keep = torch.tensor(
            [str(log_name) in allowed_logs for log_name in source["log_names"]],
            dtype=torch.bool,
        )
        if not bool(keep.any()):
            continue
        kept_indices = keep.nonzero(as_tuple=False).flatten()
        tokens.extend(source_tokens[index] for index in kept_indices.tolist())
        log_names.extend(
            str(source["log_names"][index]) for index in kept_indices.tolist()
        )
        proposals.append(source["proposals"][keep].float())
        base_scores.append(source["base_scores"][keep].float())
        factor_logits.append(source["factor_logits"][keep].float())
        candidate_features.append(source["candidate_features"][keep])
        scene_features.append(source["scene_features"][keep])
        ego_features.append(source["ego_features"][keep])
        target_factors.append(labels["target_factors"][keep].float())
        if max_scenes > 0 and len(tokens) >= max_scenes:
            break

    if not tokens:
        raise RuntimeError("No joined source/label scenes matched the requested logs")
    result = CacheTensorSet(
        tokens=tokens,
        log_names=log_names,
        proposals=torch.cat(proposals)[: max_scenes or None],
        base_scores=torch.cat(base_scores)[: max_scenes or None],
        factor_logits=torch.cat(factor_logits)[: max_scenes or None],
        candidate_features=torch.cat(candidate_features)[: max_scenes or None],
        scene_features=torch.cat(scene_features)[: max_scenes or None],
        ego_features=torch.cat(ego_features)[: max_scenes or None],
        target_factors=torch.cat(target_factors)[: max_scenes or None],
    )
    if max_scenes > 0:
        result.tokens = result.tokens[:max_scenes]
        result.log_names = result.log_names[:max_scenes]
    expected = len(result)
    for name in (
        "proposals",
        "base_scores",
        "factor_logits",
        "candidate_features",
        "scene_features",
        "ego_features",
        "target_factors",
    ):
        if len(getattr(result, name)) != expected:
            raise RuntimeError(f"Joined tensor length mismatch for {name}")
    if result.target_factors.shape[1:] != (64, 7):
        raise RuntimeError(f"Unexpected target shape {result.target_factors.shape}")
    return result


def _gather_candidates(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    while view.ndim < value.ndim:
        view = view.unsqueeze(-1)
    return value.gather(1, view.expand(-1, -1, *value.shape[2:]))


def weighted_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_delta: float,
) -> torch.Tensor:
    count = prediction.shape[1]
    left, right = torch.triu_indices(count, count, offset=1, device=prediction.device)
    target_delta = target[:, left] - target[:, right]
    valid = target_delta.abs() >= minimum_delta
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    prediction_delta = prediction[:, left] - prediction[:, right]
    weight = target_delta.abs()
    loss = F.softplus(-target_delta.sign() * prediction_delta)
    return (loss[valid] * weight[valid]).sum() / weight[valid].sum()


def listwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_temperature: float,
) -> torch.Tensor:
    target_distribution = F.softmax(target / target_temperature, dim=-1)
    return -(
        target_distribution * F.log_softmax(prediction, dim=-1)
    ).sum(dim=-1).mean()


def top_set_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: float,
    prediction_temperature: float,
) -> torch.Tensor:
    """Top-focused CE with all near-tied oracle candidates as positives."""

    best = target.max(dim=-1, keepdim=True).values
    positives = target >= best - tolerance
    target_distribution = positives.float() / positives.sum(dim=-1, keepdim=True)
    log_distribution = F.log_softmax(
        prediction / prediction_temperature, dim=-1
    )
    return -(target_distribution * log_distribution).sum(dim=-1).mean()


def expected_regret_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prediction_temperature: float,
) -> torch.Tensor:
    probabilities = F.softmax(prediction / prediction_temperature, dim=-1)
    expected = (probabilities * target).sum(dim=-1)
    return (target.max(dim=-1).values - expected).mean()


def binary_factor_loss(
    predicted_factors: torch.Tensor,
    target_six: torch.Tensor,
    safety_negative_weight: float,
) -> torch.Tensor:
    """EpisodeDrive-compatible binary targets with rare-safety weighting."""

    if predicted_factors.shape != target_six.shape:
        raise ValueError("predicted and target factor shapes must match")
    if safety_negative_weight <= 0:
        raise ValueError("safety_negative_weight must be positive")
    binary_target = target_six.clone()
    # Match EpisodeDriveLoss.three_to_two_classes: partial NOC/DDC credit is a
    # failure for the corresponding binary compliance classifier.
    binary_target[..., 0] = (binary_target[..., 0] == 1.0).to(
        binary_target.dtype
    )
    binary_target[..., 2] = (binary_target[..., 2] == 1.0).to(
        binary_target.dtype
    )
    binary_indices = torch.tensor(
        [0, 1, 2, 3, 5], device=target_six.device
    )
    selected_target = binary_target.index_select(-1, binary_indices)
    element = F.binary_cross_entropy_with_logits(
        predicted_factors.index_select(-1, binary_indices),
        selected_target,
        reduction="none",
    )
    weight = torch.ones_like(element)
    # In the selected binary order these are NOC, DAC and TTC. Violations are
    # rare but determine whether a high-progress proposal is deployably safe.
    for index in (0, 1, 3):
        weight[..., index] = torch.where(
            selected_target[..., index] < 0.5,
            safety_negative_weight,
            1.0,
        )
    return (element * weight).sum() / weight.sum()


def compute_training_loss(
    model: PublicBaseResidualRanker,
    batch: Sequence[torch.Tensor],
    *,
    minimum_pair_delta: float,
    target_temperature: float,
    pairwise_weight: float,
    listwise_weight: float,
    factor_weight: float,
    residual_l2_weight: float,
    top_set_weight: float,
    expected_regret_weight: float,
    top_set_tolerance: float,
    prediction_temperature: float,
    safety_negative_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        proposals,
        base_scores,
        factor_logits,
        candidate_features,
        scene_features,
        ego_features,
        target_factors,
    ) = batch
    output = model(
        candidate_features,
        proposals,
        factor_logits,
        base_scores,
        scene_features,
        ego_features,
    )
    top_indices = base_scores.topk(min(model.config.top_k, base_scores.shape[1]), dim=1).indices
    predicted = _gather_candidates(output["refined_scores"], top_indices)
    target_scores = _gather_candidates(target_factors[..., -1], top_indices)
    pairwise = weighted_pairwise_loss(predicted, target_scores, minimum_pair_delta)
    listwise = listwise_loss(predicted, target_scores, target_temperature)
    top_set = top_set_cross_entropy(
        predicted,
        target_scores,
        tolerance=top_set_tolerance,
        prediction_temperature=prediction_temperature,
    )
    expected_regret = expected_regret_loss(
        predicted,
        target_scores,
        prediction_temperature=prediction_temperature,
    )

    # Target order -> source/refined-logit order.
    reorder = torch.tensor([0, 1, 5, 3, 2, 4], device=target_factors.device)
    target_six = target_factors.index_select(-1, reorder)
    predicted_factors = output["refined_factor_logits"]
    binary = binary_factor_loss(
        predicted_factors,
        target_six,
        safety_negative_weight,
    )
    progress = F.smooth_l1_loss(predicted_factors[..., 4].sigmoid(), target_six[..., 4])
    factor = binary + 2.0 * progress
    residual_l2 = output["residual"].square().mean()
    total = (
        pairwise_weight * pairwise
        + listwise_weight * listwise
        + factor_weight * factor
        + residual_l2_weight * residual_l2
        + top_set_weight * top_set
        + expected_regret_weight * expected_regret
    )
    return total, {
        "loss": float(total.detach()),
        "pairwise_loss": float(pairwise.detach()),
        "listwise_loss": float(listwise.detach()),
        "factor_loss": float(factor.detach()),
        "residual_l2": float(residual_l2.detach()),
        "top_set_loss": float(top_set.detach()),
        "expected_regret_loss": float(expected_regret.detach()),
    }


def _log_bootstrap_ci(
    values: np.ndarray,
    log_names: Sequence[str],
    seed: int,
    replicates: int = 1000,
) -> Tuple[float, float]:
    grouped: Dict[str, List[float]] = {}
    for value, log_name in zip(values, log_names):
        grouped.setdefault(str(log_name), []).append(float(value))
    ordered_logs = sorted(grouped)
    if not ordered_logs:
        return float("nan"), float("nan")
    # Resample whole logs, then retain the scene-weighted estimand reported by
    # selected_pdms_delta.  Averaging per-log means would silently switch to a
    # different estimand when logs contain different numbers of scenes.
    log_sums = np.asarray([np.sum(grouped[name]) for name in ordered_logs])
    log_counts = np.asarray([len(grouped[name]) for name in ordered_logs])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(ordered_logs),
        size=(replicates, len(ordered_logs)),
    )
    samples = log_sums[sampled].sum(axis=1) / log_counts[sampled].sum(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


@dataclass
class EvaluationOutputs:
    base_scores: torch.Tensor
    residual: torch.Tensor
    refined_factor_logits: torch.Tensor
    top_k_mask: torch.Tensor
    target_factors: torch.Tensor


@torch.inference_mode()
def collect_evaluation_outputs(
    model: PublicBaseResidualRanker,
    data: CacheTensorSet,
    device: torch.device,
    batch_size: int,
) -> EvaluationOutputs:
    model.eval()
    base_scores_output: List[torch.Tensor] = []
    residual_output: List[torch.Tensor] = []
    refined_factor_output: List[torch.Tensor] = []
    top_k_output: List[torch.Tensor] = []
    target_output: List[torch.Tensor] = []
    dataset = TensorDataset(
        data.proposals,
        data.base_scores,
        data.factor_logits,
        data.candidate_features,
        data.scene_features,
        data.ego_features,
        data.target_factors,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        (
            proposals,
            base_scores,
            factor_logits,
            candidate_features,
            scene_features,
            ego_features,
            target_factors,
        ) = [value.to(device, non_blocking=True) for value in batch]
        output = model(
            candidate_features,
            proposals,
            factor_logits,
            base_scores,
            scene_features,
            ego_features,
        )
        base_scores_output.append(base_scores.float().cpu())
        residual_output.append(output["residual"].float().cpu())
        refined_factor_output.append(
            output["refined_factor_logits"].float().cpu()
        )
        top_k_output.append(output["top_k_mask"].cpu())
        target_output.append(target_factors.float().cpu())
    return EvaluationOutputs(
        base_scores=torch.cat(base_scores_output),
        residual=torch.cat(residual_output),
        refined_factor_logits=torch.cat(refined_factor_output),
        top_k_mask=torch.cat(top_k_output),
        target_factors=torch.cat(target_output),
    )


def evaluate_collected(
    outputs: EvaluationOutputs,
    log_names: Sequence[str],
    seed: int,
    *,
    residual_scale: float,
    switch_penalty: float,
    safety_floor: float,
    safety_relative_tolerance: float,
    preserve_ddc: bool,
) -> Dict[str, object]:
    base_scores = outputs.base_scores
    target_factors = outputs.target_factors
    target_scores = target_factors[..., -1]
    base_index = base_scores.argmax(dim=1)
    row_index = torch.arange(len(base_index))
    base_selected_mask = torch.zeros_like(base_scores, dtype=torch.bool)
    base_selected_mask.scatter_(1, base_index[:, None], True)
    refined_scores = base_scores + residual_scale * outputs.residual
    refined_scores = refined_scores - (
        (~base_selected_mask).to(refined_scores.dtype) * switch_penalty
    )

    safety_indices = [0, 1, 3]
    if preserve_ddc:
        safety_indices.append(2)
    probabilities = outputs.refined_factor_logits.sigmoid()[..., safety_indices]
    base_probabilities = probabilities.gather(
        1,
        base_index[:, None, None].expand(-1, 1, len(safety_indices)),
    )
    safety_mask = (probabilities >= safety_floor).all(dim=-1)
    safety_mask &= (
        probabilities >= base_probabilities - safety_relative_tolerance
    ).all(dim=-1)
    eligible = outputs.top_k_mask & safety_mask
    eligible |= base_selected_mask
    selection_scores = torch.where(eligible, refined_scores, base_scores - 100.0)
    model_index = selection_scores.argmax(dim=1)
    oracle_index = target_scores.argmax(dim=1)

    base_values = target_scores[row_index, base_index]
    model_values_tensor = target_scores[row_index, model_index]
    oracle_values = target_scores[row_index, oracle_index]
    top_target = target_scores.masked_fill(~outputs.top_k_mask, -1.0)
    topk_values = top_target.max(dim=1).values
    factor_width = target_factors.shape[-1]
    base_factors_tensor = target_factors.gather(
        1, base_index[:, None, None].expand(-1, 1, factor_width)
    ).squeeze(1)
    model_factors_tensor = target_factors.gather(
        1, model_index[:, None, None].expand(-1, 1, factor_width)
    ).squeeze(1)
    oracle_factors_tensor = target_factors.gather(
        1, oracle_index[:, None, None].expand(-1, 1, factor_width)
    ).squeeze(1)

    # Pairwise accuracy remains a diagnostic over all frozen Base top-K
    # candidates; final selection can additionally apply conservative gates.
    top_k = int(outputs.top_k_mask.sum(dim=1).min())
    ordered_indices = base_scores.topk(top_k, dim=1).indices
    top_prediction = _gather_candidates(refined_scores, ordered_indices)
    top_targets = _gather_candidates(target_scores, ordered_indices)
    left, right = torch.triu_indices(top_k, top_k, offset=1)
    target_delta = top_targets[:, left] - top_targets[:, right]
    valid_pairs = target_delta.abs() >= 0.02
    prediction_delta = top_prediction[:, left] - top_prediction[:, right]
    pair_correct = int(
        ((prediction_delta.sign() == target_delta.sign()) & valid_pairs).sum()
    )
    pair_total = int(valid_pairs.sum())

    base = base_values.numpy()
    model_values = model_values_tensor.numpy()
    oracle = oracle_values.numpy()
    topk = topk_values.numpy()
    base_factors = base_factors_tensor.numpy()
    model_factors = model_factors_tensor.numpy()
    oracle_factors = oracle_factors_tensor.numpy()
    switch_mask = (base_index != model_index).numpy()
    delta = model_values - base
    ci = _log_bootstrap_ci(delta, log_names, seed)
    base_factor_means = {
        key: float(base_factors[:, index].mean())
        for index, key in enumerate(LABEL_FACTOR_KEYS)
    }
    model_factor_means = {
        key: float(model_factors[:, index].mean())
        for index, key in enumerate(LABEL_FACTOR_KEYS)
    }
    oracle_factor_means = {
        key: float(oracle_factors[:, index].mean())
        for index, key in enumerate(LABEL_FACTOR_KEYS)
    }
    return {
        "scene_count": len(log_names),
        "log_count": len(set(log_names)),
        "base_selected_pdms": float(base.mean()),
        "model_selected_pdms": float(model_values.mean()),
        "selected_pdms_delta": float(delta.mean()),
        "selected_pdms_delta_log_bootstrap_95ci": list(ci),
        "best_of_topk_pdms": float(topk.mean()),
        "best_of_64_pdms": float(oracle.mean()),
        "base_top1_regret": float((oracle - base).mean()),
        "model_top1_regret": float((oracle - model_values).mean()),
        "regret_reduction_fraction": float(
            1.0 - (oracle - model_values).mean() / max((oracle - base).mean(), 1e-12)
        ),
        "pairwise_accuracy_delta_ge_0_02": float(pair_correct / max(pair_total, 1)),
        "pair_count_delta_ge_0_02": pair_total,
        "residual_scale": float(
            residual_scale
        ),
        "switch_penalty": float(switch_penalty),
        "safety_floor": float(safety_floor),
        "safety_relative_tolerance": float(safety_relative_tolerance),
        "preserve_ddc": bool(preserve_ddc),
        "selection_switch_rate": float(switch_mask.mean()),
        "improved_scene_count_delta_gt_0_01": int((delta > 0.01).sum()),
        "degraded_scene_count_delta_lt_minus_0_01": int((delta < -0.01).sum()),
        "base_selected_factors": base_factor_means,
        "model_selected_factors": model_factor_means,
        "oracle_selected_factors": oracle_factor_means,
        "selected_factor_delta": {
            key: model_factor_means[key] - base_factor_means[key]
            for key in LABEL_FACTOR_KEYS
        },
    }


def evaluate(
    model: PublicBaseResidualRanker,
    data: CacheTensorSet,
    device: torch.device,
    batch_size: int,
    seed: int,
    residual_scale: Optional[float] = None,
    switch_penalty: Optional[float] = None,
    safety_floor: Optional[float] = None,
    safety_relative_tolerance: Optional[float] = None,
    preserve_ddc: Optional[bool] = None,
) -> Dict[str, object]:
    outputs = collect_evaluation_outputs(model, data, device, batch_size)
    return evaluate_collected(
        outputs,
        data.log_names,
        seed,
        residual_scale=(
            model.config.inference_scale if residual_scale is None else residual_scale
        ),
        switch_penalty=(
            model.config.switch_penalty if switch_penalty is None else switch_penalty
        ),
        safety_floor=(
            model.config.safety_floor if safety_floor is None else safety_floor
        ),
        safety_relative_tolerance=(
            model.config.safety_relative_tolerance
            if safety_relative_tolerance is None
            else safety_relative_tolerance
        ),
        preserve_ddc=(
            model.config.preserve_ddc if preserve_ddc is None else preserve_ddc
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "local",
            "set_aware",
            "scene_cross_attention",
            "scene_cross_attention_set",
        ),
        default="local",
    )
    parser.add_argument(
        "--score-mode",
        choices=("residual", "factor_aggregate", "hybrid"),
        default="residual",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-pair-delta", type=float, default=0.02)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.2)
    parser.add_argument("--factor-weight", type=float, default=0.5)
    parser.add_argument("--safety-negative-weight", type=float, default=10.0)
    parser.add_argument("--residual-l2-weight", type=float, default=0.01)
    parser.add_argument("--top-set-weight", type=float, default=0.5)
    parser.add_argument("--expected-regret-weight", type=float, default=2.0)
    parser.add_argument("--top-set-tolerance", type=float, default=0.01)
    parser.add_argument("--prediction-temperature", type=float, default=0.05)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument("--require-complete-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.repo_root, args.source_root, args.label_root, args.base_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.require_complete_cache and not _source_complete(args.source_root):
        raise RuntimeError("Source cache is incomplete")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.safety_negative_weight <= 0:
        raise ValueError("safety-negative-weight must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    train_logs, val_logs = _load_official_logs(args.repo_root)
    train_data = load_tensor_cache(
        args.source_root,
        args.label_root,
        train_logs,
        max_scenes=args.max_train_scenes,
    )
    val_data = load_tensor_cache(
        args.source_root,
        args.label_root,
        val_logs,
        max_scenes=args.max_val_scenes,
    )
    if set(train_data.log_names).intersection(val_data.log_names):
        raise RuntimeError("Train and validation cache logs overlap")

    model = PublicBaseResidualRanker(
        ResidualScorerConfig(
            mode=args.mode,
            score_mode=args.score_mode,
            top_k=args.top_k,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    train_dataset = TensorDataset(
        train_data.proposals,
        train_data.base_scores,
        train_data.factor_logits,
        train_data.candidate_features,
        train_data.scene_features,
        train_data.ego_features,
        train_data.target_factors,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    initial_metrics = evaluate(
        model, val_data, device, args.eval_batch_size, args.seed
    )
    history.append({"epoch": -1, "validation": initial_metrics})
    print("RESIDUAL_EVAL " + json.dumps(history[-1], sort_keys=True), flush=True)
    best_metric = float(initial_metrics["model_selected_pdms"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        sums: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            moved = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, details = compute_training_loss(
                model,
                moved,
                minimum_pair_delta=args.minimum_pair_delta,
                target_temperature=args.target_temperature,
                pairwise_weight=args.pairwise_weight,
                listwise_weight=args.listwise_weight,
                factor_weight=args.factor_weight,
                residual_l2_weight=args.residual_l2_weight,
                top_set_weight=args.top_set_weight,
                expected_regret_weight=args.expected_regret_weight,
                top_set_tolerance=args.top_set_tolerance,
                prediction_temperature=args.prediction_temperature,
                safety_negative_weight=args.safety_negative_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in details.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        validation = evaluate(
            model, val_data, device, args.eval_batch_size, args.seed + epoch + 1
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training": {key: value / batches for key, value in sums.items()},
            "validation": validation,
        }
        history.append(record)
        print("RESIDUAL_EVAL " + json.dumps(record, sort_keys=True), flush=True)
        metric = float(validation["model_selected_pdms"])
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state, strict=True)
    validation_outputs = collect_evaluation_outputs(
        model, val_data, device, args.eval_batch_size
    )
    scale_sweep = []
    safety_settings = (
        (0.0, 1.0, False),
        (0.95, 0.10, False),
        (0.95, 0.02, False),
        (0.95, 0.01, False),
        (0.98, 0.10, False),
        (0.95, 0.10, True),
        (0.95, 0.02, True),
    )
    for scale in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for penalty in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02):
            for safety_floor, safety_tolerance, preserve_ddc in safety_settings:
                metrics = evaluate_collected(
                    validation_outputs,
                    val_data.log_names,
                    args.seed + 1000,
                    residual_scale=scale,
                    switch_penalty=penalty,
                    safety_floor=safety_floor,
                    safety_relative_tolerance=safety_tolerance,
                    preserve_ddc=preserve_ddc,
                )
                scale_sweep.append(metrics)
    safety_tolerance = 5e-4
    safe_candidates = [
        value
        for value in scale_sweep
        if value["selected_factor_delta"]["no_at_fault_collisions"]
        >= -safety_tolerance
        and value["selected_factor_delta"]["drivable_area_compliance"]
        >= -safety_tolerance
        and value["selected_factor_delta"]["time_to_collision_within_bound"]
        >= -safety_tolerance
    ]
    best_scale_metrics = max(
        safe_candidates or scale_sweep,
        key=lambda value: (
            float(value["model_selected_pdms"]),
            -float(value["residual_scale"]),
            float(value["switch_penalty"]),
        ),
    )
    deployed_model = PublicBaseResidualRanker(
        replace(
            model.config,
            inference_scale=float(best_scale_metrics["residual_scale"]),
            switch_penalty=float(best_scale_metrics["switch_penalty"]),
            safety_floor=float(best_scale_metrics["safety_floor"]),
            safety_relative_tolerance=float(
                best_scale_metrics["safety_relative_tolerance"]
            ),
            preserve_ddc=bool(best_scale_metrics["preserve_ddc"]),
        )
    ).to(device)
    deployed_model.load_state_dict(model.state_dict(), strict=True)
    final_metrics = evaluate(
        deployed_model, val_data, device, args.eval_batch_size, args.seed + 1000
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "seed": args.seed,
        "source_root": str(args.source_root.resolve()),
        "label_root": str(args.label_root.resolve()),
        "train_scene_count": len(train_data),
        "train_log_count": len(set(train_data.log_names)),
        "val_scene_count": len(val_data),
        "val_log_count": len(set(val_data.log_names)),
        "validation": final_metrics,
        "residual_scale_sweep": scale_sweep,
        "training_args": vars(args),
        "future_inputs_used": False,
        "official_scores_used_at_inference": False,
    }
    metadata["training_args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["training_args"].items()
    }
    artifact = build_residual_artifact(
        deployed_model, args.base_checkpoint, metadata=metadata
    )
    artifact_path = args.output_dir / "best_residual_scorer.pt"
    temporary = artifact_path.with_name(f".{artifact_path.name}.tmp-{os.getpid()}")
    torch.save(artifact, temporary)
    os.replace(temporary, artifact_path)
    results = {
        "schema_version": 1,
        "model_config": asdict(deployed_model.config),
        "best_epoch": best_epoch,
        "initial_validation": initial_metrics,
        "best_validation": final_metrics,
        "residual_scale_sweep": scale_sweep,
        "history": history,
        "artifact_path": str(artifact_path.resolve()),
        "metadata": metadata,
    }
    _atomic_json_dump(results, args.output_dir / "training_results.json")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
