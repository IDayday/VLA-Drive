"""Train the candidate-relative temporal consequence scorer on frozen caches."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from local_stage2.temporal_consequence_scorer import (
    ACTOR_STATE_DIM,
    AREA_KINDS,
    HORIZON_COUNT,
    RISK_KINDS,
    TemporalConsequenceConfig,
    TemporalConsequenceRanker,
    build_temporal_consequence_artifact,
)
from local_stage2.materialize_temporal_cv_policy import POLICY_TO_CONFIG
from local_stage2.train_public_base_residual_scorer import (
    LABEL_FACTOR_KEYS,
    _log_bootstrap_ci,
    base_pairwise_loss,
    binary_factor_loss,
    expected_regret_loss,
    listwise_loss,
    relative_safety_targets,
    top_set_cross_entropy,
    weighted_pairwise_loss,
)


@dataclass
class TemporalCacheTensorSet:
    tokens: List[str]
    log_names: List[str]
    proposals: torch.Tensor
    base_scores: torch.Tensor
    factor_logits: torch.Tensor
    candidate_features: torch.Tensor
    scene_features: torch.Tensor
    ego_features: torch.Tensor
    target_factors: torch.Tensor
    risk_event_by_horizon: torch.Tensor
    ego_area_violation: torch.Tensor
    key_actor_valid: torch.Tensor
    key_actor_state: torch.Tensor

    def __len__(self) -> int:
        return len(self.tokens)

    def subset(self, indices: Sequence[int]) -> "TemporalCacheTensorSet":
        index = torch.as_tensor(indices, dtype=torch.long)
        return TemporalCacheTensorSet(
            tokens=[self.tokens[value] for value in indices],
            log_names=[self.log_names[value] for value in indices],
            proposals=self.proposals[index],
            base_scores=self.base_scores[index],
            factor_logits=self.factor_logits[index],
            candidate_features=self.candidate_features[index],
            scene_features=self.scene_features[index],
            ego_features=self.ego_features[index],
            target_factors=self.target_factors[index],
            risk_event_by_horizon=self.risk_event_by_horizon[index],
            ego_area_violation=self.ego_area_violation[index],
            key_actor_valid=self.key_actor_valid[index],
            key_actor_state=self.key_actor_state[index],
        )


@dataclass
class TemporalEvaluationOutputs:
    base_scores: torch.Tensor
    residual: torch.Tensor
    refined_factor_logits: torch.Tensor
    predicted_safety: torch.Tensor
    top_k_mask: torch.Tensor
    target_factors: torch.Tensor
    risk_logits: torch.Tensor
    risk_targets: torch.Tensor
    area_logits: torch.Tensor
    area_targets: torch.Tensor
    actor_valid_logits: torch.Tensor
    actor_valid_targets: torch.Tensor
    actor_state: torch.Tensor
    actor_state_targets: torch.Tensor


def _atomic_json_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _gather(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    while view.ndim < value.ndim:
        view = view.unsqueeze(-1)
    return value.gather(1, view.expand(-1, -1, *value.shape[2:]))


def _load_temporal_cache(
    source_root: Path,
    factor_root: Path,
    consequence_root: Path,
    *,
    max_scenes: int = 0,
) -> TemporalCacheTensorSet:
    values: Dict[str, List[torch.Tensor]] = {
        key: []
        for key in (
            "proposals",
            "base_scores",
            "factor_logits",
            "candidate_features",
            "scene_features",
            "ego_features",
            "target_factors",
            "risk_event_by_horizon",
            "ego_area_violation",
            "key_actor_valid",
            "key_actor_state",
        )
    }
    tokens: List[str] = []
    log_names: List[str] = []
    parity_max = 0.0
    for source_path in sorted(source_root.glob("*_shard_*-of-*/chunk_*.pt")):
        relative = source_path.relative_to(source_root)
        factor_path = factor_root / relative
        consequence_path = consequence_root / relative
        if not factor_path.is_file() or not consequence_path.is_file():
            continue
        source = torch.load(source_path, map_location="cpu")
        factors = torch.load(factor_path, map_location="cpu")
        consequence = torch.load(consequence_path, map_location="cpu")
        source_tokens = [str(value) for value in source["tokens"]]
        if source_tokens != [str(value) for value in factors["tokens"]]:
            raise RuntimeError(f"Source/factor token mismatch: {relative}")
        if source_tokens != [str(value) for value in consequence["tokens"]]:
            raise RuntimeError(f"Source/consequence token mismatch: {relative}")
        candidate_indices = consequence["candidate_indices"].long()
        base_scores = source["base_scores"].float()
        expected = torch.argsort(base_scores, dim=1, descending=True, stable=True)[
            :, : candidate_indices.shape[1]
        ]
        if not torch.equal(candidate_indices, expected):
            raise RuntimeError(f"Consequence candidate order mismatch: {relative}")

        values["proposals"].append(_gather(source["proposals"].float(), candidate_indices))
        values["base_scores"].append(_gather(base_scores, candidate_indices))
        values["factor_logits"].append(
            _gather(source["factor_logits"].float(), candidate_indices)
        )
        values["candidate_features"].append(
            _gather(source["candidate_features"], candidate_indices)
        )
        values["scene_features"].append(source["scene_features"])
        values["ego_features"].append(source["ego_features"])
        values["target_factors"].append(
            _gather(factors["target_factors"].float(), candidate_indices)
        )
        for key in (
            "risk_event_by_horizon",
            "ego_area_violation",
            "key_actor_valid",
            "key_actor_state",
        ):
            values[key].append(consequence[key])
        tokens.extend(source_tokens)
        log_names.extend(str(value) for value in source["log_names"])
        parity_max = max(parity_max, float(consequence["factor_parity_max_abs_error"]))
        if max_scenes > 0 and len(tokens) >= max_scenes:
            break
    if not tokens:
        raise RuntimeError("No complete source/factor/consequence chunks found")
    limit = max_scenes or None
    result = TemporalCacheTensorSet(
        tokens=tokens[:limit],
        log_names=log_names[:limit],
        **{key: torch.cat(parts)[:limit] for key, parts in values.items()},
    )
    expected_shapes = {
        "proposals": (16, HORIZON_COUNT, 3),
        "base_scores": (16,),
        "factor_logits": (16, 6),
        "candidate_features": (16, 256),
        "scene_features": (16, 256),
        "ego_features": (1, 256),
        "target_factors": (16, 7),
        "risk_event_by_horizon": (16, HORIZON_COUNT, RISK_KINDS),
        "ego_area_violation": (16, HORIZON_COUNT, AREA_KINDS),
        "key_actor_valid": (16, HORIZON_COUNT, RISK_KINDS),
        "key_actor_state": (16, HORIZON_COUNT, RISK_KINDS, ACTOR_STATE_DIM),
    }
    for key, shape in expected_shapes.items():
        tensor = getattr(result, key)
        if tuple(tensor.shape[1:]) != shape or len(tensor) != len(result):
            raise RuntimeError(f"Unexpected joined shape for {key}: {tuple(tensor.shape)}")
    if parity_max > 1e-6:
        raise RuntimeError(f"Consequence/factor parity failed: {parity_max}")
    return result


def assign_balanced_log_folds(
    log_names: Sequence[str],
    num_folds: int,
    seed: int,
) -> Dict[str, int]:
    """Greedily balance scene counts while assigning whole logs to folds."""

    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    counts = Counter(str(value) for value in log_names)
    if len(counts) < num_folds:
        raise ValueError("fewer logs than folds")
    generator = random.Random(seed)
    tie_break = {name: generator.random() for name in sorted(counts)}
    ordered = sorted(counts, key=lambda name: (-counts[name], tie_break[name], name))
    fold_scenes = [0] * num_folds
    fold_logs = [0] * num_folds
    assignment: Dict[str, int] = {}
    for name in ordered:
        fold = min(range(num_folds), key=lambda value: (fold_scenes[value], fold_logs[value], value))
        assignment[name] = fold
        fold_scenes[fold] += counts[name]
        fold_logs[fold] += 1
    return assignment


def load_full_data_cv_policy(path: Path) -> Tuple[int, Dict[str, float], Mapping[str, object]]:
    """Load the epoch and one deployment policy fixed only by complete Navtrain CV."""

    summary = json.loads(path.read_text())
    fold_audit = summary.get("fold_audit", {})
    if not bool(fold_audit.get("complete")):
        raise ValueError("Full-data training requires a complete cross-validation summary")
    if not bool(summary.get("robust_deployment_available")):
        raise ValueError("Full-data training requires a robust common deployment")
    retained_epoch = int(summary["common_epoch"]["epoch"])
    deployment = summary["common_deployment"]
    policy = {
        policy_key: float(deployment[policy_key])
        for policy_key in POLICY_TO_CONFIG
    }
    return retained_epoch, policy, summary


def _weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    targets = targets.to(logits.dtype)
    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weight = torch.where(targets > 0.5, positive_weight, 1.0)
    return (element * weight).sum() / weight.sum()


def compute_temporal_training_loss(
    model: TemporalConsequenceRanker,
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        proposals,
        base_scores,
        factor_logits,
        candidate_features,
        scene_features,
        ego_features,
        target_factors,
        risk_targets,
        area_targets,
        actor_valid_targets,
        actor_state_targets,
    ) = batch
    output = model(
        candidate_features,
        proposals,
        factor_logits,
        base_scores,
        scene_features,
        ego_features,
    )
    prediction = output["refined_scores"]
    target_scores = target_factors[..., -1]
    pairwise = weighted_pairwise_loss(prediction, target_scores, args.minimum_pair_delta)
    base_pairwise = base_pairwise_loss(prediction, target_scores, args.minimum_pair_delta)
    listwise = listwise_loss(prediction, target_scores, args.target_temperature)
    top_set = top_set_cross_entropy(
        prediction,
        target_scores,
        tolerance=args.top_set_tolerance,
        prediction_temperature=args.prediction_temperature,
    )
    expected_regret = expected_regret_loss(
        prediction,
        target_scores,
        prediction_temperature=args.prediction_temperature,
    )

    reorder = torch.tensor([0, 1, 5, 3, 2, 4], device=target_factors.device)
    target_six = target_factors.index_select(-1, reorder)
    factor = binary_factor_loss(
        output["refined_factor_logits"],
        target_six,
        args.safety_negative_weight,
    )
    progress = F.smooth_l1_loss(
        output["refined_factor_logits"][..., 4].sigmoid(),
        target_six[..., 4],
    )
    factor = factor + 2.0 * progress
    if model.config.use_relative_safety_head:
        relative_target = relative_safety_targets(target_six, base_scores)
        relative_element = F.binary_cross_entropy_with_logits(
            output["relative_safety_logits"],
            relative_target,
            reduction="none",
        )
        relative_weight = torch.where(
            relative_target < 0.5,
            args.safety_negative_weight,
            1.0,
        )
        relative_safety = (
            relative_element * relative_weight
        ).sum() / relative_weight.sum().clamp_min(1.0)
    else:
        relative_safety = output["residual"].sum() * 0.0
    risk = _weighted_bce(output["risk_logits"], risk_targets, args.risk_positive_weight)
    area = _weighted_bce(output["area_logits"], area_targets, args.area_positive_weight)
    actor_valid = _weighted_bce(
        output["actor_valid_logits"],
        actor_valid_targets,
        args.actor_positive_weight,
    )
    actor_mask = actor_valid_targets.bool().unsqueeze(-1)
    actor_scale = actor_state_targets.new_tensor((30.0, 15.0, 10.0, 5.0, 1.0, 1.0))
    if bool(actor_mask.any()):
        actor_state = F.smooth_l1_loss(
            output["actor_state"][actor_mask.expand_as(actor_state_targets)]
            / actor_scale.expand_as(actor_state_targets)[actor_mask.expand_as(actor_state_targets)],
            actor_state_targets[actor_mask.expand_as(actor_state_targets)]
            / actor_scale.expand_as(actor_state_targets)[actor_mask.expand_as(actor_state_targets)],
        )
    else:
        actor_state = output["actor_state"].sum() * 0.0
    risk_probability = output["risk_logits"].sigmoid()
    monotonic = F.relu(risk_probability[..., :-1, :] - risk_probability[..., 1:, :]).mean()
    residual_l2 = output["residual"].square().mean()
    total = (
        args.pairwise_weight * pairwise
        + args.base_pairwise_weight * base_pairwise
        + args.listwise_weight * listwise
        + args.top_set_weight * top_set
        + args.expected_regret_weight * expected_regret
        + args.factor_weight * factor
        + args.relative_safety_weight * relative_safety
        + args.risk_weight * risk
        + args.area_weight * area
        + args.actor_valid_weight * actor_valid
        + args.actor_state_weight * actor_state
        + args.monotonic_weight * monotonic
        + args.residual_l2_weight * residual_l2
    )
    details = {
        "loss": total,
        "pairwise_loss": pairwise,
        "base_pairwise_loss": base_pairwise,
        "listwise_loss": listwise,
        "top_set_loss": top_set,
        "expected_regret_loss": expected_regret,
        "factor_loss": factor,
        "relative_safety_loss": relative_safety,
        "risk_loss": risk,
        "area_loss": area,
        "actor_valid_loss": actor_valid,
        "actor_state_loss": actor_state,
        "monotonic_loss": monotonic,
        "residual_l2": residual_l2,
    }
    return total, {key: float(value.detach()) for key, value in details.items()}


def _dataset(data: TemporalCacheTensorSet) -> TensorDataset:
    return TensorDataset(
        data.proposals,
        data.base_scores,
        data.factor_logits,
        data.candidate_features,
        data.scene_features,
        data.ego_features,
        data.target_factors,
        data.risk_event_by_horizon,
        data.ego_area_violation,
        data.key_actor_valid,
        data.key_actor_state,
    )


@torch.inference_mode()
def collect_outputs(
    model: TemporalConsequenceRanker,
    data: TemporalCacheTensorSet,
    device: torch.device,
    batch_size: int,
) -> TemporalEvaluationOutputs:
    model.eval()
    keys = (
        "base_scores",
        "residual",
        "refined_factor_logits",
        "predicted_safety",
        "top_k_mask",
        "target_factors",
        "risk_logits",
        "risk_targets",
        "area_logits",
        "area_targets",
        "actor_valid_logits",
        "actor_valid_targets",
        "actor_state",
        "actor_state_targets",
    )
    collected: Dict[str, List[torch.Tensor]] = {key: [] for key in keys}
    loader = DataLoader(_dataset(data), batch_size=batch_size, shuffle=False)
    for batch in loader:
        moved = [value.to(device, non_blocking=True) for value in batch]
        output = model(
            moved[3],
            moved[0],
            moved[2],
            moved[1],
            moved[4],
            moved[5],
        )
        batch_values = {
            "base_scores": moved[1],
            "residual": output["residual"],
            "refined_factor_logits": output["refined_factor_logits"],
            "predicted_safety": output["predicted_safety"],
            "top_k_mask": output["top_k_mask"],
            "target_factors": moved[6],
            "risk_logits": output["risk_logits"],
            "risk_targets": moved[7],
            "area_logits": output["area_logits"],
            "area_targets": moved[8],
            "actor_valid_logits": output["actor_valid_logits"],
            "actor_valid_targets": moved[9],
            "actor_state": output["actor_state"],
            "actor_state_targets": moved[10],
        }
        for key, value in batch_values.items():
            tensor = value.detach().cpu()
            if key != "top_k_mask":
                tensor = tensor.float()
            collected[key].append(tensor)
    return TemporalEvaluationOutputs(
        **{key: torch.cat(parts) for key, parts in collected.items()}
    )


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool).reshape(-1)
    scores = scores.astype(np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _prediction_diagnostics(
    outputs: TemporalEvaluationOutputs,
) -> Dict[str, object]:
    """Compute consequence diagnostics that are invariant to selection policy."""

    risk_probability = outputs.risk_logits.sigmoid().numpy()
    risk_target = outputs.risk_targets.numpy()
    risk_metrics = {}
    for index, name in enumerate(("collision", "ttc")):
        final_probability = risk_probability[..., -1, index]
        final_target = risk_target[..., -1, index]
        risk_metrics[name] = {
            "auroc": _binary_auc(final_target, final_probability),
            "brier": float(np.mean(np.square(final_probability - final_target))),
            "target_rate": float(final_target.mean()),
            "prediction_mean": float(final_probability.mean()),
        }
    actor_mask = outputs.actor_valid_targets.bool().unsqueeze(-1)
    actor_scale = outputs.actor_state_targets.new_tensor(
        (30.0, 15.0, 10.0, 5.0, 1.0, 1.0)
    )
    if bool(actor_mask.any()):
        actor_error = (
            (outputs.actor_state - outputs.actor_state_targets).abs()
            / actor_scale
        )[actor_mask.expand_as(outputs.actor_state)].mean()
    else:
        actor_error = torch.tensor(float("nan"))
    return {
        "risk_prediction": risk_metrics,
        "area_brier": float(
            torch.square(outputs.area_logits.sigmoid() - outputs.area_targets).mean()
        ),
        "actor_valid_brier": float(
            torch.square(
                outputs.actor_valid_logits.sigmoid() - outputs.actor_valid_targets
            ).mean()
        ),
        "actor_state_normalized_l1": float(actor_error),
    }


def _pairwise_accuracy(
    outputs: TemporalEvaluationOutputs,
    refined_scores: torch.Tensor,
) -> float:
    target_scores = outputs.target_factors[..., -1]
    left, right = torch.triu_indices(
        refined_scores.shape[1], refined_scores.shape[1], offset=1
    )
    target_delta = target_scores[:, left] - target_scores[:, right]
    prediction_delta = refined_scores[:, left] - refined_scores[:, right]
    valid = target_delta.abs() >= 0.02
    return float(
        ((target_delta.sign() == prediction_delta.sign()) & valid).sum().item()
        / max(int(valid.sum()), 1)
    )


def evaluate_outputs(
    outputs: TemporalEvaluationOutputs,
    log_names: Sequence[str],
    *,
    residual_scale: float,
    switch_penalty: float,
    safety_floor: float,
    safety_relative_tolerance: float,
    seed: int,
    bootstrap_replicates: int = 1000,
    prediction_diagnostics: Optional[Mapping[str, object]] = None,
    pairwise_accuracy: Optional[float] = None,
) -> Dict[str, object]:
    base_scores = outputs.base_scores
    target_factors = outputs.target_factors
    target_scores = target_factors[..., -1]
    base_index = base_scores.argmax(dim=1)
    rows = torch.arange(len(base_index))
    base_mask = torch.zeros_like(base_scores, dtype=torch.bool)
    base_mask.scatter_(1, base_index[:, None], True)
    predicted_safety = outputs.predicted_safety
    base_safety = predicted_safety.gather(
        1, base_index[:, None, None].expand(-1, 1, RISK_KINDS)
    )
    safe = (predicted_safety >= safety_floor).all(dim=-1)
    safe &= (
        predicted_safety >= base_safety - safety_relative_tolerance
    ).all(dim=-1)
    eligible = outputs.top_k_mask & safe
    eligible |= base_mask
    refined = base_scores + residual_scale * outputs.residual
    refined -= (~base_mask).to(refined.dtype) * switch_penalty
    selection = torch.where(eligible, refined, base_scores - 100.0)
    model_index = selection.argmax(dim=1)
    oracle_index = target_scores.argmax(dim=1)
    base_value = target_scores[rows, base_index]
    model_value = target_scores[rows, model_index]
    oracle_value = target_scores[rows, oracle_index]
    delta = (model_value - base_value).numpy()
    factor_width = target_factors.shape[-1]
    base_factors = target_factors.gather(
        1, base_index[:, None, None].expand(-1, 1, factor_width)
    ).squeeze(1)
    model_factors = target_factors.gather(
        1, model_index[:, None, None].expand(-1, 1, factor_width)
    ).squeeze(1)

    pairwise = (
        _pairwise_accuracy(outputs, refined)
        if pairwise_accuracy is None
        else float(pairwise_accuracy)
    )
    ci = _log_bootstrap_ci(delta, log_names, seed, replicates=bootstrap_replicates)
    base_means = {
        name: float(base_factors[:, index].mean())
        for index, name in enumerate(LABEL_FACTOR_KEYS)
    }
    model_means = {
        name: float(model_factors[:, index].mean())
        for index, name in enumerate(LABEL_FACTOR_KEYS)
    }
    diagnostics = dict(
        prediction_diagnostics
        if prediction_diagnostics is not None
        else _prediction_diagnostics(outputs)
    )
    base_regret = float((oracle_value - base_value).mean())
    model_regret = float((oracle_value - model_value).mean())
    return {
        "scene_count": len(log_names),
        "log_count": len(set(log_names)),
        "base_selected_pdms": float(base_value.mean()),
        "model_selected_pdms": float(model_value.mean()),
        "selected_pdms_delta": float(delta.mean()),
        "selected_pdms_delta_log_bootstrap_95ci": list(ci),
        "best_of_16_pdms": float(target_scores.max(dim=1).values.mean()),
        "base_top1_regret": base_regret,
        "model_top1_regret": model_regret,
        "regret_reduction_fraction": float(1.0 - model_regret / max(base_regret, 1e-12)),
        "pairwise_accuracy_delta_ge_0_02": pairwise,
        "selection_switch_rate": float((model_index != base_index).float().mean()),
        "improved_scene_count_delta_gt_0_01": int((model_value - base_value > 0.01).sum()),
        "degraded_scene_count_delta_lt_minus_0_01": int((model_value - base_value < -0.01).sum()),
        "residual_scale": residual_scale,
        "switch_penalty": switch_penalty,
        "safety_floor": safety_floor,
        "safety_relative_tolerance": safety_relative_tolerance,
        "base_selected_factors": base_means,
        "model_selected_factors": model_means,
        "selected_factor_delta": {
            key: model_means[key] - base_means[key] for key in LABEL_FACTOR_KEYS
        },
        **diagnostics,
    }


def deployment_sweep(
    outputs: TemporalEvaluationOutputs,
    log_names: Sequence[str],
    seed: int,
) -> List[Dict[str, object]]:
    results = []
    diagnostics = _prediction_diagnostics(outputs)
    base_scores = outputs.base_scores
    base_index = base_scores.argmax(dim=1)
    base_mask = torch.zeros_like(base_scores, dtype=torch.bool)
    base_mask.scatter_(1, base_index[:, None], True)
    for scale in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for penalty in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02):
            refined = base_scores + scale * outputs.residual
            refined -= (~base_mask).to(refined.dtype) * penalty
            pairwise = _pairwise_accuracy(outputs, refined)
            for floor, tolerance in (
                (0.0, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.8, 1.0),
                (0.9, 1.0),
                (0.95, 1.0),
                (0.98, 1.0),
                (0.99, 1.0),
                (0.995, 1.0),
                (0.8, 0.10),
                (0.9, 0.10),
                (0.95, 0.10),
                (0.95, 0.05),
                (0.98, 0.05),
                (0.99, 0.05),
                (0.98, 0.02),
                (0.99, 0.02),
            ):
                results.append(
                    evaluate_outputs(
                        outputs,
                        log_names,
                        residual_scale=scale,
                        switch_penalty=penalty,
                        safety_floor=floor,
                        safety_relative_tolerance=tolerance,
                        seed=seed,
                        bootstrap_replicates=0,
                        prediction_diagnostics=diagnostics,
                        pairwise_accuracy=pairwise,
                    )
                )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--factor-root", type=Path, required=True)
    parser.add_argument("--consequence-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260901)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--scheduler-epochs",
        type=int,
        help=(
            "Cosine scheduler horizon. Defaults to --epochs. Locked-epoch CV "
            "replay can stop early while preserving the discovery LR curve."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-pair-delta", type=float, default=0.02)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--prediction-temperature", type=float, default=0.05)
    parser.add_argument("--top-set-tolerance", type=float, default=0.01)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--base-pairwise-weight", type=float, default=2.0)
    parser.add_argument("--listwise-weight", type=float, default=0.2)
    parser.add_argument("--top-set-weight", type=float, default=0.5)
    parser.add_argument("--expected-regret-weight", type=float, default=1.0)
    parser.add_argument("--factor-weight", type=float, default=0.2)
    parser.add_argument("--relative-safety-weight", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--area-weight", type=float, default=0.5)
    parser.add_argument("--actor-valid-weight", type=float, default=0.25)
    parser.add_argument("--actor-state-weight", type=float, default=0.25)
    parser.add_argument("--monotonic-weight", type=float, default=0.1)
    parser.add_argument("--residual-l2-weight", type=float, default=0.01)
    parser.add_argument("--safety-negative-weight", type=float, default=10.0)
    parser.add_argument("--risk-positive-weight", type=float, default=10.0)
    parser.add_argument("--area-positive-weight", type=float, default=5.0)
    parser.add_argument("--actor-positive-weight", type=float, default=5.0)
    parser.add_argument("--use-base-candidate-features", action="store_true")
    parser.add_argument("--use-relative-safety-head", action="store_true")
    parser.add_argument(
        "--safety-gate-mode",
        choices=("absolute", "relative"),
        default="absolute",
    )
    parser.add_argument(
        "--score-mode",
        choices=("residual", "factor_aggregate", "hybrid"),
        default="residual",
        help=(
            "How the temporal head corrects the frozen Base score. The factor "
            "modes use only factor logits predicted from current observations."
        ),
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train on every Navtrain log using an epoch/policy fixed by complete CV.",
    )
    parser.add_argument(
        "--cv-summary",
        type=Path,
        help="Complete temporal CV summary required by --train-all.",
    )
    parser.add_argument(
        "--retained-epoch",
        type=int,
        help=(
            "For CV replay, retain this exact epoch instead of selecting a "
            "different pairwise-best epoch per fold. The deployment sweep and "
            "saved artifact are then guaranteed to use the locked common epoch."
        ),
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_scheduler_epochs(epochs: int, scheduler_epochs: Optional[int]) -> int:
    resolved = scheduler_epochs or epochs
    if epochs < 1 or resolved < epochs:
        raise ValueError(
            "scheduler-epochs must be at least epochs and both must be positive"
        )
    return resolved


def main() -> None:
    args = parse_args()
    scheduler_epochs = resolve_scheduler_epochs(args.epochs, args.scheduler_epochs)
    for path in (
        args.source_root,
        args.factor_root,
        args.consequence_root,
        args.base_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.train_all and not 0 <= args.fold_index < args.num_folds:
        raise ValueError("fold-index must be in [0, num-folds)")
    if args.train_all and args.cv_summary is None:
        raise ValueError("--train-all requires --cv-summary")
    if args.train_all and args.retained_epoch is not None:
        raise ValueError("--retained-epoch is only valid for CV replay")
    if args.retained_epoch is not None and not 0 <= args.retained_epoch < args.epochs:
        raise ValueError("--retained-epoch must be in [0, epochs)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

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

    all_data = _load_temporal_cache(
        args.source_root,
        args.factor_root,
        args.consequence_root,
        max_scenes=args.max_scenes,
    )
    cv_summary: Optional[Mapping[str, object]] = None
    retained_epoch: Optional[int] = None
    common_policy: Optional[Dict[str, float]] = None
    if args.train_all:
        assert args.cv_summary is not None
        retained_epoch, common_policy, cv_summary = load_full_data_cv_policy(
            args.cv_summary
        )
        if not 0 <= retained_epoch < args.epochs:
            raise ValueError(
                f"CV retained epoch {retained_epoch} is outside {args.epochs} epochs"
            )
        train_data = all_data
        val_data = None
        fold_payload = {
            "mode": "all_logs",
            "num_folds": int(cv_summary["fold_audit"]["declared_num_folds"]),
            "fold_seed": int(cv_summary["fold_audit"]["fold_seed"]),
            "fold_index": None,
            "train_logs": sorted(set(train_data.log_names)),
            "validation_logs": [],
            "train_scene_count": len(train_data),
            "validation_scene_count": 0,
        }
    else:
        assignment = assign_balanced_log_folds(
            all_data.log_names,
            args.num_folds,
            args.fold_seed,
        )
        val_indices = [
            index
            for index, name in enumerate(all_data.log_names)
            if assignment[name] == args.fold_index
        ]
        train_indices = [
            index
            for index, name in enumerate(all_data.log_names)
            if assignment[name] != args.fold_index
        ]
        train_data = all_data.subset(train_indices)
        val_data = all_data.subset(val_indices)
        if set(train_data.log_names).intersection(val_data.log_names):
            raise RuntimeError("Log-level fold leakage")
        fold_payload = {
            "num_folds": args.num_folds,
            "fold_seed": args.fold_seed,
            "fold_index": args.fold_index,
            "train_logs": sorted(set(train_data.log_names)),
            "validation_logs": sorted(set(val_data.log_names)),
            "train_scene_count": len(train_data),
            "validation_scene_count": len(val_data),
        }
        del all_data

    model = TemporalConsequenceRanker(
        TemporalConsequenceConfig(
            top_k=16,
            use_base_candidate_features=args.use_base_candidate_features,
            score_mode=args.score_mode,
            use_relative_safety_head=args.use_relative_safety_head,
            safety_gate_mode=args.safety_gate_mode,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_epochs,
        eta_min=args.learning_rate * 0.05,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        _dataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json_dump(fold_payload, args.output_dir / "fold.json")

    if val_data is not None:
        initial_outputs = collect_outputs(model, val_data, device, args.eval_batch_size)
        initial: Optional[Mapping[str, object]] = evaluate_outputs(
            initial_outputs,
            val_data.log_names,
            residual_scale=0.0,
            switch_penalty=0.0,
            safety_floor=0.0,
            safety_relative_tolerance=1.0,
            seed=args.seed,
        )
        history: List[Dict[str, object]] = [{"epoch": -1, "validation": initial}]
        print("TEMPORAL_CONSEQUENCE_EVAL " + json.dumps(history[-1], sort_keys=True), flush=True)
    else:
        initial = None
        history = []
    best_pairwise = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        sums: Dict[str, float] = {}
        batches = 0
        for batch in loader:
            moved = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, details = compute_temporal_training_loss(model, moved, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in details.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training": {key: value / max(batches, 1) for key, value in sums.items()},
        }
        if val_data is not None:
            outputs = collect_outputs(model, val_data, device, args.eval_batch_size)
            validation = evaluate_outputs(
                outputs,
                val_data.log_names,
                residual_scale=1.0,
                switch_penalty=0.0,
                safety_floor=0.0,
                safety_relative_tolerance=1.0,
                seed=args.seed + epoch + 1,
            )
            record["validation"] = validation
        history.append(record)
        prefix = "TEMPORAL_CONSEQUENCE_EVAL" if val_data is not None else "TEMPORAL_CONSEQUENCE_TRAIN"
        print(prefix + " " + json.dumps(record, sort_keys=True), flush=True)
        if val_data is not None:
            pairwise = float(validation["pairwise_accuracy_delta_ge_0_02"])
            if args.retained_epoch is not None:
                retain = epoch == args.retained_epoch
            else:
                retain = pairwise > best_pairwise
        else:
            retain = epoch == retained_epoch
        if retain:
            if val_data is not None:
                best_pairwise = pairwise
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No trained epoch was retained")
    model.load_state_dict(best_state, strict=True)
    if val_data is not None:
        outputs = collect_outputs(model, val_data, device, args.eval_batch_size)
        sweep = deployment_sweep(outputs, val_data.log_names, args.seed + 1000)
        sweep = [dict(item, weight_epoch=best_epoch) for item in sweep]
        safety_tolerance = 5e-4
        safe = [
            item
            for item in sweep
            if item["selected_factor_delta"]["no_at_fault_collisions"] >= -safety_tolerance
            and item["selected_factor_delta"]["drivable_area_compliance"] >= -safety_tolerance
            and item["selected_factor_delta"]["time_to_collision_within_bound"] >= -safety_tolerance
        ]
        best_deployment = max(
            safe or sweep,
            key=lambda item: (
                float(item["model_selected_pdms"]),
                -float(item["selection_switch_rate"]),
                -float(item["residual_scale"]),
            ),
        )
    else:
        assert common_policy is not None and cv_summary is not None
        sweep = []
        best_deployment = dict(cv_summary["common_deployment"])
    deployed = TemporalConsequenceRanker(
        replace(
            model.config,
            inference_scale=float(best_deployment["residual_scale"]),
            switch_penalty=float(best_deployment["switch_penalty"]),
            safety_floor=float(best_deployment["safety_floor"]),
            safety_relative_tolerance=float(
                best_deployment["safety_relative_tolerance"]
            ),
        )
    ).to(device)
    deployed.load_state_dict(model.state_dict(), strict=True)
    if val_data is not None:
        final_outputs = collect_outputs(deployed, val_data, device, args.eval_batch_size)
        validation: Optional[Mapping[str, object]] = evaluate_outputs(
            final_outputs,
            val_data.log_names,
            residual_scale=deployed.config.inference_scale,
            switch_penalty=deployed.config.switch_penalty,
            safety_floor=deployed.config.safety_floor,
            safety_relative_tolerance=deployed.config.safety_relative_tolerance,
            seed=args.seed + 2000,
        )
    else:
        assert cv_summary is not None
        validation = dict(cv_summary["common_deployment"])
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "retained_epoch": best_epoch,
        "forced_retained_epoch": args.retained_epoch,
        "scheduler_epochs": scheduler_epochs,
        "fold": fold_payload,
        "source_root": str(args.source_root.resolve()),
        "factor_root": str(args.factor_root.resolve()),
        "consequence_root": str(args.consequence_root.resolve()),
        "train_scene_count": len(train_data),
        "train_log_count": len(set(train_data.log_names)),
        "val_scene_count": len(val_data) if val_data is not None else 0,
        "val_log_count": len(set(val_data.log_names)) if val_data is not None else 0,
        "validation": validation,
        "deployment_sweep": sweep,
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "future_inputs_used": False,
        "future_targets_training_only": True,
        "official_scores_used_at_inference": False,
        "full_data_training": args.train_all,
        "cv_summary_path": str(args.cv_summary.resolve()) if args.cv_summary else None,
        "cv_common_epoch": retained_epoch if args.train_all else None,
        "cv_common_deployment": common_policy if args.train_all else None,
        "navtest_used_for_training_or_selection": False,
        "inference_signature": [
            "current_scene_features",
            "current_ego_feature",
            "candidate_trajectories",
            "public_base_candidate_features",
            "public_base_factor_logits",
            "public_base_scores",
        ],
    }
    artifact = build_temporal_consequence_artifact(
        deployed,
        args.base_checkpoint,
        metadata=metadata,
    )
    artifact_path = args.output_dir / "best_temporal_consequence_scorer.pt"
    temporary = artifact_path.with_name(f".{artifact_path.name}.tmp-{os.getpid()}")
    torch.save(artifact, temporary)
    os.replace(temporary, artifact_path)
    results = {
        "schema_version": 1,
        "model_config": asdict(deployed.config),
        "best_epoch": best_epoch,
        "initial_validation": initial,
        "best_validation": validation,
        "deployment_sweep": sweep,
        "history": history,
        "artifact_path": str(artifact_path.resolve()),
        "metadata": metadata,
    }
    _atomic_json_dump(results, args.output_dir / "training_results.json")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
