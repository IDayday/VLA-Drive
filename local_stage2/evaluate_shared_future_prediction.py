#!/usr/bin/env python3
"""Evaluate M0 current-observation shared-future predictions on held-out logs."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from local_stage2.train_independent_scorer import (
    ReplaySource,
    _atomic_json_dump,
    _sha256,
    load_current_actor_target_table,
    load_replay_sources,
)
from local_stage2.train_m0_private_residual_scorer import (
    ResidualReplayDataset,
    load_replay_base_factor_logits,
    load_shared_future_target_table,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentRankerConfig,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)


def _physical_log_set(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    train = {str(value) for value in payload["train_physical_logs"]}
    validation = {str(value) for value in payload["validation_physical_logs"]}
    if not train or not validation or train.intersection(validation):
        raise RuntimeError("split manifest does not define disjoint physical logs")
    return validation


def _decode_predicted_state(normalized: torch.Tensor) -> torch.Tensor:
    """Decode x/y/vx/vy/heading/length/width from the eight-field head."""

    return torch.stack(
        (
            normalized[..., 0] * 50.0,
            normalized[..., 1] * 50.0,
            normalized[..., 2] * 20.0,
            normalized[..., 3] * 20.0,
            torch.atan2(normalized[..., 4], normalized[..., 5]),
            normalized[..., 6].abs() * 10.0,
            normalized[..., 7].abs() * 5.0,
        ),
        dim=-1,
    )


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _presence_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    target_float = target.float()
    probability = logits.sigmoid()
    predicted = probability >= 0.5
    true_positive = int((predicted & target).sum())
    false_positive = int((predicted & ~target).sum())
    false_negative = int((~predicted & target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "bce": float(F.binary_cross_entropy_with_logits(logits, target_float)),
        "brier": float((probability - target_float).square().mean()),
        "accuracy": float((predicted == target).float().mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1.0e-12)),
    }


def _state_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, object]:
    if predicted.shape != (*valid.shape, 7) or target.shape != predicted.shape:
        raise ValueError("actor state tensors do not align")
    if not bool(valid.any()):
        raise RuntimeError("no valid actor future is available")
    position = torch.linalg.vector_norm(predicted[..., :2] - target[..., :2], dim=-1)
    velocity = torch.linalg.vector_norm(predicted[..., 2:4] - target[..., 2:4], dim=-1)
    heading = _wrap(predicted[..., 4] - target[..., 4]).abs()
    size = (predicted[..., 5:7] - target[..., 5:7]).abs().mean(dim=-1)
    per_horizon = []
    for horizon in range(valid.shape[1]):
        mask = valid[:, horizon]
        per_horizon.append(
            {
                "horizon_seconds": 0.5 * (horizon + 1),
                "actor_count": int(mask.sum()),
                "position_l2_mae_m": float(position[:, horizon][mask].mean()),
                "velocity_l2_mae_mps": float(velocity[:, horizon][mask].mean()),
                "heading_abs_mae_rad": float(heading[:, horizon][mask].mean()),
                "size_abs_mae_m": float(size[:, horizon][mask].mean()),
            }
        )
    return {
        "actor_step_count": int(valid.sum()),
        "position_l2_mae_m": float(position[valid].mean()),
        "velocity_l2_mae_mps": float(velocity[valid].mean()),
        "heading_abs_mae_rad": float(heading[valid].mean()),
        "size_abs_mae_m": float(size[valid].mean()),
        "per_horizon": per_horizon,
    }


def _baseline_states(
    current: torch.Tensor,
    horizons: int,
    constant_velocity: bool,
) -> torch.Tensor:
    # Current layout: type,x,y,vx,vy,heading,length,width.
    state = current[..., 1:].unsqueeze(1).expand(-1, horizons, -1, -1).clone()
    if constant_velocity:
        seconds = torch.arange(
            1,
            horizons + 1,
            dtype=state.dtype,
        ).view(1, horizons, 1) * 0.5
        state[..., 0] += current[:, None, :, 3] * seconds
        state[..., 1] += current[:, None, :, 4] * seconds
    return state


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != "M0PrivateResidualRanker":
        raise RuntimeError("artifact is not an M0 private residual ranker")
    private_config = IndependentRankerConfig(**artifact["private_config"])
    if not private_config.shared_future_auxiliary:
        raise RuntimeError("artifact has no shared-future prediction head")
    residual_config = M0PrivateResidualConfig(**artifact["residual_config"])
    model = M0PrivateResidualRanker(private_config, residual_config)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.to(args.device).eval()

    forbidden_names = {"future", "official", "metric"}
    signature = set(inspect.signature(model.forward).parameters)
    if any(any(term in name for term in forbidden_names) for name in signature):
        raise RuntimeError("model forward signature contains a future/evaluator input")

    source = ReplaySource(args.source_name, args.feature_root, args.label_root)
    data, source_lineage = load_replay_sources(
        [source],
        private_observation_root=args.private_observation_root,
        retain_m0_context=residual_config.m0_context_fusion,
    )
    factor_tokens, base_factor_logits = load_replay_base_factor_logits([source])
    if factor_tokens != data.tokens:
        raise RuntimeError("factor logits do not align with replay rows")
    future = load_shared_future_target_table(args.shared_future_target_root)
    current = load_current_actor_target_table(args.current_actor_target_root)
    future_for_token = {token: index for index, token in enumerate(future.tokens)}
    current_for_token = {token: index for index, token in enumerate(current.tokens)}
    future_rows = torch.tensor(
        [future_for_token.get(token, -1) for token in data.tokens], dtype=torch.long
    )
    current_rows = torch.tensor(
        [current_for_token.get(token, -1) for token in data.tokens], dtype=torch.long
    )
    validation_logs = _physical_log_set(args.split_manifest)
    indices = [
        index
        for index, physical_log in enumerate(data.physical_logs)
        if physical_log in validation_logs
        and int(future_rows[index]) >= 0
        and bool(future.supervision_valid[int(future_rows[index])])
        and int(current_rows[index]) >= 0
        and bool(current.supervision_valid[int(current_rows[index])])
    ]
    if not indices:
        raise RuntimeError("held-out split has no aligned future targets")
    loader = DataLoader(
        ResidualReplayDataset(
            data,
            base_factor_logits,
            indices,
            include_m0_context=residual_config.m0_context_fusion,
            shared_future_table=future,
            shared_future_row_indices=future_rows,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predicted_state_parts: List[torch.Tensor] = []
    presence_parts: List[torch.Tensor] = []
    type_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    target_mask_parts: List[torch.Tensor] = []
    current_parts: List[torch.Tensor] = []
    current_mask_parts: List[torch.Tensor] = []
    predicted_current_state_parts: List[torch.Tensor] = []
    current_presence_parts: List[torch.Tensor] = []
    current_type_parts: List[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            proposals, observation, observation_mask, status = batch[:4]
            base_scores, factor_logits = batch[4:6]
            source_indices = batch[7]
            cursor = 8
            m0_context = {}
            if residual_config.m0_context_fusion:
                m0_context = {
                    "m0_scene_features": batch[cursor].to(
                        args.device, non_blocking=True
                    ).float(),
                    "m0_ego_features": batch[cursor + 1].to(
                        args.device, non_blocking=True
                    ).float(),
                }
                cursor += 2
            output = model(
                observation.to(args.device, non_blocking=True).float(),
                status.to(args.device, non_blocking=True).float(),
                proposals.to(args.device, non_blocking=True),
                factor_logits.to(args.device, non_blocking=True),
                base_scores.to(args.device, non_blocking=True),
                observation_valid_mask=observation_mask.to(
                    args.device, non_blocking=True
                ),
                **m0_context,
            )
            predicted_state_parts.append(
                _decode_predicted_state(
                    output["shared_future_actor_state"]
                ).float().cpu()
            )
            presence_parts.append(
                output["shared_future_presence_logits"].float().cpu()
            )
            type_parts.append(output["shared_future_type_logits"].float().cpu())
            if private_config.current_actor_auxiliary:
                predicted_current_state_parts.append(
                    _decode_predicted_state(
                        output["current_actor_state"]
                    ).float().cpu()
                )
                current_presence_parts.append(
                    output["current_actor_presence_logits"].float().cpu()
                )
                current_type_parts.append(
                    output["current_actor_type_logits"].float().cpu()
                )
            target_parts.append(batch[cursor].float())
            target_mask_parts.append(batch[cursor + 1].bool())
            current_index = current_rows.index_select(0, source_indices.long())
            current_parts.append(current.actor_states.index_select(0, current_index))
            current_mask_parts.append(current.actor_masks.index_select(0, current_index))

    predicted = torch.cat(predicted_state_parts)
    presence_logits = torch.cat(presence_parts)
    type_logits = torch.cat(type_parts)
    target_raw = torch.cat(target_parts)
    target_mask = torch.cat(target_mask_parts)
    current_raw = torch.cat(current_parts)
    current_mask = torch.cat(current_mask_parts)
    target_state = target_raw[..., 1:]
    target_type = target_raw[..., 0].round().long()
    type_accuracy = float(
        (type_logits.argmax(dim=-1)[target_mask] == target_type[target_mask])
        .float()
        .mean()
    )
    constant = _baseline_states(current_raw, target_mask.shape[1], False)
    constant_velocity = _baseline_states(current_raw, target_mask.shape[1], True)
    repeated_presence = current_mask[:, None].expand_as(target_mask)

    model_errors = _state_errors(predicted, target_state, target_mask)
    constant_errors = _state_errors(constant, target_state, target_mask)
    cv_errors = _state_errors(constant_velocity, target_state, target_mask)
    current_model = None
    if private_config.current_actor_auxiliary:
        predicted_current = torch.cat(predicted_current_state_parts)
        current_presence = torch.cat(current_presence_parts)
        current_type_logits = torch.cat(current_type_parts)
        current_state_errors = _state_errors(
            predicted_current[:, None],
            current_raw[..., 1:][:, None],
            current_mask[:, None],
        )
        current_state_errors["per_horizon"][0]["horizon_seconds"] = 0.0
        current_type = current_raw[..., 0].round().long()
        current_model = {
            "presence": _presence_metrics(current_presence, current_mask),
            "type_accuracy_on_present_actors": float(
                (
                    current_type_logits.argmax(dim=-1)[current_mask]
                    == current_type[current_mask]
                )
                .float()
                .mean()
            ),
            "state": current_state_errors,
        }
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "artifact_epoch": int(artifact["epoch"]),
        "scene_count": len(indices),
        "physical_log_count": len({data.physical_logs[index] for index in indices}),
        "candidate_count": 64,
        "model_inference_inputs": list(artifact["inference_input_schema"]),
        "future_target_used_only_after_forward": True,
        "official_score_used_as_model_input": False,
        "external_model_representation_or_weight_used": False,
        "shared_future_relabeling": private_config.shared_future_relabeling,
        "shared_future_constant_velocity_residual": (
            private_config.shared_future_constant_velocity_residual
        ),
        "model": {
            "presence": _presence_metrics(presence_logits, target_mask),
            "type_accuracy_on_present_actors": type_accuracy,
            "state": model_errors,
        },
        "constant_position_baseline": {
            "presence": _presence_metrics(
                torch.where(
                    repeated_presence,
                    torch.full_like(target_mask, 12, dtype=torch.float32),
                    torch.full_like(target_mask, -12, dtype=torch.float32),
                ),
                target_mask,
            ),
            "state": constant_errors,
        },
        "constant_velocity_baseline": {
            "presence": _presence_metrics(
                torch.where(
                    repeated_presence,
                    torch.full_like(target_mask, 12, dtype=torch.float32),
                    torch.full_like(target_mask, -12, dtype=torch.float32),
                ),
                target_mask,
            ),
            "state": cv_errors,
        },
        "position_gain_over_constant_m": (
            constant_errors["position_l2_mae_m"]
            - model_errors["position_l2_mae_m"]
        ),
        "position_gain_over_constant_velocity_m": (
            cv_errors["position_l2_mae_m"]
            - model_errors["position_l2_mae_m"]
        ),
        "lineage": {
            "source": source_lineage,
            "private_observation_root": str(args.private_observation_root.resolve()),
            "shared_future_target": future.lineage,
            "current_actor_target": current.lineage,
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": _sha256(args.split_manifest),
        },
    }
    if current_model is not None:
        result["model_current_actor"] = current_model
    if not all(
        math.isfinite(float(value))
        for value in (
            result["position_gain_over_constant_m"],
            result["position_gain_over_constant_velocity_m"],
        )
    ):
        raise RuntimeError("shared-future evaluation produced non-finite metrics")
    _atomic_json_dump(result, args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-name", default="public_base")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--shared-future-target-root", type=Path, required=True)
    parser.add_argument("--current-actor-target-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.artifact,
        args.feature_root,
        args.label_root,
        args.private_observation_root,
        args.shared_future_target_root,
        args.current_actor_target_root,
        args.split_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
