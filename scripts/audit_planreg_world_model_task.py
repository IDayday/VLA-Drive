#!/usr/bin/env python3
"""Evaluate whether the future-register predictor learned more than copying now.

This is a diagnostic-only held-out Navtest audit. Future frames are used only
to measure the training-only world-model task; deployment/scorer evaluation
continues to consume the current frame alone.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from planreg_audit_runtime import (
    build_navtest_samples,
    collate_samples,
    load_formal_training_agent,
    select_representative_tokens,
    sha256_file,
)


HORIZON_NAMES = ("0p5", "1p5", "3p0")


def _prediction_errors(agent, predicted, current, target_current, target_future):
    predictor = agent.future_register_predictor
    current_n = predictor.normalize_register_state(current)
    target_current_n = predictor.normalize_register_state(target_current)
    target_future_n = predictor.normalize_register_state(target_future)
    cosine = 1.0 - F.cosine_similarity(
        predicted, target_future_n[:, None], dim=-1, eps=1e-8
    ).mean(dim=(1, 3))
    delta = F.smooth_l1_loss(
        predicted - current_n[:, None, None],
        target_future_n[:, None] - target_current_n[:, None, None],
        reduction="none",
    ).mean(dim=(1, 3, 4))
    return cosine, delta


def _append(rows: List[dict], values: torch.Tensor, mask: torch.Tensor, tokens, field: str):
    values = values.detach().float().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)
    for batch_index, token in enumerate(tokens):
        for horizon_index, horizon in enumerate(HORIZON_NAMES):
            if mask[batch_index, horizon_index]:
                rows.append(
                    {
                        "token": str(token),
                        "horizon": horizon,
                        "field": field,
                        "value": float(values[batch_index, horizon_index]),
                    }
                )


def _aggregate(rows: List[dict]) -> dict:
    result = {}
    fields = sorted({row["field"] for row in rows})
    for field in fields:
        result[field] = {}
        for horizon in HORIZON_NAMES:
            values = np.asarray(
                [row["value"] for row in rows if row["field"] == field and row["horizon"] == horizon],
                dtype=np.float64,
            )
            result[field][horizon] = {
                "count": int(len(values)),
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
            }
    return result


def _dtype_counts(module) -> dict:
    counts: Dict[str, int] = {}
    for parameter in module.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def _wm_gradient_audit(agent, features, targets) -> dict:
    agent.train()
    agent.zero_grad(set_to_none=True)
    predictions = agent.forward(features)
    current = predictions["planning_registers"]
    current.retain_grad()
    target_current, target_future, valid = agent._encode_ema_register_targets(
        features, targets, current.shape[0]
    )
    status = features["status_feature"]
    if status.ndim == 1:
        status = status.unsqueeze(0)
    speed = torch.linalg.vector_norm(
        status[:, 4:6].to(device=current.device, dtype=current.dtype), dim=-1
    )
    wm = agent._compute_world_model_loss_from_registers(
        current,
        targets["trajectory"],
        target_current,
        target_future,
        valid,
        current_speed=speed,
    )
    wm["wm_loss"].backward()

    layer_rows = []
    for layer_index, block in enumerate(agent.backbone.model.vision_model.encoder.layers):
        qkv = block.attn.qkv
        branch_norms = {}
        for branch in ("q_lora_a", "q_lora_b", "v_lora_a", "v_lora_b"):
            gradients = [
                parameter.grad.detach().float().square().sum()
                for parameter in getattr(qkv, branch).parameters()
                if parameter.grad is not None
            ]
            branch_norms[branch] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
        layer_rows.append({"layer": layer_index, **branch_norms})

    def module_norm(module) -> float:
        values = [
            parameter.grad.detach().float().square().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        return float(torch.stack(values).sum().sqrt()) if values else 0.0

    qv_nonzero = [
        row
        for row in layer_rows
        if all(row[name] > 0 for name in ("q_lora_a", "q_lora_b", "v_lora_a", "v_lora_b"))
    ]
    llm_grad_count = sum(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in agent.backbone.model.language_model.parameters()
    )
    result = {
        "wm_loss": float(wm["wm_loss"].detach()),
        "current_register_output_grad_norm": float(current.grad.detach().float().norm()),
        "planning_register_parameter_grad_norm": float(
            agent.backbone.planning_register_adapter.planning_registers.grad.detach().float().norm()
        ),
        "future_predictor_grad_norm": module_norm(agent.future_register_predictor),
        "vision_qv_lora_layers_with_all_four_nonzero_branches": len(qv_nonzero),
        "vision_qv_lora_layer_gradients": layer_rows,
        "frozen_llm_nonzero_gradient_tensor_count": int(llm_grad_count),
        "scorer_grad_norm_from_wm_only": module_norm(agent.action_head.scorer),
        "semantic_qformer_grad_norm_from_wm_only": module_norm(agent.action_head.q_former),
    }
    agent.zero_grad(set_to_none=True)
    agent.eval()
    return result


def audit(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The exact InternVL3-2B world-model audit requires CUDA")
    device = torch.device(args.device)
    cfg, agent, checkpoint_metadata = load_formal_training_agent(
        args.resolved_config,
        args.checkpoint,
        device=device,
        compute_dtype="float32",
    )
    if not agent.world_model_enabled or agent.future_register_predictor is None or agent.ema_register_target is None:
        raise RuntimeError("World-model audit requires the complete training checkpoint topology")

    tokens, token_metadata = select_representative_tokens(
        args.candidate_bank, args.metric_cache, args.scene_count
    )
    samples = build_navtest_samples(
        agent,
        tokens,
        token_metadata,
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
    )
    rows: List[dict] = []
    per_scene = {token: dict(token_metadata[token]) for token in tokens}
    first_batch = None
    agent.eval()
    horizons = tuple(float(value) for value in agent.world_model_config.horizons_sec)
    if horizons != (0.5, 1.5, 3.0):
        raise RuntimeError(f"Unexpected world-model horizons: {horizons}")

    for start in range(0, len(tokens), 2):
        batch_tokens = tokens[start:start + 2]
        if len(batch_tokens) < 2:
            batch_tokens = [batch_tokens[0], tokens[0]]
        features, targets = collate_samples([samples[token] for token in batch_tokens])
        if first_batch is None:
            first_batch = (features, targets)
        with torch.inference_mode():
            prediction = agent.forward(features)
            current = prediction["planning_registers"]
            target_current, target_future, valid = agent._encode_ema_register_targets(
                features, targets, current.shape[0]
            )
            if not bool(valid.all()):
                invalid = (~valid).nonzero(as_tuple=False).cpu().tolist()
                raise RuntimeError(f"Selected Navtest audit samples lack future frames: {invalid}")
            trajectory = targets["trajectory"].to(device=current.device, dtype=current.dtype)[:, None]
            status = features["status_feature"]
            speed = torch.linalg.vector_norm(
                status[:, 4:6].to(device=current.device, dtype=current.dtype), dim=-1
            )
            predictor = agent.future_register_predictor
            correct = predictor(current, trajectory, horizons, use_action_condition=True, current_speed=speed)
            no_action = predictor(current, trajectory, horizons, use_action_condition=False, current_speed=speed)
            shuffled_action = predictor(
                current,
                torch.roll(trajectory, shifts=1, dims=0),
                horizons,
                use_action_condition=True,
                current_speed=torch.roll(speed, shifts=1, dims=0),
            )
            current_n = predictor.normalize_register_state(current)
            target_current_n = predictor.normalize_register_state(target_current)
            target_future_n = predictor.normalize_register_state(target_future)
            copy_student = current_n[:, None, None].expand_as(correct)
            copy_teacher = target_current_n[:, None, None].expand_as(correct)

            for name, value in (
                ("correct", correct),
                ("no_action", no_action),
                ("shuffled_action", shuffled_action),
                ("copy_student_current", copy_student),
                ("copy_teacher_current", copy_teacher),
            ):
                cosine, delta = _prediction_errors(
                    agent, value, current, target_current, target_future
                )
                _append(rows, cosine, valid, batch_tokens, f"{name}_cosine_loss")
                _append(rows, delta, valid, batch_tokens, f"{name}_delta_loss")

            shuffled_target = torch.roll(target_future_n, shifts=1, dims=0)
            shuffled_target_cosine = 1.0 - F.cosine_similarity(
                correct, shuffled_target[:, None], dim=-1, eps=1e-8
            ).mean(dim=(1, 3))
            _append(rows, shuffled_target_cosine, valid, batch_tokens, "correct_prediction_shuffled_target_cosine_loss")

            action_effect = (correct - no_action).float().square().mean(dim=(1, 3, 4)).sqrt()
            shuffled_action_effect = (correct - shuffled_action).float().square().mean(dim=(1, 3, 4)).sqrt()
            target_delta = (target_future_n - target_current_n[:, None]).float().square().mean(dim=(2, 3)).sqrt()
            student_teacher_current = 1.0 - F.cosine_similarity(
                current_n, target_current_n, dim=-1, eps=1e-8
            ).mean(dim=1)
            _append(rows, action_effect, valid, batch_tokens, "prediction_rms_change_remove_action")
            _append(rows, shuffled_action_effect, valid, batch_tokens, "prediction_rms_change_shuffle_action")
            _append(rows, target_delta, valid, batch_tokens, "teacher_future_delta_rms")
            for index, token in enumerate(batch_tokens):
                per_scene[token]["student_teacher_current_cosine_loss"] = float(student_teacher_current[index].cpu())
                per_scene[token]["future_valid_mask"] = valid[index].cpu().tolist()

        del prediction, current, target_current, target_future
        torch.cuda.empty_cache()

    if first_batch is None:
        raise RuntimeError("No world-model audit batch was constructed")
    gradient_audit = _wm_gradient_audit(agent, *first_batch)
    aggregate = _aggregate(rows)
    for horizon in HORIZON_NAMES:
        aggregate.setdefault("derived", {})[horizon] = {
            "cosine_gain_over_student_current_copy": (
                aggregate["copy_student_current_cosine_loss"][horizon]["mean"]
                - aggregate["correct_cosine_loss"][horizon]["mean"]
            ),
            "cosine_gain_from_action_condition": (
                aggregate["no_action_cosine_loss"][horizon]["mean"]
                - aggregate["correct_cosine_loss"][horizon]["mean"]
            ),
            "cosine_gain_vs_shuffled_action": (
                aggregate["shuffled_action_cosine_loss"][horizon]["mean"]
                - aggregate["correct_cosine_loss"][horizon]["mean"]
            ),
            "correct_vs_shuffled_target_margin": (
                aggregate["correct_prediction_shuffled_target_cosine_loss"][horizon]["mean"]
                - aggregate["correct_cosine_loss"][horizon]["mean"]
            ),
        }

    report = {
        "schema_version": 1,
        "checkpoint": checkpoint_metadata,
        "resolved_config": str(args.resolved_config.resolve()),
        "resolved_config_sha256": sha256_file(args.resolved_config),
        "candidate_bank_for_scene_stratification": str(args.candidate_bank.resolve()),
        "candidate_bank_sha256": sha256_file(args.candidate_bank),
        "split": "Navtest held-out from the 103k trainval fit",
        "scene_count": len(tokens),
        "tokens": per_scene,
        "future_frames_used_for_diagnostic_only": True,
        "deployment_inference_uses_future_frames": False,
        "model_parameter_dtypes": _dtype_counts(agent),
        "metrics": aggregate,
        "gradient_routing_from_wm_loss_only": gradient_audit,
        "raw_rows": rows,
        "interpretation": {
            "learned_future_gate": "correct must beat both current-copy baselines on held-out scenes",
            "learned_action_gate": "correct must beat no-action and shuffled-action controls, not merely change numerically",
            "representation_gate": "correct-target loss must be lower than shuffled-target loss",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": checkpoint_metadata, "metrics": aggregate.get("derived", {}), "gradient": gradient_audit}, indent=2, sort_keys=True))
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
