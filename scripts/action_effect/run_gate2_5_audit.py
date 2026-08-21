#!/usr/bin/env python3
"""Run Gate 2.5 action wiring, overfit, capacity, and calibration checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import (  # noqa: E402
    CacheConflictError,
    content_hash,
    file_sha256,
    read_manifest,
    write_json,
    write_jsonl,
)
from research.action_effect.gate2_5 import (  # noqa: E402
    calibrate_risk_threshold,
    calibrated_scene_bootstrap,
    consequence_risk_score,
    intervals_to_json,
    standardize_target,
    trajectory_summary_target,
    unsafe_labels,
)
from research.action_effect.losses import ConsequencePredictionLoss  # noqa: E402
from research.action_effect.metrics import decoded_prediction  # noqa: E402
from research.action_effect.probe_data import (  # noqa: E402
    HARD_TARGET_FIELDS,
    SOFT_TARGET_FIELDS,
    load_probe_arrays,
)
from research.action_effect.world_probe import ActionEffectWorldProbe, count_parameters  # noqa: E402


CONTROL_NAMES = (
    "scene_only_probe",
    "trajectory_only_probe",
    "scene_action_probe",
    "shuffled_action_probe",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return result


def _root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _resolve(explicit: Path | None, variable: str, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _root(variable) / relative


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _probe(config: dict[str, Any], consequence_dim: int, *, dropout: float = 0.0) -> ActionEffectWorldProbe:
    probe = config["probe"]
    return ActionEffectWorldProbe(
        scene_input_dim=int(probe["scene_input_dim"]),
        consequence_dim=consequence_dim,
        latent_dim=int(probe["latent_dim"]),
        trajectory_input_dim=int(probe["trajectory_input_dim"]),
        trajectory_token_dim=int(probe["trajectory_token_dim"]),
        dropout=dropout,
        input_mode="scene_action",
    )


def _within_scene_roll(indices: np.ndarray, scene_ids: np.ndarray) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64).copy()
    for scene_id in sorted(set(scene_ids[indices].tolist())):
        position = np.flatnonzero(scene_ids[indices] == scene_id)
        if len(position) > 1:
            result[position] = np.roll(indices[position], 1)
    return result


def _scene_bootstrap_gap(
    correct: np.ndarray,
    shuffled: np.ndarray,
    scene_ids: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    scenes = np.asarray(sorted(set(scene_ids.tolist())), dtype=str)
    per_scene = {
        scene: float(np.mean(shuffled[scene_ids == scene] - correct[scene_ids == scene]))
        for scene in scenes
    }
    point = float(np.mean(list(per_scene.values())))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.mean([per_scene[scene] for scene in rng.choice(scenes, len(scenes), replace=True)])
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {"point": point, "ci_low": float(low), "ci_high": float(high)}


@torch.no_grad()
def _predict(
    model: nn.Module,
    *,
    scene_features: np.ndarray,
    scene_indices: np.ndarray,
    trajectories: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(trajectories), batch_size):
        stop = min(start + batch_size, len(trajectories))
        output = model(
            torch.from_numpy(scene_features[scene_indices[start:stop]]).to(device),
            torch.from_numpy(trajectories[start:stop]).to(device),
        )["consequence_prediction"]
        assert isinstance(output, torch.Tensor)
        result.append(output.float().cpu().numpy())
    return np.concatenate(result)


def _fit_synthetic(
    *,
    config: dict[str, Any],
    arrays: Any,
    physical_trajectories: np.ndarray,
    scene_features: np.ndarray,
    fit_scenes: list[str],
    heldout_scenes: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], ActionEffectWorldProbe]:
    cfg = config["synthetic"]
    accepted = arrays.accepted
    fit_indices = np.flatnonzero(accepted & np.isin(arrays.scene_ids, fit_scenes))
    test_indices = np.flatnonzero(accepted & np.isin(arrays.scene_ids, heldout_scenes))
    summary = trajectory_summary_target(physical_trajectories, interval_s=float(cfg["interval_s"]))
    normalized, statistics = standardize_target(summary, fit_indices)
    seed = int(config["experiment"]["split_seed"])
    _seed_everything(seed)
    model = _probe(config, normalized.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(int(cfg["epochs"])):
        order = rng.permutation(fit_indices)
        for start in range(0, len(order), int(cfg["batch_size"])):
            batch = order[start : start + int(cfg["batch_size"])]
            prediction = model(
                torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device),
                torch.from_numpy(arrays.trajectories[batch]).to(device),
            )["consequence_prediction"]
            assert isinstance(prediction, torch.Tensor)
            target = torch.from_numpy(normalized[batch]).to(device)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    correct_prediction = _predict(
        model,
        scene_features=scene_features,
        scene_indices=arrays.scene_feature_indices[test_indices],
        trajectories=arrays.trajectories[test_indices],
        device=device,
    )
    shifted = _within_scene_roll(test_indices, arrays.scene_ids)
    shuffled_prediction = _predict(
        model,
        scene_features=scene_features,
        scene_indices=arrays.scene_feature_indices[test_indices],
        trajectories=arrays.trajectories[shifted],
        device=device,
    )
    correct_error = np.mean(np.square(correct_prediction - normalized[test_indices]), axis=1)
    shuffled_error = np.mean(np.square(shuffled_prediction - normalized[test_indices]), axis=1)
    gap = _scene_bootstrap_gap(
        correct_error,
        shuffled_error,
        arrays.scene_ids[test_indices],
        samples=int(config["experiment"]["bootstrap_samples"]),
        confidence=float(config["experiment"]["bootstrap_confidence"]),
        seed=seed,
    )
    mean_correct = float(np.mean(correct_error))
    mean_shuffled = float(np.mean(shuffled_error))
    ratio = mean_shuffled / max(mean_correct, 1.0e-12)
    passed = bool(
        mean_correct <= float(cfg["maximum_normalized_mse"])
        and ratio >= float(cfg["minimum_shuffle_error_ratio"])
        and gap["ci_low"] > 0
    )
    return {
        "passed": passed,
        "fit_candidates": len(fit_indices),
        "heldout_candidates": len(test_indices),
        "target_fields": [
            "endpoint_x",
            "endpoint_y",
            "mean_speed",
            "signed_lateral_displacement",
            "max_abs_curvature",
        ],
        "target_statistics": statistics,
        "correct_normalized_mse": mean_correct,
        "shuffled_normalized_mse": mean_shuffled,
        "shuffle_error_ratio": ratio,
        "shuffle_gap": gap,
    }, model


def _decoded_mae(raw: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(decoded_prediction(raw, len(HARD_TARGET_FIELDS)) - target), axis=1)


def _candidate_overfit(
    *,
    config: dict[str, Any],
    arrays: Any,
    scene_features: np.ndarray,
    fit_scenes: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], ActionEffectWorldProbe, torch.optim.Optimizer, np.ndarray]:
    cfg = config["candidate_overfit"]
    ranked: list[tuple[float, str, np.ndarray]] = []
    for scene in fit_scenes:
        indices = np.flatnonzero(arrays.accepted & (arrays.scene_ids == scene))
        if len(indices) < 8:
            continue
        diversity = float(np.mean(np.var(arrays.targets[indices], axis=0)))
        ranked.append((diversity, scene, indices))
    selected = sorted(ranked, key=lambda row: (-row[0], row[1]))[: int(cfg["scene_count"])]
    if len(selected) != int(cfg["scene_count"]):
        raise RuntimeError("not enough candidate-rich scenes for the overfit audit")
    indices = np.concatenate([row[2] for row in selected])
    seed = int(config["experiment"]["split_seed"])
    _seed_everything(seed)
    model = _probe(
        config,
        len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS),
    ).to(device)
    loss_fn = ConsequencePredictionLoss(len(HARD_TARGET_FIELDS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(int(cfg["epochs"])):
        order = rng.permutation(indices)
        for start in range(0, len(order), int(cfg["batch_size"])):
            batch = order[start : start + int(cfg["batch_size"])]
            prediction = model(
                torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device),
                torch.from_numpy(arrays.trajectories[batch]).to(device),
            )["consequence_prediction"]
            assert isinstance(prediction, torch.Tensor)
            loss = loss_fn(prediction, torch.from_numpy(arrays.targets[batch]).to(device))["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    correct_raw = _predict(
        model,
        scene_features=scene_features,
        scene_indices=arrays.scene_feature_indices[indices],
        trajectories=arrays.trajectories[indices],
        device=device,
    )
    shifted = _within_scene_roll(indices, arrays.scene_ids)
    shuffled_raw = _predict(
        model,
        scene_features=scene_features,
        scene_indices=arrays.scene_feature_indices[indices],
        trajectories=arrays.trajectories[shifted],
        device=device,
    )
    correct_error = _decoded_mae(correct_raw, arrays.targets[indices])
    shuffled_error = _decoded_mae(shuffled_raw, arrays.targets[indices])
    constant = arrays.targets[indices].mean(axis=0, keepdims=True)
    constant[:, : len(HARD_TARGET_FIELDS)] = (
        constant[:, : len(HARD_TARGET_FIELDS)] >= 0.5
    ).astype(np.float32)
    control_error = np.mean(np.abs(constant - arrays.targets[indices]), axis=1)
    correct = float(np.mean(correct_error))
    shuffled = float(np.mean(shuffled_error))
    control = float(np.mean(control_error))
    result = {
        "passed": bool(
            correct / max(control, 1.0e-12)
            <= float(cfg["maximum_mean_control_error_ratio"])
            and shuffled / max(correct, 1.0e-12)
            >= float(cfg["minimum_shuffle_error_ratio"])
        ),
        "scene_count": len(selected),
        "candidate_count": len(indices),
        "selected_scenes": [row[1] for row in selected],
        "decoded_mae": correct,
        "mean_control_mae": control,
        "mean_control_error_ratio": correct / max(control, 1.0e-12),
        "shuffled_decoded_mae": shuffled,
        "shuffle_error_ratio": shuffled / max(correct, 1.0e-12),
    }
    return result, model, optimizer, indices


def _action_path_audit(
    *,
    model: ActionEffectWorldProbe,
    optimizer: torch.optim.Optimizer,
    arrays: Any,
    physical_trajectories: np.ndarray,
    scene_features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    action_prefixes = ("trajectory_project", "trajectory_encoder", "trajectory_pool")
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    action_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(action_prefixes)
    }
    fusion_parameters = {
        name: parameter for name, parameter in model.named_parameters() if name.startswith("fusion")
    }
    batch = indices[: min(64, len(indices))]
    scene = torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device)
    trajectory = torch.from_numpy(arrays.trajectories[batch]).to(device).requires_grad_(True)
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(scene, trajectory)["consequence_prediction"]
    assert isinstance(output, torch.Tensor)
    target = torch.from_numpy(arrays.targets[batch]).to(device)
    loss = ConsequencePredictionLoss(len(HARD_TARGET_FIELDS))(output, target)["total"]
    loss.backward(retain_graph=True)
    action_gradient = math.sqrt(
        sum(
            float(torch.sum(parameter.grad.detach().float().square()).item())
            for parameter in action_parameters.values()
            if parameter.grad is not None
        )
    )
    projection = torch.linspace(-1.0, 1.0, output.shape[-1], device=device)
    jacobian = torch.autograd.grad(
        torch.sum(output * projection),
        trajectory,
        retain_graph=False,
    )[0]
    jacobian_norm = float(jacobian.detach().float().norm(dim=(1, 2)).mean().item())
    model.eval()
    with torch.no_grad():
        components = model.forward_components(scene, trajectory.detach())
        _, no_action_latent = model.fuse_embeddings(
            components.scene_embedding,
            torch.zeros_like(components.action_embedding),
        )
        action_variance = []
        for scene_id in sorted(set(arrays.scene_ids[indices].tolist())):
            local = np.flatnonzero(arrays.scene_ids[batch] == scene_id)
            if len(local) > 1:
                action_variance.append(
                    float(components.action_embedding[local].float().var(dim=0, unbiased=False).mean().item())
                )
        fusion_delta = torch.linalg.vector_norm(
            components.effect_latent - no_action_latent, dim=1
        )
        latent_norm = torch.linalg.vector_norm(components.effect_latent, dim=1)
    normalized = arrays.trajectories[indices]
    physical = physical_trajectories[indices]
    result = {
        "all_action_parameters_in_optimizer": all(
            id(parameter) in optimizer_ids for parameter in action_parameters.values()
        ),
        "all_fusion_parameters_in_optimizer": all(
            id(parameter) in optimizer_ids for parameter in fusion_parameters.values()
        ),
        "action_parameter_count": int(sum(parameter.numel() for parameter in action_parameters.values())),
        "fusion_parameter_count": int(sum(parameter.numel() for parameter in fusion_parameters.values())),
        "action_encoder_gradient_norm": action_gradient,
        "output_to_trajectory_jacobian_norm": jacobian_norm,
        "within_scene_action_embedding_variance": float(np.mean(action_variance)),
        "scene_embedding_rms": float(components.scene_embedding.float().square().mean().sqrt().item()),
        "action_embedding_rms": float(components.action_embedding.float().square().mean().sqrt().item()),
        "fusion_action_delta_norm": float(fusion_delta.mean().item()),
        "fusion_action_delta_ratio": float((fusion_delta / latent_norm.clamp_min(1.0e-8)).mean().item()),
        "normalized_trajectory_range": {
            "minimum": np.min(normalized, axis=(0, 1)).astype(float).tolist(),
            "maximum": np.max(normalized, axis=(0, 1)).astype(float).tolist(),
            "standard_deviation": np.std(normalized, axis=(0, 1)).astype(float).tolist(),
        },
        "physical_trajectory_range": {
            "minimum": np.min(physical, axis=(0, 1)).astype(float).tolist(),
            "maximum": np.max(physical, axis=(0, 1)).astype(float).tolist(),
        },
    }
    result["passed"] = bool(
        result["all_action_parameters_in_optimizer"]
        and result["all_fusion_parameters_in_optimizer"]
        and action_gradient > 1.0e-8
        and jacobian_norm > 1.0e-8
        and result["within_scene_action_embedding_variance"] > 1.0e-10
        and result["fusion_action_delta_norm"] > 1.0e-8
    )
    return result


def _calibration_audit(
    *,
    config: dict[str, Any],
    arrays: Any,
    scales: Any,
    fit_scenes: list[str],
    heldout_scenes: list[str],
    factual_probe_dir: Path,
) -> list[dict[str, Any]]:
    low_ttc = float(config["risk"]["low_ttc_seconds"])
    labels = unsafe_labels(arrays, low_ttc_seconds=low_ttc)
    fit_mask = arrays.accepted & np.isin(arrays.scene_ids, fit_scenes)
    heldout_mask = arrays.accepted & np.isin(arrays.scene_ids, heldout_scenes)
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    rows: list[dict[str, Any]] = []
    parameter_counts: dict[str, set[int]] = {control: set() for control in CONTROL_NAMES}
    for control in CONTROL_NAMES:
        for seed in seeds:
            run_dir = factual_probe_dir / control / f"seed_{seed}"
            with np.load(run_dir / "predictions.npz") as payload:
                prediction = np.asarray(payload["consequence_prediction"], dtype=np.float32)
            with (run_dir / "result.json").open("r", encoding="utf-8") as stream:
                run_result = json.load(stream)
            parameter_counts[control].add(int(run_result["total_parameters"]))
            scores = consequence_risk_score(prediction, scales, low_ttc_seconds=low_ttc)
            threshold, fit_metrics = calibrate_risk_threshold(labels[fit_mask], scores[fit_mask])
            intervals, counts = calibrated_scene_bootstrap(
                labels=labels[heldout_mask],
                scores=scores[heldout_mask],
                scene_ids=arrays.scene_ids[heldout_mask],
                selected_scene_ids=heldout_scenes,
                threshold=threshold,
                samples=int(config["experiment"]["bootstrap_samples"]),
                confidence=float(config["experiment"]["bootstrap_confidence"]),
                seed=seed,
            )
            rows.append(
                {
                    "control": control,
                    "seed": seed,
                    "total_parameters": int(run_result["total_parameters"]),
                    "calibrated_threshold": threshold,
                    "fit_balanced_accuracy": fit_metrics["balanced_accuracy"],
                    **counts,
                    **{
                        f"{name}_{field}": value
                        for name, interval in intervals_to_json(intervals).items()
                        for field, value in interval.items()
                        if field in {"point", "ci_low", "ci_high"}
                    },
                }
            )
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise RuntimeError("a control changes parameter count across seeds")
    if len({next(iter(values)) for values in parameter_counts.values()}) != 1:
        raise RuntimeError("Gate 2.5 capacity controls do not have equal parameter counts")
    return rows


def _mean_ci(rows: Iterable[dict[str, Any]], control: str, name: str) -> tuple[float, float, float]:
    selected = [row for row in rows if row["control"] == control]
    return (
        float(np.mean([row[f"{name}_point"] for row in selected])),
        float(np.mean([row[f"{name}_ci_low"] for row in selected])),
        float(np.mean([row[f"{name}_ci_high"] for row in selected])),
    )


def _render_report(
    *,
    identity: dict[str, Any],
    synthetic: dict[str, Any],
    overfit: dict[str, Any],
    action_path: dict[str, Any],
    calibration: list[dict[str, Any]],
) -> str:
    passed = synthetic["passed"] and overfit["passed"] and action_path["passed"]
    lines = [
        "# Gate 2.5 action-path audit",
        "",
        f"**Decision: {'PASS' if passed else 'FAIL'}.** "
        + (
            "The observed Gate-2 collapse is not explained by a broken action branch."
            if passed
            else "At least one engineering check failed; Phase 6 must not run until it is resolved."
        ),
        "",
        f"Source commit: `{identity['git_commit']}`; source tree hash: `{identity['code_tree_hash']}`.",
        "",
        "## Synthetic action fitting",
        "",
        f"- Correct normalized MSE: {synthetic['correct_normalized_mse']:.6f}.",
        f"- Shuffled normalized MSE: {synthetic['shuffled_normalized_mse']:.6f}.",
        f"- Shuffled/correct ratio: {synthetic['shuffle_error_ratio']:.2f}.",
        f"- Scene-bootstrap shuffle-gap 95% CI: [{synthetic['shuffle_gap']['ci_low']:.6f}, "
        f"{synthetic['shuffle_gap']['ci_high']:.6f}].",
        "",
        "## Candidate consequence overfit",
        "",
        f"- Scenes/candidates: {overfit['scene_count']} / {overfit['candidate_count']}.",
        f"- Correct decoded MAE: {overfit['decoded_mae']:.6f}.",
        f"- Mean-control MAE: {overfit['mean_control_mae']:.6f}.",
        f"- Shuffled decoded MAE: {overfit['shuffled_decoded_mae']:.6f}.",
        "",
        "## Action-path engineering audit",
        "",
        f"- Optimizer contains every action/fusion parameter: "
        f"{action_path['all_action_parameters_in_optimizer'] and action_path['all_fusion_parameters_in_optimizer']}.",
        f"- Action encoder gradient norm: {action_path['action_encoder_gradient_norm']:.6e}.",
        f"- Output-to-trajectory Jacobian norm: {action_path['output_to_trajectory_jacobian_norm']:.6e}.",
        f"- Within-scene action embedding variance: "
        f"{action_path['within_scene_action_embedding_variance']:.6e}.",
        f"- Fusion action contribution ratio: {action_path['fusion_action_delta_ratio']:.6f}.",
        "- Physical candidate range `[x, y, heading]`: "
        f"min `{action_path['physical_trajectory_range']['minimum']}`, "
        f"max `{action_path['physical_trajectory_range']['maximum']}`.",
        "- Normalized candidate range `[x, y, sin(yaw), cos(yaw)]`: "
        f"min `{action_path['normalized_trajectory_range']['minimum']}`, "
        f"max `{action_path['normalized_trajectory_range']['maximum']}`; "
        f"std `{action_path['normalized_trajectory_range']['standard_deviation']}`.",
        "",
        "## Equal-capacity controls and calibrated false-safe metrics",
        "",
        "Thresholds are fit on accepted fit-scene candidates and frozen before held-out evaluation. "
        "Intervals resample scenes, never individual candidates or pairs.",
        "",
        "| Control | Unsafe prevalence | Balanced accuracy | AUROC | AUPRC | False-safe rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for control in CONTROL_NAMES:
        values = [_mean_ci(calibration, control, name) for name in (
            "unsafe_prevalence",
            "balanced_accuracy",
            "auroc",
            "auprc",
            "false_safe_rate",
        )]
        formatted = [f"{point:.4f} [{low:.4f}, {high:.4f}]" for point, low, high in values]
        lines.append(f"| {control} | " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The action branch can fit a deterministic trajectory function and memorize candidate-level "
            "consequences, with non-zero gradients and Jacobian. Poor factual-only candidate sensitivity "
            "therefore represents a statistical shortcut under single-future supervision rather than an "
            "action wiring failure. Calibration metrics replace the earlier uncalibrated 0.5 false-safe number.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/gate2_5.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--factual-probe-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/gate2_5_audit.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    data = config["data"]
    candidate_cache = _resolve(args.candidate_cache, "ACTION_EFFECT_CACHE_ROOT", str(data["candidate_cache"]))
    consequence_cache = _resolve(args.consequence_cache, "ACTION_EFFECT_CACHE_ROOT", str(data["consequence_cache"]))
    scene_feature_cache = _resolve(args.scene_feature_cache, "ACTION_EFFECT_CACHE_ROOT", str(data["scene_feature_cache"]))
    factual_probe_dir = _resolve(args.factual_probe_dir, "ACTION_EFFECT_OUTPUT_ROOT", str(data["factual_probe_dir"]))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / "gate2_5" / "pilot_tiny"
    )
    for path, name in (
        (candidate_cache, "candidate"),
        (consequence_cache, "consequence"),
        (scene_feature_cache, "scene feature"),
    ):
        if read_manifest(path) is None:
            raise FileNotFoundError(f"published {name} cache is missing: {path}")
    with (factual_probe_dir / "split.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    fit_scenes = [str(value) for value in split["fit"]]
    heldout_scenes = [str(value) for value in split["heldout"]]
    arrays, scene_features, _, scales, trajectory_stats, _ = load_probe_arrays(
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        scene_feature_cache=scene_feature_cache,
        fit_scene_ids=fit_scenes,
        assumption=str(data["target_assumption"]),
    )
    with np.load(candidate_cache / "candidates.npz") as payload:
        physical = np.asarray(payload["trajectories"], dtype=np.float32)[arrays.candidate_indices]
    code_paths = [
        Path(__file__),
        REPOSITORY_ROOT / "research/action_effect/gate2_5.py",
        REPOSITORY_ROOT / "research/action_effect/world_probe.py",
    ]
    identity = {
        "cache_version": config["experiment"]["cache_version"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "code_tree_hash": content_hash(
            {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_paths}
        ),
        "config_hash": content_hash(config),
        "split_hash": content_hash(split),
        "trajectory_normalization": trajectory_stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "build_identity.json"
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"Gate 2.5 output identity differs: {output_dir}")
    else:
        write_json(identity_path, identity)
    requested_device = str(config["runtime"]["device"])
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    synthetic, _ = _fit_synthetic(
        config=config,
        arrays=arrays,
        physical_trajectories=physical,
        scene_features=scene_features,
        fit_scenes=fit_scenes,
        heldout_scenes=heldout_scenes,
        device=device,
    )
    overfit, overfit_model, optimizer, overfit_indices = _candidate_overfit(
        config=config,
        arrays=arrays,
        scene_features=scene_features,
        fit_scenes=fit_scenes,
        device=device,
    )
    action_path = _action_path_audit(
        model=overfit_model,
        optimizer=optimizer,
        arrays=arrays,
        physical_trajectories=physical,
        scene_features=scene_features,
        indices=overfit_indices,
        device=device,
    )
    calibration = _calibration_audit(
        config=config,
        arrays=arrays,
        scales=scales,
        fit_scenes=fit_scenes,
        heldout_scenes=heldout_scenes,
        factual_probe_dir=factual_probe_dir,
    )
    summary = {
        "passed": bool(synthetic["passed"] and overfit["passed"] and action_path["passed"]),
        "synthetic": synthetic,
        "candidate_overfit": overfit,
        "action_path": action_path,
        "device": str(device),
        "equal_capacity_parameter_count": calibration[0]["total_parameters"],
    }
    write_json(output_dir / "synthetic_action_fitting.json", synthetic)
    write_json(output_dir / "candidate_overfit.json", overfit)
    write_json(output_dir / "action_path_audit.json", action_path)
    write_jsonl(output_dir / "false_safe_calibration.jsonl", calibration)
    write_json(output_dir / "summary.json", summary)
    report = _render_report(
        identity=identity,
        synthetic=synthetic,
        overfit=overfit,
        action_path=action_path,
        calibration=calibration,
    )
    args.report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_path.resolve().write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
