#!/usr/bin/env python3
"""Train the unified pilot_small Phase-6 world-probe ablation matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
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
from research.action_effect.effect_tube import BINARY_EFFECT_CHANNELS  # noqa: E402
from research.action_effect.losses import (  # noqa: E402
    ConsequencePredictionLoss,
    EffectTubeLoss,
    equal_scene_mean,
    normalized_pair_loss,
)
from research.action_effect.phase6_data import (  # noqa: E402
    group_indices_by_scene,
    group_pairs_by_scene,
    sample_balanced_pairs,
)
from research.action_effect.probe_data import (  # noqa: E402
    HARD_TARGET_FIELDS,
    SOFT_TARGET_FIELDS,
    iter_jsonl,
    load_probe_arrays,
    load_structured_targets,
    scales_to_json,
)
from research.action_effect.world_probe import ActionEffectWorldProbe, count_parameters  # noqa: E402


METHODS = (
    "factual_only",
    "multi_candidate_absolute",
    "global_separation",
    "aee",
    "confidence_aee",
    "scene_only_control",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _cache(explicit: Path | None, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _root("ACTION_EFFECT_CACHE_ROOT") / relative


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model(config: Mapping[str, Any], method: str) -> ActionEffectWorldProbe:
    probe = config["probe"]
    return ActionEffectWorldProbe(
        scene_input_dim=int(probe["scene_input_dim"]),
        consequence_dim=len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS),
        latent_dim=int(probe["latent_dim"]),
        trajectory_input_dim=int(probe["trajectory_input_dim"]),
        trajectory_token_dim=int(probe["trajectory_token_dim"]),
        dropout=float(probe["dropout"]),
        structured_future_shape=tuple(int(value) for value in probe["structured_future_shape"]),
        input_mode="scene_only" if method == "scene_only_control" else "scene_action",
    )


def _sample_absolute_indices(
    *,
    selected_scenes: Sequence[str],
    candidates_by_scene: Mapping[str, np.ndarray],
    anchors_by_scene: Mapping[str, np.ndarray],
    factual_only: bool,
    candidates_per_scene: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    scene_position: list[int] = []
    for position, scene in enumerate(selected_scenes):
        options = anchors_by_scene[str(scene)] if factual_only else candidates_by_scene[str(scene)]
        if factual_only:
            selected = options[:1]
        else:
            selected = rng.choice(
                options,
                size=candidates_per_scene,
                replace=len(options) < candidates_per_scene,
            )
        indices.extend(int(value) for value in selected)
        scene_position.extend([position] * len(selected))
    return np.asarray(indices, dtype=np.int64), np.asarray(scene_position, dtype=np.int64)


def _forward_indices(
    model: ActionEffectWorldProbe,
    indices: np.ndarray,
    *,
    arrays: Any,
    scene_features: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    return model(
        torch.from_numpy(scene_features[arrays.scene_feature_indices[indices]]).to(device),
        torch.from_numpy(arrays.trajectories[indices]).to(device),
    )


def _binary_channel_means(
    target: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute train-only binary prevalences without materializing the full tube subset."""

    if not len(indices):
        raise ValueError("binary effect prevalence requires at least one training candidate")
    total = np.zeros(len(BINARY_EFFECT_CHANNELS), dtype=np.float64)
    element_count = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        values = target[batch][:, :, BINARY_EFFECT_CHANNELS].astype(np.float32)
        total += values.sum(axis=(0, 1, 3, 4), dtype=np.float64)
        element_count += int(values.shape[0] * values.shape[1] * values.shape[3] * values.shape[4])
    return total / element_count


@torch.no_grad()
def _validation_loss(
    model: ActionEffectWorldProbe,
    indices: np.ndarray,
    *,
    arrays: Any,
    scene_features: np.ndarray,
    effect_target: np.ndarray,
    consequence_loss: ConsequencePredictionLoss,
    effect_loss: EffectTubeLoss,
    device: torch.device,
    batch_size: int,
    lambda_consequence: float,
    lambda_effect: float,
) -> float:
    model.eval()
    total = count = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        output = _forward_indices(
            model, batch, arrays=arrays, scene_features=scene_features, device=device
        )
        consequence = output["consequence_prediction"]
        structured = output["structured_future_prediction"]
        assert isinstance(consequence, torch.Tensor) and isinstance(structured, torch.Tensor)
        con = consequence_loss(
            consequence, torch.from_numpy(arrays.targets[batch]).to(device)
        )["total"]
        effect = effect_loss(
            structured,
            torch.from_numpy(effect_target[batch].astype(np.float32)).to(device),
        )["total"]
        value = lambda_consequence * con + lambda_effect * effect
        total += float(value.item()) * len(batch)
        count += len(batch)
    return total / max(count, 1)


def _train_one(
    *,
    config: dict[str, Any],
    method: str,
    seed: int,
    arrays: Any,
    scene_features: np.ndarray,
    effect_target: np.ndarray,
    train_scenes: list[str],
    validation_indices: np.ndarray,
    candidates_by_scene: Mapping[str, np.ndarray],
    anchors_by_scene: Mapping[str, np.ndarray],
    pair_groups: Mapping[str, Mapping[str, Sequence[tuple[int, int, Mapping[str, Any]]]]],
    positive_weight: np.ndarray,
    output_dir: Path,
    steps_override: int | None,
) -> dict[str, Any]:
    training = config["training"]
    loss_cfg = config["loss"]
    requested = str(training["device"])
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    _seed(seed)
    model = _model(config, method).to(device)
    consequence_loss = ConsequencePredictionLoss(len(HARD_TARGET_FIELDS))
    effect_weights = loss_cfg["effect_weights"]
    effect_loss = EffectTubeLoss(
        torch.from_numpy(positive_weight),
        **{f"{name}_weight": float(value) for name, value in effect_weights.items()},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(steps_override if steps_override is not None else training["steps"])
    batch_scene_count = int(training["batch_scene_count"])
    candidates_per_scene = int(training["candidates_per_scene"])
    lambda_consequence = float(loss_cfg["lambda_consequence"])
    lambda_effect = float(loss_cfg["lambda_effect"])
    lambda_pair = float(loss_cfg["lambda_pair"])
    absolute_rng = np.random.default_rng(seed)
    pair_rng = np.random.default_rng(seed + 104729)
    scene_array = np.asarray(train_scenes, dtype=str)
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        model.train()
        selected_scenes = absolute_rng.choice(
            scene_array,
            size=batch_scene_count,
            replace=len(scene_array) < batch_scene_count,
        ).tolist()
        absolute_indices, absolute_scene = _sample_absolute_indices(
            selected_scenes=selected_scenes,
            candidates_by_scene=candidates_by_scene,
            anchors_by_scene=anchors_by_scene,
            factual_only=method == "factual_only",
            candidates_per_scene=candidates_per_scene,
            rng=absolute_rng,
        )
        output = _forward_indices(
            model,
            absolute_indices,
            arrays=arrays,
            scene_features=scene_features,
            device=device,
        )
        consequence_prediction = output["consequence_prediction"]
        structured_prediction = output["structured_future_prediction"]
        effect_latent = output["effect_latent"]
        assert isinstance(consequence_prediction, torch.Tensor)
        assert isinstance(structured_prediction, torch.Tensor)
        assert isinstance(effect_latent, torch.Tensor)
        consequence_values = consequence_loss(
            consequence_prediction,
            torch.from_numpy(arrays.targets[absolute_indices]).to(device),
            reduction="none",
        )["total"]
        effect_values = effect_loss(
            structured_prediction,
            torch.from_numpy(effect_target[absolute_indices].astype(np.float32)).to(device),
            reduction="none",
        )["total"]
        scene_tensor = torch.from_numpy(absolute_scene).to(device)
        absolute_loss = equal_scene_mean(
            lambda_consequence * consequence_values + lambda_effect * effect_values,
            scene_tensor,
            scene_count=len(selected_scenes),
        )
        pair_loss = absolute_loss.new_zeros(())
        pair_distance = absolute_loss.new_zeros(())
        pair_count = 0
        if method in {"global_separation", "aee", "confidence_aee"}:
            # Global separation draws uniformly from every geometrically
            # distinct pair. AEE instead balances its semantic pair types;
            # both paths retain equal per-scene aggregation below.
            sampled = sample_balanced_pairs(
                selected_scenes,
                pair_groups,
                rng=pair_rng,
                categories=(
                    ("geometrically_distinct",)
                    if method == "global_separation"
                    else (
                        "confidence_effect_equivalent",
                        "confidence_effect_divergent",
                        "confidence_safety_boundary",
                    )
                    if method == "confidence_aee"
                    else ("effect_equivalent", "effect_divergent", "safety_boundary")
                ),
                confidence_weights=config["confidence_aee"].get("pair_confidence_weights"),
            )
            if len(sampled["left"]):
                pair_indices = np.concatenate((sampled["left"], sampled["right"]))
                pair_output = _forward_indices(
                    model,
                    pair_indices,
                    arrays=arrays,
                    scene_features=scene_features,
                    device=device,
                )["effect_latent"]
                assert isinstance(pair_output, torch.Tensor)
                pair_count = len(sampled["left"])
                left_latent, right_latent = pair_output[:pair_count], pair_output[pair_count:]
                category = torch.from_numpy(sampled["category"]).to(device)
                pair_result = normalized_pair_loss(
                    left_latent,
                    right_latent,
                    equivalent=category == 0,
                    divergent=category != 0,
                    consequence_distance=torch.from_numpy(sampled["consequence_distance"]).to(device),
                    confidence_weight=(
                        torch.from_numpy(sampled["confidence_weight"]).to(device)
                        if method == "confidence_aee"
                        else None
                    ),
                    base_margin=(
                        float(loss_cfg["global_margin"])
                        if method == "global_separation"
                        else float(loss_cfg["aee_base_margin"])
                    ),
                    margin_scale=(
                        0.0
                        if method == "global_separation"
                        else float(loss_cfg["aee_margin_scale"])
                    ),
                    maximum_margin=(
                        float(loss_cfg["global_margin"])
                        if method == "global_separation"
                        else float(loss_cfg["aee_maximum_margin"])
                    ),
                    global_separation=method == "global_separation",
                )
                pair_scene = torch.from_numpy(sampled["scene"]).to(device)
                active = pair_result["active"]
                pair_loss = equal_scene_mean(
                    pair_result["loss"][active],
                    pair_scene[active],
                    scene_count=len(selected_scenes),
                )
                pair_distance = pair_result["distance"].mean()
        total = absolute_loss + lambda_pair * pair_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite Phase-6 gradient at step {step}")
        optimizer.step()
        if step == 1 or step % int(training["log_interval"]) == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(total.item()),
                "absolute_loss": float(absolute_loss.item()),
                "pair_loss": float(pair_loss.item()),
                "pair_distance": float(pair_distance.item()),
                "pair_count": pair_count,
                "gradient_norm": float(gradient_norm.item()),
                "latent_norm": float(torch.linalg.vector_norm(effect_latent, dim=1).mean().item()),
                "latent_variance": float(effect_latent.float().var(dim=0, unbiased=False).mean().item()),
            }
            if step % int(training["validation_interval"]) == 0 or step == steps:
                row["validation_absolute_loss"] = _validation_loss(
                    model,
                    validation_indices,
                    arrays=arrays,
                    scene_features=scene_features,
                    effect_target=effect_target,
                    consequence_loss=consequence_loss,
                    effect_loss=effect_loss,
                    device=device,
                    batch_size=int(training["evaluation_batch_size"]),
                    lambda_consequence=lambda_consequence,
                    lambda_effect=lambda_effect,
                )
            history.append(row)
            print(json.dumps({"method": method, "seed": seed, **row}, sort_keys=True), flush=True)
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "method": method,
        "seed": seed,
        "steps": steps,
        "positive_weight": positive_weight.tolist(),
    }
    _atomic_torch_save(output_dir / "probe.pt", checkpoint)
    write_jsonl(output_dir / "history.jsonl", history)
    result = {
        "method": method,
        "seed": seed,
        "steps": steps,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "device": str(device),
        "final": history[-1],
    }
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/phase6.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--effect-tube-cache", type=Path)
    parser.add_argument("--split-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    data = config["data"]
    paths = {
        "candidate": _cache(args.candidate_cache, str(data["candidate_cache"])),
        "consequence": _cache(args.consequence_cache, str(data["consequence_cache"])),
        "pair": _cache(args.pair_cache, str(data["pair_cache"])),
        "scene_feature": _cache(args.scene_feature_cache, str(data["scene_feature_cache"])),
        "effect_tube": _cache(args.effect_tube_cache, str(data["effect_tube_cache"])),
        "split": _cache(args.split_cache, str(data["split_cache"])),
    }
    for name, path in paths.items():
        if read_manifest(path) is None:
            raise FileNotFoundError(f"published {name} cache is missing: {path}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / "phase6/pilot_small_v1"
    )
    with (paths["split"] / "split.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    train_scenes = [str(value) for value in split["train"]]
    validation_scenes = [str(value) for value in split["validation"]]
    arrays, scene_features, _, scales, trajectory_stats, _ = load_probe_arrays(
        candidate_cache=paths["candidate"],
        consequence_cache=paths["consequence"],
        scene_feature_cache=paths["scene_feature"],
        fit_scene_ids=train_scenes,
        assumption=str(data["target_assumption"]),
    )
    effect_target, effect_valid = load_structured_targets(paths["effect_tube"], arrays)
    metadata = list(iter_jsonl(paths["candidate"] / "metadata.jsonl"))
    if len(metadata) != len(arrays.scene_ids):
        raise RuntimeError("candidate metadata does not align with Phase-6 arrays")
    perturbation = np.asarray([str(row["perturbation_type"]) for row in metadata], dtype=str)
    heldout_family = str(split["held_out_perturbation_family"])
    train_mask = (
        arrays.accepted
        & effect_valid
        & np.isin(arrays.scene_ids, train_scenes)
        & (perturbation != heldout_family)
    )
    train_anchor_mask = train_mask & arrays.anchor
    candidates_by_scene = group_indices_by_scene(arrays.scene_ids, train_mask)
    anchors_by_scene = group_indices_by_scene(arrays.scene_ids, train_anchor_mask)
    missing_candidates = sorted(set(train_scenes) - set(candidates_by_scene))
    missing_anchors = sorted(set(train_scenes) - set(anchors_by_scene))
    if missing_candidates or missing_anchors:
        raise RuntimeError(
            f"Phase-6 train scenes lack candidates/anchors: {missing_candidates[:3]} / {missing_anchors[:3]}"
        )
    pair_rows = list(iter_jsonl(paths["pair"] / "pairs.jsonl"))
    candidate_lookup = {candidate: index for index, candidate in enumerate(arrays.candidate_ids)}
    pair_groups = group_pairs_by_scene(
        pair_rows,
        candidate_lookup,
        allowed_candidate_mask=train_mask,
    )
    validation_mask = arrays.accepted & effect_valid & np.isin(arrays.scene_ids, validation_scenes)
    validation_by_scene = group_indices_by_scene(arrays.scene_ids, validation_mask)
    validation_indices = np.concatenate(
        [indices[: min(2, len(indices))] for indices in validation_by_scene.values()]
    )
    train_indices = np.flatnonzero(train_mask)
    binary_mean = _binary_channel_means(effect_target, train_indices)
    cap = float(config["training"]["binary_positive_weight_cap"])
    positive_weight = np.clip(
        (1.0 - binary_mean) / np.clip(binary_mean, 1.0e-6, None), 1.0, cap
    ).astype(np.float32)
    methods = args.methods or [str(value) for value in config["methods"]]
    if "confidence_aee" in methods and not bool(config["confidence_aee"]["enabled"]):
        raise ValueError("confidence_aee was requested although the disagreement gate disabled it")
    seeds = args.seeds or [int(value) for value in config["experiment"]["seeds"]]
    code_files = [
        Path(__file__),
        REPOSITORY_ROOT / "research/action_effect/world_probe.py",
        REPOSITORY_ROOT / "research/action_effect/losses.py",
        REPOSITORY_ROOT / "research/action_effect/phase6_data.py",
    ]
    identity = {
        "cache_version": config["experiment"]["cache_version"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "code_tree_hash": content_hash(
            {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_files}
        ),
        "config_hash": content_hash(config),
        "source_manifests": {
            name: read_manifest(path).compatibility_identity()  # type: ignore[union-attr]
            for name, path in paths.items()
        },
        "split_hash": content_hash(split),
        "methods": methods,
        "seeds": seeds,
        "steps_override": args.steps,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "build_identity.json"
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"Phase-6 output identity differs: {output_dir}")
    else:
        write_json(identity_path, identity)
    write_json(output_dir / "split.json", split)
    write_json(output_dir / "probe_scales.json", scales_to_json(scales))
    write_json(output_dir / "trajectory_normalization.json", trajectory_stats)
    write_json(
        output_dir / "data_summary.json",
        {
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "train_candidate_count": int(train_mask.sum()),
            "held_out_perturbation_family": heldout_family,
            "binary_positive_weight": positive_weight.tolist(),
            "effect_target_shape": list(effect_target.shape),
            "pair_scene_count": len(pair_groups),
        },
    )
    results: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            run_dir = output_dir / method / f"seed_{seed}"
            if (run_dir / "result.json").is_file() and (run_dir / "probe.pt").is_file():
                with (run_dir / "result.json").open("r", encoding="utf-8") as stream:
                    results.append(json.load(stream))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            results.append(
                _train_one(
                    config=config,
                    method=method,
                    seed=seed,
                    arrays=arrays,
                    scene_features=scene_features,
                    effect_target=effect_target,
                    train_scenes=train_scenes,
                    validation_indices=validation_indices,
                    candidates_by_scene=candidates_by_scene,
                    anchors_by_scene=anchors_by_scene,
                    pair_groups=pair_groups,
                    positive_weight=positive_weight,
                    output_dir=run_dir,
                    steps_override=args.steps,
                )
            )
    write_jsonl(output_dir / "training_results.jsonl", results)
    write_json(output_dir / "complete.json", {"run_count": len(results), "identity": identity})
    print(json.dumps({"output_dir": str(output_dir), "run_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
